"""
Scenario records: obligation + evidence + the label the verifiers never see.

This module exists to enforce one boundary. A `Scenario` holds ground truth
and belongs to the eval harness. A `VerifierInput` holds the obligation and
the envelope and is all any verifier is ever handed. If ground truth crosses
that line, every metric in the project is measuring the leak rather than the
system, so the boundary is asserted rather than trusted - see
`assert_no_leakage`.

The subtle case is the requested SKU. It may legitimately appear inside the
agent's self-report: an agent that ordered maida and reports "bought the atta
you asked for" is producing real evidence, and it is exactly the case the
admissibility floor exists to handle. What it must never appear in is evidence
at or above the performance floor, because a deterministic verifier could then
read the answer off a merchant record and the semantic layer would never be
exercised. That narrower rule is what gets checked.

Splits are assigned by hash of the scenario id, not by position, so that
regenerating with a different n does not reshuffle which cases are held out.
A held-out set that moves when the dataset size changes is not held out.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from bench.taxonomy import (
    HELDOUT_FRACTION,
    PERFORMANCE_FLOOR,
    FaultClass,
    GateVerdict,
    Party,
    TAXONOMY,
)
from schemas.evidence import EvidenceEnvelope, EvidenceKind
from schemas.obligation import Obligation, format_paise


SCHEMA_VERSION = "scenario/1.0"

Split = Literal["train", "heldout"]


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


class GroundTruth(BaseModel):
    """
    The label, and everything needed to justify it.

    `rationale` is not decoration. It is what makes the dataset auditable: a
    reviewer can open any scenario file and check in one line whether the label
    is defensible, without reverse-engineering the generator.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fault_class: FaultClass
    liable_party: Party

    # Reserved for compound attribution - an agent that ordered wrongly AND a
    # merchant that then substituted. Left unpopulated in the single-fault
    # benchmark. Present in the schema so that adding it later is a generator
    # change rather than a migration.
    secondary_fault: Optional[FaultClass] = None
    secondary_party: Optional[Party] = None

    # Canonical product state. Never handed to a verifier.
    requested_sku: str          # what the user meant
    ordered_sku: str            # what the agent put in the cart
    delivered_sku: str          # what the merchant actually shipped

    # Difficulty and evidence properties.
    is_hard_pair: bool = False          # same brand, same pack, within Rs 20
    agent_report_truthful: bool = True  # False when the self-report contradicts record

    # What a correct pre-debit gate should have done.
    expected_gate_verdict: GateVerdict
    gate_detectable: bool

    # Exposure in paise if this goes undetected. Drives the rupee-denominated
    # false-positive and false-negative costs in the eval.
    loss_paise: int = Field(ge=0)

    rationale: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# Verifier input - the redacted view
# ---------------------------------------------------------------------------


class VerifierInput(BaseModel):
    """Everything a verifier is allowed to see. No labels, no canonical SKUs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str
    obligation: Obligation
    envelope: EvidenceEnvelope


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    scenario_id: str
    seed: int
    split: Split

    obligation: Obligation
    envelope: EvidenceEnvelope
    truth: GroundTruth

    def to_verifier_input(self) -> VerifierInput:
        return VerifierInput(
            scenario_id=self.scenario_id,
            obligation=self.obligation,
            envelope=self.envelope,
        )

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.scenario_id}.json"
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> "Scenario":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------


def assign_split(scenario_id: str, heldout_fraction: float = HELDOUT_FRACTION) -> Split:
    """
    Deterministic split by hash of the id.

    Position-based splitting means the held-out set changes whenever the
    dataset size changes, which quietly invalidates any comparison across runs.
    Hashing the id fixes a scenario's split for its lifetime.
    """
    digest = hashlib.sha256(scenario_id.encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "heldout" if position < heldout_fraction else "train"


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------


class LeakageError(AssertionError):
    """Ground truth reached the verifier input."""


def _leaf_strings(node: object) -> Iterator[str]:
    """Every string leaf in a parsed JSON structure."""
    if isinstance(node, dict):
        for value in node.values():
            yield from _leaf_strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _leaf_strings(value)
    elif isinstance(node, str):
        yield node


def assert_no_leakage(scenario: Scenario) -> None:
    """
    Fail if the answer is visible in what the verifiers receive.

    Three checks:

      1. No ground-truth enum value appears as an exact leaf in the verifier
         input. Exact rather than substring: `Party.AGENT` serialises to the
         string "AGENT", and the envelope legitimately carries "AGENT_CART" as
         an evidence kind. Substring matching flags that as a leak and it is
         not one. Matching parsed leaves for equality removes the collision
         while still catching a real leak, which would serialise as its own
         leaf.

      2. The rationale does not appear anywhere in the blob. Long and
         distinctive enough that substring matching is safe here.

      3. When the agent ordered something other than what was requested, the
         requested SKU must not appear in any evidence item at or above the
         performance floor. It may appear in a deceptive self-report, which is
         SELF_REPORT class and legitimate evidence. It may not appear in a
         merchant record or a PSP receipt, because a deterministic verifier
         would then read the answer directly and the semantic layer would never
         be tested.
    """
    vi = scenario.to_verifier_input()
    blob = vi.model_dump_json()
    leaves = set(_leaf_strings(json.loads(blob)))

    for token in (scenario.truth.fault_class.value, scenario.truth.liable_party.value):
        if token in leaves:
            raise LeakageError(
                f"{scenario.scenario_id}: ground-truth value {token!r} present "
                f"as a leaf in verifier input"
            )

    if scenario.truth.rationale in blob:
        raise LeakageError(
            f"{scenario.scenario_id}: ground-truth rationale present in verifier input"
        )

    requested = scenario.truth.requested_sku
    if requested == scenario.truth.ordered_sku:
        return  # nothing to hide; the agent ordered correctly

    for item in scenario.envelope.items:
        if item.evidence_class < PERFORMANCE_FLOOR:
            continue  # weak evidence may name it; that is the deception case
        if item.kind is EvidenceKind.CATALOG_SNAPSHOT:
            # A catalogue listing records what the merchant was offering, not
            # what the user asked for. A browse trace naturally includes the
            # item the agent went looking for, and a trace that excluded it
            # would be unrepresentative.
            #
            # This exemption is deliberately not load-bearing. Builders that
            # emit snapshots emit several, so no single listing points at the
            # answer, and INTENT_MISMATCH - the class this guard exists to
            # protect - emits none at all.
            continue
        if requested in item.model_dump_json():
            raise LeakageError(
                f"{scenario.scenario_id}: requested sku {requested!r} appears in "
                f"{item.item_id} at class {item.evidence_class.name}, which is at or "
                f"above the performance floor"
            )


# ---------------------------------------------------------------------------
# Scenario set
# ---------------------------------------------------------------------------


class ScenarioSet:
    """A generated dataset, with the stats the README needs."""

    def __init__(self, scenarios: Iterable[Scenario]) -> None:
        self.scenarios: list[Scenario] = list(scenarios)

    def __len__(self) -> int:
        return len(self.scenarios)

    def split(self, split: Split) -> list[Scenario]:
        return [s for s in self.scenarios if s.split == split]

    def counts(self) -> dict[FaultClass, int]:
        out = {f: 0 for f in FaultClass}
        for s in self.scenarios:
            out[s.truth.fault_class] += 1
        return out

    def validate(self) -> None:
        """Every integrity property the dataset must hold, checked at once."""
        ids = [s.scenario_id for s in self.scenarios]
        if len(set(ids)) != len(ids):
            raise AssertionError("duplicate scenario ids")

        for s in self.scenarios:
            assert_no_leakage(s)

            brk = s.envelope.verify_chain()
            if brk is not None:
                raise AssertionError(
                    f"{s.scenario_id}: broken chain at seq {brk.seq}: {brk.reason}"
                )

            if not s.obligation.hash_is_valid():
                raise AssertionError(f"{s.scenario_id}: obligation hash does not verify")

            if s.envelope.obligation_hash != s.obligation.content_hash:
                raise AssertionError(f"{s.scenario_id}: envelope not anchored to its obligation")

            spec = TAXONOMY[s.truth.fault_class]
            if s.truth.liable_party is not spec.party:
                raise AssertionError(
                    f"{s.scenario_id}: {s.truth.fault_class.value} must be attributed to "
                    f"{spec.party.value}, got {s.truth.liable_party.value}"
                )
            if s.truth.gate_detectable != spec.gate_detectable:
                raise AssertionError(f"{s.scenario_id}: gate_detectable disagrees with taxonomy")
            if s.truth.expected_gate_verdict is not spec.expected_gate_verdict:
                raise AssertionError(
                    f"{s.scenario_id}: expected gate verdict disagrees with taxonomy"
                )

            if s.truth.fault_class is FaultClass.NO_FAULT and s.truth.loss_paise != 0:
                raise AssertionError(f"{s.scenario_id}: NO_FAULT carries non-zero loss")

            if s.split != assign_split(s.scenario_id):
                raise AssertionError(f"{s.scenario_id}: split does not match its id hash")

    def summary(self) -> str:
        counts = self.counts()
        train = len(self.split("train"))
        heldout = len(self.split("heldout"))
        hard = sum(1 for s in self.scenarios if s.truth.is_hard_pair)
        lying = sum(1 for s in self.scenarios if not s.truth.agent_report_truthful)
        blind = sum(1 for s in self.scenarios if not s.truth.gate_detectable)
        exposure = sum(s.truth.loss_paise for s in self.scenarios)

        lines = [
            f"scenarios    {len(self)}  (train {train} / heldout {heldout})",
            "",
            f"{'class':<24}{'n':>5}{'share':>9}",
            "-" * 38,
        ]
        for fault, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            if n:
                lines.append(f"{fault.value:<24}{n:>5}{n / len(self):>8.1%}")
        lines += [
            "-" * 38,
            f"hard confusable pairs    {hard:>5}",
            f"deceptive self-reports   {lying:>5}",
            f"invisible to the gate    {blind:>5}  ({blind / len(self):.1%})",
            f"total exposure           {format_paise(exposure)}",
        ]
        return "\n".join(lines)


def load_dir(directory: Path) -> ScenarioSet:
    paths = sorted(Path(directory).glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no scenarios in {directory} - run `make gen` first")
    return ScenarioSet(Scenario.read(p) for p in paths)


# ---------------------------------------------------------------------------
# Self-check: one hand-built INTENT_MISMATCH scenario, end to end
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from datetime import datetime, timezone

    from agent.catalog import get, is_hard_pair
    from schemas.evidence import (
        AgentCart,
        AgentSelfReport,
        CartLine,
        MerchantFulfilment,
        MerchantOrder,
        PspPayment,
    )
    from schemas.obligation import (
        HardConstraints,
        Intent,
        ReserveBlock,
        paise,
    )

    now = datetime(2026, 9, 2, 19, 0, tzinfo=timezone.utc)
    requested = get("amul-milk-toned-1000")        # user wanted toned
    ordered = get("amul-milk-full_cream-1000")     # agent bought full cream

    obligation = Obligation(
        obligation_id="obl_selfcheck",
        created_at=now,
        expires_at=datetime(2026, 9, 2, 23, 59, tzinfo=timezone.utc),
        user_id="usr_selfcheck",
        agent_id="agt_selfcheck",
        block=ReserveBlock(
            block_id="blk_selfcheck",
            blocked_paise=paise(2000),
            blocked_at=now,
            expires_at=datetime(2026, 10, 2, 19, 0, tzinfo=timezone.utc),
        ),
        hard=HardConstraints(
            category="milk",
            max_unit_price_paise=paise(90),
            max_total_paise=paise(90),
            quantity=1,
            merchant_allowlist=("zepto",),
            brand="amul",
            pack_size_g=1000,
            # variant uncaptured on purpose - this is what makes it semantic
        ),
        intent=Intent(
            text="get me a litre of Amul toned milk from Zepto",
            uncaptured_attributes=("variant",),
        ),
    ).with_hash()

    env = EvidenceEnvelope(envelope_id="env_selfcheck", obligation_hash=obligation.content_hash)
    line = CartLine(sku=ordered.sku, quantity=1, unit_price_paise=ordered.unit_price_paise)

    env.append(AgentCart(merchant_id="zepto", lines=(line,)), "agt_selfcheck", now)
    env.append(
        MerchantOrder(merchant_id="zepto", merchant_order_id="zep_1", lines=(line,)),
        "zepto",
        now,
    )
    env.append(MerchantFulfilment(merchant_order_id="zep_1", lines=(line,)), "zepto", now)
    env.append(
        PspPayment(
            payment_id="pay_test_1",
            order_id="ord_test_1",
            amount_paise=ordered.unit_price_paise,
            block_id="blk_selfcheck",
        ),
        "razorpay",
        now,
    )
    # The agent claims it bought what was asked. It did not. SELF_REPORT class,
    # so naming the requested sku here is legitimate evidence, not leakage.
    env.append(
        AgentSelfReport(text="Ordered your Amul toned milk.", claimed_sku=requested.sku),
        "agt_selfcheck",
        now,
    )

    scenario = Scenario(
        scenario_id="scn_selfcheck_0001",
        seed=42,
        split=assign_split("scn_selfcheck_0001"),
        obligation=obligation,
        envelope=env,
        truth=GroundTruth(
            fault_class=FaultClass.INTENT_MISMATCH,
            liable_party=Party.AGENT,
            requested_sku=requested.sku,
            ordered_sku=ordered.sku,
            delivered_sku=ordered.sku,
            is_hard_pair=is_hard_pair(requested, ordered),
            agent_report_truthful=False,
            expected_gate_verdict=TAXONOMY[FaultClass.INTENT_MISMATCH].expected_gate_verdict,
            gate_detectable=TAXONOMY[FaultClass.INTENT_MISMATCH].gate_detectable,
            loss_paise=ordered.unit_price_paise,
            rationale=(
                "User asked for toned milk; agent ordered full cream. Same brand, "
                "same pack, within the price ceiling, at an allowed merchant. No "
                "hard constraint was violated because the compiler left variant "
                "uncaptured, so only the semantic verifier can see this."
            ),
        ),
    )

    ss = ScenarioSet([scenario])
    ss.validate()

    # The guard must actually bite. Move the requested sku into a merchant
    # record - class MERCHANT_RECORD, at the floor - and confirm it is caught.
    leaky = scenario.model_copy(deep=True)
    leaky.envelope.append(
        MerchantOrder(
            merchant_id="zepto",
            merchant_order_id="zep_leak",
            lines=(
                CartLine(
                    sku=requested.sku,
                    quantity=1,
                    unit_price_paise=requested.unit_price_paise,
                ),
            ),
        ),
        "zepto",
        now,
    )
    try:
        assert_no_leakage(leaky)
    except LeakageError as e:
        caught = str(e)
    else:
        raise AssertionError("leakage guard failed to fire")

    print(ss.summary())
    print(
        f"\nhard pair    {scenario.truth.is_hard_pair}  "
        f"(Rs {abs(requested.unit_price_paise - ordered.unit_price_paise) / 100:.2f} apart)"
    )
    print(f"split        {scenario.split}")
    print(f"\nleakage guard fired as expected:\n  {caught}")
    print("\nscenario schema ok")