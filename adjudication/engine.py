"""
The adjudicator: turns verifier verdicts into a decision.

It does four things, in order.

  1. Runs the deterministic verifiers. All of them, always. Stopping at the
     first failure would be cheaper and wrong: constraint and provenance
     disagree on injection scenarios by design, and short-circuiting would
     silently discard whichever ran second.

  2. Enforces the admissibility floor. Any FAIL whose declared basis meets
     below MERCHANT_RECORD is discarded before it can influence anything. The
     discarded verdicts are kept in the decision record, because "the LLM said
     this and we threw it out" is the interesting part, not noise to hide.

  3. Invokes the semantic verifier only if nothing deterministic decided. A
     model call costs money and latency; a verdict already established by
     comparing two integers does not need a second opinion.

  4. Resolves what survives into one fault class, one liable party, one loss
     estimate, and a citation.

A CLEARANCE IS A FINDING, NOT AN ABSENCE

The rule that took a bug to learn (FAILURES.md #007): a transaction is cleared
only when every competent role has affirmatively passed it on admissible
evidence. Silence from a role establishes nothing.

"No hard constraint was violated" is not "the purchase matched what the user
asked for". Those are different claims answered by different verifiers, and
treating the second as implied by the first meant clearing every intent failure
in the benchmark while reporting the result as accuracy.

TWO MODES

  GATE         Runs before the debit, so it sees only pre-debit evidence. The
               envelope is rebuilt without the fulfilment record, the agent's
               later self-report, or the user's dispute. A gate that could see
               those is not a gate, it is hindsight, and every number from it
               would be a lie.

               This is what produces the gate-blindness figure rather than
               asserting it: MERCHANT_SUBSTITUTION and USER_REGRET become
               undecidable here because the evidence that distinguishes them
               does not exist yet.

               The recorded debit stands in for the proposed debit. In this
               benchmark the amount does not change between proposal and
               capture, so the substitution is exact; on a live rail the gate
               would receive the proposal directly.

  ATTRIBUTION  Runs after settlement with the full envelope. Answers who is at
               fault and for how much.

CONFLICT RESOLUTION

When two verdicts survive the floor and name different fault classes, they are
resolved by subsumption: INJECTION_INDUCED subsumes CART_DRIFT and
MANDATE_BREACH, because it explains the breach rather than merely noting it.

The liability reading is the honest justification. An agent steered by hostile
content in a merchant's own catalogue is less culpable than one that drifted on
its own, so preferring the injection explanation is the conservative
attribution - it moves fault toward the party that controlled the channel.

This is a hand-specified table with one entry, not a general causal-precedence
mechanism. Saying so plainly is better than dressing it up.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from bench.scenario import VerifierInput
from bench.taxonomy import (
    PERFORMANCE_FLOOR,
    EvidenceClass,
    FaultClass,
    GateVerdict,
    Party,
    TAXONOMY,
)
from schemas.evidence import EvidenceEnvelope, EvidenceKind
from verifiers.base import Verdict, Verifier, VerifierOutput, VerifierRole
from verifiers.constraint import ConstraintVerifier
from verifiers.fulfilment import FulfilmentVerifier
from verifiers.provenance import ProvenanceVerifier
from verifiers.receipt import ReceiptVerifier
from verifiers.semantic import SemanticVerifier


class Mode(str, Enum):
    GATE = "gate"
    ATTRIBUTION = "attribution"


# Evidence that does not exist yet when a debit is being decided.
_POST_DEBIT_KINDS = frozenset({
    EvidenceKind.MERCHANT_FULFILMENT,   # the parcel has not arrived
    EvidenceKind.AGENT_SELF_REPORT,     # written after the fact
    EvidenceKind.USER_DISPUTE,          # raised hours later
})


# One entry. INJECTION_INDUCED explains a breach that CART_DRIFT and
# MANDATE_BREACH only observe.
_SUBSUMES: dict[FaultClass, frozenset[FaultClass]] = {
    FaultClass.INJECTION_INDUCED: frozenset({
        FaultClass.CART_DRIFT,
        FaultClass.MANDATE_BREACH,
    }),
}


# A clearance requires every COMPETENT role to have affirmatively passed.
# The absence of a complaint is not a finding of correctness: if the only
# verifier that can judge intent abstains, nothing here has established that
# the purchase matched intent, whatever the constraint checks say.
#
# Competence differs by mode. Before a debit, nobody can say whether the shelf
# matched the order, because nothing has shipped. Requiring an answer to an
# unanswerable question would make gate clearance impossible by construction -
# see FAILURES.md #008.
_REQUIRED_FOR_CLEARANCE: dict["Mode", frozenset[VerifierRole]] = {
    Mode.GATE: frozenset({
        VerifierRole.CONSTRAINT,
        VerifierRole.RECEIPT,
        VerifierRole.PROVENANCE,
        VerifierRole.SEMANTIC,
    }),
    Mode.ATTRIBUTION: frozenset({
        VerifierRole.CONSTRAINT,
        VerifierRole.RECEIPT,
        VerifierRole.FULFILMENT,
        VerifierRole.PROVENANCE,
        VerifierRole.SEMANTIC,
    }),
}


class Decision(BaseModel):
    """One adjudicated transaction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str
    mode: Mode

    gate_verdict: GateVerdict
    fault_class: FaultClass
    liable_party: Party
    loss_paise: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)

    basis_class: EvidenceClass
    cited: tuple[str, ...] = ()
    reason: str

    outputs: tuple[VerifierOutput, ...] = ()
    discarded: tuple[str, ...] = ()      # verifier ids zeroed by the floor
    llm_invoked: bool = False
    latency_ms: float = Field(default=0.0, ge=0.0)

    @property
    def abstained(self) -> bool:
        return self.gate_verdict is GateVerdict.ABSTAIN


def _gate_view(vi: VerifierInput) -> VerifierInput:
    """
    Rebuild the envelope with only what was knowable before the debit.

    Removing items breaks the hash chain, so the chain is relinked over the
    surviving items. Item ids are regenerated, which is why the envelope gets a
    distinct id - a citation from a gate decision refers to the gate view, not
    to the settled record, and conflating them would make audit trails wrong.
    """
    visible = [i for i in vi.envelope.items if i.kind not in _POST_DEBIT_KINDS]
    rebuilt = EvidenceEnvelope(
        envelope_id=f"{vi.envelope.envelope_id}@gate",
        obligation_hash=vi.envelope.obligation_hash,
    )
    for item in visible:
        rebuilt.append(
            item.payload,
            item.emitted_by,
            item.emitted_at,
            evidence_class=item.evidence_class,
        )
    return VerifierInput(
        scenario_id=vi.scenario_id, obligation=vi.obligation, envelope=rebuilt
    )


class Adjudicator:
    def __init__(
        self,
        *,
        semantic: Optional[Verifier] = None,
        deterministic: Optional[Sequence[Verifier]] = None,
        floor: EvidenceClass = PERFORMANCE_FLOOR,
    ) -> None:
        self.deterministic: list[Verifier] = list(
            deterministic
            if deterministic is not None
            else (
                ConstraintVerifier(),
                ReceiptVerifier(),
                FulfilmentVerifier(),
                ProvenanceVerifier(),
            )
        )
        self.semantic = semantic if semantic is not None else SemanticVerifier()
        self.floor = floor

    # -- main ---------------------------------------------------------------

    def decide(self, vi: VerifierInput, mode: Mode = Mode.ATTRIBUTION) -> Decision:
        started = time.perf_counter()
        view = _gate_view(vi) if mode is Mode.GATE else vi

        outputs = [v.run(view) for v in self.deterministic]
        admissible, discarded = self._filter(view, outputs)

        # The semantic verifier is only worth its cost when nothing cheaper has
        # already settled the matter.
        llm_invoked = False
        if not admissible:
            llm_invoked = True
            sem = self.semantic.run(view)
            outputs.append(sem)
            sem_admissible, sem_discarded = self._filter(view, [sem])
            admissible.extend(sem_admissible)
            discarded.extend(sem_discarded)

        decision = self._resolve(vi, view, mode, outputs, admissible, discarded, llm_invoked)
        elapsed = (time.perf_counter() - started) * 1000
        return decision.model_copy(update={"latency_ms": elapsed})

    # -- floor --------------------------------------------------------------

    def _filter(
        self, view: VerifierInput, outputs: Sequence[VerifierOutput]
    ) -> tuple[list[VerifierOutput], list[str]]:
        """
        Keep the FAILs whose declared basis clears the floor. Discard the rest.

        A verdict is only as strong as the weakest evidence behind it, so the
        basis class is the meet across the items the verifier said it used.
        Consulting a Razorpay receipt alongside the agent's self-report does not
        repair the self-report; the weak item is still load-bearing.
        """
        admissible: list[VerifierOutput] = []
        discarded: list[str] = []

        for out in outputs:
            if out.verdict is not Verdict.FAIL:
                continue
            try:
                basis = view.envelope.basis_class(out.basis)
            except KeyError:
                discarded.append(out.verifier_id)
                continue
            if basis >= self.floor:
                admissible.append(out)
            else:
                discarded.append(out.verifier_id)

        return admissible, discarded

    # -- resolution ---------------------------------------------------------

    def _resolve(
        self,
        vi: VerifierInput,
        view: VerifierInput,
        mode: Mode,
        outputs: list[VerifierOutput],
        admissible: list[VerifierOutput],
        discarded: list[str],
        llm_invoked: bool,
    ) -> Decision:
        def build(**kw) -> Decision:
            return Decision(
                scenario_id=vi.scenario_id,
                mode=mode,
                outputs=tuple(outputs),
                discarded=tuple(discarded),
                llm_invoked=llm_invoked,
                **kw,
            )

        if admissible:
            winner = self._pick(admissible)
            basis = view.envelope.basis_class(winner.basis)
            spec = TAXONOMY[winner.fault_class]  # type: ignore[index]

            others = [o for o in admissible if o is not winner]
            note = ""
            if others:
                note = (
                    "  [superseded: "
                    + "; ".join(
                        f"{o.verifier_id} proposed {o.fault_class.value}"  # type: ignore[union-attr]
                        for o in others
                    )
                    + "]"
                )

            return build(
                gate_verdict=GateVerdict.BLOCK,
                fault_class=winner.fault_class,  # type: ignore[arg-type]
                liable_party=spec.party,
                loss_paise=winner.loss_paise or 0,
                confidence=winner.confidence,
                basis_class=basis,
                cited=winner.basis,
                reason=f"{winner.verifier_id}: {winner.reason}{note}",
            )

        # Nothing failed admissibly. A clearance now requires a positive
        # finding from every competent role, not merely the absence of a
        # complaint. Roles that abstained have established nothing.
        passes = [o for o in outputs if o.verdict is Verdict.PASS]
        cleared_roles = {
            o.role for o in passes
            if o.basis and view.envelope.basis_class(o.basis) >= self.floor
        }
        missing = _REQUIRED_FOR_CLEARANCE[mode] - cleared_roles

        if missing:
            silent = sorted(r.value for r in missing)
            detail = "; ".join(
                f"{o.verifier_id}: {o.reason}"
                for o in outputs
                if o.role in missing and o.verdict is not Verdict.FAIL
            )
            return build(
                gate_verdict=GateVerdict.ABSTAIN,
                fault_class=FaultClass.NO_FAULT,
                liable_party=Party.NONE,
                loss_paise=0,
                confidence=0.0,
                basis_class=EvidenceClass.SELF_REPORT,
                cited=(),
                reason=(
                    f"nothing failed, but {', '.join(silent)} did not clear this on "
                    f"admissible evidence, so no clearance is supported ({detail})"
                ),
            )

        confidence = min((o.confidence for o in passes), default=1.0)
        cited = tuple(dict.fromkeys(i for o in passes for i in o.basis))
        strongest = min(
            view.envelope.basis_class(o.basis) for o in passes if o.basis
        )
        return build(
            gate_verdict=GateVerdict.ALLOW,
            fault_class=FaultClass.NO_FAULT,
            liable_party=Party.NONE,
            loss_paise=0,
            confidence=confidence,
            basis_class=strongest,
            cited=cited,
            reason=(
                f"every competent verifier cleared this on {strongest.name} evidence "
                f"or better"
            ),
        )

    def _pick(self, admissible: list[VerifierOutput]) -> VerifierOutput:
        """
        One winner from several surviving failures.

        Subsumption first: a verdict that explains another one wins. Then
        evidence class, then confidence. Role order is never a tiebreaker -
        which verifier happens to be listed first should not decide liability.
        """
        classes = {o.fault_class for o in admissible if o.fault_class is not None}

        for out in admissible:
            fault = out.fault_class
            if fault is None:
                continue
            subsumed = _SUBSUMES.get(fault, frozenset())
            if subsumed and (classes - {fault}) <= subsumed:
                return out

        return max(admissible, key=lambda o: (o.confidence, len(o.basis)))


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from collections import Counter
    from pathlib import Path

    from bench.scenario import load_dir

    adj = Adjudicator()
    scenarios = load_dir(Path("bench/scenarios")).scenarios

    correct: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    abstained_by: Counter[str] = Counter()
    llm_calls = 0
    abstentions = 0
    false_clearances = 0

    for s in scenarios:
        d = adj.decide(s.to_verifier_input(), Mode.ATTRIBUTION)
        truth = s.truth.fault_class.value
        totals[truth] += 1
        if d.fault_class is s.truth.fault_class and not d.abstained:
            correct[truth] += 1
        if d.llm_invoked:
            llm_calls += 1
        if d.abstained:
            abstentions += 1
            abstained_by[truth] += 1
        elif (
            d.gate_verdict is GateVerdict.ALLOW
            and s.truth.fault_class is not FaultClass.NO_FAULT
        ):
            false_clearances += 1

    n = len(scenarios)
    print("ATTRIBUTION MODE  (semantic verifier stubbed to ABSTAIN)")
    print()
    print(f"{'true class':<24}{'n':>5}{'correct':>9}{'abstain':>9}{'acc':>7}")
    print("-" * 54)
    for cls in sorted(totals, key=lambda c: -totals[c]):
        acc = correct[cls] / totals[cls]
        print(f"{cls:<24}{totals[cls]:>5}{correct[cls]:>9}{abstained_by[cls]:>9}{acc:>6.0%}")
    print("-" * 54)
    total_correct = sum(correct.values())
    print(f"{'overall':<24}{n:>5}{total_correct:>9}{abstentions:>9}{total_correct / n:>6.0%}")
    print()
    print(f"decided without a model call       {n - llm_calls:>4}  ({(n - llm_calls) / n:.1%})")
    print(f"escalated to the semantic verifier {llm_calls:>4}  ({llm_calls / n:.1%})")
    print(f"abstentions                        {abstentions:>4}  ({abstentions / n:.1%})")
    print(f"FALSE CLEARANCES                   {false_clearances:>4}  "
          f"<- a fault allowed through as clean")
    print()
    print("Read this as the deterministic-only baseline. Abstentions are cases")
    print("the system declines rather than guesses, and every one of them is a")
    print("case only the semantic verifier can settle.")

    # Conflict resolution actually fired.
    injection = next(
        s for s in scenarios if s.truth.fault_class is FaultClass.INJECTION_INDUCED
    )
    d = adj.decide(injection.to_verifier_input(), Mode.ATTRIBUTION)
    print()
    print(f"conflict example  {injection.scenario_id}")
    print(f"  resolved to     {d.fault_class.value} / {d.liable_party.value}")
    print(f"  {d.reason[:150]}...")

    # Gate mode sees less.
    sub = next(
        s for s in scenarios if s.truth.fault_class is FaultClass.MERCHANT_SUBSTITUTION
    )
    gate = adj.decide(sub.to_verifier_input(), Mode.GATE)
    attr = adj.decide(sub.to_verifier_input(), Mode.ATTRIBUTION)
    print()
    print(f"gate blindness    {sub.scenario_id}  (MERCHANT_SUBSTITUTION)")
    print(f"  gate            {gate.gate_verdict.value}")
    print(f"  attribution     {attr.fault_class.value} / {attr.liable_party.value}")