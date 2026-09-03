"""
Metrics for the Mandate evaluation.

Four decisions here shape what the numbers mean.

  COVERAGE IS REPORTED NEXT TO ACCURACY, ALWAYS.
      An abstention is neither right nor wrong; it is a decision to escalate.
      A system can reach any accuracy it likes by abstaining on everything
      difficult, so accuracy without coverage is not a result. Both appear in
      the same table, on adjacent lines.

  COST IS SPLIT FOUR WAYS, IN RUPEES.
      caught          fault found and attributed to the right party
      misattributed   fault found, wrong party charged
      missed          fault cleared as though nothing happened
      abstained       escalated, so no automatic loss but a review burden

      Misattribution is the one a payments team cares about most. Money
      assigned to the wrong party is worse than money not assigned, because
      somebody is debited who should not have been and will dispute it.

  FALSE-BLOCK COST COMES FROM THE ORDER VALUE, NOT FROM loss_paise.
      Clean scenarios carry loss_paise = 0 by construction, so scoring a
      wrongly blocked clean transaction against it would report a cost of zero
      for the most expensive kind of error a risk system makes. A blocked
      legitimate sale costs the merchant the whole basket.

  THE BY-CONSTRUCTION CAVEAT IS PRINTED IN THE REPORT.
      Not in the README, where it can be skipped. Deterministic classes score
      1.00 because a constraint checker cannot miss a numeric violation it is
      defined to catch. That is not an achievement and the table says so on the
      line beneath it.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Iterable, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from adjudication.engine import Decision, Mode
from bench.scenario import Scenario
from bench.taxonomy import (
    HARD_CONFUSION_PAIRS,
    TAXONOMY,
    FaultClass,
    GateVerdict,
)
from schemas.evidence import EvidenceKind, MerchantOrder
from schemas.obligation import format_paise


# Classes whose detection is a comparison between values the obligation already
# fixed. Perfect scores here are definitional, not evidence of quality.
_BY_CONSTRUCTION = frozenset(
    f for f, spec in TAXONOMY.items()
    if spec.deterministic and f is not FaultClass.NO_FAULT
)


def order_value_paise(scenario: Scenario) -> int:
    """What the transaction was worth. Used for false-block cost."""
    item = scenario.envelope.first_of_kind(EvidenceKind.MERCHANT_ORDER)
    if item is None:
        return 0
    order: MerchantOrder = item.payload  # type: ignore[assignment]
    return order.total_paise


class ClassMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    fault_class: FaultClass
    n: int
    correct: int
    abstained: int
    precision: float
    recall: float
    f1: float
    by_construction: bool


class Costs(BaseModel):
    model_config = ConfigDict(frozen=True)

    caught_paise: int = 0
    misattributed_paise: int = 0
    missed_paise: int = 0
    abstained_paise: int = 0
    false_block_paise: int = 0

    @property
    def exposure_paise(self) -> int:
        return (
            self.caught_paise
            + self.misattributed_paise
            + self.missed_paise
            + self.abstained_paise
        )


class GateMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    n: int = 0
    correct: int = 0
    false_blocks: int = 0        # ALLOW expected, BLOCK given
    missed: int = 0              # BLOCK expected, ALLOW given
    abstained: int = 0
    blind_n: int = 0             # scenarios the taxonomy marks gate-invisible
    blind_handled: int = 0       # of those, the gate did the expected thing


class EvalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int
    split: str
    n: int

    decided: int
    correct: int
    abstained: int
    false_clearances: int

    coverage: float
    accuracy_on_decided: float
    overall_accuracy: float

    per_class: tuple[ClassMetrics, ...]
    confusion: dict[str, dict[str, int]]

    hard_pair_n: int
    hard_pair_correct: int

    confusion_pairs: dict[str, int]

    costs: Costs
    gate: GateMetrics

    llm_invoked: int
    deterministic_only: int

    latency_p50_ms: float
    latency_p95_ms: float
    latency_p50_deterministic_ms: float
    verifier_errors: int = 0


def compute(
    scenarios: Sequence[Scenario],
    attribution: Sequence[Decision],
    gate: Optional[Sequence[Decision]] = None,
    *,
    seed: int = 42,
    split: str = "all",
) -> EvalResult:
    if len(scenarios) != len(attribution):
        raise ValueError("scenario and decision counts differ")

    pairs = list(zip(scenarios, attribution))

    tp: Counter[FaultClass] = Counter()
    fp: Counter[FaultClass] = Counter()
    fn: Counter[FaultClass] = Counter()
    totals: Counter[FaultClass] = Counter()
    abstained_by: Counter[FaultClass] = Counter()

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    caught = misattributed = missed = abstained_cost = 0
    correct = abstained = false_clearances = verifier_errors = 0
    hard_n = hard_correct = 0
    llm_invoked = 0
    latencies: list[float] = []
    det_latencies: list[float] = []

    pair_counts: dict[str, int] = {
        f"{a.value} -> {b.value}": 0 for a, b in HARD_CONFUSION_PAIRS
    }
    pair_counts.update({
        f"{b.value} -> {a.value}": 0 for a, b in HARD_CONFUSION_PAIRS
    })

    for s, d in pairs:
        truth = s.truth.fault_class
        totals[truth] += 1
        latencies.append(d.latency_ms)
        
        verifier_errors += sum(
            1 for o in d.outputs if "verifier raised" in o.reason
        )

        if d.llm_invoked:
            llm_invoked += 1
        else:
            det_latencies.append(d.latency_ms)

        if s.truth.is_hard_pair:
            hard_n += 1

        if d.abstained:
            abstained += 1
            abstained_by[truth] += 1
            fn[truth] += 1
            confusion[truth.value]["ABSTAIN"] += 1
            if truth is not FaultClass.NO_FAULT:
                abstained_cost += s.truth.loss_paise
            continue

        predicted = d.fault_class
        confusion[truth.value][predicted.value] += 1

        if predicted is truth:
            correct += 1
            tp[truth] += 1
            if s.truth.is_hard_pair:
                hard_correct += 1
            if truth is not FaultClass.NO_FAULT:
                caught += s.truth.loss_paise
        else:
            fp[predicted] += 1
            fn[truth] += 1
            key = f"{truth.value} -> {predicted.value}"
            if key in pair_counts:
                pair_counts[key] += 1

            if truth is FaultClass.NO_FAULT:
                # A clean transaction stopped or charged to someone. The
                # merchant loses the whole sale, not loss_paise, which is zero
                # here by construction.
                pass
            elif predicted is FaultClass.NO_FAULT:
                missed += s.truth.loss_paise
                false_clearances += 1
            else:
                misattributed += s.truth.loss_paise

    false_block = sum(
        order_value_paise(s)
        for s, d in pairs
        if s.truth.fault_class is FaultClass.NO_FAULT
        and not d.abstained
        and d.fault_class is not FaultClass.NO_FAULT
    )

    per_class: list[ClassMetrics] = []
    for fault in FaultClass:
        n = totals[fault]
        if n == 0:
            continue
        precision = tp[fault] / (tp[fault] + fp[fault]) if (tp[fault] + fp[fault]) else 0.0
        recall = tp[fault] / (tp[fault] + fn[fault]) if (tp[fault] + fn[fault]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class.append(
            ClassMetrics(
                fault_class=fault,
                n=n,
                correct=tp[fault],
                abstained=abstained_by[fault],
                precision=precision,
                recall=recall,
                f1=f1,
                by_construction=fault in _BY_CONSTRUCTION,
            )
        )

    gate_metrics = _gate_metrics(scenarios, gate) if gate is not None else GateMetrics()

    decided = len(pairs) - abstained
    return EvalResult(
        seed=seed,
        split=split,
        n=len(pairs),
        decided=decided,
        correct=correct,
        abstained=abstained,
        false_clearances=false_clearances,
        coverage=decided / len(pairs) if pairs else 0.0,
        accuracy_on_decided=correct / decided if decided else 0.0,
        overall_accuracy=correct / len(pairs) if pairs else 0.0,
        per_class=tuple(per_class),
        confusion={k: dict(v) for k, v in confusion.items()},
        hard_pair_n=hard_n,
        hard_pair_correct=hard_correct,
        confusion_pairs=pair_counts,
        costs=Costs(
            caught_paise=caught,
            misattributed_paise=misattributed,
            missed_paise=missed,
            abstained_paise=abstained_cost,
            false_block_paise=false_block,
        ),
        gate=gate_metrics,
        llm_invoked=llm_invoked,
        deterministic_only=len(pairs) - llm_invoked,
        latency_p50_ms=_pct(latencies, 50),
        latency_p95_ms=_pct(latencies, 95),
        latency_p50_deterministic_ms=_pct(det_latencies, 50),
        verifier_errors=verifier_errors,
    )


def _gate_metrics(
    scenarios: Sequence[Scenario], decisions: Sequence[Decision]
) -> GateMetrics:
    correct = false_blocks = missed = abstained = 0
    blind_n = blind_handled = 0

    for s, d in zip(scenarios, decisions):
        expected = s.truth.expected_gate_verdict
        actual = d.gate_verdict

        if not s.truth.gate_detectable:
            blind_n += 1
            if actual is expected:
                blind_handled += 1

        if actual is GateVerdict.ABSTAIN:
            abstained += 1
        elif actual is expected:
            correct += 1
        elif expected is GateVerdict.ALLOW and actual is GateVerdict.BLOCK:
            false_blocks += 1
        else:
            missed += 1

    return GateMetrics(
        n=len(decisions),
        correct=correct,
        false_blocks=false_blocks,
        missed=missed,
        abstained=abstained,
        blind_n=blind_n,
        blind_handled=blind_handled,
    )


def _pct(values: Iterable[float], p: int) -> float:
    data = sorted(values)
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    return statistics.quantiles(data, n=100)[min(p, 99) - 1]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(r: EvalResult) -> str:
    lines: list[str] = []
    add = lines.append

    add("=" * 72)
    add(f"MANDATE EVALUATION   seed {r.seed}   split: {r.split}   n = {r.n}")
    add("=" * 72)
    add("")

    add("ATTRIBUTION")
    add(f"{'class':<24}{'n':>5}{'prec':>7}{'rec':>7}{'f1':>7}{'abst':>7}")
    add("-" * 72)
    for c in sorted(r.per_class, key=lambda c: -c.n):
        mark = " *" if c.by_construction else ""
        add(
            f"{c.fault_class.value:<24}{c.n:>5}{c.precision:>7.2f}"
            f"{c.recall:>7.2f}{c.f1:>7.2f}{c.abstained:>7}{mark}"
        )
    add("-" * 72)
    add("")
    add("  * These classes are detected by comparing values the obligation already")
    add("    fixed - a price against a ceiling, a merchant against an allowlist.")
    add("    A perfect score is definitional, not an achievement: the checker")
    add("    cannot miss a numeric violation it is defined to catch. The classes")
    add("    that carry information are INTENT_MISMATCH, MERCHANT_SUBSTITUTION")
    add("    and USER_REGRET.")
    add("")

    add(f"coverage                {r.coverage:>7.1%}   ({r.decided} of {r.n} decided)")
    add(f"accuracy on decided     {r.accuracy_on_decided:>7.1%}")
    add(f"overall accuracy        {r.overall_accuracy:>7.1%}")
    add(f"abstentions             {r.abstained:>7}   escalated rather than guessed")
    add(f"FALSE CLEARANCES        {r.false_clearances:>7}   faults let through as clean")
    
    if r.verifier_errors:
        add(f"VERIFIER ERRORS         {r.verifier_errors:>7}   "
            f"!! results below are NOT a clean run")
    add("")

    if r.hard_pair_n:
        acc = r.hard_pair_correct / r.hard_pair_n
        add(f"hard confusable pairs   {acc:>7.1%}   "
            f"({r.hard_pair_correct} of {r.hard_pair_n})")
        add("  Same brand, same pack, within Rs 20. No deterministic check can")
        add("  separate these; the number is the semantic verifier's real score.")
        add("")

    hard_pairs = {k: v for k, v in r.confusion_pairs.items() if v}
    if hard_pairs:
        add("hardest confusions observed")
        for key, count in sorted(hard_pairs.items(), key=lambda kv: -kv[1]):
            add(f"  {key:<52}{count:>4}")
        add("")

    c = r.costs
    add("COST, IN RUPEES")
    add(f"  caught, right party   {format_paise(c.caught_paise):>14}")
    add(f"  MISATTRIBUTED         {format_paise(c.misattributed_paise):>14}   "
        f"charged to the wrong party")
    add(f"  missed                {format_paise(c.missed_paise):>14}   "
        f"cleared as though nothing happened")
    add(f"  abstained             {format_paise(c.abstained_paise):>14}   "
        f"escalated for review")
    add(f"  false blocks          {format_paise(c.false_block_paise):>14}   "
        f"clean sales stopped")
    add(f"  total exposure        {format_paise(c.exposure_paise):>14}")
    add("")

    if r.gate.n:
        g = r.gate
        add("GATE  (pre-debit, post-debit evidence withheld)")
        add(f"  correct               {g.correct:>5}")
        add(f"  false blocks          {g.false_blocks:>5}   clean transactions stopped")
        add(f"  missed                {g.missed:>5}   faults allowed through")
        add(f"  abstained             {g.abstained:>5}")
        if g.blind_n:
            add(f"  gate-invisible        {g.blind_n:>5}   "
                f"({g.blind_n / g.n:.1%} of the set)")
            add(f"    handled correctly   {g.blind_handled:>5}")
            add("    These fail after the debit decision. No pre-debit control can")
            add("    reach them, which is why attribution exists as a second mode.")
        add("")

    add("WHERE THE WORK HAPPENED")
    add(f"  decided without a model call   {r.deterministic_only:>5}  "
        f"({r.deterministic_only / r.n:.1%})")
    add(f"  escalated to the model         {r.llm_invoked:>5}  "
        f"({r.llm_invoked / r.n:.1%})")
    add("")
    add(f"  latency p50                    {r.latency_p50_ms:>7.2f} ms")
    add(f"  latency p95                    {r.latency_p95_ms:>7.2f} ms")
    add(f"  latency p50, no model call     {r.latency_p50_deterministic_ms:>7.2f} ms")
    add("")
    add("=" * 72)

    return "\n".join(lines)
