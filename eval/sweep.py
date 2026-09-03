"""
Confidence threshold sweep.

    uv run python -m eval.sweep --seed 42 --model <id>

Runs the full adjudication at a range of `min_confidence` values and reports
what each buys and costs. Entirely from the response cache: `min_confidence` is
applied after the cache lookup, so every point is a complete re-adjudication
that never touches the network. Ten points take about twenty seconds and cost
nothing.

WHY THIS EXISTS

The confidence floor was set to 0.6 and never fired. Across every escalated
case the model returned high confidence, including on every case it got wrong,
so the abstention mechanism did no work at all. That is a calibration finding
rather than a success, and it raises a question the accuracy number cannot
answer: at what threshold should this actually run?

THREE OPERATING POINTS, NOT ONE

They are different questions and they do not have the same answer.

  maximum accuracy   the most cases decided correctly
  minimum rupee cost the least money in the wrong place, given a stated cost
                     per manual review
  minimum harm       the fewest INTENT_MISMATCH cases misread as USER_REGRET

That last one matters more than its count suggests. Reading an intent failure
as an unfounded complaint tells a merchant that a buyer who was genuinely
wronged is lying: the buyer is out of pocket and the complaint goes on record
as bogus. The opposite error - refunding a complaint that was in fact
unfounded - costs a merchant a small sum and makes the buyer whole. Same line
in a confusion matrix, very different harm, and a payments system should not be
indifferent between them.

THE COST MODEL IS AN ASSUMPTION

Rupee-optimality depends on what a manual review costs, and that number is a
choice rather than a fact. It is a flag with a stated default, printed in the
report, so a reader who disagrees can recompute rather than having to trust it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from adjudication.engine import Adjudicator, Mode
from bench.scenario import Scenario, load_dir
from bench.taxonomy import FaultClass
from eval.metrics import compute, order_value_paise
from schemas.obligation import format_paise
from verifiers.semantic import DEFAULT_CACHE, DEFAULT_MODEL, ResponseCache, SemanticVerifier


DEFAULT_REVIEW_COST_PAISE = 2_000   # Rs 20 of operations time per escalation


@dataclass
class Point:
    threshold: float
    coverage: float
    accuracy: float
    accuracy_on_decided: float
    abstentions: int
    false_clearances: int
    harmful: int              # INTENT_MISMATCH read as USER_REGRET
    lenient: int              # USER_REGRET read as INTENT_MISMATCH
    missed_paise: int
    misattributed_paise: int
    false_block_paise: int
    review_paise: int

    @property
    def total_cost_paise(self) -> int:
        return (
            self.missed_paise
            + self.misattributed_paise
            + self.false_block_paise
            + self.review_paise
        )


def evaluate_at(
    scenarios: Sequence[Scenario],
    threshold: float,
    *,
    cache: ResponseCache,
    model: str,
    keyring: dict[str, str],
    review_cost_paise: int,
) -> Point:
    semantic = SemanticVerifier(
        model=model, cache=cache, cache_only=True, min_confidence=threshold
    )
    adj = Adjudicator(semantic=semantic, keyring=keyring)

    decisions = [adj.decide(s.to_verifier_input(), Mode.ATTRIBUTION) for s in scenarios]
    result = compute(scenarios, decisions, seed=0, split="sweep")

    directions: Counter[tuple[str, str]] = Counter()
    for s, d in zip(scenarios, decisions):
        if d.abstained or d.fault_class is s.truth.fault_class:
            continue
        directions[(s.truth.fault_class.value, d.fault_class.value)] += 1

    harmful = directions[(FaultClass.INTENT_MISMATCH.value, FaultClass.USER_REGRET.value)]
    lenient = directions[(FaultClass.USER_REGRET.value, FaultClass.INTENT_MISMATCH.value)]

    return Point(
        threshold=threshold,
        coverage=result.coverage,
        accuracy=result.overall_accuracy,
        accuracy_on_decided=result.accuracy_on_decided,
        abstentions=result.abstained,
        false_clearances=result.false_clearances,
        harmful=harmful,
        lenient=lenient,
        missed_paise=result.costs.missed_paise,
        misattributed_paise=result.costs.misattributed_paise,
        false_block_paise=result.costs.false_block_paise,
        review_paise=result.abstained * review_cost_paise,
    )


def render(points: Sequence[Point], review_cost_paise: int) -> str:
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add("CONFIDENCE THRESHOLD SWEEP")
    add("=" * 78)
    add("")
    add(f"{'thresh':>7}{'cover':>8}{'acc':>8}{'abst':>7}{'harmful':>9}"
        f"{'lenient':>9}{'total cost':>15}")
    add("-" * 78)
    for p in points:
        add(
            f"{p.threshold:>7.2f}{p.coverage:>8.1%}{p.accuracy:>8.1%}"
            f"{p.abstentions:>7}{p.harmful:>9}{p.lenient:>9}"
            f"{format_paise(p.total_cost_paise):>15}"
        )
    add("-" * 78)
    add("")

    best_acc = max(points, key=lambda p: p.accuracy)
    best_cost = min(points, key=lambda p: p.total_cost_paise)
    best_harm = min(points, key=lambda p: (p.harmful, p.total_cost_paise))

    add("OPERATING POINTS")
    add(f"  maximum accuracy      threshold {best_acc.threshold:.2f}   "
        f"{best_acc.accuracy:.1%} accuracy, {best_acc.coverage:.1%} coverage, "
        f"{best_acc.harmful} harmful")
    add(f"  minimum rupee cost    threshold {best_cost.threshold:.2f}   "
        f"{format_paise(best_cost.total_cost_paise)}, "
        f"{best_cost.coverage:.1%} coverage, {best_cost.harmful} harmful")
    add(f"  minimum harm          threshold {best_harm.threshold:.2f}   "
        f"{best_harm.harmful} harmful, {best_harm.coverage:.1%} coverage, "
        f"{format_paise(best_harm.total_cost_paise)}")
    add("")

    if best_acc.threshold != best_harm.threshold:
        delta_cov = best_acc.coverage - best_harm.coverage
        add(f"  Accuracy-optimal and harm-optimal are different points. Moving from")
        add(f"  {best_acc.threshold:.2f} to {best_harm.threshold:.2f} gives up "
            f"{delta_cov:.1%} coverage and removes "
            f"{best_acc.harmful - best_harm.harmful} case(s) where a buyer who was")
        add(f"  wronged is recorded as having complained without cause.")
    else:
        add("  Accuracy-optimal and harm-optimal coincide at this threshold.")
    add("")

    add("HARM DIRECTION")
    add("  harmful   INTENT_MISMATCH read as USER_REGRET. The agent bought the")
    add("            wrong thing and the buyer is recorded as lying about it.")
    add("            Buyer out of pocket, complaint on record as unfounded.")
    add("  lenient   USER_REGRET read as INTENT_MISMATCH. An unfounded complaint")
    add("            is refunded. Merchant loses the sale, buyer made whole.")
    add("")
    add(f"  Cost model: missed + misattributed + false blocks, plus "
        f"{format_paise(review_cost_paise)} per")
    add("  escalation for operations time. That review figure is an assumption,")
    add("  not a measurement - recompute with --review-cost to test another.")
    add("")
    add("=" * 78)
    return "\n".join(lines)


def plot(points: Sequence[Point], out: Path, review_cost_paise: int) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    thresholds = [p.threshold for p in points]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.plot(thresholds, [p.accuracy * 100 for p in points], marker="o", label="accuracy")
    ax1.plot(thresholds, [p.coverage * 100 for p in points], marker="s", label="coverage")
    ax1.set_ylabel("percent")
    ax1.set_title("Accuracy and coverage move in opposite directions")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(thresholds, [p.total_cost_paise / 100 for p in points],
             marker="o", color="crimson", label="total cost (Rs)")
    ax2.set_ylabel("rupees")
    ax2.set_xlabel("min_confidence")
    ax2b = ax2.twinx()
    ax2b.plot(thresholds, [p.harmful for p in points],
              marker="^", color="darkorange", linestyle="--",
              label="harmful misattributions")
    ax2b.set_ylabel("count")
    ax2.set_title(f"Cost at {format_paise(review_cost_paise)} per review, and harm")
    ax2.grid(alpha=0.3)

    handles = ax2.get_legend_handles_labels()[0] + ax2b.get_legend_handles_labels()[0]
    labels = ax2.get_legend_handles_labels()[1] + ax2b.get_legend_handles_labels()[1]
    ax2.legend(handles, labels, loc="upper left")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep the semantic confidence threshold.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=("all", "train", "heldout"), default="all")
    ap.add_argument("--scenarios", type=Path, default=Path("bench/scenarios"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--start", type=float, default=0.50)
    ap.add_argument("--stop", type=float, default=0.95)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--review-cost", type=int, default=DEFAULT_REVIEW_COST_PAISE,
                    dest="review_cost", help="paise per manual escalation")
    ap.add_argument("--out", type=Path, default=Path("experiments/reports"))
    args = ap.parse_args()

    cache = ResponseCache(args.cache)
    if len(cache) == 0:
        raise SystemExit(
            f"cache at {args.cache} is empty. Run a live evaluation first:\n"
            f"  uv run python -m eval.runner --seed {args.seed} --live "
            f"--model {args.model}"
        )

    dataset = load_dir(args.scenarios)
    scenarios = dataset.scenarios if args.split == "all" else dataset.split(args.split)
    scenarios = sorted(scenarios, key=lambda s: s.scenario_id)

    keyring_path = args.scenarios / "_keyring.json"
    keyring = (
        json.loads(keyring_path.read_text(encoding="utf-8"))
        if keyring_path.exists() else {}
    )

    thresholds: list[float] = []
    t = args.start
    while t <= args.stop + 1e-9:
        thresholds.append(round(t, 4))
        t += args.step

    print(f"sweeping {len(thresholds)} thresholds over {len(scenarios)} scenarios "
          f"from {len(cache)} cached verdicts")

    points: list[Point] = []
    for i, threshold in enumerate(thresholds, 1):
        points.append(evaluate_at(
            scenarios, threshold,
            cache=cache, model=args.model, keyring=keyring,
            review_cost_paise=args.review_cost,
        ))
        print(f"  {i}/{len(thresholds)}  min_confidence={threshold:.2f}",
              end="\r", flush=True)
    print(" " * 50, end="\r")

    report = render(points, args.review_cost)
    print()
    print(report)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "threshold_sweep.txt").write_text(report, encoding="utf-8")
    (args.out / "threshold_sweep.json").write_text(
        json.dumps([p.__dict__ | {"total_cost_paise": p.total_cost_paise}
                    for p in points], indent=2),
        encoding="utf-8",
    )
    chart = plot(points, args.out / "threshold_sweep.png", args.review_cost)

    print(f"written to {args.out}")
    if chart is None:
        print("matplotlib unavailable; chart skipped")


if __name__ == "__main__":
    main()
