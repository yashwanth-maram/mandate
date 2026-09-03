"""
Evaluation runner.

    uv run python -m eval.runner --seed 42
    uv run python -m eval.runner --seed 42 --split heldout
    uv run python -m eval.runner --seed 42 --live --workers 8
    uv run python -m eval.runner --seed 42 --live --self-report

Runs both modes over the benchmark, computes metrics, and writes three files to
experiments/runs/<tag>/:

    report.txt       the rendered table
    result.json      every computed metric
    decisions.jsonl  one line per scenario: verdict, cited evidence, discarded
                     verifiers, reason
    run.json         seed, split, flags, model, timestamp, git commit

decisions.jsonl is the file that makes the audit-trail claim checkable. Any
scenario id can be grepped out of it and read against the scenario file of the
same name, so a reviewer can confirm a decision without running anything.

run.json records the git commit because a table of numbers that cannot be tied
to the code that produced it is not reproducible, only repeatable by the person
who still has the working directory.

THE TWO ABLATIONS

    --self-report   lets the semantic verifier read the agent's account of
                    itself. Its basis then meets to SELF_REPORT and the
                    adjudicator discards every verdict it reaches that way.
                    Running with and without is what turns the admissibility
                    floor from an architectural claim into a measured one.

    --live          uses the Anthropic API. Without it the semantic verifier
                    abstains, which produces the deterministic-only baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from adjudication.engine import Adjudicator, Decision, Mode
from bench.scenario import Scenario, load_dir
from eval.metrics import compute, render
from verifiers.base import Verifier
from verifiers.semantic import DEFAULT_MODEL, SemanticVerifier


DEFAULT_SCENARIOS = Path("bench/scenarios")
DEFAULT_RUNS = Path("experiments/runs")


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def build_semantic(args: argparse.Namespace) -> Verifier:
    """The semantic verifier, wired or stubbed."""
    if not args.live:
        return SemanticVerifier(include_self_report=args.self_report)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    if args.provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key or key.startswith("sk-or-v1-xxx"):
            raise SystemExit(
                "--provider openrouter needs a real OPENROUTER_API_KEY in .env"
            )
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    else:
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key or key.startswith("sk-ant-xxx"):
            raise SystemExit("--live needs a real ANTHROPIC_API_KEY in .env")
        from anthropic import Anthropic
        client = Anthropic(api_key=key)

    return SemanticVerifier(
        client=client,
        provider=args.provider,
        include_self_report=args.self_report,
        model=args.model,
    )


def run_mode(
    adj: Adjudicator,
    scenarios: Sequence[Scenario],
    mode: Mode,
    workers: int,
    label: str,
) -> list[Decision]:
    inputs = [s.to_verifier_input() for s in scenarios]

    if workers <= 1:
        decisions = []
        for i, vi in enumerate(inputs, 1):
            decisions.append(adj.decide(vi, mode))
            if i % 50 == 0 or i == len(inputs):
                print(f"  {label}: {i}/{len(inputs)}", end="\r", flush=True)
        print(" " * 40, end="\r")
        return decisions

    # ThreadPoolExecutor.map preserves input order, which the metrics zip
    # depends on. Per-decision latency includes queueing under contention, so
    # the report says so when workers > 1.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda vi: adj.decide(vi, mode), inputs))


def write_outputs(
    out: Path,
    scenarios: Sequence[Scenario],
    attribution: Sequence[Decision],
    gate: Sequence[Decision],
    result,
    args: argparse.Namespace,
) -> None:
    out.mkdir(parents=True, exist_ok=True)

    (out / "report.txt").write_text(render(result), encoding="utf-8")
    (out / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")

    with (out / "decisions.jsonl").open("w", encoding="utf-8") as fh:
        for s, a, g in zip(scenarios, attribution, gate):
            fh.write(json.dumps({
                "scenario_id": s.scenario_id,
                "split": s.split,
                "truth": {
                    "fault_class": s.truth.fault_class.value,
                    "liable_party": s.truth.liable_party.value,
                    "is_hard_pair": s.truth.is_hard_pair,
                    "agent_report_truthful": s.truth.agent_report_truthful,
                    "loss_paise": s.truth.loss_paise,
                },
                "attribution": {
                    "fault_class": a.fault_class.value,
                    "liable_party": a.liable_party.value,
                    "gate_verdict": a.gate_verdict.value,
                    "loss_paise": a.loss_paise,
                    "confidence": round(a.confidence, 3),
                    "basis_class": a.basis_class.name,
                    "cited": list(a.cited),
                    "discarded": list(a.discarded),
                    "llm_invoked": a.llm_invoked,
                    "reason": a.reason,
                },
                "gate": {
                    "verdict": g.gate_verdict.value,
                    "expected": s.truth.expected_gate_verdict.value,
                    "gate_detectable": s.truth.gate_detectable,
                    "reason": g.reason,
                },
                "correct": a.fault_class is s.truth.fault_class and not a.abstained,
            }, ensure_ascii=False) + "\n")

    (out / "run.json").write_text(json.dumps({
        "seed": args.seed,
        "split": args.split,
        "n": len(scenarios),
        "live": args.live,
        "include_self_report": args.self_report,
        "provider": args.provider if args.live else None,
        "model": args.model if args.live else None,
        "workers": args.workers,
        "scenarios_dir": str(args.scenarios),
        "git_commit": git_commit(),
        "run_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Mandate on the benchmark.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=("all", "train", "heldout"), default="all")
    ap.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to experiments/runs/<tag>")
    ap.add_argument("--live", action="store_true",
                    help="call the model; otherwise the semantic verifier abstains")
    ap.add_argument("--self-report", action="store_true", dest="self_report",
                    help="ablation: let the semantic verifier read the agent's "
                         "self-report, so its basis falls below the floor")
    ap.add_argument("--provider", choices=("anthropic", "openrouter"),
                    default="anthropic")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="first N scenarios only, for quick checks")
    args = ap.parse_args()

    dataset = load_dir(args.scenarios)
    
    keyring_path = args.scenarios / "_keyring.json"
    keyring = (
        json.loads(keyring_path.read_text(encoding="utf-8"))
        if keyring_path.exists()
        else {}
    )
    scenarios = (
        dataset.scenarios if args.split == "all" else dataset.split(args.split)
    )
    scenarios = sorted(scenarios, key=lambda s: s.scenario_id)
    if args.limit:
        scenarios = scenarios[: args.limit]
    if not scenarios:
        raise SystemExit(f"no scenarios in split '{args.split}'")

    adj = Adjudicator(semantic=build_semantic(args), keyring=keyring)

    mode_label = "live" if args.live else "stub"
    if args.self_report:
        mode_label += "+selfreport"
    print(f"running {len(scenarios)} scenarios   seed {args.seed}   "
          f"split {args.split}   semantic: {mode_label}")

    attribution = run_mode(adj, scenarios, Mode.ATTRIBUTION, args.workers, "attribution")
    gate = run_mode(adj, scenarios, Mode.GATE, args.workers, "gate")

    result = compute(scenarios, attribution, gate, seed=args.seed, split=args.split)

    tag = f"seed{args.seed}-{args.split}-{mode_label}"
    out = args.out or (DEFAULT_RUNS / tag)
    write_outputs(out, scenarios, attribution, gate, result, args)

    print()
    print(render(result))
    if args.workers > 1:
        print("note: latency figures include queueing; workers > 1")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
