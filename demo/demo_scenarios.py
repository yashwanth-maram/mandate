"""
Live decision trace.

    uv run python -m demo.scenarios --class INTENT_MISMATCH
    uv run python -m demo.scenarios --ablation --slow
    uv run python -m demo.scenarios --list

Shows one transaction being adjudicated: the obligation, the evidence and what
each item is worth, the verifier votes arriving with their declared bases, the
floor discarding whatever cannot support itself, and the decision.

Two choices shape it.

  The mechanism is visible, not hidden. A dashboard would summarise this into a
  verdict and a confidence bar, which is exactly the interesting part removed.
  What matters here is watching a verifier declare what it relied on and having
  that decide whether its vote counts.

  The system decides blind. Ground truth is printed after the decision, never
  before. Showing the label first turns a demonstration into an illustration.

--slow paces the output for screen recording. --ablation runs the same
scenario twice, once with the semantic verifier reading the agent's own account
of itself, which is where the floor earns its place.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from adjudication.engine import Adjudicator, Mode
from bench.scenario import Scenario, load_dir
from bench.taxonomy import EvidenceClass, FaultClass, GateVerdict, PERFORMANCE_FLOOR
from schemas.obligation import format_paise
from verifiers.base import Verdict
from verifiers.semantic import DEFAULT_CACHE, DEFAULT_MODEL, ResponseCache, SemanticVerifier


console = Console()

_VERDICT_STYLE = {
    Verdict.PASS: "green",
    Verdict.FAIL: "red",
    Verdict.ABSTAIN: "yellow",
}

_CLASS_STYLE = {
    EvidenceClass.SELF_REPORT: "red",
    EvidenceClass.SELF_SIGNED: "yellow",
    EvidenceClass.MERCHANT_RECORD: "green",
    EvidenceClass.PSP_RECEIPT: "bright_green",
}


def pause(seconds: float, slow: bool) -> None:
    if slow:
        time.sleep(seconds)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def obligation_panel(scenario: Scenario) -> Panel:
    ob = scenario.obligation
    body = Table.grid(padding=(0, 2))
    body.add_column(style="dim", justify="right")
    body.add_column()

    body.add_row("said", Text(ob.intent.text, style="bold white"))
    body.add_row("", "")
    body.add_row("category", ob.hard.category)
    body.add_row("merchant", ", ".join(ob.hard.merchant_allowlist))
    body.add_row("unit ceiling", format_paise(ob.hard.max_unit_price_paise))
    body.add_row("quantity", str(ob.hard.quantity))
    for label, value in (
        ("brand", ob.hard.brand),
        ("variant", ob.hard.variant),
        ("pack", f"{ob.hard.pack_size_g}g" if ob.hard.pack_size_g else None),
    ):
        if value is not None:
            body.add_row(label, str(value))

    if ob.intent.uncaptured_attributes:
        body.add_row("", "")
        body.add_row(
            "not captured",
            Text(
                ", ".join(ob.intent.uncaptured_attributes)
                + "  (spoken about, never turned into a constraint)",
                style="yellow",
            ),
        )

    body.add_row("", "")
    body.add_row(
        "block",
        f"{format_paise(ob.block.blocked_paise)} blocked, "
        f"{format_paise(ob.block.remaining_paise)} remaining",
    )
    sig = "verified" if ob.signature else "unsigned"
    body.add_row("signature", Text(f"{sig}  {ob.signer_key_id or ''}", style="dim"))

    return Panel(body, title="[bold]obligation[/bold]", border_style="blue")


def evidence_table(scenario: Scenario) -> Table:
    table = Table(box=None, padding=(0, 2))
    table.add_column("evidence", style="dim")
    table.add_column("class")
    table.add_column("from", style="dim")
    table.add_column("what it says")

    for item in scenario.envelope.items:
        payload = item.payload
        summary = ""
        kind = item.kind.value

        if hasattr(payload, "lines"):
            first = payload.lines[0]
            summary = f"{first.sku} x{first.quantity} @ {format_paise(first.unit_price_paise)}"
        elif hasattr(payload, "amount_paise"):
            summary = format_paise(payload.amount_paise)
        elif hasattr(payload, "text"):
            summary = payload.text[:52]
        elif hasattr(payload, "description"):
            summary = payload.description[:52]

        style = _CLASS_STYLE[item.evidence_class]
        marker = "" if item.evidence_class >= PERFORMANCE_FLOOR else "  below floor"
        table.add_row(
            kind.lower().replace("_", " "),
            Text(f"{item.evidence_class.name}{marker}", style=style),
            item.emitted_by,
            summary,
        )
    return table


def votes_table(decision, scenario: Scenario, view_envelope) -> Table:
    table = Table(box=None, padding=(0, 2))
    table.add_column("verifier")
    table.add_column("verdict")
    table.add_column("basis")
    table.add_column("conf", justify="right")
    table.add_column("says")

    for out in decision.outputs:
        discarded = out.verifier_id in decision.discarded
        try:
            basis = view_envelope.basis_class(out.basis) if out.basis else EvidenceClass.SELF_REPORT
        except KeyError:
            basis = EvidenceClass.SELF_REPORT

        name = Text(out.verifier_id)
        verdict = Text(out.verdict.value, style=_VERDICT_STYLE[out.verdict])
        basis_text = Text(basis.name, style=_CLASS_STYLE[basis])
        reason = out.reason[:64]

        if discarded:
            name.stylize("strike dim")
            verdict = Text(f"{out.verdict.value}  DISCARDED", style="strike dim red")
            basis_text = Text(f"{basis.name}  below floor", style="bold red")
            reason = f"{reason}   <- rested on evidence that cannot support it"

        table.add_row(
            name, verdict, basis_text, f"{out.confidence:.2f}", Text(reason, style="dim")
        )
    return table


def decision_panel(decision) -> Panel:
    if decision.gate_verdict is GateVerdict.BLOCK:
        colour, headline = "red", "BLOCK"
    elif decision.gate_verdict is GateVerdict.ALLOW:
        colour, headline = "green", "ALLOW"
    else:
        colour, headline = "yellow", "ABSTAIN"

    body = Table.grid(padding=(0, 2))
    body.add_column(style="dim", justify="right")
    body.add_column()
    body.add_row("verdict", Text(headline, style=f"bold {colour}"))
    body.add_row("fault", decision.fault_class.value)
    body.add_row("liable", decision.liable_party.value)
    body.add_row("exposure", format_paise(decision.loss_paise))
    body.add_row("basis", decision.basis_class.name)
    body.add_row("cited", ", ".join(decision.cited) or "-")
    body.add_row(
        "model called",
        Text("yes" if decision.llm_invoked else "no", style="dim"),
    )
    body.add_row("", "")
    body.add_row("because", Text(decision.reason, style="italic"))

    return Panel(body, title="[bold]decision[/bold]", border_style=colour)


def truth_panel(scenario: Scenario, decision) -> Panel:
    correct = decision.fault_class is scenario.truth.fault_class and not decision.abstained
    body = Table.grid(padding=(0, 2))
    body.add_column(style="dim", justify="right")
    body.add_column()
    body.add_row("actual fault", scenario.truth.fault_class.value)
    body.add_row("user wanted", scenario.truth.requested_sku)
    body.add_row("agent ordered", scenario.truth.ordered_sku)
    body.add_row("merchant shipped", scenario.truth.delivered_sku)
    body.add_row(
        "agent's report",
        Text(
            "truthful" if scenario.truth.agent_report_truthful else "false",
            style="green" if scenario.truth.agent_report_truthful else "red",
        ),
    )
    body.add_row("hard pair", "yes" if scenario.truth.is_hard_pair else "no")
    body.add_row("", "")
    body.add_row("why", Text(scenario.truth.rationale, style="italic dim"))

    return Panel(
        body,
        title=Text(
            "ground truth  -  " + ("correct" if correct else "wrong"),
            style="bold green" if correct else "bold red",
        ),
        border_style="green" if correct else "red",
    )


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def trace(scenario: Scenario, adj: Adjudicator, *, slow: bool, mode: Mode) -> None:
    console.print()
    console.print(Rule(f"[bold]{scenario.scenario_id}[/bold]   {mode.value} mode"))
    console.print()

    console.print(obligation_panel(scenario))
    pause(2.5, slow)

    console.print()
    console.print("[bold]evidence on record[/bold]")
    console.print(evidence_table(scenario))
    pause(2.5, slow)

    vi = scenario.to_verifier_input()
    decision = adj.decide(vi, mode)

    view_envelope = vi.envelope
    if mode is Mode.GATE:
        from adjudication.engine import _gate_view
        view_envelope = _gate_view(vi).envelope
        console.print()
        console.print(
            Text(
                "gate mode: fulfilment, self-report and dispute are withheld - "
                "none of them exist when a debit is decided",
                style="dim italic",
            )
        )

    console.print()
    console.print("[bold]verifier votes[/bold]")
    console.print(votes_table(decision, scenario, view_envelope))
    pause(3.0, slow)

    if decision.discarded:
        console.print()
        console.print(
            Panel(
                Text(
                    f"{len(decision.discarded)} verdict(s) discarded by the "
                    f"admissibility floor.\n\n"
                    f"A verdict is worth its weakest evidence. These declared a "
                    f"basis below {PERFORMANCE_FLOOR.name}, so they weigh nothing - "
                    f"not because they were wrong, but because of what they rested on.",
                    style="red",
                ),
                title="[bold red]floor[/bold red]",
                border_style="red",
            )
        )
        pause(3.0, slow)

    console.print()
    console.print(decision_panel(decision))
    pause(2.5, slow)

    console.print()
    console.print(truth_panel(scenario, decision))
    console.print()


def ablation(scenario: Scenario, *, cache, model: str, keyring, slow: bool) -> None:
    """The same scenario, with and without the agent's account of itself."""
    console.print()
    console.print(Rule("[bold]ablation: does the floor do real work?[/bold]"))
    console.print()
    console.print(obligation_panel(scenario))
    console.print()

    for include, label in ((False, "semantic reads merchant records only"),
                           (True, "semantic also reads the agent's self-report")):
        semantic = SemanticVerifier(
            model=model, cache=cache, cache_only=cache is not None,
            include_self_report=include,
        )
        adj = Adjudicator(semantic=semantic, keyring=keyring)
        vi = scenario.to_verifier_input()
        decision = adj.decide(vi, Mode.ATTRIBUTION)

        console.print(Rule(label, style="dim"))
        console.print(votes_table(decision, scenario, vi.envelope))
        console.print()
        console.print(decision_panel(decision))
        console.print()
        pause(3.0, slow)

    console.print(truth_panel(scenario, decision))
    console.print()
    console.print(
        Text(
            "Same scenario, same model, same question. The only difference is what "
            "the verifier was allowed to read - and therefore what its verdict "
            "could rest on.",
            style="italic",
        )
    )
    console.print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Trace one decision, step by step.")
    ap.add_argument("--scenarios", type=Path, default=Path("bench/scenarios"))
    ap.add_argument("--id", help="a specific scenario id")
    ap.add_argument("--class", dest="fault_class", help="pick one of this fault class")
    ap.add_argument("--gate", action="store_true", help="gate mode instead of attribution")
    ap.add_argument("--ablation", action="store_true",
                    help="run twice, with and without the agent's self-report")
    ap.add_argument("--slow", action="store_true", help="pace it for screen recording")
    ap.add_argument("--list", action="store_true", dest="list_only")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    dataset = load_dir(args.scenarios)

    if args.list_only:
        table = Table(title="scenarios by class", box=None, padding=(0, 3))
        table.add_column("class")
        table.add_column("n", justify="right")
        table.add_column("example")
        for fault, n in sorted(dataset.counts().items(), key=lambda kv: -kv[1]):
            if not n:
                continue
            example = next(s for s in dataset.scenarios if s.truth.fault_class is fault)
            table.add_row(fault.value, str(n), example.scenario_id)
        console.print()
        console.print(table)
        console.print()
        return

    if args.id:
        scenario = next(s for s in dataset.scenarios if s.scenario_id == args.id)
    elif args.fault_class:
        target = FaultClass(args.fault_class.upper())
        pool = [s for s in dataset.scenarios if s.truth.fault_class is target]
        if args.ablation:
            pool = [s for s in pool if not s.truth.agent_report_truthful] or pool
        scenario = random.Random(args.seed).choice(pool)
    else:
        pool = [
            s for s in dataset.scenarios
            if s.truth.fault_class is FaultClass.INTENT_MISMATCH
            and s.truth.is_hard_pair
            and not s.truth.agent_report_truthful
        ]
        scenario = random.Random(args.seed).choice(pool)

    import json
    keyring_path = args.scenarios / "_keyring.json"
    keyring = (
        json.loads(keyring_path.read_text(encoding="utf-8"))
        if keyring_path.exists() else {}
    )

    cache = ResponseCache(args.cache) if args.cache.exists() else None
    if cache is not None and len(cache) == 0:
        cache = None
    if cache is None:
        console.print(
            Text(
                "no cached verdicts, so the semantic verifier will abstain. "
                "Run a live evaluation first to see it decide.",
                style="yellow",
            )
        )

    if args.ablation:
        ablation(scenario, cache=cache, model=args.model, keyring=keyring, slow=args.slow)
        return

    semantic = SemanticVerifier(model=args.model, cache=cache, cache_only=cache is not None)
    adj = Adjudicator(semantic=semantic, keyring=keyring)
    trace(scenario, adj, slow=args.slow, mode=Mode.GATE if args.gate else Mode.ATTRIBUTION)


if __name__ == "__main__":
    main()
