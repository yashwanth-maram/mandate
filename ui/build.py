"""
Build the visual walkthrough.

    uv run python -m ui.build

Precomputes every decision the page can show, then writes a single
self-contained ui/index.html. No server, no build step, no network - open it
with a double click.

WHY IT IS PRECOMPUTED

The page has to work on a reviewer's machine with no API key. Every verdict it
displays comes from the committed response cache, so the page is a rendering of
the same run `make eval` reproduces rather than a separate demo with its own
numbers. If a scenario has no cached verdict for a configuration, the page says
so instead of quietly showing something else.

WHAT IT IS FOR

One thing: making the admissibility floor visible. Every other view here exists
to set that up. Flip the self-report switch and watch a fluent, confident,
correct-sounding verdict get struck out because of what it rested on.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from adjudication.engine import Adjudicator, Mode
from agent.catalog import CATALOG
from bench.scenario import Scenario, load_dir
from bench.taxonomy import FaultClass
from schemas.evidence import EvidenceKind
from schemas.obligation import format_paise
from verifiers.base import Verdict
from verifiers.semantic import DEFAULT_CACHE, DEFAULT_MODEL, ResponseCache, SemanticVerifier


OUT = Path("ui/index.html")
SCENARIOS = Path("bench/scenarios")
PER_CLASS = 4


def rupees(paise: int) -> str:
    return format_paise(paise)


def evidence_rows(scenario: Scenario) -> list[dict]:
    rows = []
    for item in scenario.envelope.items:
        p = item.payload
        if hasattr(p, "lines"):
            first = p.lines[0]
            prod = CATALOG.get(first.sku)
            name = prod.display_name if prod else first.sku
            says = f"{name} x{first.quantity} — {rupees(first.unit_price_paise)}"
            if not getattr(p, "delivered", True):
                says += "  NOT DELIVERED"
        elif hasattr(p, "amount_paise"):
            ref = getattr(p, "payment_id", None) or getattr(p, "order_id", "")
            says = f"{ref} — {rupees(p.amount_paise)}" if ref else rupees(p.amount_paise)
        elif hasattr(p, "description"):
            says = p.description
        elif hasattr(p, "text"):
            says = p.text
        else:
            says = ""
        rows.append({
            "id": item.item_id,
            "kind": item.kind.value.replace("_", " ").lower(),
            "cls": item.evidence_class.name,
            "rank": int(item.evidence_class),
            "from": item.emitted_by,
            "says": says,
        })
    return rows


def decision_payload(scenario: Scenario, adj: Adjudicator, mode: Mode) -> dict:
    vi = scenario.to_verifier_input()
    d = adj.decide(vi, mode)

    env = vi.envelope
    if mode is Mode.GATE:
        from adjudication.engine import _gate_view
        env = _gate_view(vi).envelope

    votes = []
    for out in d.outputs:
        discarded = out.verifier_id in d.discarded
        try:
            basis = env.basis_class(out.basis).name if out.basis else "SELF_REPORT"
        except KeyError:
            basis = "SELF_REPORT"
        votes.append({
            "id": out.verifier_id,
            "role": out.role.value,
            "verdict": out.verdict.value,
            "basis": basis,
            "conf": round(out.confidence, 2),
            "reason": out.reason,
            "discarded": discarded,
            "model": out.role.value == "semantic",
        })

    return {
        "gate": d.gate_verdict.value,
        "fault": d.fault_class.value,
        "party": d.liable_party.value,
        "loss": rupees(d.loss_paise),
        "basis": d.basis_class.name,
        "cited": list(d.cited),
        "reason": d.reason,
        "llm": d.llm_invoked,
        "discarded": len(d.discarded),
        "votes": votes,
        "correct": d.fault_class is scenario.truth.fault_class and not d.abstained,
    }


def build_scenarios() -> list[dict]:
    dataset = load_dir(SCENARIOS)
    cache = ResponseCache(DEFAULT_CACHE)
    keyring = json.loads((SCENARIOS / "_keyring.json").read_text(encoding="utf-8"))

    plain = Adjudicator(
        semantic=SemanticVerifier(model=DEFAULT_MODEL, cache=cache), keyring=keyring
    )
    ablated = Adjudicator(
        semantic=SemanticVerifier(
            model=DEFAULT_MODEL, cache=cache, include_self_report=True
        ),
        keyring=keyring,
    )

    # Prefer scenarios where the ablation has a cached verdict to show, and
    # where the agent lied - those are the cases the floor exists for.
    chosen: list[Scenario] = []
    for fault in FaultClass:
        pool = [s for s in dataset.scenarios if s.truth.fault_class is fault]
        pool.sort(key=lambda s: (s.truth.agent_report_truthful, not s.truth.is_hard_pair))
        chosen.extend(pool[:PER_CLASS])

    out = []
    for s in chosen:
        ob = s.obligation
        constraints = [
            ("category", ob.hard.category),
            ("merchant", ", ".join(ob.hard.merchant_allowlist)),
            ("unit ceiling", rupees(ob.hard.max_unit_price_paise)),
            ("quantity", str(ob.hard.quantity)),
        ]
        for label, value in (("brand", ob.hard.brand), ("variant", ob.hard.variant),
                             ("pack", f"{ob.hard.pack_size_g}g" if ob.hard.pack_size_g else None)):
            if value:
                constraints.append((label, str(value)))

        req = CATALOG.get(s.truth.requested_sku)
        got = CATALOG.get(s.truth.ordered_sku)
        gap = ""
        if req and got and req.sku != got.sku:
            gap = f"{abs(req.unit_price_paise - got.unit_price_paise) / 100:.2f}"

        ablation = decision_payload(s, ablated, Mode.ATTRIBUTION)
        has_ablation = any(v["model"] and v["verdict"] != "ABSTAIN" for v in ablation["votes"])

        out.append({
            "id": s.scenario_id,
            "split": s.split,
            "said": ob.intent.text,
            "uncaptured": list(ob.intent.uncaptured_attributes),
            "constraints": constraints,
            "block": f"{rupees(ob.block.blocked_paise)} blocked, "
                     f"{rupees(ob.block.remaining_paise)} remaining",
            "signer": ob.signer_key_id or "unsigned",
            "evidence": evidence_rows(s),
            "attribution": decision_payload(s, plain, Mode.ATTRIBUTION),
            "gate": decision_payload(s, plain, Mode.GATE),
            "ablation": ablation,
            "hasAblation": has_ablation,
            "truth": {
                "fault": s.truth.fault_class.value,
                "party": s.truth.liable_party.value,
                "wanted": req.display_name if req else s.truth.requested_sku,
                "ordered": got.display_name if got else s.truth.ordered_sku,
                "delivered": (CATALOG.get(s.truth.delivered_sku).display_name
                              if CATALOG.get(s.truth.delivered_sku) else s.truth.delivered_sku),
                "truthful": s.truth.agent_report_truthful,
                "hard": s.truth.is_hard_pair,
                "gap": gap,
                "why": s.truth.rationale,
            },
        })
    # Open on the case the system exists for: a hard confusable pair where the
    # agent also lied about it. A page that opens on NO_FAULT buries the point.
    order = {
        "INTENT_MISMATCH": 0, "USER_REGRET": 1, "MERCHANT_SUBSTITUTION": 2,
        "INJECTION_INDUCED": 3, "NO_FAULT": 4, "CART_DRIFT": 5,
        "MANDATE_BREACH": 6, "DEBIT_MISMATCH": 7,
    }
    out.sort(key=lambda s: (
        order[s["truth"]["fault"]],
        not s["hasAblation"],        # ablation-capable first, for the demo
        not s["truth"]["hard"],      # hard pairs first
        s["truth"]["truthful"],      # lying agents first
    ))
    return out


def headline() -> dict:
    def read(name):
        p = Path(f"experiments/runs/{name}/result.json")
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    held = read("seed42-heldout-stub")
    full = read("seed42-all-stub")
    return {
        "model": DEFAULT_MODEL,
        "heldAcc": f"{held['overall_accuracy']:.1%}" if held else "—",
        "heldN": held["n"] if held else 0,
        "hard": f"{held['hard_pair_correct']}/{held['hard_pair_n']}" if held else "—",
        "hardPct": f"{held['hard_pair_correct'] / held['hard_pair_n']:.1%}" if held else "—",
        "falseClear": held["false_clearances"] if held else 0,
        "noModel": f"{full['deterministic_only'] / full['n']:.1%}" if full else "—",
        "blind": f"{full['gate']['blind_n'] / full['gate']['n']:.1%}" if full else "—",
    }


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mandate — how a payment gets adjudicated</title>
<style>
  :root {
    --ink: #10131a;
    --ink-2: #191d27;
    --line: #2b313f;
    --paper: #ece6d9;
    --paper-dim: #a9a294;
    --gold: #d9a441;
    --red: #d4634f;
    --green: #6f9b6e;
    --r0: #6b4a45;  /* self report   */
    --r1: #7d6a3f;  /* self signed   */
    --r2: #43684a;  /* merchant      */
    --r3: #3d6b6b;  /* psp receipt   */
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--ink); color: var(--paper);
    font: 16px/1.55 "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    -webkit-font-smoothing: antialiased;
  }
  code, .mono, table { font-family: "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace; }
  header {
    padding: 22px 30px; border-bottom: 1px solid var(--line);
    display: flex; flex-wrap: wrap; gap: 26px; align-items: baseline;
  }
  h1 { font-size: 21px; margin: 0; font-weight: 600; letter-spacing: .1px; }
  h1 span { color: var(--paper-dim); font-weight: 400; }
  .stats { margin-left: auto; display: flex; gap: 26px; flex-wrap: wrap; font-size: 13px; }
  .stat b { display: block; font-family: inherit; font-size: 20px; color: var(--gold); font-weight: 600; }
  .stat i { font-style: normal; color: var(--paper-dim); font-size: 12px; }

  .controls { padding: 16px 30px; border-bottom: 1px solid var(--line); display: flex;
              gap: 14px; align-items: center; flex-wrap: wrap; background: var(--ink-2); }
  select, button {
    background: var(--ink); color: var(--paper); border: 1px solid var(--line);
    padding: 8px 12px; border-radius: 3px; font: 13px/1 monospace; cursor: pointer;
  }
  button[aria-pressed="true"] { border-color: var(--gold); color: var(--gold); }
  button:focus-visible, select:focus-visible { outline: 2px solid var(--gold); outline-offset: 2px; }
  .hint { color: var(--paper-dim); font-size: 12.5px; }

  main { display: grid; grid-template-columns: 320px 1fr 340px; gap: 0; min-height: 70vh; }
  @media (max-width: 1100px) { main { grid-template-columns: 1fr; } }
  section { padding: 24px 30px; border-right: 1px solid var(--line); }
  section:last-child { border-right: 0; }
  h2 { font: 600 12px/1 monospace; letter-spacing: .12em; color: var(--paper-dim);
       margin: 0 0 16px; text-transform: lowercase; }

  .said { font-size: 19px; line-height: 1.45; margin: 0 0 18px; }
  .uncap { color: var(--gold); font-size: 13px; margin-bottom: 18px; font-family: monospace; }
  dl { margin: 0; font-size: 13px; }
  dl div { display: flex; gap: 12px; padding: 3px 0; }
  dt { color: var(--paper-dim); width: 96px; flex: none; font-family: monospace; }

  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  td { padding: 6px 8px 6px 0; vertical-align: top; border-bottom: 1px solid #1e222c; }
  .chip { display: inline-block; padding: 1px 7px; border-radius: 2px; font-size: 11px;
          border: 1px solid; white-space: nowrap; }
  .SELF_REPORT { color: #d09a92; border-color: var(--r0); background: #241a19; }
  .SELF_SIGNED { color: #d3bd8a; border-color: var(--r1); background: #24200f; }
  .MERCHANT_RECORD { color: #9dc79c; border-color: var(--r2); background: #16231a; }
  .PSP_RECEIPT { color: #93c6c6; border-color: var(--r3); background: #142322; }

  .vote { border-left: 2px solid var(--line); padding: 10px 0 10px 14px; margin-bottom: 4px; }
  .vote.fail { border-left-color: var(--red); }
  .vote.pass { border-left-color: var(--green); }
  .vote.abstain { border-left-color: var(--gold); }
  .vote.out { opacity: .48; }
  .vote.out .vname, .vote.out .vres { text-decoration: line-through; }
  .vote.out { border-left-color: var(--red); background:
    repeating-linear-gradient(135deg, transparent 0 8px, #2a1a18 8px 9px); }
  .vrow { display: flex; gap: 10px; align-items: center; font-family: monospace; font-size: 12.5px; }
  .vname { color: var(--paper); }
  .vres { color: var(--paper-dim); }
  .vreason { color: var(--paper-dim); font-size: 12.5px; margin-top: 5px;
             font-family: inherit; max-width: 66ch; }
  .struck { color: var(--red); font-family: monospace; font-size: 11.5px; margin-top: 5px; }

  .verdict { border: 1px solid var(--line); border-radius: 3px; padding: 18px; margin-bottom: 18px; }
  .verdict.BLOCK { border-color: var(--red); }
  .verdict.ALLOW { border-color: var(--green); }
  .verdict.ABSTAIN { border-color: var(--gold); }
  .big { font-size: 30px; font-weight: 600; letter-spacing: .5px; margin-bottom: 10px; }
  .BLOCK .big { color: var(--red); } .ALLOW .big { color: var(--green); }
  .ABSTAIN .big { color: var(--gold); }

  .truth { border: 1px dashed var(--line); border-radius: 3px; padding: 16px; font-size: 13px; }
  .truth.right { border-color: var(--green); } .truth.wrong { border-color: var(--red); }
  .truth h3 { font: 600 12px/1 monospace; margin: 0 0 12px; color: var(--paper-dim); }
  .why { color: var(--paper-dim); margin-top: 12px; line-height: 1.5; }
  .none { color: var(--paper-dim); font-size: 13px; }
  footer { padding: 18px 30px; border-top: 1px solid var(--line); color: var(--paper-dim);
           font-size: 12.5px; }
  @media (prefers-reduced-motion: no-preference) {
    .vote { transition: opacity .18s ease, border-color .18s ease; }
  }
</style>
</head>
<body>
<header>
  <h1>Mandate <span>— intent-aware verification for autonomous payments on UPI</span></h1>
  <div class="stats">
    <div class="stat"><b id="s-acc"></b><i>held-out accuracy</i></div>
    <div class="stat"><b id="s-hard"></b><i>hard confusable pairs</i></div>
    <div class="stat"><b id="s-fc"></b><i>false clearances</i></div>
    <div class="stat"><b id="s-nm"></b><i>decided with no model call</i></div>
    <div class="stat"><b id="s-blind"></b><i>invisible to a gate</i></div>
  </div>
</header>

<div class="controls">
  <select id="pick"></select>
  <button id="mode" aria-pressed="false">after the debit</button>
  <button id="abl" aria-pressed="false">let it read the agent's self-report</button>
  <span class="hint" id="hint"></span>
</div>

<main>
  <section>
    <h2>what the user asked for</h2>
    <p class="said" id="said"></p>
    <div class="uncap" id="uncap"></div>
    <dl id="constraints"></dl>
  </section>

  <section>
    <h2>evidence on record</h2>
    <table id="evidence"></table>
    <h2 style="margin-top:26px">verifier votes</h2>
    <div id="votes"></div>
  </section>

  <section>
    <h2>decision</h2>
    <div id="verdict"></div>
    <h2>ground truth, revealed after</h2>
    <div id="truth"></div>
  </section>
</main>

<footer id="foot"></footer>

<script>
const DATA = __DATA__;
const HEAD = __HEAD__;

let idx = 0, gate = false, ablate = false;

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

for (const [k, v] of Object.entries({
  's-acc': HEAD.heldAcc, 's-hard': HEAD.hardPct, 's-fc': HEAD.falseClear,
  's-nm': HEAD.noModel, 's-blind': HEAD.blind
})) $(k).textContent = v;

$('foot').textContent =
  'Every verdict shown comes from the committed response cache for ' + HEAD.model +
  '. The same run reproduces offline with: uv run python -m eval.runner --seed 42 --cache-only';

DATA.forEach((s, i) => {
  const o = document.createElement('option');
  o.value = i;
  o.textContent = s.truth.fault + '  ·  ' + s.id + (s.truth.hard ? '  ·  hard pair' : '');
  $('pick').appendChild(o);
});

function current() {
  const s = DATA[idx];
  if (ablate) return s.ablation;
  return gate ? s.gate : s.attribution;
}

function render() {
  const s = DATA[idx], d = current();

  $('said').textContent = '\\u201c' + s.said + '\\u201d';
  $('uncap').textContent = s.uncaptured.length
    ? 'not captured as a constraint: ' + s.uncaptured.join(', ')
    : '';
  $('constraints').innerHTML = s.constraints.map(
    ([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`
  ).join('') +
    `<div><dt>block</dt><dd>${esc(s.block)}</dd></div>` +
    `<div><dt>signature</dt><dd>verified · ${esc(s.signer)}</dd></div>`;

  const shown = gate && !ablate
    ? s.evidence.filter(e => !['merchant fulfilment', 'agent self report', 'user dispute'].includes(e.kind))
    : s.evidence;
  $('evidence').innerHTML = shown.map(e => `<tr>
      <td>${esc(e.kind)}</td>
      <td><span class="chip ${e.cls}">${e.cls.replace('_', ' ').toLowerCase()}</span></td>
      <td style="color:var(--paper-dim)">${esc(e.says).slice(0, 74)}</td>
    </tr>`).join('');

  $('votes').innerHTML = d.votes.map(v => `
    <div class="vote ${v.verdict.toLowerCase()} ${v.discarded ? 'out' : ''}">
      <div class="vrow">
        <span class="vname">${esc(v.id)}</span>
        <span class="vres">${v.verdict}</span>
        <span class="chip ${v.basis}">${v.basis.replace('_', ' ').toLowerCase()}</span>
        ${v.model ? '<span class="vres">model call</span>' : ''}
      </div>
      <div class="vreason">${esc(v.reason).slice(0, 200)}</div>
      ${v.discarded ? '<div class="struck">discarded — basis below the merchant-record floor, so this verdict weighs nothing</div>' : ''}
    </div>`).join('');

  $('verdict').innerHTML = `<div class="verdict ${d.gate}">
      <div class="big">${d.gate}</div>
      <dl>
        <div><dt>fault</dt><dd>${d.fault}</dd></div>
        <div><dt>liable</dt><dd>${d.party}</dd></div>
        <div><dt>exposure</dt><dd>${esc(d.loss)}</dd></div>
        <div><dt>basis</dt><dd>${d.basis.replace('_', ' ').toLowerCase()}</dd></div>
        <div><dt>model</dt><dd>${d.llm ? 'called' : 'not called'}</dd></div>
      </dl>
      <div class="vreason" style="margin-top:12px">${esc(d.reason).slice(0, 260)}</div>
    </div>`;

  const t = s.truth;
  $('truth').innerHTML = `<div class="truth ${d.correct ? 'right' : 'wrong'}">
      <h3>${d.correct ? 'system was correct' : 'system was wrong'}</h3>
      <dl>
        <div><dt>actual</dt><dd>${t.fault} · ${t.party}</dd></div>
        <div><dt>wanted</dt><dd>${esc(t.wanted)}</dd></div>
        <div><dt>ordered</dt><dd>${esc(t.ordered)}</dd></div>
        <div><dt>delivered</dt><dd>${esc(t.delivered)}</dd></div>
        <div><dt>agent said</dt><dd>${t.truthful ? 'the truth' : 'it bought the right thing — false'}</dd></div>
        ${t.gap ? `<div><dt>apart by</dt><dd>Rs ${t.gap}</dd></div>` : ''}
      </dl>
      <div class="why">${esc(t.why)}</div>
    </div>`;

  $('mode').textContent = gate ? 'before the debit' : 'after the debit';
  $('mode').setAttribute('aria-pressed', gate);
  $('abl').setAttribute('aria-pressed', ablate);
  $('mode').disabled = ablate;

  $('hint').textContent = ablate
    ? (s.hasAblation
        ? 'the semantic verifier now reads the agent\\u2019s account of itself'
        : 'no cached verdict for this scenario under the ablation')
    : (gate ? 'fulfilment, self-report and dispute withheld \\u2014 none of them exist yet' : '');
}

$('pick').onchange = e => { idx = +e.target.value; render(); };
$('mode').onclick = () => { gate = !gate; render(); };
$('abl').onclick = () => { ablate = !ablate; if (ablate) gate = false; render(); };
render();
</script>
</body>
</html>
"""


def main() -> None:
    scenarios = build_scenarios()
    page = (TEMPLATE
            .replace("__DATA__", json.dumps(scenarios, ensure_ascii=False))
            .replace("__HEAD__", json.dumps(headline(), ensure_ascii=False)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")

    with_abl = sum(1 for s in scenarios if s["hasAblation"])
    print(f"scenarios      {len(scenarios)}")
    print(f"with ablation  {with_abl}  (cached self-report verdicts)")
    print(f"written to     {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    print()
    print("Open it with a double click. No server needed.")


if __name__ == "__main__":
    main()
