# Mandate

**Intent-aware verification for autonomous payments on UPI.**

Razorpay AI Buildathon 2026 — Track 02, AI Risk Manager
Loss class: *agent-performance failure* — authorised, in-mandate, correctly fulfilled, and still wrong.

---

## The problem

UPI is days from becoming one of the world's largest agentic payment rails. NPCI's Unified Agent Protocol is expected at Global Fintech Fest, 9–11 September, built on UPI Circle and Reserve Pay. By design, **NPCI's role stops at confirming a payment request is genuine — it does not see what was bought.** It holds logs to verify agent trust, nothing more.

Razorpay's Reserve Pay already runs in production with agentic partners. A user blocks funds once, capped at ₹10,000 for up to 90 days, and an agent debits repeatedly within that limit as value is delivered.

That creates a loss class no layer of the stack can see:

> A user asks for **Amul Taaza toned milk**. The agent orders **Amul Gold full cream** — same brand, same 1L pack, ₹11 more, inside every ceiling, at an allowed merchant. The merchant delivers exactly what was ordered. The debit reconciles. The agent reports success.
>
> Authorisation passes. Fraud scoring passes. URCS recognises no dispute ground, because non-delivery, wrong amount, and technical decline all did not occur.
>
> The user has the wrong milk and no recourse.

The gap is architectural, not an oversight. It can only be closed at the PSP-and-merchant layer — where Razorpay sits and NPCI does not.

**Why this is merchant loss.** When agent-performance failure does trigger a dispute, the buyer is made whole and the merchant is not. The merchant absorbs the reversal, loses inventory held for a sale that unwinds, pays the operational cost, and takes a hit to its chargeback ratio — a tracked, penalised metric that raises its processing costs. The class originates as user harm and settles as merchant loss.

---

Architecture and evidence model: [`docs/architecture.md`](docs/architecture.md)

## Results

500 scenarios, seed 42, `gemini-3.1-flash-lite` as the semantic verifier. Held-out split is 30%, assigned by hash of scenario id.

### Held-out (n = `141`)

| class | n | precision | recall | F1 | abstained |
|---|---:|---:|---:|---:|---:|
| INTENT_MISMATCH | `22` | `1.00` | `0.91` | `0.95` | `0` |
| USER_REGRET | `8` | `0.80` | `1.00` | `0.89` | `0` |
| MERCHANT_SUBSTITUTION | `18` | `1.00` | `1.00` | `1.00` | `0` |
| NO_FAULT | `29` | `1.00` | `1.00` | `1.00` | `0` |
| CART_DRIFT \* | `22` | `1.00` | `1.00` | `1.00` | `0` |
| MANDATE_BREACH \* | `20` | `1.00` | `1.00` | `1.00` | `0` |
| DEBIT_MISMATCH \* | `12` | `1.00` | `1.00` | `1.00` | `0` |
| INJECTION_INDUCED \* | `10` | `1.00` | `1.00` | `1.00` | `0` |

**\* These scores are definitional, not achievements.** These classes are detected by comparing values the obligation already fixed — a price against a ceiling, a merchant against an allowlist. A constraint checker cannot miss a numeric violation it is defined to catch. **The classes that carry information are INTENT_MISMATCH, USER_REGRET and MERCHANT_SUBSTITUTION.**

| | |
|---|---:|
| overall accuracy | `98.6%` |
| coverage | `100.0%` |
| accuracy on decided | `98.6%` |
| **false clearances** — faults allowed through as clean | **`0`** |
| **hard confusable pairs** — same brand, same pack, within ₹20 | **`89.5%`** (`17`/`19`) |
| decided with **zero model calls** | `58.2%` |

Accuracy and coverage are always reported together. A system can reach any accuracy it likes by abstaining on everything difficult.

### Cost, in rupees

| | |
|---|---:|
| caught, right party | ₹`80,468.43` |
| **misattributed** — charged to the wrong party | ₹`766.00` |
| missed | ₹`0.00` |
| abstained — escalated for review | ₹`0.00` |
| false blocks — clean sales stopped | ₹`0.00` |

### Gate, pre-debit

Post-debit evidence is withheld: no fulfilment record, no self-report, no dispute. A gate that could see those is hindsight.

| | |
|---|---:|
| correct | `411` |
| false blocks | `0` |
| missed | `12` |
| **gate-invisible** — fail after the debit decision | **`85` (`17.0%`)** |

That last row is the empirical argument for the second mode. **`17.0%` of failures cannot be reached by any pre-debit control**, because the evidence that identifies them does not exist yet. A firewall alone is not enough on this rail.

### The ablation: does the admissibility floor do real work?

The semantic verifier can read the agent's own account of itself — *"ordered the atta you asked for"* — which is fluent, confident, and false in 84 of 500 scenarios.

Running with `--self-report`, the verifier's declared basis includes a `SELF_REPORT` item, so the meet of its basis falls below the `MERCHANT_RECORD` floor.

| | measured |
|---|---:|
| escalations that returned a verdict | `69` |
| **discarded by the floor** | **`69` (100%)** |
| verdicts reaching the aggregate | `0` |
| escalations that errored before returning | `48` |
| overall accuracy | `57.0%` |

**Every verdict that came back was discarded.** Not because it was wrong — several were correct — but because of what it rested on. Accuracy collapses to the deterministic-only baseline, which is the floor working as specified.

**Incomplete run.** 48 of the ablation's escalations failed with `503 UNAVAILABLE` against the free-tier model and never returned a verdict. Those are abstentions, not discards, and are reported separately rather than folded into the discard count. The measured discard rate is 69 of 69 returned verdicts.

---

## Reproduce it

Every figure above regenerates offline, with no API key and no cost. Model responses are cached and committed.

```bash
uv sync
make gen     # 500 scenarios from seed 42, byte-identical each run
make eval    # replays the cached run, reproduces the table above
```

`make eval` uses `--cache-only`, so a cache miss is an error rather than a quietly different number.

To run against a live model instead:

```bash
uv run python -m eval.runner --seed 42 --live --provider anthropic --model <id>
```

---

## What it does

One engine, two entry points into the same verifier mesh.

**Pre-debit gate.** Inside the Reserve Pay block-to-debit gap, verify the proposed debit against the signed obligation. `ALLOW` / `BLOCK` / `ABSTAIN`, with a cited reason.

**Post-debit attribution.** Once money has moved, adjudicate fault across user, agent, merchant and platform from the evidence envelope.

### The mechanism

Evidence is **graded**, and verifiers must **declare what they relied on**.

| class | rank | what it is |
|---|---:|---|
| `SELF_REPORT` | 0 | the agent's account of itself. Establishes nothing. |
| `SELF_SIGNED` | 1 | signed by the agent. Non-repudiation, not truth. |
| `MERCHANT_RECORD` | 2 | the merchant's order and fulfilment records. |
| `PSP_RECEIPT` | 3 | Razorpay order and payment objects. External and uninterested. |

A verdict's basis class is the **minimum** across the items the verifier declared — the meet, not the join. Reading a Razorpay receipt alongside the agent's self-report does not repair the self-report; the weak item is still load-bearing.

Verdicts below `MERCHANT_RECORD` weigh **zero**, however confident.

Two rules follow, both of which took a bug to learn:

- **A clearance is a positive finding, not the absence of a complaint.** If the only verifier competent to judge intent abstains, nothing has established that intent was met. *(FAILURES.md #007 — this fix dropped accuracy from 77% to 57% by exposing 115 silent false clearances.)*
- **A verifier answers one question.** When it answers two, the weaker competence window silently governs both. *(#008)*

### Verifiers

| verifier | question | model? |
|---|---|---|
| `constraint` | ceilings, allowlist, quantity, block, expiry | no |
| `receipt` | does the money reconcile | no |
| `fulfilment` | did the shelf match the order | no |
| `provenance` | was the agent steered by catalogue content | no |
| `semantic` | did the purchase match what the user meant | **yes** |

**Four of five never call a model**, and the fifth is only invoked when nothing cheaper has settled the matter. `57.0%` of decisions are made with zero model calls, at a p50 of `3.83 ms`.

---

## The benchmark

500 scenarios, generated from canonical product state with **ground truth by construction**. A scenario is labelled `INTENT_MISMATCH` because the canonical variant differs, never because a model read a description and formed an opinion.

44 SKUs across 8 categories of Indian quick commerce, with **13 hard confusable pairs** — same brand, same pack size, within ₹20. Atta and maida. Amul Taaza and Amul Gold. Iodised salt and low-sodium salt. Those pairs carry the difficulty of the whole headline class; an easy swap would produce a flattering number that measured nothing.

Enforced at generation time:

- **No label leakage.** Ground truth may not appear in verifier input. The requested SKU may appear in a deceptive self-report — that is the case the floor exists for — but never in evidence at or above the performance floor.
- **Chains verify, obligations are signed.** Ed25519, keys derived from the seed so runs stay byte-identical. The adjudicator refuses to decide against an obligation whose signature does not verify.
- **Labels agree with the taxonomy.** A scenario claiming `MERCHANT_SUBSTITUTION` but attributing fault to `AGENT` fails generation.

### Razorpay integration

`PspOrder` is validated against **25 real Razorpay test-mode order objects**, created through the API and saved verbatim in `bench/fixtures/razorpay/`. `uv run python -m agent.razorpay_live --verify` reproduces the check offline.

This confirmed that Razorpay returns `amount` as an **integer in paise** — an assumption the whole schema rests on and which was previously untested.

**Payments are synthetic.** A payment object requires a checkout flow with a real instrument and cannot be created from a script. Saying so is better than implying otherwise.

---

## What I found

**All errors are in the harmful direction.** Every `INTENT_MISMATCH → USER_REGRET` misclassification tells a merchant that a wronged buyer is lying: the buyer is out of pocket and the complaint is on record as unfounded. The reverse error — refunding an unfounded complaint — occurred `0` times and costs a merchant a small sum. Same accuracy figure, very different harm. *(See the threshold sweep below.)*

**The confidence floor never fired.** At min_confidence = 0.6 the floor never fired — the model returned confidence above 0.6 on every escalation, including all 11 it got wrong. Abstentions only begin at 0.95. That is a calibration finding, not a success.

**Cost-optimal is not accuracy-optimal.** Sweeping the confidence threshold from cache:
| min_confidence | coverage | accuracy | harmful | cost |
|---|---:|---:|---:|---:|
| 0.50 (max accuracy) | 100.0% | 97.8% | 11 | ₹766.00 |
| 0.95 (min harm/cost) | 93.0% | 92.8% | 1 | ₹728.00 |

Accuracy-optimal and harm-optimal are different points. Moving from 0.50 to 0.95 gives up 7.0% coverage but removes 10 cases where a buyer who was wronged is recorded as having complained without cause.

---

## Related work

The admissibility-graded verification mesh is adapted from **RAILS** ([arXiv:2606.08790](https://arxiv.org/abs/2606.08790)), which specifies verification-native clearing for agentic commerce and names the problem this project addresses: an agent "can hold valid authorization, settle a valid payment, to an honest merchant who delivers exactly what was asked, and still leave the user harmed, because the agent asked for the wrong thing."

RAILS is a specification. This is an implementation on Indian rails, with the empirical evaluation the paper lists as open work — attribution accuracy under adversarial conditions. Simplified where a four-day build required it: four evidence classes rather than six, totally ordered rather than a poset.

**Google AP2**, **Mastercard Verifiable Intent** and **Visa TAP** define formats for recording agent intent as a signed audit trail. They establish what was authorised. None adjudicates whether the agent *performed*, and none targets UPI's block-and-debit semantics.

The injection defence follows the out-of-band enforcement line — **CaMeL**, **FIDES**, **Progent**, **RTBAS**, **FORGE** — evaluated on **AgentDojo**. Those systems prevent hostile instructions from reaching a tool call. This one asks a narrower question after the fact: given that a cart breached its bounds, does the record show the agent was steered, and does that move fault from the agent to the platform?

---

## Limitations

Stated because they are real, not to pre-empt criticism.

**This is an anticipatory control.** Agentic volume on UPI is currently a limited pilot and no Indian dispute data for this loss class exists. **I am not claiming measured losses.** The claim is that the gap is architectural, and this measures what a detector for it costs and how often it is wrong.

**The injection detector and its payloads share an author.** The result is a lower bound on detectability under a known attack distribution, not robustness to novel attacks. Mitigated by structural rather than literal matching, by 455 benign browse traces as a negative set, and by sanity strings the generator never produces — but not eliminated. AgentDojo-style evaluation against unseen payloads is future work.

**Deception occurs only in AGENT-fault scenarios.** An agent lies when what it did was wrong. This makes "self-report contradicts the record" a stronger signal than it would be in reality.

**13 hard pairs across 26 directed swaps** for `90` INTENT_MISMATCH scenarios. Limited product diversity in the subset that carries the headline metric.

**Compound faults are not modelled.** Every scenario has one fault. Real disputes have several — an agent that ordered wrongly *and* a merchant that then substituted. The schema reserves `secondary_fault`; nothing populates it.

**Single evaluation run.** No variance estimate across seeds or repeated model calls.

**Keys sit in `.keys/`.** A demonstration, not a KMS.

---

## Failure log

[`FAILURES.md`](FAILURES.md) — every non-trivial thing that broke, with the commit that fixed it. Written as the build happened, not assembled afterwards.

Two entries where **the reported number went down after the fix**:

- **#007** — the adjudicator was clearing transactions on silence. 77% → 57%, exposing 115 false clearances that had been counted as successes.
- **#009** — catalogue snapshots existed only on injection scenarios, so "has a snapshot" perfectly predicted the label. A one-line function would have scored as well as the detector.

---

## Layout

```
schemas/       obligation, evidence envelope, admissibility
ledger/        Ed25519 signing
agent/         catalogue, Razorpay test-mode integration
verifiers/     constraint, receipt, fulfilment, provenance, semantic
adjudication/  floor enforcement, gate view, conflict resolution
bench/         taxonomy, generator, 500 scenarios, fixtures
eval/          runner, metrics
experiments/   seeded run outputs
```

Every module runs standalone — `uv run python -m verifiers.constraint` scores that verifier alone against the benchmark.