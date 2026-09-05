# Video script — 5 minutes

Numbers marked `__` get filled after Friday's runs. Do not record with any left in.

**Setup**
- Terminal at 120 columns, dark, font large enough to read on a phone
- Recording: OBS or the built-in Windows recorder. Screen and voice together.
- Have every command pre-typed in a second terminal so you paste rather than type
- One clean take beats five edited ones. If you fluff a line, pause two seconds and say it again — cut later or leave it, nobody minds

**Tone.** Explaining to a colleague, not pitching. Slower than feels natural. The instinct at five minutes is to rush; resist it.

---

## 0:00 – 0:35 · The problem

*On screen: nothing yet, or the README title.*

> A user tells a shopping agent: get me a litre of Amul toned milk from Zepto.
>
> The agent orders Amul Gold. Full cream. Same brand, same one-litre pack, eleven rupees more. Inside every spending limit, at an allowed merchant. The merchant delivers exactly what was ordered. The debit reconciles perfectly. The agent reports success.
>
> Authorisation passed. Fraud scoring passed. And UPI recognises no dispute ground, because there was no non-delivery, no wrong amount, no technical decline.
>
> The user has the wrong milk and no recourse.

*Beat.*

> This isn't hypothetical for much longer. NPCI's Unified Agent Protocol is expected at Global Fintech Fest next week. And by design, NPCI confirms a payment request is genuine — it does not see what was bought.
>
> So this failure is invisible to the rail. It can only be caught where Razorpay sits.

---

## 0:35 – 1:05 · What it is

*On screen: `docs/architecture.md`, first diagram.*

> Mandate is a verification layer for agent payments on UPI. It runs in two places.
>
> Before a debit fires inside a Reserve Pay block, it decides whether to allow it. After money has moved, it decides who was at fault.
>
> Five verifiers. Four of them never call a model — they compare a price to a ceiling, a debit to an order, a shipment to what was ordered. Only one asks a language model anything, and it only runs when nothing cheaper has settled the matter.

---

## 1:05 – 2:05 · Live decision

*Run: `uv run python -m demo.scenarios --slow --class INTENT_MISMATCH`*

Narrate over it. Do not read the screen aloud — point at what matters.

> Here's the obligation. The user's actual words at the top. Underneath, what a compiler managed to extract as hard constraints — category, merchant, brand, pack size.
>
> Note this line. **Variant was not captured.** The user said "toned" and the compiler didn't turn it into a constraint. That's the gap this whole system lives in.

*Evidence table appears.*

> Every piece of evidence carries a class. The merchant's records and the Razorpay receipt sit high. The agent's own account of itself sits at the bottom, marked below floor.

*Votes appear.*

> Constraint passes — nothing breached. Receipt passes — the money reconciles. Fulfilment passes — they shipped what was ordered. Every deterministic check is clean, and they're all correct.
>
> Only the semantic verifier catches it.

*Decision, then truth panel.*

> And now ground truth, which the system never saw. Toned requested, full cream ordered. Eleven rupees apart.

---

## 2:05 – 2:50 · The floor

*Run: `uv run python -m demo.scenarios --ablation --slow --class INTENT_MISMATCH`*

**This is the centre of the video. Slow down here.**

> Same scenario, twice. The only thing that changes is what the semantic verifier is allowed to read.
>
> First pass: merchant records only. It gets it right.
>
> Second pass, it also reads the agent's self-report — which says *I ordered the toned milk you asked for*. That's a lie, and it's fluent and confident.

*The struck-through row appears.*

> The model is persuaded. And its vote is thrown out.
>
> Not because it was wrong. Because of what it rested on. A verdict is worth its weakest evidence — the agent's account of its own performance can't establish that the agent performed.
>
> Across the benchmark, `__` verdicts were discarded this way. Every one of them persuasive.

---

## 2:50 – 3:20 · Two modes

*Run: `uv run python -m demo.scenarios --gate --class MERCHANT_SUBSTITUTION`*

> Same transaction at the gate, before the debit. Fulfilment, self-report and dispute are all withheld — none of them exist yet.
>
> The gate allows it, and that's correct. The merchant substituted the brand *after* the debit decision.
>
> `__` percent of failures in the benchmark are like this: unreachable by any pre-debit control, because the evidence that identifies them doesn't exist when the money moves. That's the measured argument for having a second mode. A firewall alone is not enough on this rail.

---

## 3:20 – 4:10 · Numbers

*On screen: the held-out report.*

> Five hundred scenarios, ground truth by construction — labels come from canonical product state, never from a model reading a description.
>
> On the held-out split: `__` percent accuracy at `__` percent coverage. **Zero false clearances** — no fault was ever allowed through as clean.

*Point at the asterisked rows.*

> These four classes score perfectly and that's meaningless. A constraint checker can't miss a numeric violation it's defined to catch. The number that matters is this one —

*Point at hard pairs.*

> `__` percent on hard confusable pairs. Same brand, same pack, within twenty rupees. Atta and maida. Toned and full cream. Nothing deterministic separates those.
>
> And `__` percent of all decisions were made with **zero model calls**.

*Beat.*

> One more thing. Every misclassification went the same direction: telling a merchant that a buyer who was genuinely wronged was lying. The opposite error happened `__` times.
>
> Same accuracy number, very different harm. So the threshold sweep reports three operating points, not one — maximum accuracy, minimum cost, and minimum harm. They're different thresholds, and a payments system should choose the third.

---

## 4:10 – 4:50 · What broke

*On screen: `FAILURES.md`.*

> Eleven entries, written as it happened.

*Scroll to #007.*

> This one is the important one. The system was running at 77 percent. Then I noticed the semantic verifier was abstaining and the adjudicator was treating that silence as a clearance.
>
> A hundred and fifteen faults were being cleared as clean, and counted as successes.
>
> The fix dropped accuracy from 77 to 57. **The lower number was the honest one.**

*Scroll to #009.*

> And this one. My injection detector scored forty-five out of forty-five. Then I realised only injection scenarios had catalogue snapshots at all — so a function that just counted them would have scored exactly as well. I rebuilt the benchmark so every scenario has a browse trace, and the score only means something now because there are four hundred and fifty-five benign ones to be wrong about.

---

## 4:50 – 5:00 · Close

*On screen: terminal.*

```
make eval
```

> Every number I just showed reproduces from this command. Offline, no API key, no cost — the model responses are cached and committed.
>
> The code is on GitHub. Thanks for watching.

---

## Recording notes

**Rehearse the ablation section three times.** It's the part that carries the argument and the part where rushing loses people.

**Say the numbers out loud, don't just point.** People watch on phones with the sound on and the screen small.

**Don't apologise for anything.** No "this is just a prototype", no "I only had four days". The work stands.

**If you're short on time, cut the two-modes section**, not the ablation and not the failures.

**Pre-flight**
- [ ] no `__` left anywhere
- [ ] cache populated so semantic decides rather than abstains
- [ ] terminal wide enough that no table wraps
- [ ] `--slow` on every demo command
- [ ] audio actually recording — check thirty seconds first, not at the end