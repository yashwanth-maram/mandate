# Failure Log

Append-only. Every non-trivial thing that broke during the build, what caused it,
and the commit that fixed it.

## 2026-09-02

### Entry #000 - baseline decision, recorded before any code
Chose a deterministic-first architecture: the LLM is one verifier among four,
not the pipeline. Constraint, receipt, and policy checks run in plain Python;
the semantic verifier is invoked only on cases the deterministic layer cannot
resolve, and its vote is discarded when it rests on the agent's own self-report.

Recorded now so later results can be read against the original intent rather
than rationalised after the fact.

### Entry #001 - uv not on PATH in PowerShell
Component: environment
Observed: `uv` resolved in cmd but not in PowerShell after install.
Cause: installer wrote to the User PATH; the running session and VS Code had
already snapshotted the old environment at launch.
Fix: appended %USERPROFILE%\.local\bin to the User PATH and restarted the shell.
Lesson: environment changes on Windows do not propagate to running processes.

### Entry #002 - leakage guard false positive
Component: bench/scenario.py
Observed: assert_no_leakage rejected a hand-built scenario, reporting that the
ground-truth value 'AGENT' was present in the verifier input.
Cause: the guard substring-searched serialised JSON for Party enum values.
"AGENT" is a substring of the evidence kind "AGENT_CART", which is legitimate
content. Short enum values collide under substring matching.
Fix: parse the JSON and match exact string leaves for enum values; keep
substring matching only for the rationale, which is long enough to be safe.
Lesson: a guard that has never been observed to fail is not a guard. This one
failed on its own first real input, which is the cheapest possible time.


### Entry #003 - leakage guard over-broad; injection snapshot was a label in disguise
Component: bench/scenario.py, bench/generator.py
Observed: generation failed on the first INJECTION_INDUCED scenario. The guard
reported the requested SKU present in a CatalogSnapshot at MERCHANT_RECORD class.
Cause: two problems at once. The guard treated any strong-class evidence naming
the requested SKU as leakage, but a catalogue listing records what the merchant
was offering, not what the user asked for. Separately, the builder emitted a
single snapshot - of the requested item - which really was a pointer to the
answer whatever the guard said.
Fix: narrowed the guard to exempt CATALOG_SNAPSHOT, and changed the builder to
emit a browse trace of four listings so the exemption is not load-bearing.
INTENT_MISMATCH emits no snapshots, so the class the guard protects is unaffected.
Lesson: when a guard fires, the honest question is whether the guard is wrong or
the data is. Here both were, and fixing only the guard would have been
special-casing to make a test pass.

### Entry #004 - shuffled a throwaway list
Component: bench/generator.py, _build_cart_drift
Observed: found while fixing #003. `rng.shuffle(list(modes))` builds a copy,
shuffles it, and discards it; `modes` is unchanged.
Cause: rng.shuffle mutates in place and returns None, so wrapping the argument
in list() silently does nothing.
Impact: the quantity mode always won, so brand and pack drift would never have
appeared in the benchmark. No error, no crash, just a class missing two thirds
of its variety.
Fix: shuffle a named list.
Lesson: silent nondeterminism bugs do not announce themselves. This one was
only caught because an unrelated failure sent me back into the same function.