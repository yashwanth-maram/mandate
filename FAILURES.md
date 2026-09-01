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

