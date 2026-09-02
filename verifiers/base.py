"""
The verifier contract.

Every verifier - deterministic or semantic - returns the same five things:
a verdict, a confidence, the evidence it relied on, what it thinks went wrong,
and why. The adjudicator consumes nothing else.

Two properties in this file do real work.

  The declared basis is mandatory and validated.
      A verifier returning FAIL with an empty basis is asserting a violation it
      cannot point at, and the model rejects it. This is what makes the
      admissibility floor enforceable at all: the adjudicator weighs a verdict
      by the class of the evidence the verifier said it used, so a verifier
      that declares nothing can be weighed at nothing. Honest basis reporting
      is the price of admission - a verifier that names evidence it did not
      read, or omits evidence it did, has broken the contract and the floor
      stops protecting anything.

  A crash becomes an abstention.
      `run()` wraps `verify()`, times it, and converts any exception into
      ABSTAIN with the error as the reason. One malformed scenario must not
      kill a 500-scenario evaluation, and a verifier that errored should count
      as "did not decide" rather than silently disappearing from the aggregate.
      Silent disappearance would inflate the apparent agreement of whatever
      verifiers remained.

Adapted from the verifier signature in RAILS (arXiv:2606.08790), simplified for
this deployment: four role tags rather than five, no reliability priors, and a
totally ordered admissibility scale.
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bench.scenario import VerifierInput
from bench.taxonomy import FaultClass


class VerifierRole(str, Enum):
    """What kind of question this verifier answers."""

    CONSTRAINT = "constraint"     # hard constraints the obligation captured
    RECEIPT = "receipt"           # does the money reconcile
    FULFILMENT = "fulfilment"     # did the shelf match the order
    POLICY = "policy"             # NPCI and RBI rules, versioned
    PROVENANCE = "provenance"     # was the agent steered by untrusted content
    SEMANTIC = "semantic"         # did the action match what the user meant


class Verdict(str, Enum):
    PASS = "PASS"           # this verifier found nothing wrong
    FAIL = "FAIL"           # this verifier found a specific violation
    ABSTAIN = "ABSTAIN"     # this verifier cannot decide


class VerifierOutput(BaseModel):
    """
    One verifier's contribution. Never a decision on its own.

    `basis` is the load-bearing field. It names the evidence item ids the
    verifier actually relied on, and the adjudicator computes the meet of their
    admissibility classes to decide how much this verdict counts for.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    verifier_id: str
    role: VerifierRole
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    basis: tuple[str, ...] = ()

    # What this verifier believes went wrong, when it believes anything.
    fault_class: Optional[FaultClass] = None
    loss_paise: Optional[int] = Field(default=None, ge=0)

    reason: str = Field(min_length=1)

    # Set by policy verifiers so a decision can be traced to a specific rule at
    # a specific version. Payment rules change; a verdict that cannot name the
    # rule it applied cannot be audited later.
    rule_id: Optional[str] = None
    policy_version: Optional[str] = None

    latency_ms: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _contract(self) -> "VerifierOutput":
        if self.verdict is Verdict.FAIL:
            if not self.basis:
                raise ValueError(
                    f"{self.verifier_id}: FAIL with no declared basis - a violation "
                    f"that cannot be pointed at cannot be weighed"
                )
            if self.fault_class is None:
                raise ValueError(f"{self.verifier_id}: FAIL must name a fault class")
        if self.verdict is Verdict.PASS and self.fault_class is not None:
            raise ValueError(f"{self.verifier_id}: PASS must not name a fault class")
        return self

    @property
    def decided(self) -> bool:
        return self.verdict is not Verdict.ABSTAIN


class Verifier(ABC):
    """
    Base class. Subclasses implement `verify` and nothing else.

    Deterministic subclasses should return confidence 1.0 and mean it: a
    constraint checker that compares two integers is not 90% sure. Reserve
    graded confidence for the semantic verifier, where it carries information.
    """

    role: VerifierRole
    verifier_id: str

    @abstractmethod
    def verify(self, vi: VerifierInput) -> VerifierOutput:
        """Inspect the obligation and envelope. Never sees ground truth."""

    def run(self, vi: VerifierInput) -> VerifierOutput:
        """Time the verifier and isolate its failures."""
        started = time.perf_counter()
        try:
            out = self.verify(vi)
        except Exception as exc:  # noqa: BLE001 - deliberate: isolate the run
            elapsed = (time.perf_counter() - started) * 1000
            return VerifierOutput(
                verifier_id=self.verifier_id,
                role=self.role,
                verdict=Verdict.ABSTAIN,
                confidence=0.0,
                basis=(),
                reason=f"verifier raised {type(exc).__name__}: {exc}",
                latency_ms=elapsed,
            )
        elapsed = (time.perf_counter() - started) * 1000
        return out.model_copy(update={"latency_ms": elapsed})

    # -- construction helpers ----------------------------------------------

    def _out(
        self,
        verdict: Verdict,
        reason: str,
        *,
        basis: Sequence[str] = (),
        confidence: float = 1.0,
        fault_class: Optional[FaultClass] = None,
        loss_paise: Optional[int] = None,
        rule_id: Optional[str] = None,
        policy_version: Optional[str] = None,
    ) -> VerifierOutput:
        return VerifierOutput(
            verifier_id=self.verifier_id,
            role=self.role,
            verdict=verdict,
            confidence=confidence,
            basis=tuple(basis),
            fault_class=fault_class,
            loss_paise=loss_paise,
            reason=reason,
            rule_id=rule_id,
            policy_version=policy_version,
        )

    def _fail(
        self,
        reason: str,
        fault_class: FaultClass,
        basis: Sequence[str],
        *,
        loss_paise: Optional[int] = None,
        confidence: float = 1.0,
        rule_id: Optional[str] = None,
        policy_version: Optional[str] = None,
    ) -> VerifierOutput:
        return self._out(
            Verdict.FAIL, reason, basis=basis, confidence=confidence,
            fault_class=fault_class, loss_paise=loss_paise,
            rule_id=rule_id, policy_version=policy_version,
        )

    def _pass(
        self, reason: str, basis: Sequence[str], *, confidence: float = 1.0
    ) -> VerifierOutput:
        return self._out(Verdict.PASS, reason, basis=basis, confidence=confidence)

    def _abstain(self, reason: str, basis: Sequence[str] = ()) -> VerifierOutput:
        return self._out(Verdict.ABSTAIN, reason, basis=basis, confidence=0.0)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    class _Exploding(Verifier):
        role = VerifierRole.CONSTRAINT
        verifier_id = "exploding"

        def verify(self, vi: VerifierInput) -> VerifierOutput:
            raise RuntimeError("evidence item not found")

    # A FAIL with no basis is rejected by the contract.
    try:
        VerifierOutput(
            verifier_id="bad", role=VerifierRole.CONSTRAINT, verdict=Verdict.FAIL,
            confidence=1.0, basis=(), fault_class=FaultClass.CART_DRIFT,
            reason="quantity exceeded",
        )
    except ValueError as e:
        assert "declared basis" in str(e)
    else:
        raise AssertionError("FAIL with empty basis was accepted")

    # A FAIL that names no fault class is rejected.
    try:
        VerifierOutput(
            verifier_id="bad", role=VerifierRole.CONSTRAINT, verdict=Verdict.FAIL,
            confidence=1.0, basis=("e001",), reason="something is off",
        )
    except ValueError as e:
        assert "fault class" in str(e)
    else:
        raise AssertionError("FAIL with no fault class was accepted")

    # A PASS that names a fault class is rejected - passing means nothing found.
    try:
        VerifierOutput(
            verifier_id="bad", role=VerifierRole.CONSTRAINT, verdict=Verdict.PASS,
            confidence=1.0, basis=("e001",), fault_class=FaultClass.NO_FAULT,
            reason="fine",
        )
    except ValueError as e:
        assert "must not name a fault class" in str(e)
    else:
        raise AssertionError("PASS with a fault class was accepted")

    # A crashing verifier abstains rather than taking down the run.
    from bench.scenario import load_dir
    from pathlib import Path

    sample = load_dir(Path("bench/scenarios")).scenarios[0]
    crashed = _Exploding().run(sample.to_verifier_input())
    assert crashed.verdict is Verdict.ABSTAIN
    assert "RuntimeError" in crashed.reason
    assert crashed.latency_ms >= 0.0

    print("contract enforced:")
    print("  FAIL without a declared basis    rejected")
    print("  FAIL without a fault class       rejected")
    print("  PASS naming a fault class        rejected")
    print(f"  crash -> {crashed.verdict.value}  ({crashed.reason})")
    print("\nverifier base ok")
