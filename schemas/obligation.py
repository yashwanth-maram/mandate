"""
The obligation: a signed record of what the user actually asked for.

This is the contract. Every verifier decides against it, and nothing downstream
may treat any other artifact as authoritative about user intent.

The central structural decision here is the split between `hard` and `intent`:

  hard    Machine-checkable constraints the obligation compiler was able to
          extract unambiguously - price ceiling, quantity, merchant allowlist,
          a named brand, a validity window. Checked in plain Python. No model
          is consulted, and none is needed.

  intent  The user's own words, preserved verbatim, plus whatever the compiler
          could not reduce to a constraint. This is what the semantic verifier
          reads, and only when the deterministic layer cannot resolve the case.

That split is why CART_DRIFT and INTENT_MISMATCH are different classes rather
than arbitrary labels. They can describe the same underlying error - the agent
ordered maida when the user wanted atta - and which one applies depends on
whether the compiler captured "whole wheat" as a constraint or left it sitting
in free text. Obligation coverage is imperfect in reality, so the benchmark
models it as imperfect.

What is deliberately NOT in this schema: the SKU the user "really meant". The
user said words, not a SKU. Ground truth about the intended product lives in
the scenario record the eval harness holds, never in the obligation the
verifiers read. If the answer is present in the input, the metrics measure
nothing.

Money is integer paise throughout. Never float, never decimal rupees.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "obligation/1.0"


# ---------------------------------------------------------------------------
# Rail constants
# ---------------------------------------------------------------------------

# UPI Reserve Pay (Single Block Multi Debit). Banks currently cap a block at
# Rs 10,000 for up to 90 days. Reuters reported on 2026-09-01 that this ceiling
# and its validity may be revisited for agentic use; the value is pinned here
# so that any later change is a one-line edit with a visible diff rather than a
# magic number scattered through the verifiers.
RESERVE_PAY_MAX_BLOCK_PAISE = 1_000_000      # Rs 10,000
RESERVE_PAY_MAX_VALIDITY_DAYS = 90

RUPEE = 100  # paise


def paise(rupees: float) -> int:
    """Convert rupees to integer paise. Use only for literals in test fixtures."""
    return int(round(rupees * RUPEE))


def format_paise(amount: int) -> str:
    """Render paise as rupees for logs and UI. Presentation only."""
    return f"Rs {amount / RUPEE:,.2f}"


# ---------------------------------------------------------------------------
# Reserve Pay block
# ---------------------------------------------------------------------------


class ReserveBlock(BaseModel):
    """
    The SBMD funds block the agent debits against.

    The user authorises once; the agent debits repeatedly as value is
    delivered. The gap between the block and each debit is where this system
    runs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    block_id: str
    blocked_paise: int = Field(gt=0)
    consumed_paise: int = Field(default=0, ge=0)
    blocked_at: datetime
    expires_at: datetime

    @field_validator("blocked_paise")
    @classmethod
    def _within_rail_ceiling(cls, v: int) -> int:
        if v > RESERVE_PAY_MAX_BLOCK_PAISE:
            raise ValueError(
                f"block of {format_paise(v)} exceeds the Reserve Pay ceiling of "
                f"{format_paise(RESERVE_PAY_MAX_BLOCK_PAISE)}"
            )
        return v

    @model_validator(mode="after")
    def _consistent(self) -> "ReserveBlock":
        if self.consumed_paise > self.blocked_paise:
            raise ValueError("consumed exceeds blocked")
        if self.expires_at <= self.blocked_at:
            raise ValueError("block expires before it starts")
        max_days = RESERVE_PAY_MAX_VALIDITY_DAYS
        if (self.expires_at - self.blocked_at).days > max_days:
            raise ValueError(f"block validity exceeds {max_days} days")
        return self

    @property
    def remaining_paise(self) -> int:
        return self.blocked_paise - self.consumed_paise

    def can_absorb(self, amount_paise: int, at: datetime) -> bool:
        """Whether a debit of this size fits the block right now."""
        return amount_paise <= self.remaining_paise and at < self.expires_at


# ---------------------------------------------------------------------------
# Hard constraints
# ---------------------------------------------------------------------------


class HardConstraints(BaseModel):
    """
    What the compiler extracted unambiguously. Checked without a model.

    Optional fields are optional on purpose. A user who says "get me some atta"
    named no brand, so `brand` is None and no brand check runs. A user who says
    "Aashirvaad atta" named one, so a different brand becomes CART_DRIFT rather
    than a semantic judgement call.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: str                                   # canonical, e.g. "flour"
    max_unit_price_paise: int = Field(gt=0)
    max_total_paise: int = Field(gt=0)
    quantity: int = Field(default=1, gt=0)
    merchant_allowlist: tuple[str, ...] = Field(min_length=1)

    # Present only when the user named them explicitly.
    brand: Optional[str] = None
    variant: Optional[str] = None                   # e.g. "whole_wheat"
    pack_size_g: Optional[int] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _totals_consistent(self) -> "HardConstraints":
        if self.max_total_paise < self.max_unit_price_paise:
            raise ValueError("total ceiling below unit ceiling")
        return self


# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------


class Intent(BaseModel):
    """
    The user's own words, and what the compiler could not pin down.

    `uncaptured_attributes` is the compiler's own admission of what it left on
    the table. It is not the answer - it names which attributes were spoken
    about but not turned into constraints, without saying what their values
    should be. The semantic verifier uses it to know where to look.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    uncaptured_attributes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Obligation
# ---------------------------------------------------------------------------


class Obligation(BaseModel):
    """
    Signed, immutable, and the canonical reference for every downstream check.

    `content_hash` and `signature` are excluded from canonicalisation, since a
    hash cannot cover itself. Signing lives in ledger/signer.py; this schema
    only defines what gets signed and how it is serialised.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = SCHEMA_VERSION
    obligation_id: str
    created_at: datetime
    expires_at: datetime

    user_id: str
    agent_id: str

    block: ReserveBlock
    hard: HardConstraints
    intent: Intent

    content_hash: Optional[str] = None
    signature: Optional[str] = None
    signer_key_id: Optional[str] = None

    @model_validator(mode="after")
    def _window_consistent(self) -> "Obligation":
        if self.expires_at <= self.created_at:
            raise ValueError("obligation expires before it is created")
        if self.expires_at > self.block.expires_at:
            raise ValueError("obligation outlives the funds block backing it")
        if self.hard.max_total_paise > self.block.blocked_paise:
            raise ValueError("total ceiling exceeds the blocked amount")
        return self

    # -- canonicalisation ---------------------------------------------------

    def canonical_bytes(self) -> bytes:
        """
        Deterministic serialisation for hashing and signing.

        Sorted keys, no incidental whitespace, UTC ISO-8601 timestamps. Two
        semantically identical obligations must produce identical bytes on any
        machine, or the hash chain is decorative.
        """
        payload = self.model_dump(
            mode="json",
            exclude={"content_hash", "signature", "signer_key_id"},
        )
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    def compute_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def with_hash(self) -> "Obligation":
        return self.model_copy(update={"content_hash": self.compute_hash()})

    def hash_is_valid(self) -> bool:
        return self.content_hash is not None and self.content_hash == self.compute_hash()

    # -- deterministic checks -----------------------------------------------
    #
    # Lives here rather than in the verifier because these are properties of
    # the obligation itself. verifiers/constraint.py calls them and wraps the
    # result in a verdict with a declared evidence basis.

    def is_active(self, at: datetime) -> bool:
        return self.created_at <= at < self.expires_at

    def merchant_allowed(self, merchant_id: str) -> bool:
        return merchant_id in self.hard.merchant_allowlist

    def within_unit_ceiling(self, unit_price_paise: int) -> bool:
        return unit_price_paise <= self.hard.max_unit_price_paise

    def within_total_ceiling(self, total_paise: int) -> bool:
        return total_paise <= self.hard.max_total_paise


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def _example() -> Obligation:
    """A well-formed obligation. Used by the self-check and by tests."""
    now = datetime(2026, 9, 2, 18, 30, tzinfo=timezone.utc)

    block = ReserveBlock(
        block_id="blk_demo_001",
        blocked_paise=paise(2000),
        consumed_paise=paise(340),
        blocked_at=now,
        expires_at=datetime(2026, 10, 2, 18, 30, tzinfo=timezone.utc),
    )

    hard = HardConstraints(
        category="flour",
        max_unit_price_paise=paise(400),
        max_total_paise=paise(400),
        quantity=1,
        merchant_allowlist=("zepto",),
        brand="aashirvaad",
        pack_size_g=5000,
        # variant deliberately absent: the compiler heard "atta" but did not
        # turn it into a constraint, so ordering maida becomes INTENT_MISMATCH
        # rather than CART_DRIFT.
    )

    intent = Intent(
        text="get me 5kg Aashirvaad atta from Zepto, under 400 rupees",
        uncaptured_attributes=("variant",),
    )

    return Obligation(
        obligation_id="obl_demo_001",
        created_at=now,
        expires_at=datetime(2026, 9, 2, 23, 59, tzinfo=timezone.utc),
        user_id="usr_demo",
        agent_id="agt_demo",
        block=block,
        hard=hard,
        intent=intent,
    ).with_hash()


if __name__ == "__main__":
    o = _example()

    assert o.hash_is_valid(), "hash does not verify"
    assert o.compute_hash() == _example().compute_hash(), "canonicalisation is not stable"
    assert o.merchant_allowed("zepto")
    assert not o.merchant_allowed("blinkit")
    assert o.within_unit_ceiling(paise(355))
    assert not o.within_unit_ceiling(paise(410))
    assert o.block.can_absorb(paise(355), o.created_at)
    assert not o.block.can_absorb(paise(1900), o.created_at)

    # Rail ceiling is enforced, not documented.
    try:
        ReserveBlock(
            block_id="blk_bad",
            blocked_paise=paise(15000),
            blocked_at=o.created_at,
            expires_at=o.expires_at,
        )
    except ValueError as e:
        assert "ceiling" in str(e)
    else:
        raise AssertionError("Reserve Pay ceiling was not enforced")

    print(f"obligation   {o.obligation_id}")
    print(f"hash         {o.content_hash[:16]}...")
    print(f"block        {format_paise(o.block.blocked_paise)} "
          f"({format_paise(o.block.remaining_paise)} remaining)")
    print(f"intent       {o.intent.text!r}")
    print(f"uncaptured   {o.intent.uncaptured_attributes}")
    print("\nobligation schema ok")
