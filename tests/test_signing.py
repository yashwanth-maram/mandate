"""
Signing, and the Razorpay schema mapping.

Both offline. The Razorpay tests read saved fixtures rather than calling the
API, so the suite runs on a machine with no keys and no network.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.catalog import get
from bench.scenario import ScenarioSet
from ledger.signer import Signer, verify_obligation
from schemas.obligation import (
    HardConstraints,
    Intent,
    Obligation,
    RESERVE_PAY_MAX_BLOCK_PAISE,
    ReserveBlock,
    paise,
)


NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
FIXTURES = Path("bench/fixtures/razorpay")


def _obligation() -> Obligation:
    return Obligation(
        obligation_id="obl_test",
        created_at=NOW,
        expires_at=NOW + timedelta(hours=6),
        user_id="usr_test",
        agent_id="agt_test",
        block=ReserveBlock(
            block_id="blk_test",
            blocked_paise=paise(2000),
            blocked_at=NOW,
            expires_at=NOW + timedelta(days=30),
        ),
        hard=HardConstraints(
            category="flour",
            max_unit_price_paise=paise(400),
            max_total_paise=paise(400),
            merchant_allowlist=("zepto",),
            brand="aashirvaad",
            pack_size_g=5000,
        ),
        intent=Intent(
            text="get me 5kg Aashirvaad atta from Zepto",
            uncaptured_attributes=("variant",),
        ),
    )


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------


def test_a_signed_obligation_verifies() -> None:
    signer = Signer.generate("user")
    signed = signer.sign_obligation(_obligation())

    assert signed.signature is not None
    assert signed.signer_key_id == signer.key_id
    assert verify_obligation(signed, signer.public_key_b64())


def test_another_key_does_not_verify() -> None:
    signed = Signer.generate("user").sign_obligation(_obligation())
    assert not verify_obligation(signed, Signer.generate("impostor").public_key_b64())


def test_rehashing_repairs_the_hash_but_not_the_signature() -> None:
    """
    The case that justifies having both.

    An attacker who edits an obligation and recomputes the hash passes a
    hash-only check completely clean. Raising the ceiling from Rs 400 to Rs 900
    is the edit an agent would actually want.
    """
    signer = Signer.generate("user")
    signed = signer.sign_obligation(_obligation())

    tampered = signed.model_copy(update={
        "hard": signed.hard.model_copy(update={"max_unit_price_paise": paise(900)})
    })
    assert not tampered.hash_is_valid()

    rehashed = tampered.with_hash()
    assert rehashed.hash_is_valid()
    assert not verify_obligation(rehashed, signer.public_key_b64())


def test_seeded_keys_are_deterministic() -> None:
    """Why the generator does not use os.urandom: it would break reproducibility."""
    raw = bytes(range(32))
    assert Signer.from_seed_bytes(raw).key_id == Signer.from_seed_bytes(raw).key_id
    assert Signer.from_seed_bytes(raw).key_id != Signer.from_seed_bytes(bytes(32)).key_id


def test_every_benchmark_obligation_verifies(
    dataset: ScenarioSet, keyring: dict[str, str]
) -> None:
    for s in dataset.scenarios:
        key_id = s.obligation.signer_key_id
        assert key_id in keyring, f"{s.scenario_id}: signer not in the keyring"
        assert verify_obligation(s.obligation, keyring[key_id]), s.scenario_id


def test_the_adjudicator_refuses_an_unverifiable_obligation(
    dataset: ScenarioSet, keyring: dict[str, str]
) -> None:
    """
    An obligation that cannot be authenticated is not a contract, and nothing
    downstream can be trusted if the thing every verifier decides against might
    have been issued by someone else.
    """
    from adjudication.engine import Adjudicator, Mode

    scenario = dataset.scenarios[0].model_copy(deep=True)
    forged = Signer.generate("forger")
    scenario = scenario.model_copy(update={
        "obligation": forged.sign_obligation(scenario.obligation)
    })

    decision = Adjudicator(keyring=keyring).decide(
        scenario.to_verifier_input(), Mode.ATTRIBUTION
    )
    assert decision.abstained
    assert "signature" in decision.reason.lower()


# ---------------------------------------------------------------------------
# Rail constants
# ---------------------------------------------------------------------------


def test_the_reserve_pay_ceiling_is_enforced_not_documented() -> None:
    """Banks cap a Reserve Pay block at Rs 10,000. The schema rejects more."""
    with pytest.raises(ValueError, match="ceiling"):
        ReserveBlock(
            block_id="blk_over",
            blocked_paise=RESERVE_PAY_MAX_BLOCK_PAISE + 1,
            blocked_at=NOW,
            expires_at=NOW + timedelta(days=30),
        )


def test_a_block_cannot_outlive_ninety_days() -> None:
    with pytest.raises(ValueError, match="validity"):
        ReserveBlock(
            block_id="blk_long",
            blocked_paise=paise(500),
            blocked_at=NOW,
            expires_at=NOW + timedelta(days=120),
        )


# ---------------------------------------------------------------------------
# Razorpay, from saved fixtures
# ---------------------------------------------------------------------------


def _fixtures() -> list[dict]:
    paths = sorted(FIXTURES.glob("order_*.json"))
    if not paths:
        pytest.skip("no Razorpay fixtures - run `uv run python -m agent.razorpay_live --n 25`")
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def test_razorpay_returns_integer_paise() -> None:
    """
    The assumption the whole schema rests on, checked against real responses
    rather than assumed. If Razorpay returned a decimal, every reconciliation
    in the receipt verifier would be quietly wrong.
    """
    for raw in _fixtures():
        assert isinstance(raw["amount"], int) and not isinstance(raw["amount"], bool)
        assert raw["currency"] == "INR"


def test_psp_order_maps_cleanly_from_real_responses() -> None:
    from agent.razorpay_live import check, to_psp_order

    for raw in _fixtures():
        assert check(raw) == [], raw.get("id")
        mapped = to_psp_order(raw)
        assert mapped.amount_paise == raw["amount"]
        assert mapped.receipt == raw.get("receipt")
