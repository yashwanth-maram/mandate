"""
Ed25519 signing for obligations.

The obligation schema has carried `signature` and `signer_key_id` fields since
Wednesday and both have been null. This fills them, so "signed obligation"
describes the artifact rather than the intention.

WHAT SIGNING ADDS, AND WHAT IT DOES NOT

  Hashing gives tamper-evidence. The chain in schemas/evidence.py already
  proves a record was not altered after the fact.

  Signing gives non-repudiation. It proves WHO asserted something, which the
  hash cannot.

  Neither proves the assertion was TRUE.

That last line is the whole reason SELF_SIGNED sits at class 1, below the
performance floor. An agent that cryptographically signs "I bought the atta you
asked for" has produced an artifact it cannot later disown, and the artifact is
still a lie. Cryptography establishes authorship, not honesty, and a system
that confuses the two will be persuaded by whichever party signs most
confidently.

KEY HANDLING

Keys live under .keys/, which is gitignored. A key is generated on first use
and reused after. This is a demonstration, not a KMS: a real deployment would
hold the user's key in a secure element on their device and the merchant's in an
HSM, and neither would ever sit in a project directory. Saying so is better than
implying otherwise.

Usage:
    uv run python -m ledger.signer
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from schemas.obligation import Obligation


KEY_DIR = Path(".keys")


class Signer:
    """
    An Ed25519 keypair with a stable id.

    The id is the first 16 hex characters of the public key, so a signature can
    be traced to a key without a registry, and a rotated key produces a visibly
    different id rather than silently verifying against the wrong material.
    """

    def __init__(self, private_key: Ed25519PrivateKey, name: str = "user") -> None:
        self._key = private_key
        self.name = name

    # -- construction -------------------------------------------------------

    @classmethod
    def generate(cls, name: str = "user") -> "Signer":
        return cls(Ed25519PrivateKey.generate(), name)

    @classmethod
    def from_seed_bytes(cls, raw: bytes, name: str = "user") -> "Signer":
        """
        A keypair derived from supplied bytes rather than from os.urandom.

        The benchmark generator needs this: Ed25519PrivateKey.generate() draws
        from the OS entropy pool, so signing with it would make every scenario
        file differ between runs and break the reproducibility the whole
        evaluation rests on. Deriving from the seeded PRNG keeps `--seed 42`
        meaning one dataset.

        Not for production. A user's key must come from real entropy and never
        leave their device.
        """
        if len(raw) != 32:
            raise ValueError(f"Ed25519 needs 32 seed bytes, got {len(raw)}")
        return cls(Ed25519PrivateKey.from_private_bytes(raw), name)

    @classmethod
    def load_or_create(cls, name: str = "user", directory: Path = KEY_DIR) -> "Signer":
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.ed25519"

        if path.exists():
            key = serialization.load_pem_private_key(
                path.read_bytes(), password=None
            )
            if not isinstance(key, Ed25519PrivateKey):
                raise ValueError(f"{path} is not an Ed25519 private key")
            return cls(key, name)

        signer = cls.generate(name)
        path.write_bytes(
            signer._key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        (directory / f"{name}.pub").write_text(signer.public_key_b64(), encoding="utf-8")
        return signer

    # -- identity -----------------------------------------------------------

    def public_key_bytes(self) -> bytes:
        return self._key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def public_key_b64(self) -> str:
        return base64.b64encode(self.public_key_bytes()).decode("ascii")

    @property
    def key_id(self) -> str:
        return f"{self.name}:{self.public_key_bytes().hex()[:16]}"

    # -- signing ------------------------------------------------------------

    def sign_bytes(self, payload: bytes) -> str:
        return base64.b64encode(self._key.sign(payload)).decode("ascii")

    def sign_obligation(self, obligation: Obligation) -> Obligation:
        """
        Hash, then sign the same canonical bytes the hash covers.

        Signing the hash rather than the content would work too, but signing
        the canonical bytes directly means a verifier needs one canonical form
        rather than two, and there is no gap between what was hashed and what
        was signed for an inconsistency to hide in.
        """
        hashed = obligation.with_hash()
        return hashed.model_copy(update={
            "signature": self.sign_bytes(hashed.canonical_bytes()),
            "signer_key_id": self.key_id,
        })


def verify_obligation(obligation: Obligation, public_key_b64: str) -> bool:
    """
    Check both properties, and require both.

    The hash must cover the current content and the signature must cover the
    same bytes. A valid signature over content whose hash no longer matches
    means the object was edited after signing, and reporting that as verified
    because one of the two checks passed would be worse than doing neither.
    """
    if not obligation.hash_is_valid():
        return False
    if obligation.signature is None:
        return False

    key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64))
    try:
        key.verify(base64.b64decode(obligation.signature), obligation.canonical_bytes())
    except InvalidSignature:
        return False
    return True


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from datetime import timedelta

    from agent.catalog import get
    from schemas.obligation import (
        HardConstraints,
        Intent,
        ReserveBlock,
        paise,
    )

    now = datetime(2026, 9, 4, 2, 0, tzinfo=timezone.utc)
    product = get("aashirvaad-flour-whole_wheat-5000")

    unsigned = Obligation(
        obligation_id="obl_signer_demo",
        created_at=now,
        expires_at=now + timedelta(hours=6),
        user_id="usr_demo",
        agent_id="agt_demo",
        block=ReserveBlock(
            block_id="blk_demo",
            blocked_paise=paise(2000),
            blocked_at=now,
            expires_at=now + timedelta(days=30),
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
            text="get me 5kg Aashirvaad atta from Zepto, under 400 rupees",
            uncaptured_attributes=("variant",),
        ),
    )

    user = Signer.load_or_create("user")
    signed = user.sign_obligation(unsigned)

    assert signed.signature is not None
    assert signed.signer_key_id == user.key_id
    assert verify_obligation(signed, user.public_key_b64())

    # A different key does not verify.
    other = Signer.generate("impostor")
    assert not verify_obligation(signed, other.public_key_b64())

    # Editing the content after signing breaks both checks. Raising the ceiling
    # from Rs 400 to Rs 900 is the edit an agent would actually want to make.
    tampered = signed.model_copy(update={
        "hard": signed.hard.model_copy(update={"max_unit_price_paise": paise(900)})
    })
    assert not tampered.hash_is_valid()
    assert not verify_obligation(tampered, user.public_key_b64())

    # Rehashing the tampered object repairs the hash but not the signature.
    # This is the case that matters: hashing alone would now report clean.
    rehashed = tampered.with_hash()
    assert rehashed.hash_is_valid()
    assert not verify_obligation(rehashed, user.public_key_b64())

    print(f"signer        {user.key_id}")
    print(f"public key    {user.public_key_b64()}")
    print(f"obligation    {signed.obligation_id}")
    print(f"hash          {signed.content_hash[:24]}...")
    print(f"signature     {signed.signature[:24]}...")
    print()
    print("verified                                    True")
    print("verified with a different key               False")
    print("ceiling raised to Rs 900, hash checked      False")
    print("...and rehashed to repair the hash          False  <- signature still fails")
    print()
    print("Signing proves who asserted this, not that the assertion is true.")
    print("A signed agent self-report is still SELF_REPORT class, and the")
    print("performance floor still discards it.")
