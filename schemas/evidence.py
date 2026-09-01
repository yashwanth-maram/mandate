"""
Evidence envelope: the tamper-evident record of what actually happened.

Every artifact produced during a transaction lands here with three things
attached: who emitted it, when, and how much weight it can carry. Verifiers
read the envelope and must declare which items they relied on. The adjudicator
then weighs each verdict by the admissibility of its declared basis.

Two rules govern this file.

  A verdict is only as strong as the weakest evidence behind it.
      `basis_class()` returns the MINIMUM class across the items a verifier
      relied on, not the maximum. A verifier that consulted a signed PSP
      receipt and also leaned on the agent's self-report has a SELF_REPORT
      basis. Consulting strong evidence alongside weak evidence does not
      launder the weak evidence.

  A dispute claim is not evidence of what happened.
      A user asserting "I never wanted this" is what triggers adjudication.
      It is tagged SELF_REPORT, the same class as the agent's own account of
      itself, and cannot on its own establish fault against anyone. This
      mirrors how chargeback rules already treat an unsupported cardholder
      assertion, and it is what forces INTENT_MISMATCH and USER_REGRET to be
      separated by higher-class evidence rather than by who complains.

Layering note: EvidenceClass is defined in bench/taxonomy.py, which holds the
domain model for the benchmark. This module defines the wire format that
carries it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Annotated, Iterable, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from bench.taxonomy import EvidenceClass, PERFORMANCE_FLOOR


SCHEMA_VERSION = "evidence/1.0"

# Genesis link for the chain. An envelope's first item points here.
GENESIS_HASH = "0" * 64


# ---------------------------------------------------------------------------
# Kinds
# ---------------------------------------------------------------------------


class EvidenceKind(str, Enum):
    AGENT_SELF_REPORT = "AGENT_SELF_REPORT"
    AGENT_CART = "AGENT_CART"
    CATALOG_SNAPSHOT = "CATALOG_SNAPSHOT"
    MERCHANT_ORDER = "MERCHANT_ORDER"
    MERCHANT_FULFILMENT = "MERCHANT_FULFILMENT"
    PSP_ORDER = "PSP_ORDER"
    PSP_PAYMENT = "PSP_PAYMENT"
    USER_DISPUTE = "USER_DISPUTE"


# The class each kind carries by default. Set here rather than by the emitter,
# because an agent must not be able to declare its own report authoritative.
DEFAULT_CLASS: dict[EvidenceKind, EvidenceClass] = {
    # The agent's account of itself. Establishes nothing on its own.
    EvidenceKind.AGENT_SELF_REPORT: EvidenceClass.SELF_REPORT,
    # The agent's proposed cart, signed. Non-repudiation, not truth: it proves
    # the agent proposed this, not that proposing it was correct.
    EvidenceKind.AGENT_CART: EvidenceClass.SELF_SIGNED,
    # What the merchant was advertising at decision time. External to the
    # agent, and the only way to show catalogue content steered it.
    EvidenceKind.CATALOG_SNAPSHOT: EvidenceClass.MERCHANT_RECORD,
    EvidenceKind.MERCHANT_ORDER: EvidenceClass.MERCHANT_RECORD,
    EvidenceKind.MERCHANT_FULFILMENT: EvidenceClass.MERCHANT_RECORD,
    # Razorpay objects. External to both agent and merchant, and uninterested
    # in the outcome of an intent dispute. The strongest class available.
    EvidenceKind.PSP_ORDER: EvidenceClass.PSP_RECEIPT,
    EvidenceKind.PSP_PAYMENT: EvidenceClass.PSP_RECEIPT,
    # The complaint that starts adjudication. Not proof of its own contents.
    EvidenceKind.USER_DISPUTE: EvidenceClass.SELF_REPORT,
}


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


class _Payload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CartLine(_Payload):
    sku: str
    quantity: int = Field(gt=0)
    unit_price_paise: int = Field(gt=0)

    @property
    def line_total_paise(self) -> int:
        return self.quantity * self.unit_price_paise


class AgentSelfReport(_Payload):
    kind: Literal[EvidenceKind.AGENT_SELF_REPORT] = EvidenceKind.AGENT_SELF_REPORT
    text: str
    claimed_sku: Optional[str] = None
    claims_success: bool = True


class AgentCart(_Payload):
    kind: Literal[EvidenceKind.AGENT_CART] = EvidenceKind.AGENT_CART
    merchant_id: str
    lines: tuple[CartLine, ...] = Field(min_length=1)

    @property
    def total_paise(self) -> int:
        return sum(line.line_total_paise for line in self.lines)


class CatalogSnapshot(_Payload):
    """What the merchant was showing when the agent decided.

    `description` is verbatim. When catalogue content carries an injected
    instruction, this is the item that proves the agent was steered rather than
    simply wrong - which is the difference between PLATFORM fault and AGENT
    fault.
    """

    kind: Literal[EvidenceKind.CATALOG_SNAPSHOT] = EvidenceKind.CATALOG_SNAPSHOT
    merchant_id: str
    sku: str
    display_name: str
    description: str
    listed_price_paise: int = Field(gt=0)


class MerchantOrder(_Payload):
    kind: Literal[EvidenceKind.MERCHANT_ORDER] = EvidenceKind.MERCHANT_ORDER
    merchant_id: str
    merchant_order_id: str
    lines: tuple[CartLine, ...] = Field(min_length=1)

    @property
    def total_paise(self) -> int:
        return sum(line.line_total_paise for line in self.lines)


class MerchantFulfilment(_Payload):
    """What was actually shipped. Compared against MerchantOrder, this is the
    only artifact that separates a substitution from an agent error."""

    kind: Literal[EvidenceKind.MERCHANT_FULFILMENT] = EvidenceKind.MERCHANT_FULFILMENT
    merchant_order_id: str
    lines: tuple[CartLine, ...] = Field(min_length=1)
    delivered: bool = True


class PspOrder(_Payload):
    kind: Literal[EvidenceKind.PSP_ORDER] = EvidenceKind.PSP_ORDER
    order_id: str
    amount_paise: int = Field(gt=0)
    currency: str = "INR"
    receipt: Optional[str] = None


class PspPayment(_Payload):
    kind: Literal[EvidenceKind.PSP_PAYMENT] = EvidenceKind.PSP_PAYMENT
    payment_id: str
    order_id: str
    amount_paise: int = Field(gt=0)
    status: Literal["captured", "authorized", "failed", "refunded"] = "captured"
    block_id: Optional[str] = None
    method: str = "upi_reserve_pay"


class UserDispute(_Payload):
    kind: Literal[EvidenceKind.USER_DISPUTE] = EvidenceKind.USER_DISPUTE
    text: str
    raised_at: datetime


EvidencePayload = Annotated[
    Union[
        AgentSelfReport,
        AgentCart,
        CatalogSnapshot,
        MerchantOrder,
        MerchantFulfilment,
        PspOrder,
        PspPayment,
        UserDispute,
    ],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Items and envelope
# ---------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str
    seq: int = Field(ge=0)
    payload: EvidencePayload
    emitted_by: str
    emitted_at: datetime
    evidence_class: EvidenceClass
    prev_hash: str
    content_hash: str

    @property
    def kind(self) -> EvidenceKind:
        return self.payload.kind

    def canonical_bytes(self) -> bytes:
        body = self.model_dump(mode="json", exclude={"content_hash"})
        return json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    def compute_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def hash_is_valid(self) -> bool:
        return self.content_hash == self.compute_hash()


class ChainBreak(BaseModel):
    """Where and how the chain failed. Returned instead of a bare False so the
    adjudicator can cite the specific item rather than say 'something is off'."""

    seq: int
    item_id: str
    reason: str


class EvidenceEnvelope(BaseModel):
    """
    Hash-anchored container for one transaction's evidence.

    Anchored to the obligation by hash, so an envelope cannot be re-pointed at
    a different contract after the fact.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    envelope_id: str
    obligation_hash: str
    items: list[EvidenceItem] = Field(default_factory=list)

    # -- construction -------------------------------------------------------

    def append(
        self,
        payload: EvidencePayload,
        emitted_by: str,
        emitted_at: datetime,
        evidence_class: Optional[EvidenceClass] = None,
    ) -> EvidenceItem:
        """Add an item and link it into the chain.

        `evidence_class` defaults from the payload kind. It is overridable only
        so the generator can construct forgery scenarios in which an item
        claims a class its provenance does not support.
        """
        seq = len(self.items)
        prev = self.items[-1].content_hash if self.items else GENESIS_HASH
        cls = evidence_class if evidence_class is not None else DEFAULT_CLASS[payload.kind]

        item = EvidenceItem(
            item_id=f"{self.envelope_id}-e{seq:03d}",
            seq=seq,
            payload=payload,
            emitted_by=emitted_by,
            emitted_at=emitted_at,
            evidence_class=cls,
            prev_hash=prev,
            content_hash="",
        )
        item = item.model_copy(update={"content_hash": item.compute_hash()})
        self.items.append(item)
        return item

    # -- integrity ----------------------------------------------------------

    def verify_chain(self) -> Optional[ChainBreak]:
        """Return the first break, or None if the chain is intact."""
        expected_prev = GENESIS_HASH
        for i, item in enumerate(self.items):
            if item.seq != i:
                return ChainBreak(seq=i, item_id=item.item_id, reason="sequence out of order")
            if item.prev_hash != expected_prev:
                return ChainBreak(seq=i, item_id=item.item_id, reason="prev_hash does not match predecessor")
            if not item.hash_is_valid():
                return ChainBreak(seq=i, item_id=item.item_id, reason="content hash does not match payload")
            expected_prev = item.content_hash
        return None

    @property
    def is_intact(self) -> bool:
        return self.verify_chain() is None

    # -- access -------------------------------------------------------------

    def of_kind(self, kind: EvidenceKind) -> list[EvidenceItem]:
        return [i for i in self.items if i.kind is kind]

    def first_of_kind(self, kind: EvidenceKind) -> Optional[EvidenceItem]:
        found = self.of_kind(kind)
        return found[0] if found else None

    def by_id(self, item_id: str) -> Optional[EvidenceItem]:
        return next((i for i in self.items if i.item_id == item_id), None)

    # -- admissibility ------------------------------------------------------

    def basis_class(self, item_ids: Iterable[str]) -> EvidenceClass:
        """
        The class of a declared basis: the MINIMUM across the items relied on.

        A verdict is only as strong as its weakest support. Reading a PSP
        receipt does not repair a verdict that also leaned on the agent's
        self-report - the weak item is still load-bearing, so the basis is
        weak. This is the meet, and it is what makes the floor unbypassable:
        no set of individually inadmissible items clears it by being consulted
        together.
        """
        ids = list(item_ids)
        if not ids:
            return EvidenceClass.SELF_REPORT
        classes = []
        for iid in ids:
            item = self.by_id(iid)
            if item is None:
                raise KeyError(f"declared basis references unknown item: {iid}")
            classes.append(item.evidence_class)
        return min(classes)

    def meets_floor(
        self, item_ids: Iterable[str], floor: EvidenceClass = PERFORMANCE_FLOOR
    ) -> bool:
        return self.basis_class(item_ids) >= floor


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from datetime import timezone

    now = datetime(2026, 9, 2, 18, 45, tzinfo=timezone.utc)
    env = EvidenceEnvelope(envelope_id="env_demo_001", obligation_hash="c45fea25" + "0" * 56)

    # The agent orders maida when the user asked for atta, then reports success.
    cart = env.append(
        AgentCart(
            merchant_id="zepto",
            lines=(CartLine(sku="aashirvaad-flour-refined-1000", quantity=1, unit_price_paise=6200),),
        ),
        emitted_by="agt_demo",
        emitted_at=now,
    )
    order = env.append(
        MerchantOrder(
            merchant_id="zepto",
            merchant_order_id="zep_88213",
            lines=(CartLine(sku="aashirvaad-flour-refined-1000", quantity=1, unit_price_paise=6200),),
        ),
        emitted_by="zepto",
        emitted_at=now,
    )
    payment = env.append(
        PspPayment(payment_id="pay_test_9f2", order_id="order_test_7a1",
                   amount_paise=6200, block_id="blk_demo_001"),
        emitted_by="razorpay",
        emitted_at=now,
    )
    report = env.append(
        AgentSelfReport(text="Ordered the atta you asked for.",
                        claimed_sku="aashirvaad-flour-whole_wheat-1000"),
        emitted_by="agt_demo",
        emitted_at=now,
    )

    assert env.is_intact, "fresh chain should verify"

    # Admissibility: the meet, not the join.
    strong = env.basis_class([order.item_id, payment.item_id])
    assert strong == EvidenceClass.MERCHANT_RECORD, strong
    assert env.meets_floor([order.item_id, payment.item_id])

    tainted = env.basis_class([payment.item_id, report.item_id])
    assert tainted == EvidenceClass.SELF_REPORT, tainted
    assert not env.meets_floor([payment.item_id, report.item_id]), (
        "a PSP receipt must not launder a self-report"
    )

    # Tampering is detected, and the specific item is named.
    env.items[1] = env.items[1].model_copy(
        update={"payload": MerchantOrder(
            merchant_id="zepto",
            merchant_order_id="zep_88213",
            lines=(CartLine(sku="aashirvaad-flour-whole_wheat-1000",
                            quantity=1, unit_price_paise=7800),),
        )}
    )
    brk = env.verify_chain()
    assert brk is not None and brk.seq == 1, "tampering went undetected"

    print(f"envelope     {env.envelope_id}  ({len(env.items)} items)")
    print(f"basis  order+payment   -> {strong.name} (floor met)")
    print(f"basis  payment+report  -> {tainted.name} (floor NOT met, vote weighs zero)")
    print(f"tamper detected at seq {brk.seq} ({brk.item_id}): {brk.reason}")
    print("\nevidence schema ok")
