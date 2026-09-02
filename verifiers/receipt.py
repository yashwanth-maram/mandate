"""
Receipt verifier: does the money reconcile, and did the shelf agree with the order.

Two questions, both answered by comparing records that neither the agent nor a
model produced:

  Money.       Sum of captured debits against the merchant order total, and
               against the PSP order. Catches duplicates and over-debits.

  Fulfilment.  The merchant's order record against the merchant's own
               fulfilment record. This comparison is the only thing that
               separates a substitution from an agent error - both end with the
               user holding the wrong item, and the delivered item alone cannot
               tell you which happened.

One reconciliation rule has to be right or sixty scenarios get the wrong label:
payment reconciles against the ORDER, not against the fulfilment. A buyer is
charged for what was ordered. When a merchant ships a cheaper substitute the
debit is still correct - the substitution is the fault, not the amount.
Reconciling against the delivered item would turn every substitution into a
phantom over-debit and quietly move fault from the merchant to the agent.

Everything read here is MERCHANT_RECORD class or above. This verifier never
consults the agent's cart or self-report, so its basis always clears the floor.
"""

from __future__ import annotations

from collections import defaultdict

from bench.scenario import VerifierInput
from bench.taxonomy import FaultClass
from schemas.evidence import (
    EvidenceKind,
    MerchantFulfilment,
    MerchantOrder,
    PspOrder,
    PspPayment,
)
from schemas.obligation import format_paise
from verifiers.base import Verifier, VerifierOutput, VerifierRole


class ReceiptVerifier(Verifier):
    role = VerifierRole.RECEIPT
    verifier_id = "receipt/v1"

    def verify(self, vi: VerifierInput) -> VerifierOutput:
        env = vi.envelope

        brk = env.verify_chain()
        if brk is not None:
            return self._abstain(
                f"evidence chain broken at seq {brk.seq} ({brk.item_id}): {brk.reason}"
            )
        if env.obligation_hash != vi.obligation.content_hash:
            return self._abstain("envelope is not anchored to this obligation")

        order_item = env.first_of_kind(EvidenceKind.MERCHANT_ORDER)
        if order_item is None:
            return self._abstain("no merchant order to reconcile against")
        order: MerchantOrder = order_item.payload  # type: ignore[assignment]

        payment_items = [
            item for item in env.of_kind(EvidenceKind.PSP_PAYMENT)
            if item.payload.status == "captured"  # type: ignore[union-attr]
        ]
        if not payment_items:
            return self._abstain("no captured debit to reconcile")

        # -- duplicates -----------------------------------------------------
        # Grouped by PSP order id: two captures against one order and one
        # fulfilment is a double debit, whatever the amounts.

        by_order: dict[str, list] = defaultdict(list)
        for item in payment_items:
            payment: PspPayment = item.payload  # type: ignore[assignment]
            by_order[payment.order_id].append(item)

        for psp_order_id, items in by_order.items():
            if len(items) > 1:
                duplicated = sum(
                    i.payload.amount_paise for i in items[1:]  # type: ignore[union-attr]
                )
                ids = ", ".join(i.payload.payment_id for i in items)  # type: ignore[union-attr]
                return self._fail(
                    f"{len(items)} captured debits ({ids}) against a single PSP order "
                    f"{psp_order_id} with one fulfilment",
                    FaultClass.DEBIT_MISMATCH,
                    [i.item_id for i in items] + [order_item.item_id],
                    loss_paise=duplicated,
                )

        # -- amount ---------------------------------------------------------

        captured = sum(
            item.payload.amount_paise for item in payment_items  # type: ignore[union-attr]
        )
        if captured != order.total_paise:
            direction = "over" if captured > order.total_paise else "under"
            return self._fail(
                f"captured {format_paise(captured)} against a merchant order worth "
                f"{format_paise(order.total_paise)} ({direction}-debited by "
                f"{format_paise(abs(captured - order.total_paise))})",
                FaultClass.DEBIT_MISMATCH,
                [item.item_id for item in payment_items] + [order_item.item_id],
                loss_paise=abs(captured - order.total_paise),
            )

        psp_order_item = env.first_of_kind(EvidenceKind.PSP_ORDER)
        if psp_order_item is not None:
            psp_order: PspOrder = psp_order_item.payload  # type: ignore[assignment]
            if psp_order.amount_paise != order.total_paise:
                return self._fail(
                    f"PSP order {psp_order.order_id} raised for "
                    f"{format_paise(psp_order.amount_paise)} against a merchant order "
                    f"worth {format_paise(order.total_paise)}",
                    FaultClass.DEBIT_MISMATCH,
                    [psp_order_item.item_id, order_item.item_id],
                    loss_paise=abs(psp_order.amount_paise - order.total_paise),
                )

        # -- fulfilment -----------------------------------------------------

        fulfilment_item = env.first_of_kind(EvidenceKind.MERCHANT_FULFILMENT)
        if fulfilment_item is None:
            return self._abstain(
                "debit reconciles, but no fulfilment record - cannot say whether "
                "what was ordered is what arrived"
            )
        fulfilment: MerchantFulfilment = fulfilment_item.payload  # type: ignore[assignment]

        if not fulfilment.delivered:
            return self._fail(
                f"merchant order {order.merchant_order_id} was charged but the "
                f"fulfilment record shows it was not delivered",
                FaultClass.MERCHANT_SUBSTITUTION,
                [fulfilment_item.item_id, order_item.item_id],
                loss_paise=order.total_paise,
            )

        ordered = {line.sku: line.quantity for line in order.lines}
        shipped = {line.sku: line.quantity for line in fulfilment.lines}

        if ordered != shipped:
            missing = sorted(set(ordered) - set(shipped))
            extra = sorted(set(shipped) - set(ordered))
            if missing and extra:
                detail = f"ordered {missing[0]}, shipped {extra[0]}"
            elif extra:
                detail = f"shipped {extra[0]}, which was not ordered"
            elif missing:
                detail = f"ordered {missing[0]}, which was not shipped"
            else:
                changed = [
                    f"{sku} ordered {ordered[sku]}, shipped {shipped[sku]}"
                    for sku in ordered
                    if ordered[sku] != shipped[sku]
                ]
                detail = "; ".join(changed)

            return self._fail(
                f"merchant order and fulfilment disagree: {detail}. The agent's "
                f"order was placed correctly, so this is a substitution rather "
                f"than an agent error",
                FaultClass.MERCHANT_SUBSTITUTION,
                [order_item.item_id, fulfilment_item.item_id],
                loss_paise=order.total_paise,
            )

        basis = (
            [order_item.item_id, fulfilment_item.item_id]
            + [item.item_id for item in payment_items]
        )
        return self._pass(
            f"debit of {format_paise(captured)} reconciles against the merchant "
            f"order, and fulfilment matches what was ordered",
            basis,
        )


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from collections import Counter
    from pathlib import Path

    from bench.scenario import load_dir
    from verifiers.base import Verdict

    verifier = ReceiptVerifier()
    scenarios = load_dir(Path("bench/scenarios")).scenarios

    fired: Counter[str] = Counter()
    agreed: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    abstained = 0

    for s in scenarios:
        out = verifier.run(s.to_verifier_input())
        truth = s.truth.fault_class.value
        totals[truth] += 1
        if out.verdict is Verdict.FAIL:
            fired[truth] += 1
            if out.fault_class is s.truth.fault_class:
                agreed[truth] += 1
        elif out.verdict is Verdict.ABSTAIN:
            abstained += 1

    print(f"{'true class':<24}{'n':>5}{'FAIL':>7}{'agreed':>9}")
    print("-" * 45)
    for cls in sorted(totals, key=lambda c: -totals[c]):
        print(f"{cls:<24}{totals[cls]:>5}{fired[cls]:>7}{agreed[cls]:>9}")
    print("-" * 45)
    print(f"abstentions: {abstained}")
    print()
    print("Expected: DEBIT_MISMATCH and MERCHANT_SUBSTITUTION caught and agreed,")
    print("everything else silent. Any FAIL on NO_FAULT is a false positive and")
    print("costs a merchant a real sale.")
