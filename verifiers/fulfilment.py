"""
Fulfilment verifier: did the shelf agree with the order.

Creating the fulfilment verifier split out of receipt, since order-versus-shipped is unanswerable before delivery.
"""

from __future__ import annotations

from bench.scenario import VerifierInput
from bench.taxonomy import FaultClass
from schemas.evidence import EvidenceKind, MerchantFulfilment, MerchantOrder
from verifiers.base import Verifier, VerifierOutput, VerifierRole


class FulfilmentVerifier(Verifier):
    role = VerifierRole.FULFILMENT
    verifier_id = "fulfilment/v1"

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
            return self._abstain("no merchant order to check fulfilment against")
        order: MerchantOrder = order_item.payload  # type: ignore[assignment]

        fulfilment_item = env.first_of_kind(EvidenceKind.MERCHANT_FULFILMENT)
        if fulfilment_item is None:
            return self._abstain(
                "no fulfilment record - cannot say whether what was ordered is what arrived"
            )
        fulfilment: MerchantFulfilment = fulfilment_item.payload  # type: ignore[assignment]

        if not fulfilment.delivered:
            return self._fail(
                f"merchant order {order.merchant_order_id} was not delivered",
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

        return self._pass(
            "fulfilment matches what was ordered",
            [order_item.item_id, fulfilment_item.item_id],
        )


if __name__ == "__main__":
    from collections import Counter
    from pathlib import Path

    from bench.scenario import load_dir
    from verifiers.base import Verdict

    verifier = FulfilmentVerifier()
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
