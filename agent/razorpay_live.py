"""
Razorpay test-mode integration.

Creates real orders against the Razorpay API in test mode, saves the raw
responses, and checks that the PspOrder schema this project uses maps cleanly
from them. The point is not to build a payment flow - it is to establish that
the evidence shapes the verifiers reconcile against are the shapes Razorpay
actually returns, rather than shapes I invented and then validated against
themselves.

WHAT IS REAL AND WHAT IS NOT

  Orders are real.     Created through client.order.create, fetched back from
                       the API to prove they exist server-side, and saved
                       verbatim under bench/fixtures/razorpay/.

  Payments are not.    A payment object requires a checkout flow with a real
                       instrument. It cannot be created from a script, so the
                       PspPayment objects in the benchmark are synthetic and
                       the README says so.

That distinction is stated rather than blurred. A reviewer who works on
payments will know a script cannot mint a captured payment, and claiming
otherwise would cost more credibility than the claim is worth.

WHAT THIS VALIDATES

  amount is an integer in paise      Razorpay's own field, confirming the
                                     integer-paise decision made in
                                     schemas/obligation.py rather than
                                     assuming it.
  receipt round-trips                the merchant order reference survives
  currency and status shapes         match what PspOrder expects
  created_at is a unix timestamp     not an ISO string

Usage:
    uv run python -m agent.razorpay_live --n 25      create and save fixtures
    uv run python -m agent.razorpay_live --verify    check saved fixtures only
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agent.catalog import all_products
from schemas.evidence import PspOrder
from schemas.obligation import format_paise


FIXTURES = Path("bench/fixtures/razorpay")


def client() -> Any:
    """A test-mode client, or a clear failure."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    key_id = os.environ.get("RAZORPAY_KEY_ID", "").strip()
    secret = os.environ.get("RAZORPAY_KEY_SECRET", "").strip()

    if not key_id or key_id.startswith("rzp_test_xxx"):
        raise SystemExit(
            "no RAZORPAY_KEY_ID in .env - get free test keys at "
            "dashboard.razorpay.com with Test Mode on"
        )
    if not key_id.startswith("rzp_test_"):
        # A live key here would create real orders against a real account.
        # Refusing is cheaper than explaining afterwards.
        raise SystemExit(
            f"RAZORPAY_KEY_ID is {key_id[:12]}..., which is not a test key. "
            f"This script refuses to run against live mode."
        )
    if not secret:
        raise SystemExit("no RAZORPAY_KEY_SECRET in .env")

    import razorpay

    return razorpay.Client(auth=(key_id, secret))


def to_psp_order(raw: dict[str, Any]) -> PspOrder:
    """
    Map a Razorpay order response onto the schema the verifiers read.

    Thin on purpose. If this needed reshaping or unit conversion, the schema
    would be wrong and the receipt verifier would be reconciling against a
    fiction.
    """
    return PspOrder(
        order_id=raw["id"],
        amount_paise=int(raw["amount"]),
        currency=raw["currency"],
        receipt=raw.get("receipt"),
    )


def check(raw: dict[str, Any]) -> list[str]:
    """Assumptions the rest of the codebase makes. Returns failures."""
    problems: list[str] = []

    if not isinstance(raw.get("amount"), int):
        problems.append(f"amount is {type(raw.get('amount')).__name__}, expected int")
    if raw.get("currency") != "INR":
        problems.append(f"currency is {raw.get('currency')!r}, expected 'INR'")
    if not str(raw.get("id", "")).startswith("order_"):
        problems.append(f"id is {raw.get('id')!r}, expected an order_ prefix")
    if not isinstance(raw.get("created_at"), int):
        problems.append("created_at is not a unix timestamp")
    if raw.get("status") not in ("created", "attempted", "paid"):
        problems.append(f"unexpected status {raw.get('status')!r}")

    try:
        mapped = to_psp_order(raw)
        if mapped.amount_paise != raw["amount"]:
            problems.append("amount did not survive the mapping")
        if mapped.receipt != raw.get("receipt"):
            problems.append("receipt did not survive the mapping")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"PspOrder rejected the response: {exc}")

    return problems


def create_fixtures(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """Create n real test-mode orders priced from the catalogue."""
    rng = random.Random(seed)
    products = list(all_products())
    api = client()
    saved: list[dict[str, Any]] = []

    FIXTURES.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        product = rng.choice(products)
        receipt = f"mandate_{i:03d}"
        created = api.order.create({
            "amount": product.unit_price_paise,
            "currency": "INR",
            "receipt": receipt,
            "notes": {
                "sku": product.sku,
                "source": "mandate-benchmark",
                "note": "schema validation fixture, test mode",
            },
        })

        # Fetch it back. A response object proves the request was accepted;
        # fetching proves the order exists on Razorpay's side.
        fetched = api.order.fetch(created["id"])

        path = FIXTURES / f"{fetched['id']}.json"
        path.write_text(json.dumps(fetched, indent=2, sort_keys=True), encoding="utf-8")
        saved.append(fetched)

        print(f"  {i + 1:>3}/{n}  {fetched['id']}  "
              f"{format_paise(fetched['amount']):>12}  {product.display_name[:34]}")

    (FIXTURES / "_meta.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "count": len(saved),
        "mode": "test",
        "seed": seed,
        "note": (
            "Real Razorpay test-mode order objects, saved verbatim. Payments are "
            "not included: a payment object requires a checkout flow with an "
            "instrument and cannot be created from a script, so PspPayment "
            "objects in the benchmark are synthetic."
        ),
    }, indent=2), encoding="utf-8")

    return saved


def load_fixtures() -> list[dict[str, Any]]:
    paths = sorted(p for p in FIXTURES.glob("order_*.json"))
    if not paths:
        raise SystemExit(
            f"no fixtures in {FIXTURES} - run with --n 25 first"
        )
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create Razorpay test-mode orders and validate the PspOrder schema."
    )
    ap.add_argument("--n", type=int, default=0, help="orders to create")
    ap.add_argument("--verify", action="store_true", help="check saved fixtures only")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.n:
        print(f"creating {args.n} test-mode orders")
        orders = create_fixtures(args.n, args.seed)
        print()
    elif args.verify:
        orders = load_fixtures()
        print(f"loaded {len(orders)} saved fixtures from {FIXTURES}")
        print()
    else:
        ap.error("pass --n to create fixtures or --verify to check saved ones")
        return

    failures = 0
    for raw in orders:
        problems = check(raw)
        if problems:
            failures += 1
            print(f"  {raw.get('id')}: " + "; ".join(problems))

    total = sum(int(o["amount"]) for o in orders)
    print(f"{'schema check':<28}{len(orders) - failures}/{len(orders)} clean")
    print(f"{'total value':<28}{format_paise(total)}")
    print()

    if failures:
        raise SystemExit(
            f"{failures} response(s) did not match the PspOrder schema. The "
            f"schema is wrong, not the API."
        )

    sample = orders[0]
    print("one real response, mapped:")
    print(f"  raw    id={sample['id']} amount={sample['amount']} "
          f"({type(sample['amount']).__name__}) currency={sample['currency']} "
          f"receipt={sample.get('receipt')}")
    mapped = to_psp_order(sample)
    print(f"  PspOrder  {mapped.order_id}  {format_paise(mapped.amount_paise)}  "
          f"receipt={mapped.receipt}")
    print()
    print("Orders are real. Payments are synthetic: a payment object needs a")
    print("checkout flow with an instrument and cannot be created from a script.")


if __name__ == "__main__":
    main()
