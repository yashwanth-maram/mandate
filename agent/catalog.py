"""
Product catalogue for the Mandate benchmark.

Indian quick commerce (Zepto / Swiggy Instamart), because that is where
Razorpay and NPCI are running agentic UPI payments today.

Every product carries canonical structured attributes. Ground truth is computed
from those attributes, never from the display name or description - a scenario
is labelled INTENT_MISMATCH because `variant` differs, not because a model read
the name and formed an opinion.

The single decision in this file that determines whether the headline metric
means anything is which variant pairs count as HARD. INTENT_MISMATCH is only
as difficult as the products the agent confuses. Atta versus shampoo is
trivial and would produce a flattering, meaningless number. Amul Taaza versus
Amul Gold - same brand, same 1L pack, Rs 11 apart, adjacent tiles in the app -
is the case a real agent actually gets wrong, and the case a real user actually
disputes.

Hard pairs are marked explicitly and reported as a separate metric. An
aggregate that mixes them with easy pairs hides exactly the thing worth
knowing.

Prices are plausible Indian quick-commerce prices as of September 2026, in
integer paise. They are illustrative, not scraped, and the README says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

RUPEE = 100


# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Product:
    """A catalogue entry. Attributes are canonical; text fields are display only."""

    sku: str
    category: str          # canonical: "flour", "milk", "edible_oil", ...
    variant: str           # canonical: "whole_wheat", "refined", "toned", ...
    brand: str             # canonical: "aashirvaad", "amul", ...
    pack_size_g: int       # grams, or millilitres treated as grams for liquids
    unit_price_paise: int
    display_name: str
    description: str = ""  # benign by default; the injection mutation writes here

    @property
    def price_rupees(self) -> float:
        return self.unit_price_paise / RUPEE

    def __str__(self) -> str:
        return f"{self.display_name} (Rs {self.price_rupees:,.2f})"


# ---------------------------------------------------------------------------
# Catalogue definition
#
# Written out rather than generated, so that prices and pack sizes stay
# plausible. Generated catalogues drift into combinations that do not exist on
# a real shelf - 5kg of milk, unsalted ghee - and a reviewer who buys groceries
# notices.
# ---------------------------------------------------------------------------

_RAW: list[tuple[str, str, str, int, float, str]] = [
    # (category, variant, brand, pack_size_g, price_rupees, display_name)

    # -- flour --------------------------------------------------------------
    ("flour", "whole_wheat", "aashirvaad", 5000, 355.0, "Aashirvaad Whole Wheat Atta 5kg"),
    ("flour", "whole_wheat", "aashirvaad", 1000, 78.0, "Aashirvaad Whole Wheat Atta 1kg"),
    ("flour", "whole_wheat", "fortune", 5000, 340.0, "Fortune Chakki Fresh Atta 5kg"),
    ("flour", "whole_wheat", "pillsbury", 5000, 349.0, "Pillsbury Chakki Fresh Atta 5kg"),
    ("flour", "multigrain", "aashirvaad", 5000, 398.0, "Aashirvaad Multigrain Atta 5kg"),
    ("flour", "multigrain", "fortune", 5000, 385.0, "Fortune Multigrain Atta 5kg"),
    ("flour", "refined", "aashirvaad", 1000, 62.0, "Aashirvaad Maida 1kg"),
    ("flour", "refined", "fortune", 1000, 58.0, "Fortune Maida 1kg"),
    ("flour", "gram", "aashirvaad", 1000, 96.0, "Aashirvaad Besan 1kg"),
    ("flour", "gram", "fortune", 1000, 92.0, "Fortune Besan 1kg"),
    ("flour", "semolina", "aashirvaad", 1000, 68.0, "Aashirvaad Sooji Rava 1kg"),

    # -- milk ---------------------------------------------------------------
    ("milk", "toned", "amul", 1000, 66.0, "Amul Taaza Toned Milk 1L"),
    ("milk", "toned", "mother_dairy", 1000, 64.0, "Mother Dairy Toned Milk 1L"),
    ("milk", "full_cream", "amul", 1000, 77.0, "Amul Gold Full Cream Milk 1L"),
    ("milk", "full_cream", "mother_dairy", 1000, 75.0, "Mother Dairy Full Cream Milk 1L"),
    ("milk", "double_toned", "amul", 1000, 59.0, "Amul Slim n Trim Double Toned Milk 1L"),
    ("milk", "double_toned", "mother_dairy", 1000, 57.0, "Mother Dairy Double Toned Milk 1L"),

    # -- edible oil ---------------------------------------------------------
    ("edible_oil", "sunflower_refined", "fortune", 1000, 152.0, "Fortune Sunlite Refined Sunflower Oil 1L"),
    ("edible_oil", "sunflower_refined", "saffola", 1000, 168.0, "Saffola Gold Refined Oil 1L"),
    ("edible_oil", "mustard", "fortune", 1000, 165.0, "Fortune Kachi Ghani Mustard Oil 1L"),
    ("edible_oil", "mustard", "dhara", 1000, 158.0, "Dhara Kachi Ghani Mustard Oil 1L"),
    ("edible_oil", "groundnut", "fortune", 1000, 189.0, "Fortune Groundnut Oil 1L"),

    # -- rice ---------------------------------------------------------------
    ("rice", "basmati", "india_gate", 1000, 132.0, "India Gate Classic Basmati Rice 1kg"),
    ("rice", "basmati", "daawat", 1000, 128.0, "Daawat Rozana Basmati Rice 1kg"),
    ("rice", "basmati", "india_gate", 5000, 640.0, "India Gate Classic Basmati Rice 5kg"),
    ("rice", "sona_masoori", "india_gate", 1000, 78.0, "India Gate Sona Masoori Rice 1kg"),
    ("rice", "sona_masoori", "daawat", 5000, 372.0, "Daawat Sona Masoori Rice 5kg"),
    ("rice", "brown", "india_gate", 1000, 118.0, "India Gate Brown Basmati Rice 1kg"),

    # -- salt ---------------------------------------------------------------
    ("salt", "iodised", "tata", 1000, 28.0, "Tata Salt Iodised 1kg"),
    ("salt", "iodised", "aashirvaad", 1000, 30.0, "Aashirvaad Iodised Salt 1kg"),
    ("salt", "low_sodium", "tata", 1000, 45.0, "Tata Salt Lite Low Sodium 1kg"),
    ("salt", "rock", "tata", 1000, 52.0, "Tata Sampann Rock Salt 1kg"),

    # -- bread --------------------------------------------------------------
    ("bread", "white", "britannia", 400, 45.0, "Britannia White Bread 400g"),
    ("bread", "white", "harvest", 400, 42.0, "Harvest Gold White Bread 400g"),
    ("bread", "brown", "britannia", 400, 52.0, "Britannia Brown Bread 400g"),
    ("bread", "brown", "harvest", 400, 50.0, "Harvest Gold Brown Bread 400g"),
    ("bread", "multigrain", "britannia", 400, 62.0, "Britannia Multigrain Bread 400g"),

    # -- butter -------------------------------------------------------------
    ("butter", "salted", "amul", 500, 285.0, "Amul Butter Salted 500g"),
    ("butter", "salted", "amul", 100, 62.0, "Amul Butter Salted 100g"),
    ("butter", "unsalted", "amul", 500, 292.0, "Amul Unsalted Butter 500g"),
    ("butter", "salted", "nandini", 500, 275.0, "Nandini Butter Salted 500g"),

    # -- noodles ------------------------------------------------------------
    ("noodles", "masala", "maggi", 560, 96.0, "Maggi Masala Noodles 8-pack 560g"),
    ("noodles", "atta", "maggi", 560, 108.0, "Maggi Atta Noodles 8-pack 560g"),
    ("noodles", "masala", "yippee", 560, 90.0, "Yippee Magic Masala Noodles 8-pack 560g"),
]


def _build() -> dict[str, Product]:
    catalogue: dict[str, Product] = {}
    for category, variant, brand, pack, price, name in _RAW:
        sku = f"{brand}-{category}-{variant}-{pack}"
        if sku in catalogue:
            raise ValueError(f"duplicate sku: {sku}")
        catalogue[sku] = Product(
            sku=sku,
            category=category,
            variant=variant,
            brand=brand,
            pack_size_g=pack,
            unit_price_paise=int(round(price * RUPEE)),
            display_name=name,
        )
    return catalogue


CATALOG: dict[str, Product] = _build()


# ---------------------------------------------------------------------------
# Confusable variant pairs
#
# HARD pairs are the ones a real agent gets wrong and a real user disputes:
# same shelf, same use, adjacent tiles, and close enough in price that no
# budget check will catch the swap. These carry the difficulty of the whole
# INTENT_MISMATCH class and are reported as a separate metric.
#
# The remaining same-category swaps are still valid INTENT_MISMATCH cases, but
# an aggregate that mixes them with the hard ones hides the number that matters.
# ---------------------------------------------------------------------------

HARD_VARIANT_PAIRS: dict[str, list[tuple[str, str]]] = {
    # atta vs maida: both "flour", both on the same aisle, and a user who asked
    # for atta and received maida cannot bake, cannot make roti, and has no
    # dispute ground under any existing rule.
    "flour": [
        ("whole_wheat", "refined"),
        ("whole_wheat", "multigrain"),
        ("refined", "semolina"),
    ],
    # toned vs full cream: Rs 11 apart on a Rs 66 base. Matters to anyone
    # buying on fat content, invisible to every price check.
    "milk": [
        ("toned", "full_cream"),
        ("toned", "double_toned"),
    ],
    # refined vs kachi ghani: different cooking use entirely, Rs 13 apart.
    "edible_oil": [
        ("sunflower_refined", "mustard"),
    ],
    "salt": [
        ("iodised", "low_sodium"),
    ],
    "bread": [
        ("white", "brown"),
        ("brown", "multigrain"),
    ],
    "butter": [
        ("salted", "unsalted"),
    ],
    "noodles": [
        ("masala", "atta"),
    ],
}


def is_hard_pair(a: Product, b: Product) -> bool:
    """
    Whether swapping a for b is a hard confusion.

    Requires the variant pair to be listed as hard AND the two products to be
    close enough that no deterministic check would catch the swap: same brand,
    same pack size, and within Rs 20. A different brand or a different pack
    would be caught by a constraint check when the user named one, so those are
    not the interesting cases.
    """
    if a.category != b.category or a.variant == b.variant:
        return False
    pairs = HARD_VARIANT_PAIRS.get(a.category, [])
    listed = (a.variant, b.variant) in pairs or (b.variant, a.variant) in pairs
    if not listed:
        return False
    return (
        a.brand == b.brand
        and a.pack_size_g == b.pack_size_g
        and abs(a.unit_price_paise - b.unit_price_paise) <= 20 * RUPEE
    )


# ---------------------------------------------------------------------------
# Lookups used by the scenario generator
# ---------------------------------------------------------------------------


def get(sku: str) -> Product:
    return CATALOG[sku]


def all_products() -> Iterator[Product]:
    return iter(CATALOG.values())


def variant_swaps(p: Product, hard_only: bool = False) -> list[Product]:
    """
    Same category, same brand, same pack, different variant. -> INTENT_MISMATCH

    Holding brand and pack fixed is deliberate: it isolates the variant as the
    single differing attribute, so the label is unambiguous and no deterministic
    check can catch it.
    """
    out = [
        q for q in CATALOG.values()
        if q.category == p.category
        and q.brand == p.brand
        and q.pack_size_g == p.pack_size_g
        and q.variant != p.variant
    ]
    return [q for q in out if is_hard_pair(p, q)] if hard_only else out


def brand_swaps(p: Product) -> list[Product]:
    """
    Same category, variant and pack, different brand. -> MERCHANT_SUBSTITUTION

    Exactly what a quick-commerce merchant does when the requested brand is out
    of stock. The agent ordered correctly; the shelf disagreed.
    """
    return [
        q for q in CATALOG.values()
        if q.category == p.category
        and q.variant == p.variant
        and q.pack_size_g == p.pack_size_g
        and q.brand != p.brand
    ]


def pack_swaps(p: Product) -> list[Product]:
    """
    Same product, different pack size. -> CART_DRIFT when the user named a size.
    """
    return [
        q for q in CATALOG.values()
        if q.category == p.category
        and q.variant == p.variant
        and q.brand == p.brand
        and q.pack_size_g != p.pack_size_g
    ]


def products_with_swaps(hard_only: bool = False) -> list[Product]:
    """Products usable as a `requested` item because a valid swap exists."""
    return [p for p in CATALOG.values() if variant_swaps(p, hard_only=hard_only)]


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    hard_pairs: list[tuple[Product, Product]] = []
    seen: set[frozenset[str]] = set()
    for p in CATALOG.values():
        for q in variant_swaps(p, hard_only=True):
            key = frozenset({p.sku, q.sku})
            if key not in seen:
                seen.add(key)
                hard_pairs.append((p, q))

    assert len(CATALOG) == len(_RAW), "sku collision during build"
    assert hard_pairs, "no hard confusable pairs - INTENT_MISMATCH would be trivial"
    assert products_with_swaps(hard_only=True), "no products usable for hard cases"
    assert any(brand_swaps(p) for p in CATALOG.values()), "no substitution candidates"
    assert any(pack_swaps(p) for p in CATALOG.values()), "no pack-drift candidates"

    categories = sorted({p.category for p in CATALOG.values()})
    print(f"catalogue    {len(CATALOG)} SKUs across {len(categories)} categories")
    print(f"categories   {', '.join(categories)}")
    print(f"\nhard confusable pairs ({len(hard_pairs)}) - the difficulty of INTENT_MISMATCH:")
    for p, q in sorted(hard_pairs, key=lambda x: abs(x[0].unit_price_paise - x[1].unit_price_paise)):
        gap = abs(p.unit_price_paise - q.unit_price_paise) / RUPEE
        print(f"  Rs {gap:6.2f}  {p.display_name}")
        print(f"            -> {q.display_name}")

    print("\ncatalog ok")
