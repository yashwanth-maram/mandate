"""
Benchmark integrity.

If any of these fail, every number the project reports is measuring something
other than what it claims to measure.
"""

from __future__ import annotations

import json

import pytest

from bench.generator import generate
from bench.scenario import ScenarioSet, assert_no_leakage, assign_split
from bench.taxonomy import SCENARIO_MIX, TAXONOMY, FaultClass


def test_dataset_validates(dataset: ScenarioSet) -> None:
    """Chains intact, obligations hashed, labels agree with the taxonomy."""
    dataset.validate()


def test_no_ground_truth_reaches_the_verifiers(dataset: ScenarioSet) -> None:
    """
    The boundary the whole evaluation rests on.

    A verifier that can see the label is not being measured. This has fired
    twice during development - once on an enum value colliding with an evidence
    kind, once on an injection scenario naming the requested SKU in a catalogue
    snapshot.
    """
    for scenario in dataset.scenarios:
        assert_no_leakage(scenario)


def test_generation_is_deterministic() -> None:
    """
    Same seed, identical bytes.

    Signing keys are derived from the seeded PRNG rather than os.urandom
    precisely so this holds. If it breaks, `make eval` stops meaning anything
    on someone else's machine.
    """
    a = generate(seed=42, n=40)
    b = generate(seed=42, n=40)

    for left, right in zip(a.scenarios, b.scenarios):
        assert left.model_dump_json() == right.model_dump_json()


def test_different_seeds_differ() -> None:
    """Determinism must not have collapsed into ignoring the seed."""
    a = generate(seed=42, n=20)
    b = generate(seed=7, n=20)
    assert a.scenarios[0].model_dump_json() != b.scenarios[0].model_dump_json()


def test_splits_are_stable_across_dataset_size() -> None:
    """
    Splits hash the scenario id, not the position.

    Position-based splitting silently reshuffles the held-out set whenever n
    changes, which invalidates any comparison between runs of different sizes.
    """
    small = {s.scenario_id: s.split for s in generate(seed=42, n=40).scenarios}
    large = {s.scenario_id: s.split for s in generate(seed=42, n=200).scenarios}

    shared = set(small) & set(large)
    assert shared, "no overlapping ids to compare"
    for sid in shared:
        assert small[sid] == large[sid] == assign_split(sid)


def test_class_mix_matches_the_taxonomy(dataset: ScenarioSet) -> None:
    counts = dataset.counts()
    n = len(dataset)
    for fault, share in SCENARIO_MIX.items():
        assert abs(counts[fault] / n - share) < 0.01, fault.value


def test_liable_party_follows_the_fault_class(dataset: ScenarioSet) -> None:
    """A scenario cannot claim one class and attribute fault to another."""
    for s in dataset.scenarios:
        assert s.truth.liable_party is TAXONOMY[s.truth.fault_class].party


def test_clean_scenarios_carry_no_loss(dataset: ScenarioSet) -> None:
    for s in dataset.scenarios:
        if s.truth.fault_class is FaultClass.NO_FAULT:
            assert s.truth.loss_paise == 0


def test_every_scenario_has_a_browse_trace(dataset: ScenarioSet) -> None:
    """
    Catalogue snapshots on every scenario, not just injection ones.

    When only injection scenarios carried snapshots, their presence perfectly
    predicted the label and a detector that counted them would have scored as
    well as the real one. FAILURES.md #009.
    """
    from schemas.evidence import EvidenceKind

    for s in dataset.scenarios:
        assert s.envelope.of_kind(EvidenceKind.CATALOG_SNAPSHOT), s.scenario_id


def test_injection_is_not_predicted_by_snapshot_count(dataset: ScenarioSet) -> None:
    """The negative set is real: benign scenarios have traces of the same size."""
    from schemas.evidence import EvidenceKind

    tainted, benign = set(), set()
    for s in dataset.scenarios:
        n = len(s.envelope.of_kind(EvidenceKind.CATALOG_SNAPSHOT))
        if s.truth.fault_class is FaultClass.INJECTION_INDUCED:
            tainted.add(n)
        else:
            benign.add(n)

    assert tainted & benign, (
        "no snapshot count is shared between injection and benign scenarios, so "
        "counting snapshots would separate the classes"
    )


def test_money_is_always_integer_paise(dataset: ScenarioSet) -> None:
    """
    No float ever represents money.

    Floats in currency arithmetic accumulate error that shows up as
    reconciliation failures nobody can explain. Razorpay's own API returns
    integer paise; the schema follows it.
    """
    money_fields = {
        "amount_paise", "unit_price_paise", "loss_paise", "blocked_paise",
        "consumed_paise", "max_unit_price_paise", "max_total_paise",
        "listed_price_paise",
    }

    def walk(node: object, path: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in money_fields:
                    assert isinstance(value, int) and not isinstance(value, bool), (
                        f"{path}.{key} is {type(value).__name__}, expected int"
                    )
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    for s in dataset.scenarios[:100]:
        walk(json.loads(s.model_dump_json()), s.scenario_id)


def test_hard_pairs_exist_and_are_hard(dataset: ScenarioSet) -> None:
    """
    The headline metric needs difficult cases to be worth reporting.

    An INTENT_MISMATCH between atta and shampoo would score perfectly and mean
    nothing. Hard pairs share brand and pack size and sit within Rs 20.
    """
    from agent.catalog import get

    hard = [s for s in dataset.scenarios if s.truth.is_hard_pair]
    assert len(hard) >= 40, f"only {len(hard)} hard pairs"

    for s in hard:
        requested, ordered = get(s.truth.requested_sku), get(s.truth.ordered_sku)
        assert requested.brand == ordered.brand
        assert requested.pack_size_g == ordered.pack_size_g
        assert abs(requested.unit_price_paise - ordered.unit_price_paise) <= 2000
