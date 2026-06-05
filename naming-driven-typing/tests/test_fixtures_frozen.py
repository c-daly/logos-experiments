"""Guard the committed fixtures load, validate, and stay label-free."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.fixtures_io import load_catalog, load_clusters

EXP_DIR = Path(__file__).resolve().parents[1]


def test_frozen_clusters_load_and_are_label_free() -> None:
    clusters = load_clusters(EXP_DIR / "fixtures" / "clusters.json")
    assert clusters
    for c in clusters:
        assert "label" not in c and "labels" not in c
        assert c["sample_coverage"] == 1.0  # no down-sampling (SPEC §6.1)


def test_frozen_catalog_loads_with_roots_and_vehicle_fragments() -> None:
    cat = load_catalog(EXP_DIR / "fixtures" / "catalog.json")
    assert cat["roots_present_in_live_catalog"] is True
    for root in ("entity", "concept", "process"):
        assert root in cat["by_norm"] and cat["by_norm"][root]
    assert len(cat["by_norm"]["vehicle"]) > 1  # fragments case (SPEC §4.2)


def test_no_labels_json_exists() -> None:
    # 2026-06-05 override: eval is label-free; labels.json must not exist.
    assert not (EXP_DIR / "fixtures" / "labels.json").exists()


@pytest.mark.xfail(
    reason="harness.catalog (T2) not yet landed; tightens once T2 merges",
    raises=ImportError,
)
def test_frozen_catalog_matches_t2_builder_schema() -> None:
    # The frozen catalog must carry exactly the keys the T2 builder emits
    # ({catalog_by_uuid, by_norm, roots_present_in_live_catalog}). Guarded by
    # xfail until harness.catalog (T2) merges; then this tightens for real.
    from harness.catalog import build_enriched_catalog  # noqa: F401

    cat = load_catalog(EXP_DIR / "fixtures" / "catalog.json")
    assert {"catalog_by_uuid", "by_norm", "roots_present_in_live_catalog"} <= set(cat)


@pytest.mark.xfail(
    reason="harness.catalog (T2) not yet landed; tightens once T2 merges",
    raises=ImportError,
)
def test_reseed_live_path_pins_real_catalog_symbol() -> None:
    # reseed_and_build lazily imports build_catalog_from_client from
    # harness.catalog — the live-path entry T2 actually exports. The source
    # check runs offline so a rename back to a phantom symbol fails HERE,
    # not as a NameError on the first live reseed; the hasattr check
    # tightens once T2 merges into this branch.
    import inspect

    from harness import reseed

    assert "build_catalog_from_client" in inspect.getsource(reseed.reseed_and_build)

    import harness.catalog as catalog_module

    assert hasattr(catalog_module, "build_catalog_from_client")
