"""Unit tests for HEAD_PRED_RANGES injection in episode_classifier_2d.

The classifier ships with ``HEAD_PRED_RANGES`` as a module-level dict of
``{category: (min, max) | None}`` values, defaulting to all-None (pure 1D
tail_pred classification). After a sparse benchmark run, the post-sparse
pipeline computes per-category p5/p95 distributions and writes them to
``head_pred_ranges.json``. These tests cover two small plumbing helpers that
let ``classify_all`` pick up those learned ranges without mutating the
module-level constant or rewriting the .py file.

- ``load_head_pred_ranges_from_json(path)``: loads the JSON emitted by
  ``fill_head_pred_ranges`` and converts it into the ``(min, max) | None``
  format that ``build_default_regions`` expects. ``None`` entries
  (insufficient samples) remain ``None`` so the classifier stays 1D for
  those categories.

- ``build_default_regions(head_pred_ranges=...)``: optional override argument
  that takes precedence over the module-level constant. Categories missing
  from the override fall back to the module default.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark.episode_classifier_2d import Category
from scripts.benchmark.episode_classifier_2d import build_default_regions
from scripts.benchmark.episode_classifier_2d import load_head_pred_ranges_from_json

# ---------------------------------------------------------------------------
# load_head_pred_ranges_from_json
# ---------------------------------------------------------------------------


def test_load_reads_head_min_max_into_tuples(tmp_path: Path):
    """Each present category becomes ``(head_min, head_max)``; summary stats dropped."""
    path = tmp_path / "ranges.json"
    path.write_text(
        json.dumps(
            {
                "fold_success": {
                    "head_min": -0.82,
                    "head_max": -0.70,
                    "p5": -0.82,
                    "p25": -0.80,
                    "p50": -0.78,
                    "p75": -0.75,
                    "p95": -0.70,
                    "n_samples": 107,
                    "lower_percentile": 5.0,
                    "upper_percentile": 95.0,
                },
                "flatten_success": None,
                "shuffle_success": None,
                "fold_failure": None,
                "intervention_recovery": None,
            }
        )
    )

    result = load_head_pred_ranges_from_json(path)

    assert result["fold_success"] == pytest.approx((-0.82, -0.70))
    assert result["flatten_success"] is None
    assert result["shuffle_success"] is None
    assert result["fold_failure"] is None
    assert result["intervention_recovery"] is None


def test_load_uses_head_min_max_not_p5_p95(tmp_path: Path):
    """When the user picked wider bounds (e.g. lower_percentile=1, upper=99),
    head_min/head_max differ from p5/p95. The loader must honor the bounds,
    not the summary stats."""
    path = tmp_path / "wide.json"
    path.write_text(
        json.dumps(
            {
                "fold_success": {
                    "head_min": -0.95,  # p1, wider than p5
                    "head_max": -0.55,  # p99, wider than p95
                    "p5": -0.82,
                    "p95": -0.70,
                    "n_samples": 100,
                    "lower_percentile": 1.0,
                    "upper_percentile": 99.0,
                },
            }
        )
    )

    result = load_head_pred_ranges_from_json(path)
    assert result["fold_success"] == pytest.approx((-0.95, -0.55))


def test_load_all_categories_present(tmp_path: Path):
    def _entry(lo, hi):
        return {"head_min": lo, "head_max": hi, "p5": lo, "p95": hi, "n_samples": 50}

    path = tmp_path / "all.json"
    path.write_text(
        json.dumps(
            {
                "fold_success": _entry(-0.85, -0.70),
                "flatten_success": _entry(-0.50, -0.20),
                "shuffle_success": _entry(-0.10, 0.05),
                "fold_failure": _entry(-0.95, -0.60),
                "intervention_recovery": _entry(-0.40, -0.10),
            }
        )
    )

    result = load_head_pred_ranges_from_json(path)

    assert len(result) == 5
    assert all(isinstance(v, tuple) for v in result.values())


def test_load_rejects_missing_head_min_or_head_max(tmp_path: Path):
    """A malformed entry without ``head_min``/``head_max`` raises KeyError."""
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "fold_success": {"p50": -0.78, "n_samples": 5},
            }
        )
    )

    with pytest.raises((KeyError, ValueError)):
        load_head_pred_ranges_from_json(path)


# ---------------------------------------------------------------------------
# build_default_regions with head_pred_ranges override
# ---------------------------------------------------------------------------


def test_build_regions_no_override_keeps_1d_mode_by_default():
    """Without an override, the global HEAD_PRED_RANGES (all-None) is used
    and every region's head_min/head_max is ``None``."""
    regions = build_default_regions()

    assert len(regions) == 5
    for region in regions:
        assert region.head_min is None
        assert region.head_max is None


def test_build_regions_with_override_sets_head_bounds():
    """Passing a dict of ``(min, max)`` tuples should populate head bounds on
    the corresponding regions."""
    override = {
        "fold_success": (-0.82, -0.70),
        "flatten_success": None,
        "shuffle_success": (-0.05, 0.05),
        "fold_failure": None,
        "intervention_recovery": None,
    }
    regions = build_default_regions(head_pred_ranges=override)

    by_category = {r.category: r for r in regions}
    assert by_category[Category.FOLD_SUCCESS].head_min == pytest.approx(-0.82)
    assert by_category[Category.FOLD_SUCCESS].head_max == pytest.approx(-0.70)
    assert by_category[Category.SHUFFLE_SUCCESS].head_min == pytest.approx(-0.05)
    assert by_category[Category.SHUFFLE_SUCCESS].head_max == pytest.approx(0.05)
    # Categories explicitly None in the override stay 1D
    assert by_category[Category.FLATTEN_SUCCESS].head_min is None
    assert by_category[Category.FOLD_FAILURE].head_min is None
    assert by_category[Category.INTERVENTION_RECOVERY].head_min is None


def test_build_regions_partial_override_preserves_missing_categories():
    """If the override dict only has some categories, the missing ones fall
    back to the module-level HEAD_PRED_RANGES (currently all-None)."""
    override = {"fold_success": (-0.82, -0.70)}  # Only one category provided

    regions = build_default_regions(head_pred_ranges=override)
    by_category = {r.category: r for r in regions}

    assert by_category[Category.FOLD_SUCCESS].head_min == pytest.approx(-0.82)
    # Missing categories use the module default (all None -> 1D fallback)
    assert by_category[Category.SHUFFLE_SUCCESS].head_min is None
    assert by_category[Category.FLATTEN_SUCCESS].head_min is None


def test_build_regions_override_does_not_mutate_module_constant():
    """The override must not leak into the module-level HEAD_PRED_RANGES."""
    from scripts.benchmark.episode_classifier_2d import HEAD_PRED_RANGES

    override = {"fold_success": (-0.82, -0.70)}
    build_default_regions(head_pred_ranges=override)

    assert HEAD_PRED_RANGES["fold_success"] is None  # still the default
