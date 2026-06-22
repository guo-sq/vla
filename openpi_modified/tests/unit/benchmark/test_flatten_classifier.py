"""Tests for flatten_classifier — rule-based classification of flatten vs fold repos."""

import json
from pathlib import Path

import pytest

from scripts.benchmark.flatten_classifier import Confidence
from scripts.benchmark.flatten_classifier import FlattenLabel
from scripts.benchmark.flatten_classifier import RepoClassification
from scripts.benchmark.flatten_classifier import classify_repos
from scripts.benchmark.flatten_classifier import compute_viability
from scripts.benchmark.flatten_classifier import cross_validate
from scripts.benchmark.flatten_classifier import rule_classify

# ---------------------------------------------------------------------------
# Fixtures — minimal catalog / feishu entries
# ---------------------------------------------------------------------------


def _catalog_entry(
    repo_id: str,
    task_text: str = "Fold the checkered shirt.",
    n_episodes: int = 10,
    data_type: str = "record",
    n_frames: int = 500,
) -> dict:
    return {
        "repo_id": repo_id,
        "task_text": task_text,
        "n_episodes": n_episodes,
        "data_type": data_type,
        "n_frames": n_frames,
    }


def _feishu_entry(
    feishu_label: str = "fold",
    feishu_task: str = "叠衣服",
    feishu_annotation: str = "",
    feishu_section: str = "section_a",
) -> dict:
    return {
        "feishu_label": feishu_label,
        "feishu_task": feishu_task,
        "feishu_annotation": feishu_annotation,
        "feishu_section": feishu_section,
    }


@pytest.fixture
def catalog_path(tmp_path: Path) -> Path:
    catalog = {
        "repo.straighten": _catalog_entry(
            "repo.straighten",
            task_text="Straighten the checkered shirt.",
            n_episodes=5,
        ),
        "repo.lay_flat": _catalog_entry(
            "repo.lay_flat",
            task_text="Lay the checkered shirt flat.",
            n_episodes=3,
        ),
        "repo.disarrange": _catalog_entry(
            "repo.disarrange",
            task_text="Disarrange the towel on the table.",
            n_episodes=4,
            data_type="r_policy",
        ),
        "repo.fold_only": _catalog_entry(
            "repo.fold_only",
            task_text="Fold the checkered shirt.",
            n_episodes=8,
        ),
        "repo.bimodal": _catalog_entry(
            "repo.bimodal",
            task_text="Fold the checkered shirt.",
            n_episodes=12,
        ),
        "repo.puping": _catalog_entry(
            "repo.puping",
            task_text="Fold the checkered shirt.",
            n_episodes=7,
        ),
        "repo.preq_fold": _catalog_entry(
            "repo.preq_fold",
            task_text="Fold the checkered shirt.",
            n_episodes=6,
        ),
        "repo.milestone": _catalog_entry(
            "repo.milestone",
            task_text="Fold the checkered shirt.",
            n_episodes=9,
        ),
        "repo.non_flat": _catalog_entry(
            "repo.non_flat",
            task_text="Fold the checkered shirt.",
            n_episodes=11,
        ),
        "repo.model_test": _catalog_entry(
            "repo.model_test",
            task_text="Fold the checkered shirt.",
            n_episodes=2,
        ),
        "repo.catalog_only": _catalog_entry(
            "repo.catalog_only",
            task_text="Fold the checkered shirt.",
            n_episodes=15,
        ),
    }
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps(catalog))
    return p


@pytest.fixture
def feishu_path(tmp_path: Path) -> Path:
    feishu = {
        "repo.straighten": _feishu_entry(feishu_label="flatten", feishu_task="衣服铺平"),
        "repo.lay_flat": _feishu_entry(feishu_label="flatten", feishu_task="衣服铺平"),
        "repo.disarrange": _feishu_entry(feishu_label="unknown", feishu_task=""),
        "repo.fold_only": _feishu_entry(feishu_label="fold", feishu_task="叠衣服"),
        "repo.bimodal": _feishu_entry(feishu_label="fold", feishu_task="铺平或叠好衣服"),
        "repo.puping": _feishu_entry(feishu_label="flatten", feishu_task="衣服铺平"),
        "repo.preq_fold": _feishu_entry(feishu_label="fold", feishu_task="铺平情况下叠好"),
        "repo.milestone": _feishu_entry(feishu_label="fold", feishu_task="50s是到铺平就结束了"),
        "repo.non_flat": _feishu_entry(feishu_label="fold", feishu_task="非铺平状态开始叠"),
        "repo.model_test": _feishu_entry(feishu_label="model_test", feishu_task="测试模型"),
        # feishu-only repo (not in catalog)
        "repo.feishu_only": _feishu_entry(feishu_label="flatten", feishu_task="衣服铺平"),
        # weird note key — should be handled gracefully
        "record.clothes.bipiper.v0202.policy.5的04、14是到铺平就结束了": _feishu_entry(
            feishu_label="unknown", feishu_task="备注"
        ),
    }
    p = tmp_path / "feishu.json"
    p.write_text(json.dumps(feishu, ensure_ascii=False))
    return p


# ---------------------------------------------------------------------------
# Test: FlattenLabel and Confidence enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_flatten_label_values(self):
        assert FlattenLabel.FLATTEN == "flatten"
        assert FlattenLabel.FOLD == "fold"
        assert FlattenLabel.BIMODAL == "bimodal"
        assert FlattenLabel.DISARRANGE == "disarrange"
        assert FlattenLabel.NON_TASK == "non_task"
        assert FlattenLabel.UNKNOWN == "unknown"

    def test_confidence_values(self):
        assert Confidence.HIGH == "high"
        assert Confidence.MEDIUM == "medium"
        assert Confidence.LOW == "low"


# ---------------------------------------------------------------------------
# Test: rule_classify — individual rule matching
# ---------------------------------------------------------------------------


class TestRuleClassify:
    """Test each classification rule in isolation."""

    def test_rule1_straighten_english(self):
        label, conf, reason, rule_num = rule_classify(
            task_text="Straighten the checkered shirt.",
            feishu_task="",
        )
        assert label == FlattenLabel.FLATTEN
        assert conf == Confidence.MEDIUM
        assert rule_num == 1

    def test_rule2_lay_flat_english(self):
        label, conf, reason, rule_num = rule_classify(
            task_text="Lay the checkered shirt flat.",
            feishu_task="",
        )
        assert label == FlattenLabel.FLATTEN
        assert conf == Confidence.MEDIUM
        assert rule_num == 2

    def test_rule3_disarrange(self):
        label, conf, reason, rule_num = rule_classify(
            task_text="Disarrange the towel on the table.",
            feishu_task="",
        )
        assert label == FlattenLabel.DISARRANGE
        assert conf == Confidence.HIGH
        assert rule_num == 3

    def test_rule4_bimodal(self):
        label, conf, reason, rule_num = rule_classify(
            task_text="Fold the checkered shirt.",
            feishu_task="铺平或叠好衣服",
        )
        assert label == FlattenLabel.BIMODAL
        assert conf == Confidence.LOW
        assert rule_num == 4

    def test_rule5_prerequisite_fold(self):
        label, conf, reason, rule_num = rule_classify(
            task_text="Fold the checkered shirt.",
            feishu_task="铺平情况下叠好",
        )
        assert label == FlattenLabel.FOLD
        assert conf == Confidence.HIGH
        assert rule_num == 5

    def test_rule5_slanted_prerequisite(self):
        label, conf, reason, rule_num = rule_classify(
            task_text="Fold the checkered shirt.",
            feishu_task="斜铺平情况下叠好",
        )
        assert label == FlattenLabel.FOLD
        assert conf == Confidence.HIGH
        assert rule_num == 5

    def test_rule6_milestone(self):
        label, conf, reason, rule_num = rule_classify(
            task_text="Fold the checkered shirt.",
            feishu_task="50s是到铺平就结束了",
        )
        assert label == FlattenLabel.FOLD
        assert conf == Confidence.MEDIUM
        assert rule_num == 6

    def test_rule7_non_flat(self):
        label, conf, reason, rule_num = rule_classify(
            task_text="Fold the checkered shirt.",
            feishu_task="非铺平状态开始叠",
        )
        assert label == FlattenLabel.FOLD
        assert conf == Confidence.LOW
        assert rule_num == 7

    def test_rule8_pure_flatten(self):
        label, conf, reason, rule_num = rule_classify(
            task_text="Fold the checkered shirt.",
            feishu_task="衣服铺平",
        )
        assert label == FlattenLabel.FLATTEN
        assert conf == Confidence.HIGH
        assert rule_num == 8

    def test_rule8_excludes_bimodal(self):
        """'铺平或叠好' should NOT match rule 8 — it hits rule 4 first."""
        label, _, _, rule_num = rule_classify(
            task_text="Fold the checkered shirt.",
            feishu_task="铺平或叠好衣服",
        )
        assert rule_num == 4
        assert label == FlattenLabel.BIMODAL

    def test_rule10_default_fold(self):
        label, conf, reason, rule_num = rule_classify(
            task_text="Fold the checkered shirt.",
            feishu_task="叠衣服",
        )
        assert label == FlattenLabel.FOLD
        assert conf == Confidence.HIGH
        assert rule_num == 10

    def test_rule10_no_feishu(self):
        label, conf, reason, rule_num = rule_classify(
            task_text="Fold the checkered shirt.",
            feishu_task="",
        )
        assert label == FlattenLabel.FOLD
        assert conf == Confidence.HIGH
        assert rule_num == 10


# ---------------------------------------------------------------------------
# Test: rule_classify edge cases — rule priority
# ---------------------------------------------------------------------------


class TestRulePriority:
    """Verify higher-priority rules take precedence."""

    def test_straighten_overrides_feishu_flatten(self):
        """Rule 1 (English 'Straighten') fires before any feishu rules."""
        label, _, _, rule_num = rule_classify(
            task_text="Straighten the checkered shirt.",
            feishu_task="衣服铺平",
        )
        assert rule_num == 1

    def test_lay_flat_overrides_feishu(self):
        label, _, _, rule_num = rule_classify(
            task_text="Lay the checkered shirt flat.",
            feishu_task="铺平或叠好衣服",
        )
        assert rule_num == 2

    def test_rule5_over_rule8(self):
        """'铺平情况下叠好' matches rule 5 (fold) before rule 8 (flatten)."""
        label, _, _, rule_num = rule_classify(
            task_text="Fold the checkered shirt.",
            feishu_task="铺平情况下叠好衣服",
        )
        assert label == FlattenLabel.FOLD
        assert rule_num == 5


# ---------------------------------------------------------------------------
# Test: cross_validate
# ---------------------------------------------------------------------------


class TestCrossValidate:
    def test_rule_flatten_feishu_flatten_no_conflict(self):
        final, conf, conflict, detail = cross_validate(
            rule_label=FlattenLabel.FLATTEN,
            rule_confidence=Confidence.HIGH,
            feishu_label="flatten",
            feishu_annotation="",
        )
        assert final == FlattenLabel.FLATTEN
        assert not conflict

    def test_rule_flatten_feishu_fold_conflict(self):
        final, conf, conflict, detail = cross_validate(
            rule_label=FlattenLabel.FLATTEN,
            rule_confidence=Confidence.HIGH,
            feishu_label="fold",
            feishu_annotation="",
        )
        assert conflict
        assert conf != Confidence.HIGH  # confidence lowered

    def test_rule_fold_feishu_flatten_uses_feishu(self):
        """If feishu says flatten and annotation is '衣服铺平', trust feishu."""
        final, conf, conflict, detail = cross_validate(
            rule_label=FlattenLabel.FOLD,
            rule_confidence=Confidence.HIGH,
            feishu_label="flatten",
            feishu_annotation="衣服铺平",
        )
        assert final == FlattenLabel.FLATTEN
        assert conflict

    def test_rule_fold_feishu_flatten_without_annotation(self):
        """Conflict flagged even without confirming annotation."""
        final, conf, conflict, detail = cross_validate(
            rule_label=FlattenLabel.FOLD,
            rule_confidence=Confidence.HIGH,
            feishu_label="flatten",
            feishu_annotation="",
        )
        assert conflict

    def test_bimodal_feishu_fold_expected(self):
        """Bimodal + feishu fold is expected (59 repos), no conflict."""
        final, conf, conflict, detail = cross_validate(
            rule_label=FlattenLabel.BIMODAL,
            rule_confidence=Confidence.LOW,
            feishu_label="fold",
            feishu_annotation="",
        )
        assert not conflict
        assert final == FlattenLabel.BIMODAL

    def test_no_feishu_lowers_confidence(self):
        final, conf, conflict, detail = cross_validate(
            rule_label=FlattenLabel.FLATTEN,
            rule_confidence=Confidence.HIGH,
            feishu_label="",
            feishu_annotation="",
        )
        assert final == FlattenLabel.FLATTEN
        assert conf == Confidence.MEDIUM  # lowered one level

    def test_no_feishu_medium_becomes_low(self):
        final, conf, conflict, detail = cross_validate(
            rule_label=FlattenLabel.FLATTEN,
            rule_confidence=Confidence.MEDIUM,
            feishu_label="",
            feishu_annotation="",
        )
        assert conf == Confidence.LOW

    def test_non_task_feishu_labels(self):
        """Non-task feishu labels mapped via rule 9 are kept."""
        final, conf, conflict, detail = cross_validate(
            rule_label=FlattenLabel.NON_TASK,
            rule_confidence=Confidence.HIGH,
            feishu_label="model_test",
            feishu_annotation="",
        )
        assert final == FlattenLabel.NON_TASK
        assert not conflict


# ---------------------------------------------------------------------------
# Test: classify_repos integration — full pipeline
# ---------------------------------------------------------------------------


class TestClassifyRepos:
    def test_returns_all_repos(self, catalog_path: Path, feishu_path: Path):
        results = classify_repos(catalog_path, feishu_path)
        # Should cover all catalog repos + feishu-only repos
        repo_ids = {r.repo_id for r in results}
        assert "repo.straighten" in repo_ids
        assert "repo.feishu_only" in repo_ids
        assert "repo.catalog_only" in repo_ids

    def test_straighten_classified_flatten(self, catalog_path: Path, feishu_path: Path):
        results = classify_repos(catalog_path, feishu_path)
        by_id = {r.repo_id: r for r in results}
        r = by_id["repo.straighten"]
        assert r.final_label == FlattenLabel.FLATTEN
        assert r.in_catalog
        assert r.in_feishu

    def test_disarrange_classified(self, catalog_path: Path, feishu_path: Path):
        results = classify_repos(catalog_path, feishu_path)
        by_id = {r.repo_id: r for r in results}
        r = by_id["repo.disarrange"]
        assert r.final_label == FlattenLabel.DISARRANGE

    def test_bimodal_classified(self, catalog_path: Path, feishu_path: Path):
        results = classify_repos(catalog_path, feishu_path)
        by_id = {r.repo_id: r for r in results}
        r = by_id["repo.bimodal"]
        assert r.final_label == FlattenLabel.BIMODAL

    def test_model_test_non_task(self, catalog_path: Path, feishu_path: Path):
        results = classify_repos(catalog_path, feishu_path)
        by_id = {r.repo_id: r for r in results}
        r = by_id["repo.model_test"]
        assert r.final_label == FlattenLabel.NON_TASK

    def test_fold_default(self, catalog_path: Path, feishu_path: Path):
        results = classify_repos(catalog_path, feishu_path)
        by_id = {r.repo_id: r for r in results}
        r = by_id["repo.fold_only"]
        assert r.final_label == FlattenLabel.FOLD

    def test_catalog_only_repo(self, catalog_path: Path, feishu_path: Path):
        results = classify_repos(catalog_path, feishu_path)
        by_id = {r.repo_id: r for r in results}
        r = by_id["repo.catalog_only"]
        assert r.in_catalog
        assert not r.in_feishu

    def test_feishu_only_repo(self, catalog_path: Path, feishu_path: Path):
        results = classify_repos(catalog_path, feishu_path)
        by_id = {r.repo_id: r for r in results}
        r = by_id["repo.feishu_only"]
        assert not r.in_catalog
        assert r.in_feishu
        assert r.n_episodes == 0

    def test_weird_feishu_key_excluded(self, catalog_path: Path, feishu_path: Path):
        """Weird note-like keys should not appear as classified repos."""
        results = classify_repos(catalog_path, feishu_path)
        by_id = {r.repo_id: r for r in results}
        assert "record.clothes.bipiper.v0202.policy.5的04、14是到铺平就结束了" not in by_id

    def test_matched_rule_set(self, catalog_path: Path, feishu_path: Path):
        results = classify_repos(catalog_path, feishu_path)
        for r in results:
            assert 1 <= r.matched_rule <= 10

    def test_prerequisite_fold(self, catalog_path: Path, feishu_path: Path):
        results = classify_repos(catalog_path, feishu_path)
        by_id = {r.repo_id: r for r in results}
        r = by_id["repo.preq_fold"]
        assert r.final_label == FlattenLabel.FOLD
        assert r.matched_rule == 5

    def test_milestone_fold(self, catalog_path: Path, feishu_path: Path):
        results = classify_repos(catalog_path, feishu_path)
        by_id = {r.repo_id: r for r in results}
        r = by_id["repo.milestone"]
        assert r.final_label == FlattenLabel.FOLD
        assert r.matched_rule == 6


# ---------------------------------------------------------------------------
# Test: compute_viability
# ---------------------------------------------------------------------------


class TestComputeViability:
    def test_viable_when_enough_episodes(self):
        results = [
            RepoClassification(
                repo_id=f"repo_{i}",
                task_text="",
                feishu_task="",
                feishu_annotation="",
                rule_classification=FlattenLabel.FLATTEN,
                feishu_label="flatten",
                final_label=FlattenLabel.FLATTEN,
                confidence=Confidence.HIGH,
                reason="",
                n_episodes=10,
                has_conflict=False,
                conflict_detail="",
                in_catalog=True,
                in_feishu=True,
                matched_rule=8,
            )
            for i in range(4)
        ]
        v = compute_viability(results)
        assert v["confirmed_flatten_repos"] == 4
        assert v["confirmed_flatten_episodes"] == 40
        assert v["viable"]
        assert v["verdict"] == "VIABLE"

    def test_not_viable_few_episodes(self):
        results = [
            RepoClassification(
                repo_id="repo_0",
                task_text="",
                feishu_task="",
                feishu_annotation="",
                rule_classification=FlattenLabel.FLATTEN,
                feishu_label="flatten",
                final_label=FlattenLabel.FLATTEN,
                confidence=Confidence.HIGH,
                reason="",
                n_episodes=5,
                has_conflict=False,
                conflict_detail="",
                in_catalog=True,
                in_feishu=True,
                matched_rule=8,
            ),
        ]
        v = compute_viability(results)
        assert v["confirmed_flatten_episodes"] == 5
        assert not v["viable"]
        assert v["verdict"] == "NEEDS_MORE_DATA"

    def test_bimodal_counted_separately(self):
        results = [
            RepoClassification(
                repo_id="repo_bi",
                task_text="",
                feishu_task="",
                feishu_annotation="",
                rule_classification=FlattenLabel.BIMODAL,
                feishu_label="fold",
                final_label=FlattenLabel.BIMODAL,
                confidence=Confidence.LOW,
                reason="",
                n_episodes=100,
                has_conflict=False,
                conflict_detail="",
                in_catalog=True,
                in_feishu=True,
                matched_rule=4,
            ),
        ]
        v = compute_viability(results)
        assert v["bimodal_repos"] == 1
        assert v["bimodal_episodes"] == 100
        assert v["min_flatten_tp_estimate"] == 0
        assert v["max_flatten_tp_estimate"] == 100

    def test_repos_not_in_catalog_excluded(self):
        results = [
            RepoClassification(
                repo_id="repo_feishu",
                task_text="",
                feishu_task="",
                feishu_annotation="",
                rule_classification=FlattenLabel.FLATTEN,
                feishu_label="flatten",
                final_label=FlattenLabel.FLATTEN,
                confidence=Confidence.HIGH,
                reason="",
                n_episodes=0,
                has_conflict=False,
                conflict_detail="",
                in_catalog=False,
                in_feishu=True,
                matched_rule=8,
            ),
        ]
        v = compute_viability(results)
        assert v["confirmed_flatten_repos"] == 0


# ---------------------------------------------------------------------------
# Test: CLI output files
# ---------------------------------------------------------------------------


class TestCLIOutput:
    def test_writes_classification_json(self, catalog_path: Path, feishu_path: Path, tmp_path: Path):
        from scripts.benchmark.flatten_classifier import main as fc_main

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        fc_main(
            [
                "--catalog",
                str(catalog_path),
                "--feishu",
                str(feishu_path),
                "--output_dir",
                str(output_dir),
            ]
        )
        result_path = output_dir / "flatten_classification.json"
        assert result_path.exists()
        data = json.loads(result_path.read_text())
        assert isinstance(data, list)
        assert len(data) > 0
        # Check structure
        first = data[0]
        assert "repo_id" in first
        assert "final_label" in first
        assert "matched_rule" in first

    def test_writes_report_md(self, catalog_path: Path, feishu_path: Path, tmp_path: Path):
        from scripts.benchmark.flatten_classifier import main as fc_main

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        fc_main(
            [
                "--catalog",
                str(catalog_path),
                "--feishu",
                str(feishu_path),
                "--output_dir",
                str(output_dir),
            ]
        )
        report_path = output_dir / "flatten_report.md"
        assert report_path.exists()
        content = report_path.read_text()
        assert "Summary" in content
        assert "Confidence" in content
        assert "Conflict" in content
        assert "Viability" in content
