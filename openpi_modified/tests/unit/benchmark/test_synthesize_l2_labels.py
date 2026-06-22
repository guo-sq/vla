"""Unit tests for scripts/benchmark/synthesize_l2_labels.py.

Covers priority ordering, exclusion handling, bimodal guard, task_group
inference, and confidence-based filtering.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark.synthesize_l2_labels import FIVE_CLASSES  # noqa: E402
from scripts.benchmark.synthesize_l2_labels import collect_exclusion_repos  # noqa: E402
from scripts.benchmark.synthesize_l2_labels import collect_repo_task_groups  # noqa: E402
from scripts.benchmark.synthesize_l2_labels import resolve_task_group  # noqa: E402
from scripts.benchmark.synthesize_l2_labels import synthesize  # noqa: E402


def _ep(key: str, category: str, confidence: str = "high") -> dict:
    return {
        "episode_key": key,
        "category": category,
        "confidence": confidence,
        "head_pred": -0.8,
        "tail_pred": 0.0,
        "n_frames": 100,
    }


class TestCollectExclusionRepos:
    def test_flattens_all_permanent_categories(self):
        excl = {
            "permanent_non_task": [{"repo_id": "r1"}],
            "permanent_data_quality": [{"repo_id": "r2"}],
            "permanent_structural": [{"repo_id": "r3"}],
            "temporary_upload_pending": [{"repo_id": "r4"}],
        }
        assert collect_exclusion_repos(excl) == {"r1", "r2", "r3", "r4"}

    def test_ignores_unknown_sections(self):
        assert collect_exclusion_repos({"_meta": {}, "summary": {}}) == set()


class TestCollectRepoTaskGroups:
    def test_splits_bimodal_and_task_groups(self):
        fc = [
            {"repo_id": "fa", "final_label": "fold"},
            {"repo_id": "fb", "final_label": "flatten"},
            {"repo_id": "bi", "final_label": "bimodal"},
            {"repo_id": "nt", "final_label": "non_task"},
            {"repo_id": "ds", "final_label": "disarrange"},
        ]
        bimodal, tg = collect_repo_task_groups(fc)
        assert bimodal == {"bi"}
        assert tg == {"fa": "fold", "fb": "flatten"}


class TestResolveTaskGroup:
    def test_direct_mapping_wins_over_repo_fallback(self):
        assert resolve_task_group("fold_success", "any_repo", {}) == "fold"
        assert resolve_task_group("flatten_success", "any_repo", {}) == "flatten"

    def test_shuffle_and_intervention_use_repo_fallback(self):
        repo_tg = {"ra": "flatten"}
        assert resolve_task_group("shuffle_success", "ra", repo_tg) == "flatten"
        assert resolve_task_group("intervention_recovery", "ra", repo_tg) == "flatten"

    def test_default_to_fold_when_no_repo_info(self):
        assert resolve_task_group("shuffle_success", "unknown_repo", {}) == "fold"


class TestSynthesizePriority:
    def _base_inputs(self) -> dict:
        return {
            "episode_classification": {"episodes": []},
            "ground_truth": {"episode_labels": {}, "repo_labels": {}},
            "exclusion_list": {},
            "flatten_classification": [],
        }

    def test_exclusion_drops_entire_repo(self):
        inputs = self._base_inputs()
        inputs["episode_classification"]["episodes"] = [
            _ep("bad_repo:0", "fold_success"),
            _ep("bad_repo:1", "flatten_success"),
        ]
        inputs["exclusion_list"] = {"permanent_data_quality": [{"repo_id": "bad_repo"}]}
        labels, repo_fb, stats = synthesize(**inputs)
        assert labels == {}
        assert repo_fb == {}
        assert stats["dropped_exclusion"] == 2

    def test_bimodal_repo_tagged_not_trained(self):
        inputs = self._base_inputs()
        inputs["episode_classification"]["episodes"] = [_ep("bi_repo:0", "fold_success")]
        inputs["flatten_classification"] = [{"repo_id": "bi_repo", "final_label": "bimodal"}]
        labels, _, stats = synthesize(**inputs)
        assert "bi_repo:0" in labels
        assert labels["bi_repo:0"]["l2"] is None
        assert labels["bi_repo:0"]["source"] == "excluded_bimodal_p0"
        assert stats["excluded_bimodal_p0"] == 1

    def test_self_play_qc_overrides_episode_classification(self):
        inputs = self._base_inputs()
        # episode_classification says medium flatten_success, but GT says high fold_success
        inputs["episode_classification"]["episodes"] = [_ep("r:0", "flatten_success", confidence="medium")]
        inputs["ground_truth"]["episode_labels"] = {
            "r:0": {"category": "fold_success", "confidence": "high", "source": "self_play_qc"}
        }
        labels, _, stats = synthesize(**inputs)
        assert labels["r:0"]["l2"] == "fold_success"
        assert labels["r:0"]["source"] == "self_play_qc"
        assert labels["r:0"]["confidence"] == "high"
        assert stats["self_play_qc_fold_success"] == 1

    def test_low_confidence_episodes_dropped(self):
        inputs = self._base_inputs()
        inputs["episode_classification"]["episodes"] = [
            _ep("r:0", "fold_success", confidence="low"),
            _ep("r:1", "fold_success", confidence="medium"),
            _ep("r:2", "fold_success", confidence="high"),
        ]
        labels, _, stats = synthesize(**inputs)
        assert "r:0" not in labels
        assert "r:1" in labels
        assert labels["r:1"]["confidence"] == "medium"
        assert "r:2" in labels
        assert labels["r:2"]["confidence"] == "high"
        assert stats["dropped_low_confidence"] == 1

    def test_unknown_class_dropped(self):
        inputs = self._base_inputs()
        inputs["episode_classification"]["episodes"] = [_ep("r:0", "some_weird_label", confidence="high")]
        labels, _, stats = synthesize(**inputs)
        assert labels == {}
        assert stats["ep_unknown_class"] == 1

    def test_repo_label_fallback_only_for_uncovered(self):
        inputs = self._base_inputs()
        # Repo A has an episode-level label, repo B has only repo-level
        inputs["episode_classification"]["episodes"] = [
            _ep("repoA:0", "fold_success"),
        ]
        inputs["ground_truth"]["repo_labels"] = {
            "repoA": {"category": "fold_failure", "confidence": "medium"},
            "repoB": {"category": "flatten_success", "confidence": "high"},
        }
        labels, repo_fb, stats = synthesize(**inputs)
        # repoA was covered episode-level, so its repo_label is ignored
        assert "repoA" not in repo_fb
        # repoB falls back to repo_label
        assert "repoB" in repo_fb
        assert repo_fb["repoB"]["l2"] == "flatten_success"
        assert repo_fb["repoB"]["source"] == "repo_label"
        assert repo_fb["repoB"]["task_group"] == "flatten"
        assert stats["repo_label_flatten_success"] == 1

    def test_repo_label_excluded_if_repo_in_exclusion(self):
        inputs = self._base_inputs()
        inputs["exclusion_list"] = {"permanent_non_task": [{"repo_id": "bad"}]}
        inputs["ground_truth"]["repo_labels"] = {"bad": {"category": "fold_success", "confidence": "high"}}
        _, repo_fb, _ = synthesize(**inputs)
        assert repo_fb == {}


class TestSynthesizeIntegration:
    """Smoke test: produce a minimal output and verify schema invariants."""

    def test_output_schema(self):
        inputs = {
            "episode_classification": {
                "episodes": [
                    _ep("r1:0", "fold_success"),
                    _ep("r1:1", "fold_failure"),
                    _ep("r2:0", "flatten_success"),
                    _ep("r3:0", "shuffle_success"),
                    _ep("r4:0", "intervention_recovery"),
                ]
            },
            "ground_truth": {"episode_labels": {}, "repo_labels": {}},
            "exclusion_list": {},
            "flatten_classification": [
                {"repo_id": "r1", "final_label": "fold"},
                {"repo_id": "r2", "final_label": "flatten"},
                {"repo_id": "r3", "final_label": "fold"},
                {"repo_id": "r4", "final_label": "flatten"},
            ],
        }
        labels, _, _ = synthesize(**inputs)

        # Schema: every label has 4 required keys
        for key, v in labels.items():
            assert set(v.keys()) >= {"l2", "task_group", "confidence", "source"}
            assert v["l2"] in FIVE_CLASSES
            assert v["confidence"] in {"high", "medium", "low"}
            assert ":" in key

        # task_group inference
        assert labels["r1:0"]["task_group"] == "fold"
        assert labels["r1:1"]["task_group"] == "fold"
        assert labels["r2:0"]["task_group"] == "flatten"
        # shuffle_success falls back to repo's final_label (r3 is fold repo)
        assert labels["r3:0"]["task_group"] == "fold"
        # intervention_recovery on flatten repo -> flatten
        assert labels["r4:0"]["task_group"] == "flatten"
