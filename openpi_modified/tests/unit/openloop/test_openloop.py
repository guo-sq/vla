import numpy as np
import pytest

from scripts import test_subtask as _test_subtask


class _DummyTokenizer:
    def extract_subtask(self, tokens, mask=None):
        values = np.asarray(tokens)
        if mask is not None:
            values = values[np.asarray(mask).astype(bool)]
        return " ".join(str(int(x)) for x in values.tolist())


@pytest.mark.posttrain
@pytest.mark.smoke
def test_compute_exact_match_is_case_insensitive() -> None:
    assert _test_subtask.compute_exact_match("Pick The Cup", "pick the cup") is True


@pytest.mark.posttrain
def test_compute_token_metrics_handles_empty_prediction() -> None:
    metrics = _test_subtask.compute_token_metrics("", "pick place")

    assert metrics == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


@pytest.mark.posttrain
@pytest.mark.smoke
def test_compute_subtask_metrics_aggregates_matches() -> None:
    tokenizer = _DummyTokenizer()
    metrics = _test_subtask.compute_subtask_metrics(
        pred_tokens_list=[np.array([1, 2]), np.array([5, 6])],
        gt_tokens_list=[np.array([1, 2]), np.array([5, 7])],
        gt_masks_list=[np.array([True, True]), np.array([True, True])],
        tokenizer=tokenizer,
    )

    assert metrics.total_samples == 2
    assert metrics.correct_samples == 1
    assert metrics.exact_match_accuracy == pytest.approx(0.5)
    assert 0.0 <= metrics.avg_f1 <= 1.0
