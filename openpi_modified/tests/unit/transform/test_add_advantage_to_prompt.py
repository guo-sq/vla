"""AddAdvantageToPrompt transform unit tests."""

from unittest.mock import patch

import numpy as np

from openpi.transforms import AddAdvantageToPrompt


class TestAddAdvantageToPrompt:
    def test_fixed_positive(self):
        transform = AddAdvantageToPrompt(fixed_advantage=True, training=False)
        data = {"task": "Pick up the cup. Action: move arm"}
        result = transform(data)
        assert "Advantage: positive" in result["task"]
        assert "Action:" in result["task"]

    def test_fixed_negative(self):
        transform = AddAdvantageToPrompt(fixed_advantage=False, training=False)
        data = {"task": "Pick up the cup. Action: move arm"}
        result = transform(data)
        assert "Advantage: negative" in result["task"]

    def test_indicator_true(self):
        transform = AddAdvantageToPrompt(training=False)
        data = {"task": "Pick up the cup. Action: move arm", "indicator": True}
        result = transform(data)
        assert "Advantage: positive" in result["task"]

    def test_indicator_false(self):
        transform = AddAdvantageToPrompt(training=False)
        data = {"task": "Pick up the cup. Action: move arm", "indicator": False}
        result = transform(data)
        assert "Advantage: negative" in result["task"]

    def test_indicator_numpy_scalar(self):
        transform = AddAdvantageToPrompt(training=False)
        data = {"task": "Pick up. Action: go", "indicator": np.array(True)}  # noqa: FBT003
        result = transform(data)
        assert "Advantage: positive" in result["task"]

    def test_indicator_numpy_array(self):
        transform = AddAdvantageToPrompt(training=False)
        data = {"task": "Pick up. Action: go", "indicator": np.array([False])}
        result = transform(data)
        assert "Advantage: negative" in result["task"]

    def test_no_indicator_no_fixed(self):
        transform = AddAdvantageToPrompt(training=False)
        data = {"task": "Pick up the cup. Action: move arm"}
        result = transform(data)
        assert result["task"] == "Pick up the cup. Action: move arm"  # Unchanged

    def test_insertion_before_action(self):
        transform = AddAdvantageToPrompt(fixed_advantage=True, training=False)
        data = {"task": "Task: fold cloth, State: ready;\nAction: pick up"}
        result = transform(data)
        # Should be: Task: fold cloth, State: ready;\nAdvantage: positive\nAction: pick up
        parts = result["task"].split("\n")
        action_idx = next(i for i, p in enumerate(parts) if "Action:" in p)
        advantage_idx = next(i for i, p in enumerate(parts) if "Advantage:" in p)
        assert advantage_idx < action_idx

    def test_no_action_marker_appends(self):
        transform = AddAdvantageToPrompt(fixed_advantage=True, training=False)
        data = {"task": "Just a task with no marker"}
        result = transform(data)
        assert result["task"].endswith("Advantage: positive")

    def test_prompt_field(self):
        transform = AddAdvantageToPrompt(fixed_advantage=True, training=False)
        data = {"prompt": "Pick up. Action: go"}
        result = transform(data)
        assert "Advantage: positive" in result["prompt"]

    def test_both_task_and_prompt(self):
        transform = AddAdvantageToPrompt(fixed_advantage=True, training=False)
        data = {"task": "T. Action: a", "prompt": "P. Action: b"}
        result = transform(data)
        assert "Advantage: positive" in result["task"]
        assert "Advantage: positive" in result["prompt"]

    @patch("openpi.transforms.random")
    def test_dropout_drops(self, mock_random):
        mock_random.random.return_value = 0.1  # < 0.3 dropout_rate
        transform = AddAdvantageToPrompt(dropout_rate=0.3, training=True)
        data = {"task": "T. Action: a", "indicator": True}
        result = transform(data)
        assert "Advantage" not in result["task"]  # Dropped

    @patch("openpi.transforms.random")
    def test_dropout_keeps(self, mock_random):
        mock_random.random.return_value = 0.5  # > 0.3 dropout_rate
        transform = AddAdvantageToPrompt(dropout_rate=0.3, training=True)
        data = {"task": "T. Action: a", "indicator": True}
        result = transform(data)
        assert "Advantage: positive" in result["task"]  # Kept

    def test_fixed_advantage_no_dropout(self):
        """Fixed advantage should never be dropped."""
        transform = AddAdvantageToPrompt(fixed_advantage=True, dropout_rate=1.0, training=True)
        data = {"task": "T. Action: a"}
        result = transform(data)
        assert "Advantage: positive" in result["task"]

    def test_numpy_task(self):
        transform = AddAdvantageToPrompt(fixed_advantage=True, training=False)
        data = {"task": np.array("Pick up. Action: go")}
        result = transform(data)
        assert "Advantage: positive" in result["task"]
