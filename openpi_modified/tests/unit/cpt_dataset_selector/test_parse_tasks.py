"""Tests for tasks.jsonl parsing."""

from tools.cpt_dataset_selector.parse_tasks import normalize_text
from tools.cpt_dataset_selector.parse_tasks import parse_tasks_jsonl_content


def test_parse_lerobot_style():
    content = '{"task_index": 0, "task": "Pick up the cup"}\n'
    rows = parse_tasks_jsonl_content(content)
    assert len(rows) == 1
    assert rows[0]["normalized_text"] == normalize_text("Pick up the cup")


def test_instruction_key():
    content = '{"instruction": "Pour water"}\n'
    rows = parse_tasks_jsonl_content(content)
    assert rows[0]["task_text"] == "Pour water"


def test_extract_task_index():
    content = '{"task_index": 3, "task": "x"}\n'
    rows = parse_tasks_jsonl_content(content)
    assert rows[0]["task_index"] == 3


def test_skips_bad_json():
    content = 'not json\n{"task": "ok"}\n'
    rows = parse_tasks_jsonl_content(content)
    assert len(rows) == 1
