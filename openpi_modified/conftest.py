from pathlib import Path
import os

import pytest


_MARKERS = {
    "unit": "fast unit-level coverage.",
    "integration": "cross-module pipeline coverage.",
    "smoke": "very fast regression guardrails.",
    "config": "configuration loading and contract checks.",
    "rl": "reinforcement-learning related tests.",
    "pretrain": "pretraining/model/data-pipeline tests.",
    "posttrain": "post-training/inference/evaluation tests.",
    "slow": "long-running tests.",
    "gpu": "tests requiring GPU.",
    "manual": "should be run manually.",
}

_PRETRAIN_TOKENS = (
    "/src/openpi/models/",
    "/src/openpi/shared/",
    "/src/openpi/training/",
    "/src/openpi/transforms_test.py",
    "/packages/openpi-client/",
    "/tests/unit/config/",
    "/tests/unit/transform/",
    "/tests/integration/",
)

_POSTTRAIN_TOKENS = (
    "/src/openpi/policies/",
    "/scripts/train_test.py",
    "/tests/unit/openloop/",
)

_RL_TOKENS = (
    "/tests/unit/rl/",
    "/scripts/test_rl.py",
    "/scripts/train_rl.py",
)


def _set_jax_cpu_backend_if_no_gpu() -> None:
    try:
        import pynvml

        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except Exception:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")


def pytest_configure(config: pytest.Config) -> None:
    _set_jax_cpu_backend_if_no_gpu()
    for marker, description in _MARKERS.items():
        config.addinivalue_line("markers", f"{marker}: {description}")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    for item in items:
        path = Path(str(item.fspath)).as_posix()

        if "/tests/integration/" in path or path.endswith("/scripts/train_test.py"):
            item.add_marker(pytest.mark.integration)
        elif "/tests/unit/" in path or "/src/openpi/" in path or "/packages/openpi-client/" in path:
            item.add_marker(pytest.mark.unit)

        if "/tests/unit/config/" in path:
            item.add_marker(pytest.mark.config)

        if any(token in path for token in _PRETRAIN_TOKENS):
            item.add_marker(pytest.mark.pretrain)

        if any(token in path for token in _POSTTRAIN_TOKENS):
            item.add_marker(pytest.mark.posttrain)

        if any(token in path for token in _RL_TOKENS):
            item.add_marker(pytest.mark.rl)