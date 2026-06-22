from pathlib import Path
import subprocess

import pytest


@pytest.mark.smoke
def test_setup_runtime_env_exposes_paligemma_tokenizer_in_legacy_home_cache(tmp_path: Path) -> None:
    shared_home = tmp_path / "shared-home"
    shared_model = shared_home / ".cache" / "openpi" / "big_vision" / "paligemma_tokenizer.model"
    shared_model.parent.mkdir(parents=True)
    shared_model.write_bytes(b"tokenizer-model")

    script_path = Path("scripts/ci/setup_runtime_env.sh").resolve()
    runtime_root = tmp_path / "runtime"

    command = f"""
set -euo pipefail
source "{script_path}" "{runtime_root}" "{shared_home}"
test -L "$HOME/.cache/openpi"
test -f "$HOME/.cache/openpi/big_vision/paligemma_tokenizer.model"
test -f "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
cmp "$HOME/.cache/openpi/big_vision/paligemma_tokenizer.model" "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
"""
    subprocess.run(["bash", "-lc", command], check=True)
