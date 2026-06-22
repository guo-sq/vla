# CI Pipeline

This repository now uses GitHub-hosted runners for lightweight lint jobs and the local self-hosted runner on this machine for smoke and manual validation jobs.

## Workflows

- `pre-commit.yml`: runs changed-file `uv-lock`, `black`, and `ruff` checks.
- `ci.yml`: runs changed-file `black` and `ruff` checks on GitHub-hosted runners, then runs the required `smoke` and `config-loss-drift` suites on the self-hosted runner by reusing the existing local repository, `.venv`, tokenizer cache, and FFmpeg runtime from this machine. It triggers on pull requests, on pushes to `main`, and on manual dispatch so feature branches do not double-run the same pipeline on both `push` and `pull_request`. Manual dispatch can additionally fan out optional suites such as `unit`, `config`, `integration`, `pretrain`, `posttrain`, `rl`, and `full`.
- `test.yml`: manual self-hosted validation workflow for `smoke`, `full`, hotspot integration configs, and the targeted config/RL regression tests, all reusing the current machine environment instead of rebuilding it.
- `glm-code-review.yml`: the `GLM-5 Code Review` workflow posts GLM-5 pull request review feedback on pull requests, and also supports manual triggering with a PR number. The review logic lives in `.github/workflows/glm_code_review.py`.
- `glm-assistant.yml`: the `GLM-5 Assistant` workflow replies to GitHub issues or PR comments that mention `@glm`, and also supports manual triggering with a prompt and an optional PR number. The assistant logic lives in `.github/workflows/glm_assistant.py`.

## Local self-hosted runner setup

- Runner: the self-hosted GitHub Actions runner already running on this machine
- Repository path: `/mnt/workspace/heyuan/ci_cd/openpi_modified`
- Python environment: `/mnt/workspace/heyuan/ci_cd/openpi_modified/.venv`
- Shared cache home: `/root`
- FFmpeg runtime path: `/root/miniconda3/envs/openpi-ci/lib`

The smoke and manual validation jobs do not rebuild the environment. They reuse the existing local `.venv`, the existing tokenizer cache under `/root/.cache/openpi`, and the FFmpeg shared libraries from the existing `openpi-ci` Conda environment.

The lint jobs still run on GitHub-hosted runners because they do not depend on the local model cache or FFmpeg runtime.

Default CI intentionally stays narrow:

- only changed Python files are linted by dedicated `black` and `ruff` jobs
- `smoke` and `config-loss-drift` block PRs by default
- the self-hosted smoke command currently excludes the historical `config_path2` hotspot case, which depends on a config file not present on the base branch
- historical unit debt and repository-wide formatting debt are not part of the default GitHub-runner gate

Use `ci.yml` through `workflow_dispatch` when you want to keep the default required gate and additionally opt into broader suites on the same run. Use `test.yml` when you want to run one of the older manual validation entries directly against the local runner environment without going through the default CI gate.

## Required GitHub secret

- `ZHIPU_API_KEY`: enables the GLM-5 code review and assistant workflows

## Pending

- Open-loop evaluation parity is still a TODO: keep `/mnt/oss_models/pretrain_models/pi05_anyverse/cfg_pi0.5_28_dim.all_public_datasets_exp_0216/99999/params` aligned on the 1/10000 open-loop metric before promoting the model.