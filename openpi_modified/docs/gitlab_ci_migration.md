# GitLab CI migration notes

This repository now includes a GitLab pipeline in .gitlab-ci.yml that mirrors the existing GitHub Actions setup:

- lint jobs: black, ruff, pre-commit
- shared self-hosted test jobs: smoke, config-loss-drift
- shared self-hosted integration jobs: openloop-eval, openloop-fast, full suite
- manual runner validation job for ad-hoc checks

## Required GitLab runner setup

The GitHub workflow assumes a long-lived self-hosted machine with pre-provisioned dependencies. The GitLab pipeline keeps the same assumption for parity.

Your GitLab runner should satisfy all of the following:

- tags include self-hosted, linux, x64
- shell executor is recommended
- repository worktree is available at /mnt/workspace/heyuan/ci_cd/openpi_modified
- shared virtualenv exists at /mnt/workspace/heyuan/ci_cd/openpi_modified/.venv
- shared OpenPI cache exists under /root/.cache/openpi
- tokenizer exists at /root/.cache/openpi/big_vision/paligemma_tokenizer.model
- ffmpeg library exists at /root/miniconda3/envs/openpi-ci/lib/libavformat.so.61

If your runner tags differ, update the tags section in .gitlab-ci.yml.

## Pipeline triggers

The GitLab workflow matches the GitHub behavior:

- merge request pipelines
- pushes to main
- manual web-triggered pipelines

## Manual variables

When you start a pipeline manually from GitLab UI, you can override these variables:

- OPENLOOP_FAST_CHECKPOINT_DIR
- OPENLOOP_FAST_CPFS_ROOT
- VALIDATION_SUITE

Supported VALIDATION_SUITE values:

- smoke
- full
- hotspot
- config-rl

## Notes on differences from GitHub Actions

- Lint jobs always exist so downstream needs stay valid. They compute their own changed-file list inside the job using GitLab pipeline metadata.
- Merge request pipelines diff against CI_MERGE_REQUEST_DIFF_BASE_SHA. Push pipelines diff against CI_COMMIT_BEFORE_SHA. Other cases fall back to the current commit tree so first-commit and manual pipelines still work.
- workflow_dispatch inputs are represented as normal GitLab CI variables with defaults.
- GITHUB_* runtime identifiers were mapped to GitLab CI_PIPELINE_ID and CI_JOB_ID.

## First bring-up checklist

1. Register a GitLab runner on the target machine and attach the expected tags.
2. Verify the runner user can access /mnt/workspace/heyuan/ci_cd/openpi_modified and /root/.cache/openpi.
3. Run a manual pipeline with VALIDATION_SUITE=smoke.
4. Run a manual pipeline overriding OPENLOOP_FAST_CHECKPOINT_DIR and OPENLOOP_FAST_CPFS_ROOT if the defaults differ on the GitLab host.