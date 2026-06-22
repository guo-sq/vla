# GitHub Runner Recovery on DSW

This repository now includes a workspace-persisted runner management path for DSW-like environments where `/opt` can be wiped and `systemd` is unavailable inside the container.

## Goals

- Preserve runner registration state outside ephemeral system paths.
- Rebuild `/opt/actions-runner` after DSW restart or filesystem cleanup.
- Restart the listener automatically when the local process disappears.
- Restart the listener when GitHub still reports the runner as offline.
- Keep the mechanism repo-managed so the recovery path is versioned.

## Files

- `tools/github_runner/runner_manager.py`: runner backup, restore, start, status, watchdog.
- `tools/github_runner/dsw_startup.sh`: DSW startup hook entrypoint.
- `tools/github_runner/runsvc.sh`: service launcher template restored into `/opt/actions-runner/runsvc.sh`.
- `tools/github_runner/runner.env.example`: environment template.

## Recommended layout

Use a persistent state directory under the workspace, for example:

```bash
/mnt/workspace/heyuan/ci_cd/openpi_modified/.runner-state
```

The state directory stores:

- cached runner release archives
- runner registration backup files (`.runner`, `.credentials`, `.credentials_rsaparams`, `.env`, `.path`, `.service`)
- runner stdout logs
- watchdog logs

## Configuration

1. Copy the example config.

```bash
cp tools/github_runner/runner.env.example .runner-state/runner.env
```

2. Put a GitHub token with permission to read repository runners into the configured token file.

```bash
printf '%s' '<github-token>' > .runner-state/github-token
chmod 600 .runner-state/github-token
```

3. On a healthy runner host, create the initial backup snapshot.

```bash
python3 tools/github_runner/runner_manager.py --config .runner-state/runner.env snapshot
```

This step is important. Without the first snapshot, the manager can restore runner binaries, but it cannot restore runner registration state.

## DSW startup hook

Configure the DSW startup command to run:

```bash
bash /mnt/workspace/heyuan/ci_cd/openpi_modified/tools/github_runner/dsw_startup.sh \
  /mnt/workspace/heyuan/ci_cd/openpi_modified/.runner-state/runner.env
```

What this does:

- restores or re-downloads the runner archive when `/opt/actions-runner` is missing
- restores registration files from `.runner-state`
- starts the listener if it is not running
- launches a background watchdog loop

## Watchdog behavior

`runner_manager.py watch` runs an infinite loop. Each cycle:

1. snapshots the current runner registration state when available
2. checks whether a local runner process is alive
3. checks GitHub runner status when a token or `gh` auth is available
4. restarts the listener when the process is gone
5. restarts the listener when GitHub still reports the runner as offline

## Operational commands

```bash
python3 tools/github_runner/runner_manager.py --config .runner-state/runner.env status
python3 tools/github_runner/runner_manager.py --config .runner-state/runner.env snapshot
python3 tools/github_runner/runner_manager.py --config .runner-state/runner.env restore
python3 tools/github_runner/runner_manager.py --config .runner-state/runner.env ensure-running
python3 tools/github_runner/runner_manager.py --config .runner-state/runner.env watch
```

## Failure model and limits

- If the DSW pod is fully gone, GitHub cannot directly start it from a workflow. The startup hook must be configured on the DSW side.
- If the runner token becomes invalid, restore can rebuild files but re-registration is still a manual GitHub action.
- Without a GitHub token or valid `gh` auth, watchdog can still recover from local process death, but it cannot detect the case where the process exists and GitHub marks the runner offline.

## Suggested rollout

1. create `.runner-state/runner.env`
2. place the GitHub token file
3. run `snapshot`
4. run `ensure-running`
5. configure DSW startup command to `dsw_startup.sh`
6. verify `status` shows both `local_running=true` and `github_status.status=online`