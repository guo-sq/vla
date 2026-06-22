#!/usr/bin/env python3

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import pathlib
import pwd
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

RUNTIME_FILES = (
    ".runner",
    ".credentials",
    ".credentials_rsaparams",
    ".env",
    ".path",
    ".service",
)


@dataclasses.dataclass(frozen=True)
class RunnerConfig:
    github_repo: str
    runner_name: str
    install_dir: pathlib.Path
    state_dir: pathlib.Path
    runner_user: str
    runner_version: str
    runner_archive_url: str
    github_token: str | None
    github_token_file: pathlib.Path | None
    watch_interval_seconds: int
    start_timeout_seconds: int

    @property
    def backup_dir(self) -> pathlib.Path:
        return self.state_dir / "backups"

    @property
    def current_backup_dir(self) -> pathlib.Path:
        return self.state_dir / "current"

    @property
    def cache_dir(self) -> pathlib.Path:
        return self.state_dir / "cache"

    @property
    def log_dir(self) -> pathlib.Path:
        return self.state_dir / "logs"

    @property
    def runner_log_path(self) -> pathlib.Path:
        return self.log_dir / "runner.log"


def parse_env_file(path: pathlib.Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise ValueError(f"Invalid config line in {path}: {raw_line}")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_config(config_path: pathlib.Path | None) -> RunnerConfig:
    values: dict[str, str] = {}
    if config_path is not None:
        values.update(parse_env_file(config_path))
    for key in (
        "GITHUB_REPO",
        "RUNNER_NAME",
        "RUNNER_INSTALL_DIR",
        "RUNNER_STATE_DIR",
        "RUNNER_USER",
        "RUNNER_VERSION",
        "RUNNER_ARCHIVE_URL",
        "GITHUB_TOKEN",
        "GITHUB_TOKEN_FILE",
        "WATCH_INTERVAL_SECONDS",
        "START_TIMEOUT_SECONDS",
    ):
        if key in os.environ:
            values[key] = os.environ[key]

    github_repo = values.get("GITHUB_REPO")
    runner_name = values.get("RUNNER_NAME")
    if not github_repo or not runner_name:
        raise ValueError("GITHUB_REPO and RUNNER_NAME are required")

    install_dir = pathlib.Path(values.get("RUNNER_INSTALL_DIR", "/opt/actions-runner")).expanduser().resolve()
    state_dir = pathlib.Path(values.get("RUNNER_STATE_DIR", ".runner-state")).expanduser().resolve()
    github_token_file = values.get("GITHUB_TOKEN_FILE")
    return RunnerConfig(
        github_repo=github_repo,
        runner_name=runner_name,
        install_dir=install_dir,
        state_dir=state_dir,
        runner_user=values.get("RUNNER_USER", "github-runner"),
        runner_version=values.get("RUNNER_VERSION", ""),
        runner_archive_url=values.get("RUNNER_ARCHIVE_URL", ""),
        github_token=values.get("GITHUB_TOKEN"),
        github_token_file=pathlib.Path(github_token_file).expanduser().resolve() if github_token_file else None,
        watch_interval_seconds=int(values.get("WATCH_INTERVAL_SECONDS", "30")),
        start_timeout_seconds=int(values.get("START_TIMEOUT_SECONDS", "120")),
    )


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def ensure_state_dirs(config: RunnerConfig) -> None:
    for path in (config.state_dir, config.backup_dir, config.current_backup_dir, config.cache_dir, config.log_dir):
        path.mkdir(parents=True, exist_ok=True)


def detect_runner_version(config: RunnerConfig) -> str:
    if config.runner_version:
        return config.runner_version
    for child in sorted(config.install_dir.iterdir()) if config.install_dir.exists() else []:
        if child.name.startswith("bin."):
            return child.name.split("bin.", 1)[1]
    manifest_path = config.current_backup_dir / "manifest.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return str(data.get("runner_version", ""))
    return ""


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def github_token(config: RunnerConfig) -> str | None:
    if config.github_token:
        return config.github_token
    if config.github_token_file and config.github_token_file.exists():
        return config.github_token_file.read_text(encoding="utf-8").strip()
    return None


def run_command(
    command: list[str], *, capture_output: bool = False, check: bool = True, timeout: int = 30
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=capture_output, check=check, timeout=timeout)


def list_local_processes(config: RunnerConfig) -> list[tuple[int, str]]:
    result = run_command(["ps", "-eo", "pid=,args="], capture_output=True)
    matches: list[tuple[int, str]] = []
    install_path = str(config.install_dir)
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, args = stripped.partition(" ")
        if not pid_text.isdigit():
            continue
        if install_path not in args and "Runner.Listener" not in args and "run.sh" not in args:
            continue
        if "runner_manager.py" in args:
            continue
        matches.append((int(pid_text), args))
    return matches


def local_runner_running(config: RunnerConfig) -> bool:
    return any("Runner.Listener" in args or "./run.sh" in args for _, args in list_local_processes(config))


def chown_path(path: pathlib.Path, user: str) -> None:
    if os.geteuid() != 0:
        return
    run_command(["chown", f"{user}:{user}", str(path)])


def sync_runsvc(config: RunnerConfig) -> None:
    source = pathlib.Path(__file__).resolve().with_name("runsvc.sh")
    target = config.install_dir / "runsvc.sh"
    if not source.exists():
        raise FileNotFoundError(f"Missing template: {source}")
    if not target.exists() or source.read_text(encoding="utf-8") != target.read_text(encoding="utf-8"):
        shutil.copy2(source, target)
    target.chmod(0o755)
    chown_path(target, config.runner_user)


def build_local_runner_archive(config: RunnerConfig, archive_path: pathlib.Path) -> pathlib.Path:
    if not (config.install_dir / "run.sh").exists():
        raise FileNotFoundError(f"Runner install not found at {config.install_dir}")
    log(f"Building runner archive from local install at {config.install_dir}")
    excluded_names = {"_diag", *RUNTIME_FILES, "runsvc.sh"}
    with tarfile.open(archive_path, "w:gz") as tar:
        for child in sorted(config.install_dir.iterdir()):
            if child.name in excluded_names:
                continue
            tar.add(child, arcname=child.name)
    return archive_path


def cache_runner_archive(config: RunnerConfig) -> pathlib.Path:
    ensure_state_dirs(config)
    archive_url = config.runner_archive_url
    version = detect_runner_version(config)
    if not archive_url:
        if not version:
            raise ValueError("RUNNER_ARCHIVE_URL or RUNNER_VERSION is required to restore a wiped runner install")
        archive_url = (
            f"https://github.com/actions/runner/releases/download/v{version}/"
            f"actions-runner-linux-x64-{version}.tar.gz"
        )
    archive_name = archive_url.rsplit("/", 1)[-1]
    archive_path = config.cache_dir / archive_name
    if archive_path.exists():
        return archive_path
    if config.install_dir.exists() and (config.install_dir / "run.sh").exists():
        return build_local_runner_archive(config, archive_path)
    log(f"Downloading runner archive from {archive_url}")
    with urllib.request.urlopen(archive_url) as response, archive_path.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)
    return archive_path


def ensure_installation(config: RunnerConfig) -> None:
    listener_binary = config.install_dir / "bin" / "Runner.Listener"
    if listener_binary.exists() and (config.install_dir / "run.sh").exists():
        sync_runsvc(config)
        return
    archive_path = cache_runner_archive(config)
    config.install_dir.mkdir(parents=True, exist_ok=True)
    log(f"Restoring runner binaries into {config.install_dir}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(config.install_dir)
    sync_runsvc(config)
    if os.geteuid() == 0:
        run_command(["chown", "-R", f"{config.runner_user}:{config.runner_user}", str(config.install_dir)])


def backup_runtime_state(config: RunnerConfig) -> pathlib.Path:
    ensure_state_dirs(config)
    file_hashes: dict[str, str] = {}
    copied_files: list[str] = []
    for name in (*RUNTIME_FILES, "runsvc.sh"):
        source = config.install_dir / name
        if source.exists():
            copied_files.append(name)
            file_hashes[name] = file_sha256(source)

    current_manifest_path = config.current_backup_dir / "manifest.json"
    if current_manifest_path.exists():
        current_manifest = json.loads(current_manifest_path.read_text(encoding="utf-8"))
        if current_manifest.get("file_hashes") == file_hashes:
            try:
                cache_runner_archive(config)
            except Exception as exc:
                log(f"Skipped archive cache refresh: {exc}")
            return config.current_backup_dir

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = config.backup_dir / timestamp
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    for name in (*RUNTIME_FILES, "runsvc.sh"):
        source = config.install_dir / name
        if source.exists():
            shutil.copy2(source, snapshot_dir / name)
    manifest = {
        "created_at": now_iso(),
        "github_repo": config.github_repo,
        "runner_name": config.runner_name,
        "runner_version": detect_runner_version(config),
        "files": copied_files,
        "file_hashes": file_hashes,
    }
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if config.current_backup_dir.exists():
        shutil.rmtree(config.current_backup_dir)
    shutil.copytree(snapshot_dir, config.current_backup_dir)
    try:
        cache_runner_archive(config)
    except Exception as exc:
        log(f"Skipped archive cache refresh: {exc}")
    log(f"Backed up runner state into {snapshot_dir}")
    return snapshot_dir


def latest_backup_dir(config: RunnerConfig) -> pathlib.Path | None:
    if config.current_backup_dir.exists() and (config.current_backup_dir / ".runner").exists():
        return config.current_backup_dir
    candidates = sorted((path for path in config.backup_dir.iterdir() if path.is_dir()), reverse=True) if config.backup_dir.exists() else []
    return candidates[0] if candidates else None


def restore_runtime_state(config: RunnerConfig) -> pathlib.Path:
    ensure_installation(config)
    snapshot_dir = latest_backup_dir(config)
    if snapshot_dir is None:
        raise FileNotFoundError("No runner backup found. Run snapshot once on a healthy runner first.")
    copied = False
    for name in (*RUNTIME_FILES, "runsvc.sh"):
        source = snapshot_dir / name
        if source.exists():
            shutil.copy2(source, config.install_dir / name)
            copied = True
    if not copied:
        raise FileNotFoundError(f"Backup at {snapshot_dir} does not contain runner runtime files")
    sync_runsvc(config)
    if os.geteuid() == 0:
        run_command(["chown", "-R", f"{config.runner_user}:{config.runner_user}", str(config.install_dir)])
    log(f"Restored runner state from {snapshot_dir}")
    return snapshot_dir


def github_runner_status(config: RunnerConfig) -> dict[str, object] | None:
    endpoint = f"https://api.github.com/repos/{config.github_repo}/actions/runners?per_page=100"
    token = github_token(config)
    if token:
        request = urllib.request.Request(endpoint, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = json.load(response)
        except urllib.error.URLError as exc:
            log(f"GitHub runner status request failed: {exc}")
            return None
    else:
        if shutil.which("gh") is None:
            return None
        result = run_command(["gh", "api", f"repos/{config.github_repo}/actions/runners"], capture_output=True, check=False)
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
    for runner in data.get("runners", []):
        if runner.get("name") == config.runner_name:
            return {
                "status": runner.get("status"),
                "busy": runner.get("busy"),
                "labels": [label.get("name") for label in runner.get("labels", [])],
            }
    return None


def stop_runner(config: RunnerConfig) -> None:
    processes = list_local_processes(config)
    if not processes:
        return
    for pid, _ in processes:
        os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 15
    while time.time() < deadline:
        if not list_local_processes(config):
            return
        time.sleep(1)
    for pid, _ in list_local_processes(config):
        os.kill(pid, signal.SIGKILL)


def start_runner(config: RunnerConfig) -> None:
    ensure_installation(config)
    if not (config.install_dir / ".runner").exists():
        restore_runtime_state(config)
    sync_runsvc(config)
    if local_runner_running(config):
        return
    config.log_dir.mkdir(parents=True, exist_ok=True)
    shell_command = f"cd {shlex.quote(str(config.install_dir))} && ./run.sh"
    command: list[str]
    current_user = pwd.getpwuid(os.geteuid()).pw_name
    if os.geteuid() == 0 and current_user != config.runner_user:
        command = ["su", "-s", "/bin/bash", config.runner_user, "-c", shell_command]
    else:
        command = ["/bin/bash", "-lc", shell_command]
    log(f"Starting runner process for {config.runner_name}")
    with config.runner_log_path.open("a", encoding="utf-8") as log_file:
        subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)


def wait_for_runner(config: RunnerConfig) -> bool:
    deadline = time.time() + config.start_timeout_seconds
    while time.time() < deadline:
        local_running = local_runner_running(config)
        remote_status = github_runner_status(config)
        if local_running and (remote_status is None or remote_status.get("status") == "online"):
            return True
        time.sleep(5)
    return False


def ensure_running(config: RunnerConfig) -> bool:
    ensure_state_dirs(config)
    if config.install_dir.exists() and (config.install_dir / ".runner").exists():
        backup_runtime_state(config)
    remote_status = github_runner_status(config)
    local_running = local_runner_running(config)
    if local_running and (remote_status is None or remote_status.get("status") == "online"):
        return True
    if local_running and remote_status and remote_status.get("status") != "online":
        log("Runner process exists but GitHub reports offline; restarting listener")
        stop_runner(config)
        local_running = False
    if not local_running:
        start_runner(config)
    healthy = wait_for_runner(config)
    if healthy:
        backup_runtime_state(config)
    return healthy


def status_payload(config: RunnerConfig) -> dict[str, object]:
    backup_dir = latest_backup_dir(config)
    return {
        "github_repo": config.github_repo,
        "runner_name": config.runner_name,
        "install_dir": str(config.install_dir),
        "state_dir": str(config.state_dir),
        "local_running": local_runner_running(config),
        "github_status": github_runner_status(config),
        "latest_backup": str(backup_dir) if backup_dir else None,
        "runner_version": detect_runner_version(config),
    }


def watch(config: RunnerConfig) -> int:
    ensure_state_dirs(config)
    while True:
        try:
            healthy = ensure_running(config)
            if not healthy:
                log("Runner did not become healthy before timeout")
        except Exception as exc:
            log(f"Watch cycle failed: {exc}")
        time.sleep(config.watch_interval_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage a self-hosted GitHub Actions runner on DSW-like hosts")
    parser.add_argument("--config", type=pathlib.Path, help="Path to runner env file")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    subparsers.add_parser("snapshot")
    subparsers.add_parser("restore")
    subparsers.add_parser("start")
    subparsers.add_parser("stop")
    subparsers.add_parser("ensure-running")
    subparsers.add_parser("watch")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "status":
        print(json.dumps(status_payload(config), indent=2, sort_keys=True))
        return 0
    if args.command == "snapshot":
        backup_runtime_state(config)
        return 0
    if args.command == "restore":
        restore_runtime_state(config)
        return 0
    if args.command == "start":
        start_runner(config)
        return 0
    if args.command == "stop":
        stop_runner(config)
        return 0
    if args.command == "ensure-running":
        return 0 if ensure_running(config) else 1
    if args.command == "watch":
        return watch(config)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())