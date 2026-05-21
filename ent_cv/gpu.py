"""GPU contention helpers shared by training, prediction, and ad-hoc scripts.

The host runs other GPU users (most notably ``llama-server.service``); these
helpers detect them, stop them when our work needs the memory, and restart
them on the way out. Anything that wraps a CUDA workload should use
``gpu_yield(...)`` as a context manager — that's the public surface here.
"""
from __future__ import annotations

from contextlib import contextmanager
import subprocess
import time
from typing import Iterator

from loguru import logger

# Default headroom: trainings at imgsz=1024 batch=auto typically need ~14–16 GiB
# free. Setting this lower than that lets AutoBatch wedge itself into a tiny
# slice and OOM with cryptic cascades; higher and we'd refuse runs that would
# have fit. 16 GiB matches the smart-sweep observations.
DEFAULT_TRAIN_MIN_FREE_GIB = 16.0

# Inference at imgsz≤1024 typically peaks under 4 GiB. 8 GiB leaves headroom
# for the model load + a couple of frames in flight without being so high that
# we'd kick out a 6 GiB co-tenant unnecessarily.
DEFAULT_INFER_MIN_FREE_GIB = 8.0

_WAIT_POLL_SECONDS = 60

# Systemd user services that may be paused to free GPU memory. Match is by
# substring against the nvidia-smi process_name; value is the unit name to
# stop. Our workloads are treated as higher priority than these.
YIELDABLE_SERVICES: dict[str, str] = {
    "llama-server": "llama-server.service",
}


def gpu_status(device: str) -> tuple[float, list[tuple[int, str, float]]]:
    """Return (free_gib, [(pid, name, used_gib), ...]) for the requested device.

    ``device`` matches Ultralytics' format: ``'0'``, ``'0,1'``, or ``'cpu'``.
    Only the first GPU index is inspected; multi-GPU is uncommon here.
    """
    if device == "cpu":
        return (float("inf"), [])
    idx = device.split(",")[0].strip()

    free_out = subprocess.run(
        ["nvidia-smi", f"--id={idx}",
         "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    free_gib = float(free_out.stdout.strip()) / 1024.0

    apps_out = subprocess.run(
        ["nvidia-smi", f"--id={idx}",
         "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    apps: list[tuple[int, str, float]] = []
    for line in apps_out.stdout.strip().splitlines():
        if not line:
            continue
        pid_s, name, mem_s = (s.strip() for s in line.split(",", 2))
        apps.append((int(pid_s), name, float(mem_s) / 1024.0))
    apps.sort(key=lambda a: a[2], reverse=True)
    return (free_gib, apps)


def _systemctl_user(action: str, unit: str) -> bool:
    """Run ``systemctl --user <action> <unit>``. Return True on success."""
    try:
        subprocess.run(
            ["systemctl", "--user", action, unit],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _systemctl_user_is_active(unit: str) -> bool:
    try:
        out = subprocess.run(
            ["systemctl", "--user", "is-active", unit],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() == "active"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _identify_yieldable(holders: list[tuple[int, str, float]]) -> list[str]:
    """Return systemd unit names for GPU holders that match YIELDABLE_SERVICES."""
    units: list[str] = []
    seen: set[str] = set()
    for _pid, name, _used in holders:
        for pattern, unit in YIELDABLE_SERVICES.items():
            if pattern in name and unit not in seen and _systemctl_user_is_active(unit):
                units.append(unit)
                seen.add(unit)
    return units


def _format_holders(holders: list[tuple[int, str, float]]) -> str:
    if not holders:
        return "  (no compute apps reported by nvidia-smi)"
    return "\n".join(
        f"  PID {pid:>7}  {used:>5.1f} GiB  {name}"
        for pid, name, used in holders
    )


def preflight_gpu(
    device: str,
    min_free_gib: float = DEFAULT_TRAIN_MIN_FREE_GIB,
    *,
    wait: bool = False,
    wait_timeout_min: int = 60,
    yield_services: bool = True,
) -> list[str]:
    """Refuse (or wait) when the GPU is too contested.

    When ``yield_services`` is set, any holder matching ``YIELDABLE_SERVICES`` is
    stopped here; the caller is responsible for restarting them on exit (see
    ``restore_services`` / ``gpu_yield``). Returns the list of stopped units.

    Failed AutoBatch probes from a contested GPU cascade through 4–5 retries
    over several minutes and surface as cryptic OOMs deep in Ultralytics. This
    preflight names the contender process up front.
    """
    if device == "cpu":
        return []

    free_gib, holders = gpu_status(device)
    if free_gib >= min_free_gib:
        return []

    stopped: list[str] = []
    if yield_services:
        yieldable = _identify_yieldable(holders)
        for unit in yieldable:
            if _systemctl_user("stop", unit):
                stopped.append(unit)
                logger.warning(f"Paused {unit} (yielding GPU to higher-priority work).")
            else:
                logger.warning(f"Could not stop {unit}; continuing.")
        if stopped:
            time.sleep(5)  # let GPU memory release
            free_gib, holders = gpu_status(device)
            if free_gib >= min_free_gib:
                logger.success(
                    f"After yielding {', '.join(stopped)}, GPU has {free_gib:.1f} GiB free."
                )
                return stopped

    base_msg = (
        f"GPU {device} has {free_gib:.1f} GiB free, "
        f"work needs ≥ {min_free_gib:.1f} GiB.\n"
        f"Processes holding GPU memory:\n{_format_holders(holders)}"
    )
    if not wait:
        for unit in stopped:
            _systemctl_user("start", unit)
        raise RuntimeError(
            f"{base_msg}\nStop one of the above and retry, or pass wait=True to wait."
        )

    logger.warning(base_msg)
    deadline = time.monotonic() + wait_timeout_min * 60
    logger.info(
        f"Waiting up to {wait_timeout_min} min for ≥ {min_free_gib:.1f} GiB to free up "
        f"(polling every {_WAIT_POLL_SECONDS}s)..."
    )
    while time.monotonic() < deadline:
        time.sleep(_WAIT_POLL_SECONDS)
        free_gib, holders = gpu_status(device)
        if free_gib >= min_free_gib:
            logger.success(f"GPU now has {free_gib:.1f} GiB free, proceeding.")
            return stopped
        logger.info(f"  still waiting; {free_gib:.1f} GiB free")
    for unit in stopped:
        _systemctl_user("start", unit)
    raise RuntimeError(
        f"GPU did not free up within {wait_timeout_min} min.\n"
        f"Current state: {free_gib:.1f} GiB free.\n"
        f"Processes holding GPU memory:\n{_format_holders(holders)}"
    )


def restore_services(units: list[str]) -> None:
    """Restart units previously stopped by ``preflight_gpu``."""
    for unit in units:
        if _systemctl_user("start", unit):
            logger.info(f"Restarted {unit}.")
        else:
            logger.warning(
                f"Failed to restart {unit}. Restart it manually: "
                f"`systemctl --user start {unit}`"
            )


@contextmanager
def gpu_yield(
    device: str = "0",
    min_free_gib: float = DEFAULT_INFER_MIN_FREE_GIB,
    *,
    wait: bool = False,
    wait_timeout_min: int = 60,
) -> Iterator[list[str]]:
    """Context manager: pause yieldable GPU services while the block runs.

    Defaults to the inference headroom (``DEFAULT_INFER_MIN_FREE_GIB``). Pass
    ``min_free_gib=DEFAULT_TRAIN_MIN_FREE_GIB`` for training-sized workloads.

    Yields the list of units that were stopped, in case callers want to log
    them. The same units are restarted on exit, even when the block raises.
    """
    stopped = preflight_gpu(
        device,
        min_free_gib=min_free_gib,
        wait=wait,
        wait_timeout_min=wait_timeout_min,
    )
    try:
        yield stopped
    finally:
        restore_services(stopped)
