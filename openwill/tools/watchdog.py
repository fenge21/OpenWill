"""Watchdog - monitor and automatically restart the OpenWill agent

The watchdog is an independent process responsible for:
1. Monitoring whether the OpenWill process is alive
2. Automatically restarting after a crash
3. Recording crash logs
4. Preventing frequent crashes (crash cooldown)

Usage:
    python -m openwill.watchdog          # Start the watchdog
    python -m openwill.watchdog --stop   # Stop the watchdog
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "data", "runtime")
WATCHDOG_PID_FILE = os.path.join(RUNTIME_DIR, "watchdog.pid")
CRASH_LOG_FILE = os.path.join(RUNTIME_DIR, "crash_log.json")


def _ensure_runtime_dir():
    os.makedirs(RUNTIME_DIR, exist_ok=True)


def start_watchdog() -> Optional[int]:
    """Start the watchdog process (independent of the OpenWill process)"""
    _ensure_runtime_dir()

    # Check if watchdog is already running
    if _is_watchdog_running():
        logger.info("Watchdog is already running")
        return None

    python = sys.executable
    args = [python, "-m", "openwill.watchdog"]

    try:
        if sys.platform == "win32":
            proc = subprocess.Popen(
                args, cwd=PROJECT_ROOT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
            )
        else:
            proc = subprocess.Popen(
                args, cwd=PROJECT_ROOT, start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
            )

        logger.info(f"🐕 Watchdog started (PID: {proc.pid})")
        return proc.pid
    except Exception as e:
        logger.error(f"Failed to start watchdog: {e}")
        return None


def stop_watchdog() -> bool:
    """Stop the watchdog"""
    if not os.path.exists(WATCHDOG_PID_FILE):
        return True

    try:
        with open(WATCHDOG_PID_FILE, "r") as f:
            info = json.load(f)
        pid = info.get("pid")
        if pid:
            import psutil
            proc = psutil.Process(pid)
            proc.terminate()
            logger.info(f"Watchdog stopped (PID: {pid})")
    except Exception:
        pass

    try:
        os.remove(WATCHDOG_PID_FILE)
    except Exception:
        pass

    return True


def _is_watchdog_running() -> bool:
    """Check if the watchdog is running"""
    if not os.path.exists(WATCHDOG_PID_FILE):
        return False

    try:
        with open(WATCHDOG_PID_FILE, "r") as f:
            info = json.load(f)
        pid = info.get("pid")
        import psutil
        psutil.Process(pid)
        return True
    except Exception:
        return False


def _log_crash(reason: str):
    """Record crash log"""
    _ensure_runtime_dir()
    logs = []
    if os.path.exists(CRASH_LOG_FILE):
        try:
            with open(CRASH_LOG_FILE, "r") as f:
                logs = json.load(f)
        except Exception:
            logs = []

    logs.append({
        "timestamp": time.time(),
        "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": reason,
    })
    logs = logs[-100:]  # Keep the most recent 100 entries

    try:
        with open(CRASH_LOG_FILE, "w") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def watchdog_main():
    """Watchdog main loop"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] watchdog: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _ensure_runtime_dir()

    # Write watchdog PID
    with open(WATCHDOG_PID_FILE, "w") as f:
        json.dump({
            "pid": os.getpid(),
            "start_time": time.time(),
            "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, ensure_ascii=False, indent=2)

    logger.info("🐕 Watchdog started")

    # Crash cooldown
    crash_count = 0
    last_crash_time = 0
    COOLDOWNS = [10, 30, 60, 120, 300, 600]  # Increasing cooldown times

    try:
        while True:
            # Check if OpenWill is running
            from .self_restart import is_another_instance_running, read_pid, spawn_new_version

            if not is_another_instance_running():
                # OpenWill is not running, need to start
                now = time.time()
                since_last = now - last_crash_time

                # Cooldown
                cooldown = COOLDOWNS[min(crash_count, len(COOLDOWNS) - 1)]
                if since_last < cooldown:
                    sleep_time = cooldown - since_last
                    logger.info(f"In crash cooldown, restarting in {sleep_time:.0f}s...")
                    time.sleep(sleep_time)

                logger.info("OpenWill is not running, starting...")
                _log_crash("Process not running")

                new_pid = spawn_new_version()
                if new_pid:
                    logger.info(f"OpenWill started (PID: {new_pid})")
                    # Wait a while before checking again
                    time.sleep(30)
                    # Check if still running
                    from .self_restart import is_another_instance_running
                    if is_another_instance_running():
                        crash_count = 0  # Reset crash count
                        logger.info("OpenWill running normally")
                    else:
                        crash_count += 1
                        last_crash_time = time.time()
                        logger.warning(f"OpenWill exited shortly after starting (crash count: {crash_count})")
                else:
                    crash_count += 1
                    last_crash_time = time.time()
                    logger.error(f"OpenWill failed to start (crash count: {crash_count})")
            else:
                # Running normally, check every 60 seconds
                crash_count = 0

            # Check heartbeat
            # 15-minute timeout to accommodate very long LLM API calls
            # (some models can take 10+ minutes to respond)
            from .self_restart import check_heartbeat
            if is_another_instance_running() and not check_heartbeat(max_age_seconds=600):
                # No heartbeat for 10 minutes, may be stuck
                logger.warning("⚠️ OpenWill heartbeat timeout, may be stuck")
                # Wait another 5 minutes
                time.sleep(300)
                if not check_heartbeat(max_age_seconds=900):
                    logger.warning("⚠️ Heartbeat still timed out, will restart OpenWill")
                    from .self_restart import kill_other_instance
                    kill_other_instance()
                    crash_count += 1
                    last_crash_time = time.time()
                    continue

            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Watchdog stopped")
    except Exception as e:
        logger.error(f"Watchdog exception: {e}")
    finally:
        try:
            os.remove(WATCHDOG_PID_FILE)
        except Exception:
            pass


if __name__ == "__main__":
    watchdog_main()
