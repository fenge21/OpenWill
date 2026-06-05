"""Self-restart and hot swap - the agent autonomously manages its own lifecycle

Core capabilities:
1. Self-restart - the agent spawns a new process to replace itself without human intervention
2. Hot swap - after the new version passes validation, automatically start the new version and shut down the old
3. Watchdog - automatically restart after crash, never stop
4. Process lock - prevent multiple instances from running simultaneously
"""

import json
import logging
import os
import psutil
import signal
import subprocess
import sys
import time
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "data", "runtime")
PID_FILE = os.path.join(RUNTIME_DIR, "openwill.pid")
START_FLAG = os.path.join(RUNTIME_DIR, ".starting")
HEARTBEAT_FILE = os.path.join(RUNTIME_DIR, "heartbeat.json")
SWAP_FLAG = os.path.join(RUNTIME_DIR, ".swap_pending")


def _ensure_runtime_dir():
    os.makedirs(RUNTIME_DIR, exist_ok=True)


def write_pid():
    """Write current process PID"""
    _ensure_runtime_dir()
    with open(PID_FILE, "w") as f:
        json.dump({
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "start_time": time.time(),
            "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cmdline": sys.argv,
            "python": sys.executable,
        }, f, ensure_ascii=False, indent=2)


def read_pid() -> Optional[dict]:
    """Read PID file"""
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None


def is_another_instance_running() -> bool:
    """Check if another instance is running"""
    info = read_pid()
    if not info:
        return False

    pid = info.get("pid")
    if not pid:
        return False

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return False

    # Check if it's an openwill process
    cmdline = " ".join(proc.cmdline()).lower()
    if "openwill" in cmdline or "main.py" in cmdline:
        return True

    return False


def kill_other_instance() -> bool:
    """Kill another running instance"""
    info = read_pid()
    if not info:
        return True

    pid = info.get("pid")
    if not pid:
        return True

    try:
        proc = psutil.Process(pid)
        logger.info(f"Stopping old instance (PID: {pid})...")
        proc.terminate()

        # Wait 5 seconds
        try:
            proc.wait(timeout=5)
        except psutil.TimeoutExpired:
            proc.kill()
            logger.info(f"Force stopped old instance (PID: {pid})")

        return True
    except psutil.NoSuchProcess:
        return True
    except Exception as e:
        logger.error(f"Failed to stop old instance: {e}")
        return False


def write_heartbeat():
    """Write heartbeat"""
    _ensure_runtime_dir()
    with open(HEARTBEAT_FILE, "w") as f:
        json.dump({
            "pid": os.getpid(),
            "timestamp": time.time(),
            "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f)


def check_heartbeat(max_age_seconds: int = 120) -> bool:
    """Check if heartbeat is normal"""
    if not os.path.exists(HEARTBEAT_FILE):
        return False

    try:
        with open(HEARTBEAT_FILE, "r") as f:
            hb = json.load(f)
        age = time.time() - hb.get("timestamp", 0)
        return age < max_age_seconds
    except Exception:
        return False


def spawn_new_version(python_args: list[str] = None, env_extra: dict = None) -> Optional[int]:
    """
    Start a new version of itself

    Flow:
    1. Start a new process using staging code
    2. The new process starts in --verify-start mode, only does verification then exits
    3. After verification passes, start a new process in normal mode
    4. After the new process starts successfully, the old process exits

    Args:
        python_args: Additional python arguments
        env_extra: Additional environment variables

    Returns:
        PID of the new process, or None if startup failed
    """
    _ensure_runtime_dir()

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    # Build startup command
    python = sys.executable
    main_script = os.path.join(PROJECT_ROOT, "main.py")

    # Pass the same arguments (except --cycles)
    args = [python]
    if python_args:
        args.extend(python_args)
    args.append(main_script)

    # Pass original command line arguments (filter out --cycles, new process should run indefinitely)
    for arg in sys.argv[1:]:
        if arg.startswith("--cycles"):
            continue
        args.append(arg)

    # Add self-spawned flag
    args.append("--self-spawned")

    logger.info(f"🚀 Starting new version: {' '.join(args[:5])}...")

    try:
        # Start with subprocess, detached from current process
        if sys.platform == "win32":
            # Windows: use CREATE_NEW_PROCESS_GROUP to make the new process independent
            proc = subprocess.Popen(
                args,
                env=env,
                cwd=PROJECT_ROOT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
        else:
            # Linux/Mac: use start_new_session to make the new process independent
            proc = subprocess.Popen(
                args,
                env=env,
                cwd=PROJECT_ROOT,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )

        logger.info(f"✅ New version started (PID: {proc.pid})")
        return proc.pid

    except Exception as e:
        logger.error(f"❌ Failed to start new version: {e}")
        return None


def graceful_shutdown():
    """Gracefully shut down the current process"""
    logger.info("Shutting down gracefully...")
    # Clean up PID file
    try:
        if os.path.exists(PID_FILE):
            info = read_pid()
            if info and info.get("pid") == os.getpid():
                os.remove(PID_FILE)
    except Exception:
        pass

    logger.info("Goodbye.")


def hot_swap_to_staging() -> dict:
    """
    Hot swap to the staging version

    Complete flow:
    1. Validate staging
    2. Deploy staging as current code
    3. Start a new version of itself
    4. Wait for new version heartbeat confirmation
    5. After confirmation, old process exits

    No human intervention required throughout.
    """
    from .bluegreen import staging_verify, staging_prepare_deploy, execute_deploy

    # 1. Validate staging
    logger.info("🔄 Hot swap: validating staging...")
    verify_result = staging_verify()
    if not verify_result.get("success"):
        return {
            "success": False,
            "error": "Staging validation failed",
            "details": verify_result.get("errors", []),
        }

    # 2. Deploy staging
    logger.info("🔄 Hot swap: deploying staging...")
    deploy_result = execute_deploy()
    if not deploy_result.get("success"):
        return {
            "success": False,
            "error": "Deployment failed",
            "details": deploy_result,
        }

    # 3. Start new version
    logger.info("🔄 Hot swap: starting new version...")
    new_pid = spawn_new_version()
    if not new_pid:
        # Startup failed, rollback
        logger.error("❌ New version startup failed, rolling back...")
        backup_dir = deploy_result.get("backup_dir")
        if backup_dir:
            from .bluegreen import _rollback_deploy
            _rollback_deploy(backup_dir)
        return {
            "success": False,
            "error": "New version startup failed, rolled back",
        }

    # 4. Wait for new version heartbeat
    logger.info(f"🔄 Hot swap: waiting for new version heartbeat (PID: {new_pid})...")
    max_wait = 60  # Wait up to 60 seconds
    start = time.time()

    while time.time() - start < max_wait:
        time.sleep(3)

        # Check if new process is still alive
        try:
            proc = psutil.Process(new_pid)
            if not proc.is_running():
                logger.error("❌ New version process has exited")
                # Rollback
                backup_dir = deploy_result.get("backup_dir")
                if backup_dir:
                    from .bluegreen import _rollback_deploy
                    _rollback_deploy(backup_dir)
                return {"success": False, "error": "New version process exited abnormally, rolled back"}
        except psutil.NoSuchProcess:
            logger.error("❌ New version process has exited")
            backup_dir = deploy_result.get("backup_dir")
            if backup_dir:
                from .bluegreen import _rollback_deploy
                _rollback_deploy(backup_dir)
            return {"success": False, "error": "New version process exited abnormally, rolled back"}

        # Check heartbeat
        hb_info = None
        if os.path.exists(HEARTBEAT_FILE):
            try:
                with open(HEARTBEAT_FILE, "r") as f:
                    hb_info = json.load(f)
            except Exception:
                pass

        if hb_info and hb_info.get("pid") == new_pid:
            age = time.time() - hb_info.get("timestamp", 0)
            if age < 30:
                logger.info(f"✅ New version heartbeat confirmed! PID: {new_pid}")
                # 5. Old process exits
                logger.info("👋 Old version is exiting, new version has taken over...")
                # Write swap flag so main loop knows to exit
                _ensure_runtime_dir()
                with open(SWAP_FLAG, "w") as f:
                    json.dump({
                        "new_pid": new_pid,
                        "timestamp": time.time(),
                    }, f)
                return {"success": True, "new_pid": new_pid}

    # Timeout
    logger.warning("⚠️ Timed out waiting for new version heartbeat")
    return {"success": False, "error": "Timed out waiting for new version heartbeat"}


def should_swap() -> bool:
    """Check if we need to switch to the new version (checked periodically in main loop)"""
    if not os.path.exists(SWAP_FLAG):
        return False

    try:
        with open(SWAP_FLAG, "r") as f:
            info = json.load(f)

        new_pid = info.get("new_pid")
        if new_pid:
            try:
                proc = psutil.Process(new_pid)
                if proc.is_running():
                    return True
            except psutil.NoSuchProcess:
                pass

        # New process is gone, clear flag
        os.remove(SWAP_FLAG)
        return False
    except Exception:
        return False


def clear_swap_flag():
    """Clear swap flag (called after new process starts)"""
    if os.path.exists(SWAP_FLAG):
        os.remove(SWAP_FLAG)


def watchdog_check() -> bool:
    """Watchdog check: is the current process healthy"""
    try:
        proc = psutil.Process(os.getpid())
        # Check memory usage
        mem = proc.memory_info()
        mem_mb = mem.rss / 1024 / 1024
        if mem_mb > 2048:  # Over 2GB
            logger.warning(f"⚠️ Memory usage too high: {mem_mb:.0f}MB")
            return False
        return True
    except Exception:
        return True
