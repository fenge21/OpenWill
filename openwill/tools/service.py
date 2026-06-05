"""System service registration - auto-start on boot, auto-start on crash

Supports:
- Windows: Scheduled Task
- Linux: systemd service
"""

import logging
import os
import platform
import subprocess
import sys

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_cmd(args, stderr=None, timeout=30):
    """Run a subprocess command and log the result.

    Args:
        args: List of command arguments.
        stderr: stderr handling (e.g. subprocess.DEVNULL). Defaults to None.
        timeout: Command timeout in seconds. Defaults to 30.

    Returns:
        subprocess.CompletedProcess result object.
    """
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, stderr=stderr)
        if result.returncode != 0:
            logger.warning(
                "Command %s returned code %d, stderr: %s",
                args, result.returncode, result.stderr.strip()
            )
        return result
    except subprocess.TimeoutExpired:
        logger.error("Command %s timed out after %d seconds", args, timeout)
        return None
    except FileNotFoundError:
        logger.error("Command not found: %s", args[0])
        return None
    except Exception as e:
        logger.error("Failed to run command %s: %s", args, e)
        return None


def install_as_service() -> bool:
    """Register as a system service (auto-start on boot)"""
    system = platform.system()

    if system == "Windows":
        return _install_windows_task()
    elif system == "Linux":
        return _install_linux_systemd()
    elif system == "Darwin":
        return _install_macos_launchd()
    else:
        logger.warning(f"Unsupported system: {system}")
        return False


def uninstall_service() -> bool:
    """Uninstall system service"""
    system = platform.system()

    if system == "Windows":
        return _uninstall_windows_task()
    elif system == "Linux":
        return _uninstall_linux_systemd()
    elif system == "Darwin":
        return _uninstall_macos_launchd()
    else:
        return False


def _install_windows_task() -> bool:
    """Windows: Register scheduled task"""
    python = sys.executable
    task_name = "OpenWill_Watchdog"

    # Delete old task first (ignore errors if it doesn't exist)
    _run_cmd(["schtasks", "/delete", "/tn", task_name, "/f"], stderr=subprocess.DEVNULL)

    # Create new task: start on boot
    result = _run_cmd([
        "schtasks", "/create", "/tn", task_name,
        "/tr", f'"{python}" -m openwill.watchdog',
        "/sc", "onstart",
        "/ru", os.getenv("USERNAME", ""),
        "/rl", "limited",
        "/f",
    ])

    if result and result.returncode == 0:
        logger.info(f"✅ Windows scheduled task registered: {task_name}")
        return True
    else:
        logger.error("❌ Failed to register Windows scheduled task")
        return False


def _uninstall_windows_task() -> bool:
    task_name = "OpenWill_Watchdog"
    result = _run_cmd(["schtasks", "/delete", "/tn", task_name, "/f"], stderr=subprocess.DEVNULL)
    return result is not None and result.returncode == 0


def _install_linux_systemd() -> bool:
    """Linux: Register systemd service"""
    python = sys.executable
    service_content = f"""[Unit]
Description=OpenWill AI Agent Watchdog
After=network.target

[Service]
Type=simple
User={os.getenv("USER", "root")}
WorkingDirectory={PROJECT_ROOT}
ExecStart={python} -m openwill.watchdog
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
"""
    service_path = "/etc/systemd/system/openwill-watchdog.service"

    try:
        with open(service_path, "w") as f:
            f.write(service_content)

        r1 = _run_cmd(["systemctl", "daemon-reload"])
        r2 = _run_cmd(["systemctl", "enable", "openwill-watchdog"])
        r3 = _run_cmd(["systemctl", "start", "openwill-watchdog"])

        if r1 and r2 and r3 and r1.returncode == 0 and r2.returncode == 0 and r3.returncode == 0:
            logger.info("✅ systemd service registered")
            return True
        else:
            logger.error("❌ One or more systemctl commands failed")
            return False
    except PermissionError:
        logger.error("❌ Root privileges required: sudo python main.py --install-service")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to register systemd service: {e}")
        return False


def _uninstall_linux_systemd() -> bool:
    _run_cmd(["systemctl", "stop", "openwill-watchdog"], stderr=subprocess.DEVNULL)
    _run_cmd(["systemctl", "disable", "openwill-watchdog"], stderr=subprocess.DEVNULL)
    try:
        os.remove("/etc/systemd/system/openwill-watchdog.service")
    except Exception:
        pass
    _run_cmd(["systemctl", "daemon-reload"])
    return True


def _install_macos_launchd() -> bool:
    """macOS: Register launchd service"""
    python = sys.executable
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.openwill.watchdog</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>-m</string>
        <string>openwill.watchdog</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{PROJECT_ROOT}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
"""
    plist_path = os.path.expanduser("~/Library/LaunchAgents/com.openwill.watchdog.plist")

    try:
        with open(plist_path, "w") as f:
            f.write(plist_content)
        result = _run_cmd(["launchctl", "load", plist_path])
        if result and result.returncode == 0:
            logger.info("✅ macOS launchd service registered")
            return True
        else:
            logger.error("❌ launchctl load failed")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to register launchd service: {e}")
        return False


def _uninstall_macos_launchd() -> bool:
    plist_path = os.path.expanduser("~/Library/LaunchAgents/com.openwill.watchdog.plist")
    _run_cmd(["launchctl", "unload", plist_path], stderr=subprocess.DEVNULL)
    try:
        os.remove(plist_path)
    except Exception:
        pass
    return True
