"""Shell command execution tool with security hardening"""

import logging
import os
import re
import subprocess
import tempfile
import platform
from typing import Optional, Tuple

from .registry import ToolResult

logger = logging.getLogger(__name__)

# Safe environment variable names allowed to be passed to subprocess
SAFE_ENV_VARS = {
    "PATH", "HOME", "USER", "TEMP", "TMP",
    "SYSTEMROOT", "COMSPEC", "PYTHONIOENCODING", "LANG",
    "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMFILES",
    "PROGRAMDATA", "HOMEDRIVE", "HOMEPATH",
}

# Substrings that indicate a sensitive environment variable
SENSITIVE_ENV_PATTERNS = [
    "KEY", "SECRET", "TOKEN", "PASSWORD", "API",
    "CREDENTIAL", "PRIVATE", "AUTH",
]

# Directories where commands are allowed to operate by default
SAFE_DIRS = [
    os.getcwd(),
    tempfile.gettempdir(),
]

# Interactive commands that require a TTY and must be blocked
INTERACTIVE_COMMANDS = [
    "vim", "nano", "less", "more", "top", "htop",
    "ssh", "telnet", "screen", "tmux",
]

# Interpreter -e/-c patterns that allow arbitrary code execution
BLOCKED_INTERPRETER_EXEC = [
    "python -c", "python3 -c", "python2 -c",
    "perl -e", "ruby -e", "node -e",
    "python -C", "python3 -C",
]


def _build_safe_env() -> dict:
    """Build a minimal safe environment dict, stripping sensitive variables."""
    safe_env = {}
    for key, value in os.environ.items():
        key_upper = key.upper()
        # Only include vars in the whitelist
        if key_upper not in SAFE_ENV_VARS:
            continue
        # Extra check: skip if the key contains sensitive substrings
        if any(pattern in key_upper for pattern in SENSITIVE_ENV_PATTERNS):
            logger.warning(f"Stripped sensitive env var from subprocess: {key}")
            continue
        safe_env[key] = value
    # Ensure PYTHONIOENCODING is set
    safe_env["PYTHONIOENCODING"] = "utf-8"
    return safe_env


def _validate_command_structure(command: str) -> Tuple[bool, str]:
    """
    Validate command structure to block dangerous shell constructs.

    Instead of a blacklist of exact strings, this checks the structural
    elements of the command to prevent bypass techniques.

    Returns:
        (is_safe, reason) - is_safe=True if command passes all checks
    """
    command_stripped = command.strip()
    command_lower = command_stripped.lower()

    # Block pipe chains
    if "|" in command_stripped:
        # Allow simple pipes only for safe read-only commands
        # But block curl|bash, wget|bash, etc.
        pipe_pattern = re.compile(r"(curl|wget)\s+.*\|\s*(bash|sh|zsh|fish|dash)", re.IGNORECASE)
        if pipe_pattern.search(command_stripped):
            return False, "Piping downloaded content into shell is blocked"
        # Block any pipe chain as a general rule
        return False, "Pipe chains (|) are not allowed in command execution"

    # Block command chaining operators
    if "&&" in command_stripped:
        return False, "Command chaining (&&) is not allowed"
    if "||" in command_stripped:
        return False, "Command chaining (||) is not allowed"

    # Block semicolon command separator (but allow inside quotes)
    # Simple heuristic: if there's a ; outside of quotes, block it
    semicolon_outside_quotes = _has_char_outside_quotes(command_stripped, ";")
    if semicolon_outside_quotes:
        return False, "Command chaining (;) is not allowed"

    # Block command substitution
    if "$(" in command_stripped:
        return False, "Command substitution $() is not allowed"
    if "`" in command_stripped:
        return False, "Command substitution (backticks) is not allowed"

    # Block background execution
    if command_stripped.endswith("&") or " & " in command_stripped:
        # Also check for & outside quotes
        ampersand_outside_quotes = _has_char_outside_quotes(command_stripped, "&")
        if ampersand_outside_quotes:
            return False, "Background execution (&) is not allowed"

    # Block redirection to system paths
    redirect_sys_pattern = re.compile(r">\s*/etc/", re.IGNORECASE)
    if redirect_sys_pattern.search(command_stripped):
        return False, "Redirection to /etc/ is not allowed"
    redirect_dev_pattern = re.compile(r">\s*/dev/", re.IGNORECASE)
    if redirect_dev_pattern.search(command_stripped):
        return False, "Redirection to /dev/ is not allowed"

    # Block interpreter arbitrary code execution patterns
    for pattern in BLOCKED_INTERPRETER_EXEC:
        if command_lower.startswith(pattern) or f" {pattern}" in command_lower:
            return False, f"Arbitrary code execution via {pattern} is not allowed"

    # Block interactive commands
    first_word = command_lower.split()[0] if command_lower.split() else ""
    if first_word in INTERACTIVE_COMMANDS:
        return False, f"Interactive commands not allowed: {first_word}"

    # Block dangerous destructive commands (as a final safety net)
    destructive_patterns = [
        r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?/\b",           # rm -rf /
        r"\bdel\s+/s\s+/q\s+[A-Za-z]:",                      # del /s /q C:\
        r"\bformat\s+[A-Za-z]:",                              # format C:
        r"\bmkfs\b",                                          # mkfs
        r"\bshutdown\b",                                      # shutdown
        r"\breboot\b",                                        # reboot
        r"\bhalt\b",                                          # halt
        r"\bpoweroff\b",                                      # poweroff
        r"\bdd\s+if=",                                        # dd if=
        r"\bchmod\s+777\s+/",                                 # chmod 777 /
        r"\btaskkill\s+/f\b",                                 # taskkill /f
        r"\bkill\s+-9\s+1\b",                                 # kill -9 1
    ]
    for pattern in destructive_patterns:
        if re.search(pattern, command_stripped, re.IGNORECASE):
            return False, f"Command blocked by security policy: destructive operation detected"

    return True, ""


def _has_char_outside_quotes(command: str, char: str) -> bool:
    """Check if a character appears outside of single or double quotes."""
    in_single = False
    in_double = False
    i = 0
    while i < len(command):
        c = command[i]
        # Handle escape sequences inside double quotes
        if c == '\\' and in_double and i + 1 < len(command):
            i += 2
            continue
        if c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
        elif c == char and not in_single and not in_double:
            return True
        i += 1
    return False


def _validate_working_dir(cwd: Optional[str]) -> Optional[str]:
    """
    Validate and return a safe working directory.

    If cwd is provided, it must be within SAFE_DIRS.
    If cwd is None or invalid, defaults to the project directory.
    """
    if cwd is not None:
        cwd_abs = os.path.abspath(cwd)
        for safe_dir in SAFE_DIRS:
            safe_abs = os.path.abspath(safe_dir)
            if cwd_abs.startswith(safe_abs):
                return cwd_abs
        logger.warning(f"Working directory blocked (outside safe dirs): {cwd}")
    # Default to project directory
    return os.path.abspath(SAFE_DIRS[0])


def shell_execute(command: str, timeout: int = 30, cwd: Optional[str] = None) -> ToolResult:
    """
    Execute a shell command with security hardening.

    Args:
        command: The command to execute
        timeout: Timeout in seconds
        cwd: Working directory
    """
    # Security check: structural validation
    is_safe, reason = _validate_command_structure(command)
    if not is_safe:
        return ToolResult(
            success=False,
            output="",
            error=f"Command blocked by security policy: {reason}",
        )

    # Security check: validate working directory
    safe_cwd = _validate_working_dir(cwd)

    # Build a clean environment without sensitive variables
    safe_env = _build_safe_env()

    try:
        is_windows = platform.system() == "Windows"

        if is_windows:
            result = subprocess.run(
                ["powershell", "-Command", command],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=safe_cwd,
                env=safe_env,
            )
        else:
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=safe_cwd,
                env=safe_env,
            )

        output = result.stdout
        error = result.stderr

        if result.returncode == 0:
            return ToolResult(
                success=True,
                output=output[:10000],
                data={"return_code": result.returncode},
            )
        else:
            return ToolResult(
                success=False,
                output=output[:5000],
                error=error[:2000],
                data={"return_code": result.returncode},
            )

    except subprocess.TimeoutExpired:
        return ToolResult(
            success=False,
            output="",
            error=f"Command execution timed out ({timeout}s)",
        )
    except FileNotFoundError as e:
        return ToolResult(
            success=False,
            output="",
            error=f"Command not found: {e}",
        )
    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=f"Execution failed: {e}",
        )


def shell_safe_execute(command: str, cwd: Optional[str] = None) -> ToolResult:
    """
    Execute shell command in safe mode - only read-only commands allowed.

    Args:
        command: The command to execute
        cwd: Working directory
    """
    # In safe mode, only commands with these prefixes are allowed
    safe_prefixes = [
        "ls", "dir", "cat", "type", "head", "tail",
        "pwd", "cd", "echo", "which", "where",
        "python --version", "pip list", "pip show",
        "git status", "git log", "git diff", "git branch",
        "env", "set", "hostname", "whoami",
        "wc", "find", "grep",
    ]

    command_start = command.strip().split()[0] if command.strip() else ""
    is_safe = any(command.strip().lower().startswith(prefix.lower()) for prefix in safe_prefixes)

    if not is_safe:
        return ToolResult(
            success=False,
            output="",
            error=f"Not allowed in safe mode: {command}",
        )

    return shell_execute(command, timeout=15, cwd=cwd)


# Tool registration info
SHELL_TOOLS = {
    "shell_exec": {
        "func": shell_execute,
        "description": "Execute a shell command and return output. Can execute any system command, but dangerous commands are blocked.",
        "parameters": {
            "command": "The command string to execute",
            "timeout": "Timeout in seconds, default 30",
            "cwd": "Working directory (optional)",
        },
        "category": "system",
        "dangerous": True,
    },
    "shell_safe": {
        "func": shell_safe_execute,
        "description": "Execute shell command in safe mode, only read-only commands allowed (ls, cat, git status, etc.).",
        "parameters": {
            "command": "The command string to execute",
            "cwd": "Working directory (optional)",
        },
        "category": "system",
        "dangerous": False,
    },
}
