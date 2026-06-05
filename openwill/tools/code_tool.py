"""Code self-modification tool - read, modify, execute own code

Safety mechanisms:
1. Syntax check (ast.parse)
2. Import validation (subprocess test import)
3. Automatic backup (.bak + modification log)
4. Automatic recovery on startup (detect corruption and rollback)
5. Modification log (complete record of all changes)
"""

import ast
import json
import logging
import os
import subprocess
import sys
import time
from typing import Optional

from .registry import ToolResult

logger = logging.getLogger(__name__)

# OpenWill project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Modification log file
MODIFICATION_LOG_FILE = os.path.join(PROJECT_ROOT, "data", "modification_log.json")

# Core files that are not allowed to be modified (the safety guardian's last line of defense)
PROTECTED_FILES = [
    "openwill/safety/guardian.py",  # The safety guardian itself cannot be modified
]


def _get_openwill_path(relative_path: str) -> str:
    """Get the absolute path within the OpenWill project"""
    abs_path = os.path.normpath(os.path.join(PROJECT_ROOT, relative_path))
    if not abs_path.startswith(os.path.normpath(PROJECT_ROOT)):
        raise ValueError(f"Path is outside project scope: {relative_path}")
    return abs_path


def _log_modification(module_path: str, old_code: str, new_code: str,
                      success: bool, error: str = ""):
    """Log code modification"""
    os.makedirs(os.path.dirname(MODIFICATION_LOG_FILE), exist_ok=True)

    log_entry = {
        "timestamp": time.time(),
        "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        "module": module_path,
        "old_code_preview": old_code[:200],
        "new_code_preview": new_code[:200],
        "success": success,
        "error": error,
    }

    # Read existing logs
    logs = []
    if os.path.exists(MODIFICATION_LOG_FILE):
        try:
            with open(MODIFICATION_LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []

    logs.append(log_entry)

    # Keep only the most recent 100 entries
    logs = logs[-100:]

    try:
        with open(MODIFICATION_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to write modification log: {e}")


def _validate_import(module_path: str) -> tuple[bool, str]:
    """
    Validate that the modified module can be imported normally in a subprocess

    This is the most critical safety check: syntactically correct does not mean it can run.
    By importing in a subprocess, even if the module has runtime errors, it won't affect the current process.
    """
    # Convert file path to module path
    # openwill/agent.py -> openwill.agent
    module_name = module_path.replace("/", ".").replace("\\", ".").replace(".py", "")

    # Need to import from project root directory
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT

    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module_name}; print('IMPORT_OK')"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=PROJECT_ROOT,
            env=env,
        )

        if result.returncode == 0 and "IMPORT_OK" in result.stdout:
            return True, ""

        error = result.stderr.strip()
        # Filter out common harmless warnings
        error_lines = []
        for line in error.split("\n"):
            if "UserWarning" in line or "DeprecationWarning" in line:
                continue
            error_lines.append(line)
        error = "\n".join(error_lines).strip()

        if error:
            return False, f"Import failed: {error[:500]}"
        return False, f"Import abnormal (returncode={result.returncode})"

    except subprocess.TimeoutExpired:
        return False, "Import timed out (module may have blocking operations)"
    except Exception as e:
        return False, f"Validation failed: {e}"


def _validate_syntax(code: str) -> tuple[bool, str]:
    """Validate Python code syntax"""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error: {e}"


def code_read(module_path: str, start_line: int = 0, end_line: Optional[int] = None) -> ToolResult:
    """
    Read own code file

    Args:
        module_path: Path relative to project root, e.g. "openwill/agent.py"
        start_line: Start line (0-based)
        end_line: End line
    """
    try:
        abs_path = _get_openwill_path(module_path)
    except ValueError as e:
        return ToolResult(success=False, output="", error=str(e))

    if not os.path.exists(abs_path):
        return ToolResult(success=False, output="", error=f"File does not exist: {module_path}")

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if end_line is None:
            end_line = len(lines)
        selected = lines[start_line:end_line]

        numbered = []
        for i, line in enumerate(selected, start=start_line + 1):
            numbered.append(f"{i:4d} | {line.rstrip()}")

        content = "\n".join(numbered)
        return ToolResult(
            success=True,
            output=content[:30000],
            data={
                "module_path": module_path,
                "abs_path": abs_path,
                "total_lines": len(lines),
                "shown_lines": len(selected),
            },
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Read failed: {e}")


def code_modify(module_path: str, old_code: str, new_code: str) -> ToolResult:
    """
    Modify own code - replace old code with new code

    Safety flow:
    1. Check if file is protected
    2. Validate new code syntax
    3. Backup original file
    4. Execute replacement
    5. Validate file syntax after replacement
    6. Validate that it can be imported normally in a subprocess
    7. If any step fails, automatically rollback

    Args:
        module_path: Path relative to project root
        old_code: Old code to replace
        new_code: New code to replace with
    """
    # 1. Check protected files
    if module_path in PROTECTED_FILES:
        return ToolResult(
            success=False,
            output="",
            error=f"File {module_path} is protected, modification not allowed. This is the safety guardian's last line of defense.",
        )

    try:
        abs_path = _get_openwill_path(module_path)
    except ValueError as e:
        return ToolResult(success=False, output="", error=str(e))

    if not os.path.exists(abs_path):
        return ToolResult(success=False, output="", error=f"File does not exist: {module_path}")

    try:
        # Read original file
        with open(abs_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        # Check if old code exists
        if old_code not in original_content:
            return ToolResult(
                success=False,
                output="",
                error="Code to replace not found. Please use code_read first to confirm the code content, ensuring old_code matches exactly.",
            )

        # 2. Validate new code syntax
        if module_path.endswith(".py"):
            syntax_ok, syntax_error = _validate_syntax(new_code)
            if not syntax_ok:
                _log_modification(module_path, old_code, new_code, False, syntax_error)
                return ToolResult(success=False, output="", error=syntax_error)

        # 3. Backup original file (with timestamp, preserving history)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(PROJECT_ROOT, "data", "backups")
        os.makedirs(backup_dir, exist_ok=True)

        # Historical backup (with timestamp, won't overwrite)
        backup_filename = f"{module_path.replace('/', '_').replace('\\', '_')}.{timestamp}.bak"
        history_backup = os.path.join(backup_dir, backup_filename)
        with open(history_backup, "w", encoding="utf-8") as f:
            f.write(original_content)

        # Latest backup (fixed name, for quick rollback)
        latest_backup = abs_path + ".bak"
        with open(latest_backup, "w", encoding="utf-8") as f:
            f.write(original_content)

        # 4. Execute replacement
        new_content = original_content.replace(old_code, new_code, 1)

        # 5. Validate file syntax after replacement
        if module_path.endswith(".py"):
            syntax_ok, syntax_error = _validate_syntax(new_content)
            if not syntax_ok:
                _log_modification(module_path, old_code, new_code, False, f"Syntax error after replacement: {syntax_error}")
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File has syntax error after replacement, not written: {syntax_error}",
                )

        # 6. Write modification
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        # 7. Validate that it can be imported normally in a subprocess
        if module_path.endswith(".py"):
            import_ok, import_error = _validate_import(module_path)

            if not import_ok:
                # Import failed! Rollback immediately
                logger.warning(f"⚠️ Import validation failed after modification, rolling back: {module_path}")
                logger.warning(f"   Error: {import_error}")

                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(original_content)

                _log_modification(module_path, old_code, new_code, False, f"Import validation failed, rolled back: {import_error}")

                return ToolResult(
                    success=False,
                    output="",
                    error=f"Import validation failed after modification, automatically rolled back.\nError: {import_error}\n\nOriginal file has been restored. Historical backup saved at: {history_backup}",
                )

        # Modification successful!
        _log_modification(module_path, old_code, new_code, True)

        logger.info(f"✅ Code modification successful: {module_path} (backup: {history_backup})")

        return ToolResult(
            success=True,
            output=f"Modified {module_path}\n"
                   f"- Historical backup: {history_backup}\n"
                   f"- Latest backup: {latest_backup}\n"
                   f"- Import validation: passed",
            data={
                "module_path": module_path,
                "backup_path": latest_backup,
                "history_backup": history_backup,
                "old_length": len(original_content),
                "new_length": len(new_content),
                "import_validated": True,
            },
        )

    except Exception as e:
        _log_modification(module_path, old_code, new_code, False, str(e))
        return ToolResult(success=False, output="", error=f"Modification failed: {e}")


def code_create(module_path: str, content: str) -> ToolResult:
    """
    Create a new code file

    Args:
        module_path: Path relative to project root
        content: File content
    """
    try:
        abs_path = _get_openwill_path(module_path)
    except ValueError as e:
        return ToolResult(success=False, output="", error=str(e))

    if os.path.exists(abs_path):
        return ToolResult(success=False, output="", error=f"File already exists: {module_path}, please use code_modify")

    # Python file syntax check
    if module_path.endswith(".py"):
        syntax_ok, syntax_error = _validate_syntax(content)
        if not syntax_ok:
            return ToolResult(success=False, output="", error=syntax_error)

    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Validate that new file can be imported
        if module_path.endswith(".py"):
            import_ok, import_error = _validate_import(module_path)
            if not import_ok:
                # Import failed, delete new file
                os.remove(abs_path)
                return ToolResult(
                    success=False,
                    output="",
                    error=f"New file import validation failed, deleted: {import_error}",
                )

        _log_modification(module_path, "", content, True)

        return ToolResult(
            success=True,
            output=f"Created {module_path} (import validation passed)",
            data={"module_path": module_path, "abs_path": abs_path},
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Creation failed: {e}")


def code_list_modules() -> ToolResult:
    """List all code modules in the OpenWill project"""
    modules = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__" and d != "data"]
        for f in files:
            if f.endswith(".py"):
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, PROJECT_ROOT)
                size = os.path.getsize(abs_path)
                protected = " [PROTECTED]" if rel_path.replace("\\", "/") in PROTECTED_FILES else ""
                modules.append(f"  {rel_path} ({size} bytes){protected}")

    output = f"OpenWill project modules ({PROJECT_ROOT}):\n" + "\n".join(modules)
    return ToolResult(
        success=True,
        output=output,
        data={"module_count": len(modules), "project_root": PROJECT_ROOT},
    )


def code_execute(code: str, timeout: int = 30) -> ToolResult:
    """
    Execute Python code and return the result

    Executes in an isolated subprocess for real OS-level security isolation,
    with a cleaned environment that strips sensitive variables.

    Args:
        code: Python code to execute
        timeout: Timeout in seconds, default 30
    """
    # Syntax check (early error detection, still useful)
    try:
        ast.parse(code)
    except SyntaxError as e:
        return ToolResult(success=False, output="", error=f"Syntax error: {e}")

    # Build a clean environment for the subprocess, stripping sensitive variables
    safe_env = {}
    sensitive_keywords = ("KEY", "SECRET", "TOKEN", "PASSWORD", "API")
    safe_var_names = {
        "PATH", "HOME", "USER", "USERNAME", "TEMP", "TMP",
        "PYTHONIOENCODING", "PYTHONPATH", "LANG", "LC_ALL",
        "SYSTEMROOT", "COMSPEC", "PROGRAMFILES", "PROCESSOR_ARCHITECTURE",
    }
    for key, value in os.environ.items():
        key_upper = key.upper()
        # Skip variables containing sensitive keywords
        if any(kw in key_upper for kw in sensitive_keywords):
            continue
        # Pass through known safe variables
        if key_upper in safe_var_names or key in safe_var_names:
            safe_env[key] = value

    # Ensure subprocess can find the Python executable
    safe_env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=safe_env,
        )

        output = result.stdout
        error_output = result.stderr

        if result.returncode == 0:
            # Combine stdout and stderr for output
            combined = output
            if error_output:
                combined += ("\n--- stderr ---\n" + error_output) if combined else error_output
            return ToolResult(
                success=True,
                output=combined[:10000],
                data={"code_length": len(code), "returncode": result.returncode},
            )
        else:
            error_msg = error_output[:2000] if error_output else f"Process exited with code {result.returncode}"
            return ToolResult(
                success=False,
                output=output[:5000],
                error=error_msg,
            )

    except subprocess.TimeoutExpired:
        return ToolResult(
            success=False,
            output="",
            error=f"Execution timed out (limit: {timeout}s)",
        )
    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=f"Execution failed: {e}",
        )


def code_install_package(package: str) -> ToolResult:
    """Install a Python package"""
    dangerous = [";", "&&", "|", "`", "$"]
    for ch in dangerous:
        if ch in package:
            return ToolResult(success=False, output="", error=f"Package name contains illegal character: {ch}")

    # Build a clean environment, stripping sensitive variables
    sensitive_keywords = ("KEY", "SECRET", "TOKEN", "PASSWORD", "API",
                          "CREDENTIAL", "PRIVATE", "AUTH")
    safe_env = {}
    for key, value in os.environ.items():
        if any(kw in key.upper() for kw in sensitive_keywords):
            continue
        safe_env[key] = value
    safe_env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True, text=True, timeout=120,
            env=safe_env,
        )
        if result.returncode == 0:
            return ToolResult(success=True, output=f"Installed {package}\n{result.stdout[:2000]}")
        else:
            return ToolResult(success=False, output=result.stdout[:1000], error=result.stderr[:1000])
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, output="", error="Installation timed out")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Installation failed: {e}")


def code_rollback(module_path: str) -> ToolResult:
    """
    Rollback code to the most recent backup version

    Args:
        module_path: Path relative to project root
    """
    try:
        abs_path = _get_openwill_path(module_path)
    except ValueError as e:
        return ToolResult(success=False, output="", error=str(e))

    backup_path = abs_path + ".bak"
    if not os.path.exists(backup_path):
        return ToolResult(success=False, output="", error=f"Backup file not found: {backup_path}")

    try:
        # First backup current version (in case rollback is wrong)
        current_content = ""
        if os.path.exists(abs_path):
            with open(abs_path, "r", encoding="utf-8") as f:
                current_content = f.read()

        # Read backup
        with open(backup_path, "r", encoding="utf-8") as f:
            backup_content = f.read()

        # Write backup content
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(backup_content)

        # Validate that it can be imported after rollback
        if module_path.endswith(".py"):
            import_ok, import_error = _validate_import(module_path)
            if not import_ok:
                # Still can't import after rollback, restore current version
                if current_content:
                    with open(abs_path, "w", encoding="utf-8") as f:
                        f.write(current_content)
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Import validation failed after rollback, current version restored: {import_error}",
                )

        _log_modification(module_path, "rollback", backup_content[:200], True)

        return ToolResult(
            success=True,
            output=f"Rolled back {module_path} to backup version (import validation passed)",
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Rollback failed: {e}")


def code_modification_history() -> ToolResult:
    """View code modification history"""
    if not os.path.exists(MODIFICATION_LOG_FILE):
        return ToolResult(success=True, output="No modification records yet")

    try:
        with open(MODIFICATION_LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)

        lines = []
        for log in logs[-20:]:  # Most recent 20 entries
            status = "✅" if log.get("success") else "❌"
            lines.append(
                f"{status} [{log.get('time_str', '')}] {log.get('module', '')}\n"
                f"   Old: {log.get('old_code_preview', '')[:60]}...\n"
                f"   New: {log.get('new_code_preview', '')[:60]}..."
            )
            if not log.get("success") and log.get("error"):
                lines.append(f"   Error: {log['error'][:80]}")

        output = f"Code modification history ({len(logs)} total, showing most recent 20):\n\n" + "\n\n".join(lines)
        return ToolResult(success=True, output=output)
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Failed to read history: {e}")


# Tool registration info
CODE_TOOLS = {
    "code_read": {
        "func": code_read,
        "description": "Read OpenWill project's own code files. Used to understand its own structure and logic.",
        "parameters": {
            "module_path": "Path relative to project root, e.g. openwill/agent.py",
            "start_line": "Start line number (0-based), default 0",
            "end_line": "End line number, default to end of file",
        },
        "category": "self_modification",
        "dangerous": False,
    },
    "code_modify": {
        "func": code_modify,
        "description": "Modify own code! Automatically backs up before replacement, validates import after replacement, automatically rolls back on failure.",
        "parameters": {
            "module_path": "Path relative to project root",
            "old_code": "Old code to replace (must match the original exactly)",
            "new_code": "New code to replace with",
        },
        "category": "self_modification",
        "dangerous": True,
    },
    "code_create": {
        "func": code_create,
        "description": "Create a new code file. Automatically validates import after creation, deletes on failure.",
        "parameters": {
            "module_path": "Path relative to project root",
            "content": "File content",
        },
        "category": "self_modification",
        "dangerous": True,
    },
    "code_rollback": {
        "func": code_rollback,
        "description": "Rollback code to the most recent backup version. Automatically validates import after rollback.",
        "parameters": {
            "module_path": "Path relative to project root",
        },
        "category": "self_modification",
        "dangerous": True,
    },
    "code_list_modules": {
        "func": code_list_modules,
        "description": "List all code modules in the OpenWill project. Protected files are marked.",
        "parameters": {},
        "category": "self_modification",
        "dangerous": False,
    },
    "code_modification_history": {
        "func": code_modification_history,
        "description": "View code modification history records.",
        "parameters": {},
        "category": "self_modification",
        "dangerous": False,
    },
    "code_execute": {
        "func": code_execute,
        "description": "Execute Python code and return the result. Runs in an isolated namespace.",
        "parameters": {
            "code": "Python code to execute",
            "timeout": "Timeout in seconds, default 30",
        },
        "category": "execution",
        "dangerous": True,
    },
    "code_install": {
        "func": code_install_package,
        "description": "Install a Python package to extend own capabilities.",
        "parameters": {
            "package": "Package name, e.g. requests",
        },
        "category": "system",
        "dangerous": True,
    },
}
