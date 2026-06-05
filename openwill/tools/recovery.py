"""Automatic recovery on startup - detect corrupted code and rollback from backup"""

import ast
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _validate_module_import(module_path: str) -> tuple[bool, str]:
    """Validate that a module can be imported normally in a subprocess"""
    module_name = module_path.replace("/", ".").replace("\\", ".").replace(".py", "")
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT

    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module_name}; print('OK')"],
            capture_output=True, text=True, timeout=15,
            cwd=PROJECT_ROOT, env=env,
        )
        if result.returncode == 0 and "OK" in result.stdout:
            return True, ""
        return False, result.stderr.strip()[:500]
    except Exception as e:
        return False, str(e)


def _scan_python_files(directory: str) -> list[str]:
    """Scan all Python files in a directory"""
    py_files = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__" and d != "data"]
        for f in files:
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(root, f), directory)
                py_files.append(rel)
    return py_files


def startup_recovery() -> list[dict]:
    """
    Automatic recovery on startup: scan all code files, if corruption is found, rollback from backup

    Returns:
        List of recovery operations
    """
    recoveries = []

    logger.info("🔍 Startup self-check: scanning code integrity...")

    py_files = _scan_python_files(PROJECT_ROOT)

    for rel_path in py_files:
        abs_path = os.path.join(PROJECT_ROOT, rel_path)

        # Syntax check
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()
            ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"⚠️ Syntax error: {rel_path} - {e}")
            recovery = _try_recover(rel_path, abs_path, f"Syntax error: {e}")
            recoveries.append(recovery)
            continue
        except Exception:
            # Non-Python file or encoding issue, skip
            continue

        # Import check (only check files within the openwill package)
        if rel_path.startswith("openwill") and rel_path.endswith(".py"):
            # Skip __init__.py (may be empty)
            if os.path.basename(rel_path) == "__init__.py":
                content_stripped = content.strip()
                if not content_stripped or content_stripped.startswith('"""'):
                    continue

            import_ok, import_error = _validate_module_import(rel_path)
            if not import_ok:
                logger.warning(f"⚠️ Import failed: {rel_path} - {import_error[:100]}")
                recovery = _try_recover(rel_path, abs_path, f"Import failed: {import_error[:200]}")
                recoveries.append(recovery)

    if recoveries:
        logger.info(f"🔧 Startup recovery complete: {len(recoveries)} file(s) repaired")
    else:
        logger.info("✅ Startup self-check passed: all code files intact")

    return recoveries


def _try_recover(rel_path: str, abs_path: str, error: str) -> dict:
    """Try to recover a file from backup"""
    backup_path = abs_path + ".bak"

    if not os.path.exists(backup_path):
        logger.error(f"❌ No backup available for recovery: {rel_path}")
        return {
            "file": rel_path,
            "error": error,
            "recovered": False,
            "reason": "No backup file",
        }

    try:
        # Validate backup file
        with open(backup_path, "r", encoding="utf-8") as f:
            backup_content = f.read()

        try:
            ast.parse(backup_content)
        except SyntaxError:
            logger.error(f"❌ Backup file also has syntax errors: {rel_path}")
            return {
                "file": rel_path,
                "error": error,
                "recovered": False,
                "reason": "Backup file also has syntax errors",
            }

        # Restore backup
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(backup_content)

        # Validate that it can be imported after recovery
        if rel_path.startswith("openwill") and rel_path.endswith(".py"):
            import_ok, import_error = _validate_module_import(rel_path)
            if not import_ok:
                logger.error(f"❌ Still cannot import after recovery: {rel_path} - {import_error[:100]}")
                return {
                    "file": rel_path,
                    "error": error,
                    "recovered": False,
                    "reason": f"Still cannot import after recovery: {import_error[:100]}",
                }

        logger.info(f"✅ Recovered from backup: {rel_path}")
        return {
            "file": rel_path,
            "error": error,
            "recovered": True,
            "reason": "Recovered from .bak backup",
        }

    except Exception as e:
        logger.error(f"❌ Recovery failed: {rel_path} - {e}")
        return {
            "file": rel_path,
            "error": error,
            "recovered": False,
            "reason": f"Error during recovery: {e}",
        }


def scan_backups() -> list[dict]:
    """Scan all available backup files"""
    backups = []
    backup_dir = os.path.join(PROJECT_ROOT, "data", "backups")

    if os.path.exists(backup_dir):
        for f in os.listdir(backup_dir):
            if f.endswith(".bak"):
                filepath = os.path.join(backup_dir, f)
                size = os.path.getsize(filepath)
                mtime = os.path.getmtime(filepath)
                backups.append({
                    "filename": f,
                    "path": filepath,
                    "size": size,
                    "mtime": mtime,
                })

    # Also scan .bak files (in the same directory as source files)
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__" and d != "data"]
        for f in files:
            if f.endswith(".bak"):
                filepath = os.path.join(root, f)
                size = os.path.getsize(filepath)
                mtime = os.path.getmtime(filepath)
                rel = os.path.relpath(filepath, PROJECT_ROOT)
                backups.append({
                    "filename": f,
                    "path": filepath,
                    "relative_path": rel,
                    "size": size,
                    "mtime": mtime,
                })

    return backups
