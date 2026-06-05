"""Blue-green self-modification system - Copy → Modify → Verify → Switch

Core idea: Don't modify running code directly, instead:
1. Copy the entire project to a staging directory
2. Modify the staging copy
3. Fully verify on the staging copy (import test + runtime test)
4. After verification passes, mark as "ready to switch"
5. On next startup, automatically switch from staging; if startup fails, roll back

This ensures:
- The running process is never affected
- The new version is fully verified before switching
- Failed switches can automatically roll back to the old version
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGING_DIR = os.path.join(PROJECT_ROOT, "data", "staging")
STAGING_META = os.path.join(STAGING_DIR, ".staging_meta.json")
DEPLOY_FLAG = os.path.join(PROJECT_ROOT, "data", ".pending_deploy")


def _copy_project_to_staging() -> str:
    """Copy the current project entirely to the staging directory"""
    if os.path.exists(STAGING_DIR):
        shutil.rmtree(STAGING_DIR)

    os.makedirs(STAGING_DIR, exist_ok=True)

    # Copy project files (excluding data directory and caches)
    ignore_patterns = shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".git", "data", "*.bak",
        ".trae", ".venv", "venv", "node_modules",
    )

    shutil.copytree(
        PROJECT_ROOT,
        STAGING_DIR,
        ignore=ignore_patterns,
        dirs_exist_ok=True,
    )

    # Write metadata
    meta = {
        "created_at": time.time(),
        "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": PROJECT_ROOT,
        "modifications": [],
        "verified": False,
    }
    with open(STAGING_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"📦 Project copied to staging: {STAGING_DIR}")
    return STAGING_DIR


def _get_staging_path(relative_path: str) -> str:
    """Get the file path within the staging directory"""
    return os.path.join(STAGING_DIR, relative_path)


def staging_ensure() -> dict:
    """Ensure the staging directory exists, create it if it doesn't"""
    if not os.path.exists(STAGING_META):
        _copy_project_to_staging()
        return {"status": "created", "staging_dir": STAGING_DIR}

    try:
        with open(STAGING_META, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return {"status": "exists", "staging_dir": STAGING_DIR, "meta": meta}
    except Exception:
        _copy_project_to_staging()
        return {"status": "recreated", "staging_dir": STAGING_DIR}


def staging_read(module_path: str, start_line: int = 0, end_line: Optional[int] = None) -> dict:
    """Read a code file in the staging directory"""
    abs_path = _get_staging_path(module_path)

    if not os.path.exists(abs_path):
        return {"success": False, "error": f"File does not exist: {module_path}"}

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if end_line is None:
            end_line = len(lines)
        selected = lines[start_line:end_line]

        numbered = []
        for i, line in enumerate(selected, start=start_line + 1):
            numbered.append(f"{i:4d} | {line.rstrip()}")

        return {
            "success": True,
            "content": "\n".join(numbered),
            "total_lines": len(lines),
            "shown_lines": len(selected),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def staging_modify(module_path: str, old_code: str, new_code: str) -> dict:
    """
    Modify code on the staging copy (does not affect running code)

    Args:
        module_path: Path relative to project root
        old_code: Old code to replace
        new_code: New code to replace with
    """
    import ast

    # Ensure staging exists
    staging_ensure()

    abs_path = _get_staging_path(module_path)

    if not os.path.exists(abs_path):
        return {"success": False, "error": f"File does not exist in staging: {module_path}"}

    # Read file in staging
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return {"success": False, "error": f"Read failed: {e}"}

    # Check if old code exists
    if old_code not in content:
        return {"success": False, "error": "Code to replace not found. Please use staging_read to confirm first."}

    # Syntax check
    if module_path.endswith(".py"):
        try:
            ast.parse(new_code)
        except SyntaxError as e:
            return {"success": False, "error": f"New code syntax error: {e}"}

    # Execute replacement
    new_content = content.replace(old_code, new_code, 1)

    # Syntax check after replacement
    if module_path.endswith(".py"):
        try:
            ast.parse(new_content)
        except SyntaxError as e:
            return {"success": False, "error": f"Syntax error after replacement: {e}"}

    # Write to staging
    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        return {"success": False, "error": f"Write failed: {e}"}

    # Record modification in metadata
    try:
        with open(STAGING_META, "r", encoding="utf-8") as f:
            meta = json.load(f)

        meta["modifications"].append({
            "timestamp": time.time(),
            "module": module_path,
            "old_code_preview": old_code[:100],
            "new_code_preview": new_code[:100],
        })
        meta["verified"] = False  # Need to re-verify after modification

        with open(STAGING_META, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to update staging metadata: {e}")

    logger.info(f"📝 Staging modification: {module_path}")
    return {"success": True, "module": module_path, "message": "Modified on staging copy (does not affect running code)"}


def staging_create(module_path: str, content: str) -> dict:
    """Create a new file on the staging copy"""
    import ast

    staging_ensure()

    abs_path = _get_staging_path(module_path)

    if os.path.exists(abs_path):
        return {"success": False, "error": f"File already exists in staging: {module_path}"}

    if module_path.endswith(".py"):
        try:
            ast.parse(content)
        except SyntaxError as e:
            return {"success": False, "error": f"Syntax error: {e}"}

    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        return {"success": False, "error": f"Creation failed: {e}"}

    return {"success": True, "module": module_path, "message": "Created on staging copy"}


def staging_verify() -> dict:
    """
    Validate staging copy integrity

    Validation steps:
    1. Syntax check for all .py files
    2. Import test for all openwill modules
    3. Core functionality smoke test (create Agent instance)
    """
    import ast

    if not os.path.exists(STAGING_META):
        return {"success": False, "error": "Staging directory does not exist"}

    logger.info("🔍 Validating staging copy...")

    errors = []

    # 1. Syntax check all .py files
    for root, dirs, files in os.walk(STAGING_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__" and d != "data"]
        for f in files:
            if f.endswith(".py"):
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, "r", encoding="utf-8") as fh:
                        ast.parse(fh.read())
                except SyntaxError as e:
                    rel = os.path.relpath(filepath, STAGING_DIR)
                    errors.append(f"Syntax error {rel}: {e}")

    if errors:
        return {"success": False, "errors": errors, "stage": "syntax_check"}

    # 2. Import test - run in subprocess using staging directory
    env = os.environ.copy()
    env["PYTHONPATH"] = STAGING_DIR

    # Test core module imports
    test_imports = [
        "openwill.config",
        "openwill.llm.interface",
        "openwill.memory.short_term",
        "openwill.memory.long_term",
        "openwill.memory.reflective",
        "openwill.consciousness.curiosity",
        "openwill.consciousness.reflection",
        "openwill.consciousness.values",
        "openwill.consciousness.identity",
        "openwill.consciousness.evolution",
        "openwill.safety.guardian",
        "openwill.explorer.web",
        "openwill.lifecycle.phases",
        "openwill.lifecycle.report",
        "openwill.tools.registry",
        "openwill.tools.code_tool",
        "openwill.tools.file_tool",
        "openwill.tools.shell_tool",
        "openwill.tools.web_tool",
        "openwill.agent",
    ]

    import_test_code = ";\n".join([f"import {m}" for m in test_imports])
    import_test_code += ";\nprint('ALL_IMPORTS_OK')"

    try:
        result = subprocess.run(
            [sys.executable, "-c", import_test_code],
            capture_output=True, text=True, timeout=30,
            cwd=STAGING_DIR, env=env,
        )

        if result.returncode != 0 or "ALL_IMPORTS_OK" not in result.stdout:
            errors.append(f"Import test failed:\n{result.stderr[:500]}")
            return {"success": False, "errors": errors, "stage": "import_test"}
    except subprocess.TimeoutExpired:
        errors.append("Import test timed out")
        return {"success": False, "errors": errors, "stage": "import_test"}

    # 3. Smoke test - can an Agent instance be created
    smoke_test_code = """
import sys
sys.path.insert(0, '.')
from openwill.config import AgentConfig
from openwill.agent import OpenWillAgent
config = AgentConfig()
config.llm.provider = 'ollama'  # No real API needed
config.llm.model = 'test'
config.llm.api_key = 'test'
try:
    agent = OpenWillAgent(config)
    print('SMOKE_TEST_OK')
except Exception as e:
    print(f'SMOKE_TEST_FAIL: {e}')
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", smoke_test_code],
            capture_output=True, text=True, timeout=30,
            cwd=STAGING_DIR, env=env,
        )

        if "SMOKE_TEST_OK" not in result.stdout:
            errors.append(f"Smoke test failed:\n{result.stderr[:500]}\n{result.stdout[:500]}")
            return {"success": False, "errors": errors, "stage": "smoke_test"}
    except subprocess.TimeoutExpired:
        errors.append("Smoke test timed out")
        return {"success": False, "errors": errors, "stage": "smoke_test"}

    # Validation passed! Update metadata
    try:
        with open(STAGING_META, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["verified"] = True
        meta["verified_at"] = time.time()
        with open(STAGING_META, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    logger.info("✅ Staging validation passed!")
    return {
        "success": True,
        "modifications": len(meta.get("modifications", [])),
        "message": "Staging copy validation passed, safe to switch",
    }


def staging_prepare_deploy() -> dict:
    """
    Prepare deployment: mark staging as ready to switch

    On next startup, the startup script will detect the pending switch flag,
    automatically replace current code with staging, then start.
    If startup fails, it automatically rolls back.
    """
    if not os.path.exists(STAGING_META):
        return {"success": False, "error": "Staging directory does not exist"}

    # Verify first
    verify_result = staging_verify()
    if not verify_result["success"]:
        return {
            "success": False,
            "error": "Staging validation did not pass, switching not allowed",
            "verify_errors": verify_result.get("errors", []),
        }

    # Write deployment flag
    deploy_info = {
        "staging_dir": STAGING_DIR,
        "project_root": PROJECT_ROOT,
        "created_at": time.time(),
        "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    os.makedirs(os.path.dirname(DEPLOY_FLAG), exist_ok=True)
    with open(DEPLOY_FLAG, "w", encoding="utf-8") as f:
        json.dump(deploy_info, f, ensure_ascii=False, indent=2)

    logger.info("🚀 Marked as ready for deployment. Will automatically switch to new version on next restart.")
    return {
        "success": True,
        "message": "Staging has been verified and marked as ready for deployment. Will automatically switch on next restart.",
        "deploy_info": deploy_info,
    }


def staging_status() -> dict:
    """View staging status"""
    if not os.path.exists(STAGING_META):
        return {"exists": False}

    try:
        with open(STAGING_META, "r", encoding="utf-8") as f:
            meta = json.load(f)

        pending_deploy = os.path.exists(DEPLOY_FLAG)
        if pending_deploy:
            with open(DEPLOY_FLAG, "r", encoding="utf-8") as f:
                deploy_info = json.load(f)
            meta["pending_deploy"] = deploy_info

        return {"exists": True, **meta}
    except Exception as e:
        return {"exists": True, "error": str(e)}


def staging_reset() -> dict:
    """Reset staging (discard all modifications, re-copy from current code)"""
    if os.path.exists(STAGING_DIR):
        shutil.rmtree(STAGING_DIR)
    if os.path.exists(DEPLOY_FLAG):
        os.remove(DEPLOY_FLAG)

    _copy_project_to_staging()
    return {"success": True, "message": "Staging has been reset to a copy of the current code"}


def execute_deploy() -> dict:
    """
    Execute deployment: replace current code with staging

    This function is typically called by the startup script, not by the running agent.
    Flow:
    1. Backup current code to data/backups/pre_deploy_{timestamp}/
    2. Replace current code with staging
    3. Delete deployment flag
    4. If subsequent startup fails, restore from backup
    """
    if not os.path.exists(DEPLOY_FLAG):
        return {"success": False, "error": "No staging ready for deployment"}

    # Backup current code
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(PROJECT_ROOT, "data", "backups", f"pre_deploy_{timestamp}")

    logger.info(f"📦 Backing up current code to: {backup_dir}")

    ignore_patterns = shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".git", "data", "*.bak",
        ".trae", ".venv", "venv", "node_modules",
    )

    # Only backup openwill directory and main.py
    os.makedirs(backup_dir, exist_ok=True)
    if os.path.exists(os.path.join(PROJECT_ROOT, "openwill")):
        shutil.copytree(
            os.path.join(PROJECT_ROOT, "openwill"),
            os.path.join(backup_dir, "openwill"),
            ignore=ignore_patterns,
        )
    if os.path.exists(os.path.join(PROJECT_ROOT, "main.py")):
        shutil.copy2(
            os.path.join(PROJECT_ROOT, "main.py"),
            os.path.join(backup_dir, "main.py"),
        )

    # Replace with staging
    logger.info("🔄 Replacing current code with staging...")

    # Delete current openwill directory
    current_openwill = os.path.join(PROJECT_ROOT, "openwill")
    if os.path.exists(current_openwill):
        shutil.rmtree(current_openwill)

    # Copy staging's openwill directory
    staging_openwill = os.path.join(STAGING_DIR, "openwill")
    if os.path.exists(staging_openwill):
        shutil.copytree(staging_openwill, current_openwill, ignore=ignore_patterns)

    # Copy main.py
    staging_main = os.path.join(STAGING_DIR, "main.py")
    if os.path.exists(staging_main):
        shutil.copy2(staging_main, os.path.join(PROJECT_ROOT, "main.py"))

    # Delete deployment flag
    os.remove(DEPLOY_FLAG)

    # Write deployment record
    deploy_record = os.path.join(PROJECT_ROOT, "data", "deploy_history.json")
    records = []
    if os.path.exists(deploy_record):
        try:
            with open(deploy_record, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []

    records.append({
        "timestamp": time.time(),
        "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backup_dir": backup_dir,
        "status": "deployed",
    })
    records = records[-50:]  # Keep the most recent 50 records

    os.makedirs(os.path.dirname(deploy_record), exist_ok=True)
    with open(deploy_record, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info("✅ Deployment complete!")
    return {
        "success": True,
        "backup_dir": backup_dir,
        "message": "Current code replaced with staging. Please restart to use the new version.",
    }


def check_and_deploy() -> dict:
    """
    Check on startup if there is a staging ready for deployment, and if so, execute deployment

    Called by the startup script
    """
    if not os.path.exists(DEPLOY_FLAG):
        return {"deployed": False}

    logger.info("🚀 Detected staging ready for deployment, executing deployment...")

    result = execute_deploy()

    if result["success"]:
        # Validate deployed code
        from .recovery import startup_recovery
        recoveries = startup_recovery()

        if recoveries:
            # Issues after deployment, try to roll back
            failed = [r for r in recoveries if not r["recovered"]]
            if failed:
                logger.error(f"❌ Post-deployment self-check failed, rolling back...")
                _rollback_deploy(result["backup_dir"])
                return {
                    "deployed": True,
                    "rollback": True,
                    "errors": [r["error"] for r in failed],
                    "message": "Post-deployment self-check failed, rolled back to old version",
                }

    return {"deployed": True, "rollback": False, **result}


def _rollback_deploy(backup_dir: str):
    """Roll back deployment from backup"""
    ignore_patterns = shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".git", "data", "*.bak",
    )

    current_openwill = os.path.join(PROJECT_ROOT, "openwill")
    backup_openwill = os.path.join(backup_dir, "openwill")

    if os.path.exists(current_openwill):
        shutil.rmtree(current_openwill)

    if os.path.exists(backup_openwill):
        shutil.copytree(backup_openwill, current_openwill, ignore=ignore_patterns)

    backup_main = os.path.join(backup_dir, "main.py")
    if os.path.exists(backup_main):
        shutil.copy2(backup_main, os.path.join(PROJECT_ROOT, "main.py"))

    logger.info(f"⏪ Rolled back to: {backup_dir}")
