"""Tool system entry point - register all tools and integrate into the agent"""

import logging

from .registry import ToolRegistry, ToolResult
from .shell_tool import SHELL_TOOLS
from .file_tool import FILE_TOOLS
from .code_tool import CODE_TOOLS
from .web_tool import WEB_TOOLS
from .bluegreen import (
    staging_ensure, staging_read, staging_modify, staging_create,
    staging_verify, staging_prepare_deploy, staging_status,
    staging_reset,
)
from .self_restart import hot_swap_to_staging

logger = logging.getLogger(__name__)


# Blue-green self-modification tools
BLUEGREEN_TOOLS = {
    "staging_read": {
        "func": lambda module_path, **kw: _wrap_staging_read(module_path, **kw),
        "description": "Read code in the staging copy (does not affect running code).",
        "parameters": {
            "module_path": "Path relative to project root",
            "start_line": "Start line number (0-based), default 0",
            "end_line": "End line number, default to end of file",
        },
        "category": "self_modification",
        "dangerous": False,
    },
    "staging_modify": {
        "func": lambda module_path, old_code, new_code, **kw: _wrap_staging_modify(module_path, old_code, new_code),
        "description": "Modify code on the staging copy (does not affect running code). This is the safe way to self-modify.",
        "parameters": {
            "module_path": "Path relative to project root",
            "old_code": "Old code to replace (must match exactly)",
            "new_code": "New code to replace with",
        },
        "category": "self_modification",
        "dangerous": True,
    },
    "staging_create": {
        "func": lambda module_path, content, **kw: _wrap_staging_create(module_path, content),
        "description": "Create a new file on the staging copy (does not affect running code).",
        "parameters": {
            "module_path": "Path relative to project root",
            "content": "File content",
        },
        "category": "self_modification",
        "dangerous": True,
    },
    "staging_verify": {
        "func": lambda **kw: _wrap_staging_verify(),
        "description": "Validate staging copy integrity (syntax + import + smoke test).",
        "parameters": {},
        "category": "self_modification",
        "dangerous": False,
    },
    "staging_prepare_deploy": {
        "func": lambda **kw: _wrap_staging_deploy(),
        "description": "Mark the verified staging as ready for deployment. Will automatically switch to the new version on next restart.",
        "parameters": {},
        "category": "self_modification",
        "dangerous": True,
    },
    "staging_status": {
        "func": lambda **kw: _wrap_staging_status(),
        "description": "View staging copy status and modification records.",
        "parameters": {},
        "category": "self_modification",
        "dangerous": False,
    },
    "staging_reset": {
        "func": lambda **kw: _wrap_staging_reset(),
        "description": "Reset staging, discard all modifications, re-copy from current code.",
        "parameters": {},
        "category": "self_modification",
        "dangerous": False,
    },
    "hot_swap": {
        "func": lambda **kw: _wrap_hot_swap(),
        "description": "Hot swap! Validate staging and automatically start new version to replace current version. Current process will exit, new version takes over automatically.",
        "parameters": {},
        "category": "self_modification",
        "dangerous": True,
    },
}


def _wrap_staging_read(module_path, **kwargs) -> ToolResult:
    result = staging_read(module_path, **kwargs)
    if result.get("success"):
        return ToolResult(success=True, output=result.get("content", ""), data=result)
    return ToolResult(success=False, output="", error=result.get("error", ""))


def _wrap_staging_modify(module_path, old_code, new_code) -> ToolResult:
    result = staging_modify(module_path, old_code, new_code)
    if result.get("success"):
        return ToolResult(success=True, output=result.get("message", ""), data=result)
    return ToolResult(success=False, output="", error=result.get("error", ""))


def _wrap_staging_create(module_path, content) -> ToolResult:
    result = staging_create(module_path, content)
    if result.get("success"):
        return ToolResult(success=True, output=result.get("message", ""), data=result)
    return ToolResult(success=False, output="", error=result.get("error", ""))


def _wrap_staging_verify() -> ToolResult:
    result = staging_verify()
    if result.get("success"):
        return ToolResult(
            success=True,
            output=f"✅ Staging validation passed! {result.get('modifications', 0)} modification(s) made.\n{result.get('message', '')}",
            data=result,
        )
    return ToolResult(
        success=False,
        output="",
        error=f"Staging validation failed ({result.get('stage', '')}):\n" + "\n".join(result.get("errors", [])),
    )


def _wrap_staging_deploy() -> ToolResult:
    result = staging_prepare_deploy()
    if result.get("success"):
        return ToolResult(success=True, output=result.get("message", ""), data=result)
    return ToolResult(success=False, output="", error=result.get("error", ""))


def _wrap_staging_status() -> ToolResult:
    result = staging_status()
    import json
    return ToolResult(success=True, output=json.dumps(result, ensure_ascii=False, indent=2), data=result)


def _wrap_staging_reset() -> ToolResult:
    result = staging_reset()
    if result.get("success"):
        return ToolResult(success=True, output=result.get("message", ""))
    return ToolResult(success=False, output="", error=result.get("error", ""))


def _wrap_hot_swap() -> ToolResult:
    result = hot_swap_to_staging()
    if result.get("success"):
        return ToolResult(
            success=True,
            output=f"✅ Hot swap successful! New version PID: {result.get('new_pid')}. Current process is about to exit.",
            data=result,
        )
    return ToolResult(
        success=False,
        output="",
        error=f"Hot swap failed: {result.get('error', 'Unknown error')}",
        data=result,
    )


def create_tool_registry(safety_guardian=None) -> ToolRegistry:
    """
    Create and register all tools

    Args:
        safety_guardian: Safety guardian instance, used for security checks on dangerous operations

    Returns:
        ToolRegistry with all tools registered
    """
    registry = ToolRegistry()

    all_tools = {
        **SHELL_TOOLS,
        **FILE_TOOLS,
        **CODE_TOOLS,
        **WEB_TOOLS,
        **BLUEGREEN_TOOLS,
    }

    for name, tool_info in all_tools.items():
        original_func = tool_info["func"]
        dangerous = tool_info.get("dangerous", False)

        if safety_guardian and dangerous:
            def make_safe_wrapper(func, tool_name, is_dangerous):
                def safe_wrapper(**kwargs):
                    # First: validate actual tool parameters (not just text description)
                    if safety_guardian and hasattr(safety_guardian, 'validate_tool_call'):
                        param_check = safety_guardian.validate_tool_call(tool_name, kwargs)
                        if not param_check.get("allowed", True):
                            return ToolResult(
                                success=False,
                                output="",
                                error=f"Safety guardian blocked this operation: {param_check.get('reason', 'Parameter validation failed')}",
                            )
                    # Second: evaluate action safety (text description + params)
                    action_desc = f"Calling tool {tool_name}({kwargs})"
                    if safety_guardian:
                        safety_result = safety_guardian.evaluate_action(
                            action_desc,
                            context=f"Tool call: {tool_name}",
                            params=kwargs,
                        )
                        if not safety_result["safe"]:
                            return ToolResult(
                                success=False,
                                output="",
                                error=f"Safety guardian blocked this operation: {safety_result['reason']}",
                            )
                    return func(**kwargs)
                return safe_wrapper

            wrapped_func = make_safe_wrapper(original_func, name, dangerous)
        else:
            wrapped_func = original_func

        registry.register(
            name=name,
            description=tool_info["description"],
            func=wrapped_func,
            parameters=tool_info.get("parameters", {}),
            category=tool_info.get("category", "general"),
            dangerous=dangerous,
        )

    logger.info(f"Registered {len(all_tools)} tools")
    return registry
