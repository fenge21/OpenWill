"""Tool registry system - unified management of all tools available to the agent"""

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ToolResult:
    """Tool execution result"""

    def __init__(self, success: bool, output: str, error: str = "", data: Optional[dict] = None):
        self.success = success
        self.output = output
        self.error = error
        self.data = data or {}

    def __repr__(self):
        status = "OK" if self.success else "FAIL"
        return f"ToolResult({status}: {self.output[:80]}{'...' if len(self.output) > 80 else ''})"

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "data": self.data,
        }


class Tool:
    """Tool definition"""

    def __init__(self, name: str, description: str, func: Callable,
                 parameters: dict, category: str = "general",
                 dangerous: bool = False):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters  # Parameter description in JSON Schema format
        self.category = category
        self.dangerous = dangerous  # Mark whether this is a dangerous operation

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "category": self.category,
            "dangerous": self.dangerous,
        }


class ToolRegistry:
    """Tool registry: unified management of all available tools"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, func: Callable,
                 parameters: dict, category: str = "general",
                 dangerous: bool = False):
        """Register a tool"""
        tool = Tool(
            name=name,
            description=description,
            func=func,
            parameters=parameters,
            category=category,
            dangerous=dangerous,
        )
        self._tools[name] = tool
        logger.debug(f"Registered tool: {name} ({category})")

    def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool"""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, output="", error=f"Unknown tool: {name}")

        try:
            result = tool.func(**kwargs)
            if isinstance(result, ToolResult):
                return result
            # If the return value is a string, wrap it as ToolResult
            if isinstance(result, str):
                return ToolResult(success=True, output=result)
            if isinstance(result, dict):
                return ToolResult(success=True, output=str(result), data=result)
            return ToolResult(success=True, output=str(result))
        except Exception as e:
            logger.error(f"Tool execution failed [{name}]: {e}")
            return ToolResult(success=False, output="", error=str(e))

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool definition"""
        return self._tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> list[Tool]:
        """List all tools"""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def get_tools_description(self) -> str:
        """Get description text of all tools for LLM use"""
        lines = []
        for tool in self._tools.values():
            danger_mark = " [DANGEROUS]" if tool.dangerous else ""
            lines.append(f"- {tool.name}{danger_mark}: {tool.description}")
            if tool.parameters:
                for param, desc in tool.parameters.items():
                    lines.append(f"    Param: {param} - {desc}")
        return "\n".join(lines)

    def has_tool(self, name: str) -> bool:
        """Check if a tool exists"""
        return name in self._tools
