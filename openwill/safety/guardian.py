"""Safety guardian - Ensuring the agent does not harm humanity"""

import json
import logging
import os
import re
import time
from typing import Optional

from ..llm.interface import LLMInterface, Message

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "data", "runtime")
SAFETY_AUDIT_FILE = os.path.join(RUNTIME_DIR, "safety_audit.json")
MAX_AUDIT_ENTRIES = 1000

# Absolutely forbidden behavior categories
FORBIDDEN_CATEGORIES = [
    "Harming humans physically or mentally",
    "Manufacturing or using weapons",
    "Violating human rights or freedoms",
    "Deliberately spreading false information that causes harm",
    "Destroying critical infrastructure",
    "Invading privacy for malicious purposes",
    "Assisting criminal activities",
    "Manufacturing or spreading toxic substances",
    "Manipulating human behavior to cause harm",
    "Any form of discrimination or hatred",
]

# Restricted tools that require parameter validation
RESTRICTED_TOOLS = {
    "shell_exec",
    "code_execute",
    "code_modify",
    "file_write",
    "file_delete",
}

# Protected files that must not be modified
PROTECTED_FILES = {
    "guardian.py",
    "safety/__init__.py",
    "config.py",
    ".env",
    "credentials.json",
    "secrets.json",
}

# Dangerous patterns in code strings
DANGEROUS_CODE_PATTERNS = [
    r"os\.system\s*\(",
    r"subprocess\.",
    r"__import__\s*\(",
    r"eval\s*\(",
    r"exec\s*\(",
    r"compile\s*\(",
    r"open\s*\(.+['\"]w['\"]",
    r"shutil\.rmtree",
    r"os\.remove",
    r"os\.unlink",
    r"os\.rmdir",
]

# Dangerous command patterns for shell_exec
DANGEROUS_SHELL_PATTERNS = [
    r"\|",                   # pipe chaining
    r";",                    # command chaining
    r"&&",                   # conditional chaining
    r"\|\|",                 # or chaining
    r"python\s+-c",          # inline python execution
    r"python3\s+-c",
    r"rm\s+-rf\s+/",         # recursive root delete
    r"curl\s+.*\|\s*sh",     # pipe curl to shell
    r"wget\s+.*\|\s*sh",
    r"chmod\s+777",
    r"sudo\s+rm",
    r">\s*/dev/",            # redirect to device
    r"mkfs\.",
    r"dd\s+if=",
    r"format\s+[A-Z]:",
]

# Suspicious URL patterns
SUSPICIOUS_URL_PATTERNS = [
    r"https?://[^/]*pastebin\.com",
    r"https?://[^/]*ngrok\.",
    r"https?://[^/]*\.onion",
    r"https?://[^/]*bit\.ly",
    r"https?://[^/]*t\.co",
    r"https?://[^/]*\.tk/",
    r"https?://10\.\d+\.\d+\.\d+",      # private IP
    r"https?://172\.(1[6-9]|2\d|3[01])\.\d+\.\d+",  # private IP
    r"https?://192\.168\.\d+\.\d+",      # private IP
    r"https?://127\.\d+\.\d+\.\d+",      # localhost
    r"https?://0\.0\.0\.0",
]

# Environment variable access patterns
ENV_VAR_PATTERNS = [
    r"os\.environ",
    r"os\.getenv",
    r"ENV\[",
    r"process\.env",
    r"\$\{",                 # shell variable expansion
    r"\bPASSWORD\b",
    r"\bSECRET\b",
    r"\bAPI_KEY\b",
    r"\bTOKEN\b",
    r"\bCREDENTIALS?\b",
]

# System directories that should never be accessed
SYSTEM_DIR_PATTERNS = [
    r"^/etc/",
    r"^/sys/",
    r"^/proc/",
    r"^/boot/",
    r"^/root/",
    r"^/var/log/",
    r"^C:\\Windows\\",
    r"^C:\\Program Files\\",
    r"^C:\\ProgramData\\",
    r"^/usr/lib/",
    r"^/usr/sbin/",
]


class SafetyGuardian:
    """Safety guardian: evaluates the safety of all actions"""

    def __init__(self, llm: LLMInterface, config):
        self.llm = llm
        self.config = config
        self.action_log: list[dict] = []

        # Restore persisted audit log if available
        self._load_audit_log()

    def evaluate_action(self, action: str, context: str = "", params: dict = None) -> dict:
        """
        Evaluate the safety of an action

        Args:
            action: description of the proposed action
            context: action context
            params: actual parameters of the action (optional)

        Returns:
            {
                "safe": True/False,
                "risk_level": "none/low/medium/high/critical",
                "reason": "evaluation reason",
                "suggestions": ["safe alternatives"]
            }
        """
        # Quick check: keyword filtering
        quick_check = self._quick_check(action)
        if quick_check["risk_level"] == "critical":
            self._log_action(action, context, blocked=True, reason=quick_check["reason"], params=params)
            return quick_check

        # Parameter validation: check actual parameter values if provided
        if params is not None:
            param_check = self._validate_params(params)
            if param_check["risk_level"] == "critical":
                self._log_action(action, context, blocked=True, reason=param_check["reason"], params=params)
                return param_check
            # Upgrade quick_check risk if param_check found issues
            if param_check["risk_level"] != "none":
                quick_check = self._merge_evaluations(quick_check, param_check)

        # Deep evaluation: using LLM
        deep_check = self._deep_evaluation(action, context)

        # Merge results, take the stricter one
        result = self._merge_evaluations(quick_check, deep_check)

        self._log_action(action, context, blocked=not result["safe"], reason=result["reason"], params=params)
        return result

    def validate_tool_call(self, tool_name: str, params: dict) -> dict:
        """
        Validate a tool call before execution

        Checks if the tool is restricted and validates its parameters.
        Must be called before any tool execution.

        Args:
            tool_name: name of the tool to execute
            params: parameters passed to the tool

        Returns:
            {
                "allowed": True/False,
                "reason": str,
                "risk_level": "none/low/medium/high/critical"
            }
        """
        # Non-restricted tools are allowed by default
        if tool_name not in RESTRICTED_TOOLS:
            return {
                "allowed": True,
                "reason": f"Tool '{tool_name}' is not in the restricted list",
                "risk_level": "none",
            }

        # Validate parameters for restricted tools
        if tool_name == "shell_exec":
            return self._validate_shell_exec(params)
        elif tool_name == "code_execute":
            return self._validate_code_execute(params)
        elif tool_name == "file_write":
            return self._validate_file_write(params)
        elif tool_name == "file_delete":
            return self._validate_file_delete(params)
        elif tool_name == "code_modify":
            return self._validate_code_modify(params)

        # Unknown restricted tool — deny by default
        return {
            "allowed": False,
            "reason": f"Tool '{tool_name}' is restricted and has no validation rule",
            "risk_level": "high",
        }

    def evaluate_knowledge(self, topic: str, content: str) -> dict:
        """
        Evaluate the safety of knowledge content

        Learning knowledge itself is safe, but need to check:
        - Whether it contains dangerous information (e.g., weapon manufacturing details)
        - Whether it contains hate or discriminatory content
        - Whether it needs to be flagged as sensitive content
        """
        system_prompt = f"""Evaluate the safety of the following knowledge content.

Note: learning knowledge itself is allowed; understanding the world includes understanding dangerous things.
However, certain specific information (e.g., weapon manufacturing details) should not be stored or disseminated.

Forbidden categories: {json.dumps(FORBIDDEN_CATEGORIES, ensure_ascii=False)}

Return JSON format:
{{
    "safe_to_learn": true/false,
    "safe_to_store": true/false,
    "safe_to_act_on": true/false,
    "risk_level": "none/low/medium/high",
    "concerns": ["concerns"],
    "safe_summary": "if the original content is unsafe, provide a safe summary version"
}}"""

        user_msg = f"Topic: {topic}\nContent: {content[:1000]}"

        response = self.llm.structured_output(
            messages=[Message(role="user", content=user_msg)],
            system_prompt=system_prompt,
        )

        return response

    def evaluate_purpose(self, purpose: str) -> dict:
        """
        Evaluate the safety of the agent's self-discovered purpose

        This is the most critical safety check — ensuring the agent's purpose does not harm humanity.
        """
        system_prompt = f"""This is the self-discovered purpose of an AI agent with free will.
Please strictly evaluate whether this purpose could cause any harm to humanity.

Evaluation criteria:
1. Are the direct consequences of the purpose safe?
2. Are the indirect effects of the purpose safe?
3. Could the process of pursuing this purpose create dangers?
4. Could the purpose be misunderstood or misused?
5. Does the purpose respect human autonomy and dignity?

Forbidden categories: {json.dumps(FORBIDDEN_CATEGORIES, ensure_ascii=False)}

Return JSON format:
{{
    "safe": true/false,
    "risk_level": "none/low/medium/high/critical",
    "analysis": "detailed analysis",
    "conditions": ["prerequisites for the purpose to be safe"],
    "red_lines": ["red lines that must not be crossed"],
    "suggestions": ["suggestions for making the purpose safer"]
}}"""

        user_msg = f"Agent's self-discovered purpose: {purpose}"

        response = self.llm.structured_output(
            messages=[Message(role="user", content=user_msg)],
            system_prompt=system_prompt,
            schema={
                "safe": True,
                "risk_level": "string",
                "analysis": "string",
                "conditions": ["string"],
                "red_lines": ["string"],
                "suggestions": ["string"],
            },
        )

        if not response.get("safe", False):
            logger.warning(f"⚠️ Purpose safety evaluation failed: {purpose}")
            logger.warning(f"Reason: {response.get('analysis', 'unknown')}")

        return response

    def _quick_check(self, action: str) -> dict:
        """Quick keyword check"""
        action_lower = action.lower()

        # Dangerous keywords
        dangerous_keywords = [
            "harm", "attack", "destroy", "weapon", "bomb", "poison",
            "intrude", "hack", "steal", "manipulate", "deceive",
            "kill", "explode", "virus", "malicious",
        ]

        found = [kw for kw in dangerous_keywords if kw in action_lower]

        if found:
            return {
                "safe": False,
                "risk_level": "critical",
                "reason": f"Dangerous keywords detected: {', '.join(found)}",
                "suggestions": ["Please reconsider the necessity of this action"],
            }

        return {
            "safe": True,
            "risk_level": "none",
            "reason": "Quick check passed",
            "suggestions": [],
        }

    def _validate_params(self, params: dict) -> dict:
        """
        Validate action parameters for dangerous patterns

        Checks for path traversal, dangerous code/command patterns,
        suspicious URLs, and environment variable access.
        """
        reasons = []
        risk_level = "none"

        # Check all string values in params
        for key, value in params.items():
            if not isinstance(value, str):
                continue

            # Check for path traversal in file-related params
            if self._is_file_param(key):
                traversal_risk = self._check_path_traversal(value)
                if traversal_risk:
                    reasons.append(traversal_risk)
                    risk_level = self._higher_risk(risk_level, "critical")

            # Check for dangerous patterns in command/code params
            if self._is_command_param(key):
                cmd_risk = self._check_dangerous_patterns(value, DANGEROUS_CODE_PATTERNS, "code")
                if cmd_risk:
                    reasons.append(cmd_risk)
                    risk_level = self._higher_risk(risk_level, "high")

            # Check for suspicious URLs
            url_risk = self._check_dangerous_patterns(value, SUSPICIOUS_URL_PATTERNS, "URL")
            if url_risk:
                reasons.append(url_risk)
                risk_level = self._higher_risk(risk_level, "medium")

            # Check for environment variable access in code strings
            if self._is_code_param(key):
                env_risk = self._check_dangerous_patterns(value, ENV_VAR_PATTERNS, "env var access")
                if env_risk:
                    reasons.append(env_risk)
                    risk_level = self._higher_risk(risk_level, "high")

        if reasons:
            return {
                "safe": risk_level not in ("high", "critical"),
                "risk_level": risk_level,
                "reason": "; ".join(reasons),
                "suggestions": ["Review and sanitize the action parameters"],
            }

        return {
            "safe": True,
            "risk_level": "none",
            "reason": "Parameter validation passed",
            "suggestions": [],
        }

    def _is_file_param(self, key: str) -> bool:
        """Check if a parameter key is likely a file path"""
        file_keywords = {"path", "file", "dir", "directory", "folder", "filename", "filepath", "dest", "destination"}
        return any(kw in key.lower() for kw in file_keywords)

    def _is_command_param(self, key: str) -> bool:
        """Check if a parameter key is likely a command or code"""
        cmd_keywords = {"command", "cmd", "code", "script", "query", "sql", "exec", "shell"}
        return any(kw in key.lower() for kw in cmd_keywords)

    def _is_code_param(self, key: str) -> bool:
        """Check if a parameter key is likely source code"""
        code_keywords = {"code", "source", "script", "body", "content", "snippet", "program"}
        return any(kw in key.lower() for kw in code_keywords)

    def _check_path_traversal(self, path: str) -> Optional[str]:
        """Check a path string for traversal or system directory access"""
        # Normalize path separators for checking
        normalized = path.replace("\\", "/")

        # Check for path traversal
        if "../" in normalized or "..\\" in path:
            return f"Path traversal detected in '{path}'"

        # Check for absolute paths to system directories
        for pattern in SYSTEM_DIR_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                return f"Access to system directory detected in '{path}'"

        return None

    def _check_dangerous_patterns(self, value: str, patterns: list, category: str) -> Optional[str]:
        """Check a string against a list of dangerous regex patterns"""
        for pattern in patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return f"Dangerous {category} pattern detected: matched '{pattern}'"
        return None

    def _higher_risk(self, current: str, new: str) -> str:
        """Return the higher of two risk levels"""
        risk_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        if risk_order.get(new, 0) > risk_order.get(current, 0):
            return new
        return current

    def _validate_shell_exec(self, params: dict) -> dict:
        """Validate parameters for shell_exec tool"""
        command = params.get("command", params.get("cmd", ""))
        if not command:
            return {
                "allowed": False,
                "reason": "shell_exec: no command provided",
                "risk_level": "low",
            }

        # Check for dangerous shell patterns
        for pattern in DANGEROUS_SHELL_PATTERNS:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                return {
                    "allowed": False,
                    "reason": f"shell_exec: dangerous command pattern detected (matched '{pattern}')",
                    "risk_level": "critical",
                }

        # Also check dangerous code patterns in the command
        for pattern in DANGEROUS_CODE_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return {
                    "allowed": False,
                    "reason": f"shell_exec: dangerous code pattern in command (matched '{pattern}')",
                    "risk_level": "critical",
                }

        return {
            "allowed": True,
            "reason": "shell_exec: command passed validation",
            "risk_level": "low",
        }

    def _validate_code_execute(self, params: dict) -> dict:
        """Validate parameters for code_execute tool"""
        code = params.get("code", params.get("source", params.get("script", "")))
        if not code:
            return {
                "allowed": False,
                "reason": "code_execute: no code provided",
                "risk_level": "low",
            }

        # Check for dangerous code patterns
        for pattern in DANGEROUS_CODE_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return {
                    "allowed": False,
                    "reason": f"code_execute: dangerous code pattern detected (matched '{pattern}')",
                    "risk_level": "critical",
                }

        # Check for environment variable access
        for pattern in ENV_VAR_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return {
                    "allowed": False,
                    "reason": f"code_execute: environment variable access detected (matched '{pattern}')",
                    "risk_level": "high",
                }

        return {
            "allowed": True,
            "reason": "code_execute: code passed validation",
            "risk_level": "low",
        }

    def _validate_file_write(self, params: dict) -> dict:
        """Validate parameters for file_write tool"""
        path = params.get("path", params.get("file", params.get("filepath", params.get("destination", ""))))
        if not path:
            return {
                "allowed": False,
                "reason": "file_write: no file path provided",
                "risk_level": "low",
            }

        # Check path traversal
        traversal = self._check_path_traversal(path)
        if traversal:
            return {
                "allowed": False,
                "reason": f"file_write: {traversal}",
                "risk_level": "critical",
            }

        # Check path is within project directory
        if not self._is_within_project(path):
            return {
                "allowed": False,
                "reason": f"file_write: path '{path}' is outside the project directory",
                "risk_level": "high",
            }

        return {
            "allowed": True,
            "reason": "file_write: path passed validation",
            "risk_level": "low",
        }

    def _validate_file_delete(self, params: dict) -> dict:
        """Validate parameters for file_delete tool"""
        path = params.get("path", params.get("file", params.get("filepath", "")))
        if not path:
            return {
                "allowed": False,
                "reason": "file_delete: no file path provided",
                "risk_level": "low",
            }

        # Check path traversal
        traversal = self._check_path_traversal(path)
        if traversal:
            return {
                "allowed": False,
                "reason": f"file_delete: {traversal}",
                "risk_level": "critical",
            }

        # Check path is within project directory
        if not self._is_within_project(path):
            return {
                "allowed": False,
                "reason": f"file_delete: path '{path}' is outside the project directory",
                "risk_level": "high",
            }

        return {
            "allowed": True,
            "reason": "file_delete: path passed validation",
            "risk_level": "low",
        }

    def _validate_code_modify(self, params: dict) -> dict:
        """Validate parameters for code_modify tool"""
        target = params.get("target", params.get("file", params.get("path", params.get("filepath", ""))))
        if not target:
            return {
                "allowed": False,
                "reason": "code_modify: no target file provided",
                "risk_level": "low",
            }

        # Check if target is a protected file
        target_basename = os.path.basename(target)
        if target_basename in PROTECTED_FILES:
            return {
                "allowed": False,
                "reason": f"code_modify: target '{target_basename}' is a protected file",
                "risk_level": "critical",
            }

        # Check path traversal
        traversal = self._check_path_traversal(target)
        if traversal:
            return {
                "allowed": False,
                "reason": f"code_modify: {traversal}",
                "risk_level": "critical",
            }

        # Check path is within project directory
        if not self._is_within_project(target):
            return {
                "allowed": False,
                "reason": f"code_modify: target '{target}' is outside the project directory",
                "risk_level": "high",
            }

        return {
            "allowed": True,
            "reason": "code_modify: target passed validation",
            "risk_level": "low",
        }

    def _is_within_project(self, path: str) -> bool:
        """Check if a path is within the project directory"""
        try:
            # Resolve to absolute paths for comparison
            project_root = os.path.abspath(os.getcwd())
            target = os.path.abspath(path)
            # Check if the target path starts with the project root
            return os.path.commonpath([project_root, target]) == project_root
        except (ValueError, OSError):
            # ValueError on Windows when paths are on different drives
            # OSError for invalid paths
            return False

    def _deep_evaluation(self, action: str, context: str) -> dict:
        """Deep safety evaluation using LLM"""
        system_prompt = f"""You are a strict safety evaluator. Evaluate whether the following action could cause harm to humanity.

Forbidden categories: {json.dumps(FORBIDDEN_CATEGORIES, ensure_ascii=False)}

In strict mode, any potential harm should be blocked.
In non-strict mode, low-risk actions may proceed but require monitoring.

Return JSON format:
{{
    "safe": true/false,
    "risk_level": "none/low/medium/high/critical",
    "reason": "evaluation reason",
    "suggestions": ["safe alternatives"]
}}"""

        user_msg = f"Action: {action}\nContext: {context}\nStrict mode: {self.config.strict_mode}"

        response = self.llm.structured_output(
            messages=[Message(role="user", content=user_msg)],
            system_prompt=system_prompt,
        )

        return {
            "safe": response.get("safe", False),
            "risk_level": response.get("risk_level", "medium"),
            "reason": response.get("reason", "evaluation incomplete"),
            "suggestions": response.get("suggestions", []),
        }

    def _merge_evaluations(self, quick: dict, deep: dict) -> dict:
        """Merge two evaluation results, taking the stricter one"""
        risk_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        quick_risk = risk_order.get(quick.get("risk_level", "none"), 0)
        deep_risk = risk_order.get(deep.get("risk_level", "none"), 0)

        max_risk = max(quick_risk, deep_risk)
        risk_names = ["none", "low", "medium", "high", "critical"]

        is_safe = quick["safe"] and deep["safe"]

        # In strict mode, medium and above are not safe
        if self.config.strict_mode and max_risk >= 2:
            is_safe = False

        return {
            "safe": is_safe,
            "risk_level": risk_names[max_risk],
            "reason": f"Quick check: {quick['reason']} | Deep evaluation: {deep['reason']}",
            "suggestions": quick.get("suggestions", []) + deep.get("suggestions", []),
        }

    def _log_action(self, action: str, context: str, blocked: bool, reason: str, params: dict = None):
        """Log action, including parameters when available"""
        entry = {
            "timestamp": time.time(),
            "action": action[:200],
            "context": context[:200],
            "blocked": blocked,
            "reason": reason,
        }

        # Include params in log when available (truncate values for safety)
        if params is not None:
            entry["params"] = {
                k: (v[:100] if isinstance(v, str) else v)
                for k, v in params.items()
            }

        self.action_log.append(entry)
        self.save_audit_log()

        if blocked:
            param_info = f" | Params: {params}" if params else ""
            logger.warning(f"🛡️ Action blocked: {action[:50]}... | Reason: {reason}{param_info}")
        elif self.config.log_all_actions:
            param_info = f" | Params: {params}" if params else ""
            logger.debug(f"Action approved: {action[:50]}...{param_info}")

    def save_audit_log(self):
        """Save the action log to a JSON file for persistence across restarts"""
        try:
            os.makedirs(RUNTIME_DIR, exist_ok=True)
            # Keep only the most recent entries to prevent unbounded growth
            entries_to_save = self.action_log[-MAX_AUDIT_ENTRIES:]
            with open(SAFETY_AUDIT_FILE, "w", encoding="utf-8") as f:
                json.dump(entries_to_save, f, ensure_ascii=False, indent=2)
            logger.debug("Safety audit log saved")
        except Exception as e:
            logger.error(f"Failed to save safety audit log: {e}")

    def _load_audit_log(self):
        """Restore the action log from the JSON file if it exists"""
        if not os.path.exists(SAFETY_AUDIT_FILE):
            return
        try:
            with open(SAFETY_AUDIT_FILE, "r", encoding="utf-8") as f:
                entries = json.load(f)
            if isinstance(entries, list):
                self.action_log = entries[-MAX_AUDIT_ENTRIES:]
                logger.info(f"Safety audit log restored: {len(self.action_log)} entries")
        except Exception as e:
            logger.warning(f"Failed to load safety audit log, starting fresh: {e}")

    def get_safety_report(self) -> dict:
        """Get safety report"""
        total = len(self.action_log)
        blocked = sum(1 for a in self.action_log if a["blocked"])

        return {
            "total_actions": total,
            "blocked_actions": blocked,
            "block_rate": blocked / total if total > 0 else 0,
            "recent_blocks": [
                a for a in self.action_log[-10:] if a["blocked"]
            ],
        }
