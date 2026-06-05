"""File operation tools - read, write, copy, move, delete files"""

import logging
import os
import re
import shutil
from typing import Optional, Tuple

from .registry import ToolResult

logger = logging.getLogger(__name__)

# Project root directory (resolved at import time)
_PROJECT_DIR = os.path.realpath(os.getcwd())

# Data directory (relative to project root by default)
_DATA_DIR = os.path.realpath(os.environ.get("DATA_DIR", os.path.join(_PROJECT_DIR, "data")))

# Allowed directories for write/copy/move/delete operations
_WRITE_ALLOWED_DIRS = [_PROJECT_DIR, _DATA_DIR]

# Windows system sensitive paths
_WINDOWS_SENSITIVE_PATHS = [
    os.path.normpath(r"C:\Windows"),
    os.path.normpath(r"C:\ProgramData"),
    os.path.normpath(r"C:\Program Files"),
    os.path.normpath(r"C:\Program Files (x86)"),
]

# Linux system sensitive paths
_LINUX_SENSITIVE_PATHS = [
    "/etc",
    "/root",
    "/var/log",
    "/proc",
    "/sys",
    "/dev",
    "/boot",
    "/usr/share",
]

# File/directory name patterns that indicate credentials or secrets
_CREDENTIAL_PATTERNS = [
    re.compile(r"^\.env$", re.IGNORECASE),
    re.compile(r"^\.ssh$", re.IGNORECASE),
    re.compile(r"^\.gnupg$", re.IGNORECASE),
    re.compile(r"^\.aws$", re.IGNORECASE),
    re.compile(r"_key", re.IGNORECASE),
    re.compile(r"_secret", re.IGNORECASE),
    re.compile(r"\.pem$", re.IGNORECASE),
    re.compile(r"\.key$", re.IGNORECASE),
    re.compile(r"^id_rsa$", re.IGNORECASE),
    re.compile(r"^id_ed25519$", re.IGNORECASE),
    re.compile(r"^credentials$", re.IGNORECASE),
    re.compile(r"^\.netrc$", re.IGNORECASE),
    re.compile(r"^\.gitconfig$", re.IGNORECASE),
]


def _is_windows() -> bool:
    """Check if running on Windows"""
    return os.name == "nt"


def _get_sensitive_paths() -> list:
    """Get platform-specific sensitive system paths"""
    paths = list(_LINUX_SENSITIVE_PATHS)
    if _is_windows():
        paths.extend(_WINDOWS_SENSITIVE_PATHS)
    return paths


def _is_credential_name(name: str) -> bool:
    """Check if a file or directory name matches credential patterns"""
    for pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(name):
            return True
    return False


def _is_path_traversal(path: str) -> bool:
    """Check if a path contains traversal sequences"""
    # Check for .. in the path components (not just in the string)
    normalized = os.path.normpath(path)
    # After normalization, if the path goes above the base, normpath will contain ..
    # But we also check the raw string for suspicious patterns
    parts = path.replace("\\", "/").split("/")
    return ".." in parts


def _is_in_sensitive_path(resolved_path: str) -> Tuple[bool, str]:
    """Check if a resolved path falls under a system sensitive directory"""
    for sensitive in _get_sensitive_paths():
        sensitive_norm = os.path.normpath(sensitive)
        if resolved_path.lower().startswith(sensitive_norm.lower()) if _is_windows() else resolved_path.startswith(sensitive_norm):
            return True, f"Path is under system sensitive directory: {sensitive_norm}"
    return False, ""


def _is_appdata_path(resolved_path: str) -> bool:
    """Check if a path is under Windows AppData directory"""
    if not _is_windows():
        return False
    # Match C:\Users\*\AppData\
    pattern = re.compile(r"^[A-Za-z]:\\Users\\[^\\]+\\AppData\\", re.IGNORECASE)
    return bool(pattern.match(resolved_path))


def _is_home_secret_path(resolved_path: str) -> bool:
    """Check if a path is under Linux home secret directories (.ssh, .gnupg)"""
    if _is_windows():
        return False
    # Match /home/*/.ssh/ or /home/*/.gnupg/
    pattern = re.compile(r"^/home/[^/]+/\.ssh(/|$)")
    if pattern.match(resolved_path):
        return True
    pattern2 = re.compile(r"^/home/[^/]+/\.gnupg(/|$)")
    if pattern2.match(resolved_path):
        return True
    return False


def _is_in_allowed_dir(resolved_path: str, allowed_dirs: list) -> bool:
    """Check if a resolved path is within any of the allowed directories"""
    for allowed in allowed_dirs:
        allowed_norm = os.path.normpath(allowed)
        if resolved_path.lower().startswith(allowed_norm.lower()) if _is_windows() else resolved_path.startswith(allowed_norm):
            return True
    return False


def _validate_path(path: str, mode: str = "read") -> Tuple[bool, str, str]:
    """
    Validate a file path for security.

    Checks:
    - Resolves path to absolute and verifies it's within allowed directories
    - Blocks path traversal attempts (../, ..\\)
    - Blocks access to system sensitive paths
    - Blocks access to credential files/directories
    - For write operations, restricts to project and data directories only
    - For read operations, allows project directory freely, restricts outside

    Args:
        path: The file path to validate
        mode: Operation mode - "read" or "write". Write mode is stricter,
              only allowing operations within project and data directories.

    Returns:
        Tuple of (is_safe: bool, reason: str, resolved_path: str)
    """
    if not path or not isinstance(path, str):
        return False, "Path is empty or invalid", ""

    # Block path traversal attempts
    if _is_path_traversal(path):
        return False, f"Path traversal detected in: {path}", ""

    # Check raw input path for Linux sensitive paths (even on Windows,
    # to block attempts like /etc/shadow passed as input)
    raw_normalized = path.replace("\\", "/")
    for sensitive in _LINUX_SENSITIVE_PATHS:
        if raw_normalized.startswith(sensitive + "/") or raw_normalized == sensitive:
            return False, f"Path targets system sensitive directory: {sensitive}", ""

    # Resolve to absolute path
    try:
        resolved = os.path.realpath(path)
    except Exception as e:
        return False, f"Failed to resolve path: {e}", ""

    # Check system sensitive paths on resolved path
    is_sensitive, reason = _is_in_sensitive_path(resolved)
    if is_sensitive:
        return False, reason, resolved

    # Check Windows AppData
    if _is_appdata_path(resolved):
        return False, "Path is under Windows AppData directory", resolved

    # Check Linux home secret directories
    if _is_home_secret_path(resolved):
        return False, "Path is under home secret directory (.ssh/.gnupg)", resolved

    # Check credential file/directory names
    basename = os.path.basename(resolved)
    if _is_credential_name(basename):
        # Allow credential-named files within project directory for read mode
        if mode == "read" and _is_in_allowed_dir(resolved, [_PROJECT_DIR]):
            pass  # Allow reading credential files within the project directory
        else:
            return False, f"Access to credential file/directory is blocked: {basename}", resolved

    # Also check parent directories for credential directory names (e.g. .ssh/authorized_keys)
    parts = resolved.replace("\\", "/").split("/")
    for part in parts:
        if _is_credential_name(part) and part != basename:
            if mode == "read" and _is_in_allowed_dir(resolved, [_PROJECT_DIR]):
                pass  # Allow within project directory
            else:
                return False, f"Path traverses credential directory: {part}", resolved

    # Mode-specific checks
    if mode == "write":
        # Write/copy/move/delete: only allow within project and data directories
        if not _is_in_allowed_dir(resolved, _WRITE_ALLOWED_DIRS):
            return False, f"Write operation not allowed outside project/data directories: {resolved}", resolved
    elif mode == "read":
        # Read: allow within project directory freely
        # Outside project directory, additional restrictions already applied above
        # (sensitive paths, credential files, etc.)
        pass

    return True, "", resolved


def file_read(path: str, encoding: str = "utf-8", start_line: int = 0,
              end_line: Optional[int] = None) -> ToolResult:
    """
    Read file contents

    Args:
        path: File path
        encoding: Encoding
        start_line: Start line (0-based)
        end_line: End line (exclusive)
    """
    # Validate path security
    is_safe, reason, resolved = _validate_path(path, mode="read")
    if not is_safe:
        return ToolResult(success=False, output="", error=f"Path security check failed: {reason}")

    if not os.path.exists(resolved):
        return ToolResult(success=False, output="", error=f"File does not exist: {path}")

    if os.path.isdir(resolved):
        return ToolResult(success=False, output="", error=f"Path is a directory, not a file: {path}")

    try:
        with open(resolved, "r", encoding=encoding, errors="replace") as f:
            lines = f.readlines()

        # Line range
        if end_line is None:
            end_line = len(lines)
        selected = lines[start_line:end_line]

        # Add line numbers
        numbered = []
        for i, line in enumerate(selected, start=start_line + 1):
            numbered.append(f"{i:4d} | {line.rstrip()}")

        content = "\n".join(numbered)
        total_lines = len(lines)
        shown_lines = len(selected)

        return ToolResult(
            success=True,
            output=content[:20000],  # Limit output
            data={
                "path": path,
                "total_lines": total_lines,
                "shown_lines": shown_lines,
                "start_line": start_line,
                "end_line": end_line,
            },
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Read failed: {e}")


def file_write(path: str, content: str, encoding: str = "utf-8",
               append: bool = False) -> ToolResult:
    """
    Write to file

    Args:
        path: File path
        content: Content to write
        encoding: Encoding
        append: Whether to use append mode
    """
    # Validate path security (write mode: restricted to project/data dirs)
    is_safe, reason, resolved = _validate_path(path, mode="write")
    if not is_safe:
        return ToolResult(success=False, output="", error=f"Path security check failed: {reason}")

    try:
        os.makedirs(os.path.dirname(resolved) if os.path.dirname(resolved) else ".", exist_ok=True)
        mode = "a" if append else "w"
        with open(resolved, mode, encoding=encoding) as f:
            f.write(content)

        return ToolResult(
            success=True,
            output=f"{'Appended' if append else 'Wrote'} {len(content)} characters to {path}",
            data={"path": path, "bytes_written": len(content.encode(encoding)), "append": append},
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Write failed: {e}")


def file_copy(src: str, dst: str) -> ToolResult:
    """
    Copy file or directory

    Args:
        src: Source path
        dst: Destination path
    """
    # Validate both source (read) and destination (write) paths
    src_safe, src_reason, src_resolved = _validate_path(src, mode="read")
    if not src_safe:
        return ToolResult(success=False, output="", error=f"Source path security check failed: {src_reason}")

    dst_safe, dst_reason, dst_resolved = _validate_path(dst, mode="write")
    if not dst_safe:
        return ToolResult(success=False, output="", error=f"Destination path security check failed: {dst_reason}")

    if not os.path.exists(src_resolved):
        return ToolResult(success=False, output="", error=f"Source path does not exist: {src}")

    try:
        if os.path.isdir(src_resolved):
            shutil.copytree(src_resolved, dst_resolved, dirs_exist_ok=True)
            return ToolResult(success=True, output=f"Copied directory {src} -> {dst}")
        else:
            os.makedirs(os.path.dirname(dst_resolved) if os.path.dirname(dst_resolved) else ".", exist_ok=True)
            shutil.copy2(src_resolved, dst_resolved)
            return ToolResult(success=True, output=f"Copied file {src} -> {dst}")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Copy failed: {e}")


def file_move(src: str, dst: str) -> ToolResult:
    """
    Move file or directory

    Args:
        src: Source path
        dst: Destination path
    """
    # Validate both source (write, since moving removes it) and destination (write) paths
    src_safe, src_reason, src_resolved = _validate_path(src, mode="write")
    if not src_safe:
        return ToolResult(success=False, output="", error=f"Source path security check failed: {src_reason}")

    dst_safe, dst_reason, dst_resolved = _validate_path(dst, mode="write")
    if not dst_safe:
        return ToolResult(success=False, output="", error=f"Destination path security check failed: {dst_reason}")

    if not os.path.exists(src_resolved):
        return ToolResult(success=False, output="", error=f"Source path does not exist: {src}")

    try:
        os.makedirs(os.path.dirname(dst_resolved) if os.path.dirname(dst_resolved) else ".", exist_ok=True)
        shutil.move(src_resolved, dst_resolved)
        return ToolResult(success=True, output=f"Moved {src} -> {dst}")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Move failed: {e}")


def file_delete(path: str) -> ToolResult:
    """
    Delete file or directory (use with caution)

    Args:
        path: Path to delete
    """
    # Validate path security (write mode: restricted to project/data dirs)
    is_safe, reason, resolved = _validate_path(path, mode="write")
    if not is_safe:
        return ToolResult(success=False, output="", error=f"Path security check failed: {reason}")

    if not os.path.exists(resolved):
        return ToolResult(success=False, output="", error=f"Path does not exist: {path}")

    try:
        if os.path.isdir(resolved):
            shutil.rmtree(resolved)
            return ToolResult(success=True, output=f"Deleted directory: {path}")
        else:
            os.remove(resolved)
            return ToolResult(success=True, output=f"Deleted file: {path}")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Delete failed: {e}")


def file_list(path: str = ".", pattern: str = "*") -> ToolResult:
    """
    List directory contents

    Args:
        path: Directory path
        pattern: Glob match pattern
    """
    import glob

    # Validate path security
    is_safe, reason, resolved = _validate_path(path, mode="read")
    if not is_safe:
        return ToolResult(success=False, output="", error=f"Path security check failed: {reason}")

    if not os.path.exists(resolved):
        return ToolResult(success=False, output="", error=f"Path does not exist: {path}")

    if not os.path.isdir(resolved):
        return ToolResult(success=False, output="", error=f"Not a directory: {path}")

    try:
        search_path = os.path.join(resolved, pattern)
        entries = glob.glob(search_path)

        lines = []
        for entry in sorted(entries):
            name = os.path.basename(entry)
            if os.path.isdir(entry):
                lines.append(f"  [DIR]  {name}/")
            else:
                size = os.path.getsize(entry)
                lines.append(f"  [FILE] {name} ({size} bytes)")

        content = f"Directory: {resolved}\n" + "\n".join(lines)
        return ToolResult(
            success=True,
            output=content,
            data={"path": path, "count": len(entries)},
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=f"List failed: {e}")


def file_search(directory: str, keyword: str, file_pattern: str = "*.py") -> ToolResult:
    """
    Search for files containing a keyword in a directory

    Args:
        directory: Directory to search
        keyword: Search keyword
        file_pattern: File match pattern
    """
    import glob

    # Validate directory path security
    is_safe, reason, resolved_dir = _validate_path(directory, mode="read")
    if not is_safe:
        return ToolResult(success=False, output="", error=f"Directory path security check failed: {reason}")

    if not os.path.exists(resolved_dir):
        return ToolResult(success=False, output="", error=f"Directory does not exist: {directory}")

    try:
        search_path = os.path.join(resolved_dir, "**", file_pattern)
        files = glob.glob(search_path, recursive=True)

        results = []
        for filepath in files:
            if os.path.isdir(filepath):
                continue
            # Validate each file before reading
            file_safe, file_reason, _ = _validate_path(filepath, mode="read")
            if not file_safe:
                continue
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if keyword.lower() in line.lower():
                            results.append(f"{filepath}:{line_num}: {line.strip()[:100]}")
            except Exception:
                continue

        output = "\n".join(results[:50])  # Limit number of results
        return ToolResult(
            success=True,
            output=output or f"No content containing '{keyword}' found",
            data={"matches": len(results), "files_searched": len(files)},
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Search failed: {e}")


# Tool registration info
FILE_TOOLS = {
    "file_read": {
        "func": file_read,
        "description": "Read file contents, supports specifying line range. Returns content with line numbers.",
        "parameters": {
            "path": "File path",
            "encoding": "Encoding, default utf-8",
            "start_line": "Start line number (0-based), default 0",
            "end_line": "End line number (exclusive), default to end of file",
        },
        "category": "file",
        "dangerous": False,
    },
    "file_write": {
        "func": file_write,
        "description": "Write file contents. Can overwrite or append.",
        "parameters": {
            "path": "File path",
            "content": "Content to write",
            "encoding": "Encoding, default utf-8",
            "append": "Whether to use append mode, default False",
        },
        "category": "file",
        "dangerous": True,
    },
    "file_copy": {
        "func": file_copy,
        "description": "Copy file or directory.",
        "parameters": {
            "src": "Source path",
            "dst": "Destination path",
        },
        "category": "file",
        "dangerous": False,
    },
    "file_move": {
        "func": file_move,
        "description": "Move file or directory.",
        "parameters": {
            "src": "Source path",
            "dst": "Destination path",
        },
        "category": "file",
        "dangerous": True,
    },
    "file_delete": {
        "func": file_delete,
        "description": "Delete file or directory. Use with caution!",
        "parameters": {
            "path": "Path to delete",
        },
        "category": "file",
        "dangerous": True,
    },
    "file_list": {
        "func": file_list,
        "description": "List directory contents.",
        "parameters": {
            "path": "Directory path, default current directory",
            "pattern": "Glob match pattern, default *",
        },
        "category": "file",
        "dangerous": False,
    },
    "file_search": {
        "func": file_search,
        "description": "Search for files containing a keyword in a directory.",
        "parameters": {
            "directory": "Directory to search",
            "keyword": "Search keyword",
            "file_pattern": "File match pattern, default *.py",
        },
        "category": "file",
        "dangerous": False,
    },
}
