"""MCP (Model Context Protocol) client for connecting to external MCP servers and registering their tools."""

import json
import logging
import os
import subprocess
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Default persistence path for server configs
_MCP_CONFIG_PATH = os.path.join("data", "runtime", "mcp_servers.json")


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""

    name: str
    command: Optional[str] = None
    args: list = field(default_factory=list)
    url: Optional[str] = None
    env: dict = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "command": self.command,
            "args": self.args,
            "url": self.url,
            "env": self.env,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MCPServerConfig":
        return cls(
            name=data.get("name", ""),
            command=data.get("command"),
            args=data.get("args", []),
            url=data.get("url"),
            env=data.get("env", {}),
            enabled=data.get("enabled", True),
        )


class _StdioTransport:
    """Transport for local MCP servers using stdio (subprocess + JSON-RPC 2.0)."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._lock = threading.Lock()
        self._initialized = False

    def start(self) -> bool:
        """Launch the server process and perform the MCP initialize handshake."""
        if not self.config.command:
            logger.error(f"Server '{self.config.name}': no command specified for stdio transport")
            return False

        try:
            env = {**os.environ, **self.config.env}
            self.process = subprocess.Popen(
                [self.config.command, *self.config.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                bufsize=0,
            )

            # Perform MCP initialize handshake
            result = self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "OpenWill", "version": "1.0"},
            })
            if result is None:
                logger.error(f"Server '{self.config.name}': initialize handshake failed")
                self.stop()
                return False

            # Send initialized notification (no id, no response expected)
            self._send_notification("notifications/initialized", {})

            self._initialized = True
            logger.info(f"Server '{self.config.name}': stdio transport started")
            return True
        except Exception as e:
            logger.error(f"Server '{self.config.name}': failed to start stdio transport: {e}")
            self.stop()
            return False

    def stop(self):
        """Terminate the server process."""
        if self.process:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
            except Exception as e:
                logger.warning(f"Server '{self.config.name}': error stopping process: {e}")
            finally:
                self.process = None
                self._initialized = False

    def is_healthy(self) -> bool:
        """Check if the server process is still running."""
        if not self.process:
            return False
        return self.process.poll() is None and self._initialized

    def list_tools(self) -> list[dict]:
        """Request the list of tools from the server."""
        result = self._send_request("tools/list", {})
        if result is None:
            return []
        return result.get("tools", [])

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call a tool on the server."""
        result = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        return result or {}

    def _send_request(self, method: str, params: dict) -> Optional[dict]:
        """Send a JSON-RPC 2.0 request and read the response."""
        with self._lock:
            if not self.process or self.process.poll() is not None:
                logger.error(f"Server '{self.config.name}': process not running")
                return None

            self._request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            }
            return self._write_and_read(request)

    def _send_notification(self, method: str, params: dict):
        """Send a JSON-RPC 2.0 notification (no id, no response expected)."""
        with self._lock:
            if not self.process or self.process.poll() is not None:
                return
            notification = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }
            try:
                payload = json.dumps(notification) + "\n"
                self.process.stdin.write(payload.encode("utf-8"))
                self.process.stdin.flush()
            except Exception as e:
                logger.warning(f"Server '{self.config.name}': failed to send notification: {e}")

    def _write_and_read(self, request: dict) -> Optional[dict]:
        """Write a JSON-RPC request to stdin and read the response from stdout."""
        try:
            payload = json.dumps(request) + "\n"
            self.process.stdin.write(payload.encode("utf-8"))
            self.process.stdin.flush()

            # Read response line from stdout
            response_line = self.process.stdout.readline()
            if not response_line:
                logger.error(f"Server '{self.config.name}': empty response from server")
                return None

            response = json.loads(response_line.decode("utf-8"))

            if "error" in response:
                error = response["error"]
                logger.error(
                    f"Server '{self.config.name}': JSON-RPC error "
                    f"{error.get('code')}: {error.get('message')}"
                )
                return None

            return response.get("result")
        except json.JSONDecodeError as e:
            logger.error(f"Server '{self.config.name}': invalid JSON response: {e}")
            return None
        except Exception as e:
            logger.error(f"Server '{self.config.name}': communication error: {e}")
            return None


class _HttpTransport:
    """Transport for remote MCP servers using HTTP (JSON-RPC 2.0 over HTTP POST)."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._request_id = 0
        self._lock = threading.Lock()
        self._initialized = False
        self._base_url = config.url or ""

    def start(self) -> bool:
        """Perform the MCP initialize handshake over HTTP."""
        try:
            result = self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "OpenWill", "version": "1.0"},
            })
            if result is None:
                logger.error(f"Server '{self.config.name}': HTTP initialize handshake failed")
                return False

            # Send initialized notification (fire-and-forget)
            self._send_notification("notifications/initialized", {})

            self._initialized = True
            logger.info(f"Server '{self.config.name}': HTTP transport started")
            return True
        except Exception as e:
            logger.error(f"Server '{self.config.name}': failed to start HTTP transport: {e}")
            return False

    def stop(self):
        """No persistent connection to close for HTTP transport."""
        self._initialized = False

    def is_healthy(self) -> bool:
        """Check server health by sending a simple ping request."""
        if not self._initialized:
            return False
        try:
            result = self._send_request("ping", {})
            return result is not None
        except Exception:
            return False

    def list_tools(self) -> list[dict]:
        """Request the list of tools from the server."""
        result = self._send_request("tools/list", {})
        if result is None:
            return []
        return result.get("tools", [])

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call a tool on the server."""
        result = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        return result or {}

    def _send_request(self, method: str, params: dict) -> Optional[dict]:
        """Send a JSON-RPC 2.0 request via HTTP POST."""
        with self._lock:
            self._request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            }

        try:
            data = json.dumps(request).encode("utf-8")
            req = urllib.request.Request(
                self._base_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
                response = json.loads(body)

            if "error" in response:
                error = response["error"]
                logger.error(
                    f"Server '{self.config.name}': HTTP JSON-RPC error "
                    f"{error.get('code')}: {error.get('message')}"
                )
                return None

            return response.get("result")
        except urllib.error.URLError as e:
            logger.error(f"Server '{self.config.name}': HTTP request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Server '{self.config.name}': invalid JSON in HTTP response: {e}")
            return None
        except Exception as e:
            logger.error(f"Server '{self.config.name}': HTTP transport error: {e}")
            return None

    def _send_notification(self, method: str, params: dict):
        """Send a JSON-RPC 2.0 notification via HTTP POST (fire-and-forget)."""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        try:
            data = json.dumps(notification).encode("utf-8")
            req = urllib.request.Request(
                self._base_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.warning(f"Server '{self.config.name}': failed to send HTTP notification: {e}")


class MCPClient:
    """Manages connections to MCP servers and provides tool discovery/invocation."""

    def __init__(self, config_path: str = _MCP_CONFIG_PATH):
        self._servers: dict[str, MCPServerConfig] = {}
        self._transports: dict[str, _StdioTransport | _HttpTransport] = {}
        self._discovered_tools: dict[str, list[dict]] = {}  # server_name -> tools
        self._config_path = config_path

        # Load persisted configs
        self._load_configs()

        # Load from MCP_SERVERS env var
        self._load_from_env()

    # ── Server management ──────────────────────────────────────────────

    def add_server(self, config: MCPServerConfig) -> bool:
        """Add and connect to a new MCP server."""
        if config.name in self._servers:
            logger.warning(f"Server '{config.name}' already exists, replacing")

        # Remove existing transport if any
        if config.name in self._transports:
            self._transports[config.name].stop()
            del self._transports[config.name]

        self._servers[config.name] = config
        self._discovered_tools.pop(config.name, None)

        if not config.enabled:
            logger.info(f"Server '{config.name}' added but disabled")
            self._save_configs()
            return True

        transport = self._create_transport(config)
        self._transports[config.name] = transport

        success = transport.start()
        if success:
            # Discover tools immediately after connecting
            try:
                tools = transport.list_tools()
                self._discovered_tools[config.name] = tools
                logger.info(
                    f"Server '{config.name}': discovered {len(tools)} tools"
                )
            except Exception as e:
                logger.error(f"Server '{config.name}': tool discovery failed: {e}")
        else:
            logger.error(f"Server '{config.name}': failed to connect")

        self._save_configs()
        return success

    def remove_server(self, name: str):
        """Disconnect and remove a server."""
        if name in self._transports:
            self._transports[name].stop()
            del self._transports[name]
        self._servers.pop(name, None)
        self._discovered_tools.pop(name, None)
        self._save_configs()
        logger.info(f"Server '{name}' removed")

    def list_servers(self) -> list[dict]:
        """List all configured servers with their connection status."""
        result = []
        for name, config in self._servers.items():
            transport = self._transports.get(name)
            healthy = transport.is_healthy() if transport else False
            tool_count = len(self._discovered_tools.get(name, []))
            result.append({
                "name": name,
                "enabled": config.enabled,
                "healthy": healthy,
                "transport": "stdio" if config.command else ("http" if config.url else "unknown"),
                "tool_count": tool_count,
                "command": config.command,
                "url": config.url,
            })
        return result

    # ── Tool discovery and invocation ──────────────────────────────────

    def discover_tools(self) -> list[dict]:
        """Discover all tools from all connected servers. Returns a flat list with server info."""
        all_tools = []
        for name, config in self._servers.items():
            if not config.enabled:
                continue

            transport = self._transports.get(name)
            if not transport or not transport.is_healthy():
                # Try reconnecting
                if transport:
                    logger.warning(f"Server '{name}': unhealthy, attempting reconnect")
                    transport.stop()
                transport = self._create_transport(config)
                self._transports[name] = transport
                if not transport.start():
                    logger.error(f"Server '{name}': reconnect failed")
                    continue

            try:
                tools = transport.list_tools()
                self._discovered_tools[name] = tools
            except Exception as e:
                logger.error(f"Server '{name}': tool discovery failed: {e}")
                self._discovered_tools[name] = []
                continue

            for tool in tools:
                tool_entry = {
                    "server_name": name,
                    "tool_name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "inputSchema": tool.get("inputSchema", {}),
                }
                all_tools.append(tool_entry)

        return all_tools

    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        """Call a tool on a specific server."""
        transport = self._transports.get(server_name)
        if not transport:
            return {"error": f"Server '{server_name}' not found or not connected"}

        if not transport.is_healthy():
            return {"error": f"Server '{server_name}' is not healthy"}

        try:
            result = transport.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            logger.error(f"Server '{server_name}': tool call '{tool_name}' failed: {e}")
            return {"error": str(e)}

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self):
        """Connect to all configured and enabled servers."""
        for name, config in self._servers.items():
            if not config.enabled:
                logger.info(f"Server '{name}': skipped (disabled)")
                continue

            if name in self._transports:
                transport = self._transports[name]
                if transport.is_healthy():
                    continue
                transport.stop()

            transport = self._create_transport(config)
            self._transports[name] = transport

            success = transport.start()
            if success:
                try:
                    tools = transport.list_tools()
                    self._discovered_tools[name] = tools
                    logger.info(f"Server '{name}': connected, {len(tools)} tools discovered")
                except Exception as e:
                    logger.error(f"Server '{name}': tool discovery failed: {e}")
            else:
                logger.error(f"Server '{name}': failed to connect")

    def stop(self):
        """Disconnect from all servers."""
        for name, transport in self._transports.items():
            try:
                transport.stop()
                logger.info(f"Server '{name}': disconnected")
            except Exception as e:
                logger.warning(f"Server '{name}': error during disconnect: {e}")
        self._transports.clear()
        self._discovered_tools.clear()

    # ── Tool registry bridge ───────────────────────────────────────────

    def register_to_tool_registry(self, registry):
        """Register all discovered MCP tools into an existing ToolRegistry.

        Each MCP tool becomes a callable that routes through call_tool().
        Tool names are prefixed with 'mcp_{server_name}_' to avoid collisions.
        """
        from openwill.tools.registry import ToolResult

        for server_name, tools in self._discovered_tools.items():
            for tool_info in tools:
                raw_name = tool_info.get("name", "")
                if not raw_name:
                    continue

                registered_name = f"mcp_{server_name}_{raw_name}"
                description = tool_info.get("description", f"MCP tool: {raw_name} (from {server_name})")
                input_schema = tool_info.get("inputSchema", {})

                # Build parameter description from inputSchema
                parameters = {}
                props = input_schema.get("properties", {})
                required = input_schema.get("required", [])
                for prop_name, prop_schema in props.items():
                    prop_type = prop_schema.get("type", "any")
                    prop_desc = prop_schema.get("description", "")
                    req_marker = " (required)" if prop_name in required else ""
                    parameters[prop_name] = f"{prop_type}: {prop_desc}{req_marker}"

                # Capture variables for closure
                _server = server_name
                _tool = raw_name
                _client = self

                def _make_tool_fn(srv, tl, client):
                    """Factory to create a tool function with proper closure binding."""
                    def tool_fn(**kwargs) -> ToolResult:
                        result = client.call_tool(srv, tl, kwargs)
                        if "error" in result:
                            return ToolResult(
                                success=False,
                                output="",
                                error=result["error"],
                            )
                        # MCP tools/call returns content array
                        content = result.get("content", [])
                        text_parts = []
                        data = {}
                        for item in content:
                            if item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            else:
                                text_parts.append(json.dumps(item))
                        if not content and result:
                            # Fallback: return raw result
                            text_parts.append(json.dumps(result))
                            data = result
                        output = "\n".join(text_parts)
                        return ToolResult(success=True, output=output, data=data)
                    return tool_fn

                tool_func = _make_tool_fn(_server, _tool, _client)

                registry.register(
                    name=registered_name,
                    description=description,
                    func=tool_func,
                    parameters=parameters,
                    category="mcp",
                    dangerous=False,
                )
                logger.debug(f"Registered MCP tool: {registered_name}")

    # ── Persistence ────────────────────────────────────────────────────

    def _save_configs(self):
        """Save server configurations to disk."""
        try:
            config_dir = os.path.dirname(self._config_path)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)

            data = [cfg.to_dict() for cfg in self._servers.values()]
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save MCP server configs: {e}")

    def _load_configs(self):
        """Load server configurations from disk."""
        try:
            if not os.path.exists(self._config_path):
                return

            with open(self._config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                logger.warning("Invalid MCP server config file: expected a JSON array")
                return

            for item in data:
                config = MCPServerConfig.from_dict(item)
                if config.name:
                    self._servers[config.name] = config

            logger.info(f"Loaded {len(self._servers)} MCP server config(s) from disk")
        except Exception as e:
            logger.error(f"Failed to load MCP server configs: {e}")

    def _load_from_env(self):
        """Read MCP_SERVERS env var as JSON array of server configs."""
        raw = os.getenv("MCP_SERVERS", "").strip()
        if not raw:
            return

        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                logger.warning("MCP_SERVERS env var: expected a JSON array")
                return

            for item in data:
                config = MCPServerConfig.from_dict(item)
                if config.name and config.name not in self._servers:
                    self._servers[config.name] = config
                    logger.info(f"Loaded MCP server '{config.name}' from env")

            self._save_configs()
        except json.JSONDecodeError as e:
            logger.error(f"MCP_SERVERS env var: invalid JSON: {e}")
        except Exception as e:
            logger.error(f"MCP_SERVERS env var: error loading: {e}")

    # ── Internal helpers ───────────────────────────────────────────────

    def _create_transport(self, config: MCPServerConfig) -> _StdioTransport | _HttpTransport:
        """Create the appropriate transport for a server config."""
        if config.command:
            return _StdioTransport(config)
        elif config.url:
            return _HttpTransport(config)
        else:
            logger.error(
                f"Server '{config.name}': must specify either 'command' (stdio) or 'url' (HTTP)"
            )
            # Default to stdio with a dummy command so is_healthy() returns False
            return _StdioTransport(config)
