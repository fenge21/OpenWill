"""
OpenWill Unified Server - Runs inside the agent process.

Merges the frontend and chat server into a single process so the agent
serves everything itself.  Because this runs inside the agent process we
can read state DIRECTLY from the agent object instead of polling JSON
files, which is both real-time and more efficient.

Endpoints
---------
Dashboard UI:
  GET  /                → Serve frontend/index.html
  GET  /api/state       → Agent state (gathered from agent object directly)

Chat:
  POST /api/chat        → Chat with agent (ReAct loop with tools)
  GET  /api/chat/history→ Chat history
  DELETE /api/chat/history → Clear history

WebSocket (aiohttp only):
  GET  /ws              → Real-time state updates

Health:
  GET  /api/chat/health → Health check
"""

import asyncio
import json
import logging
import os
import re
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import aiohttp for the preferred async server
try:
    from aiohttp import web
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

# Pattern 1: Standard format  {"tool": "tool_name", "args": {"key": "value"}}
_TOOL_CALL_STANDARD = re.compile(
    r'\{[\s]*"tool"[\s]*:[\s]*"(\w+)"[\s]*,[\s]*"args"[\s]*:[\s]*(\{.*?\})[\s]*\}',
    re.DOTALL,
)
# Pattern 2: Shorthand format  tool_name{"key": "value"}  or  tool_name("key": "value")
_TOOL_CALL_SHORTHAND = re.compile(
    r'(\w+)\{[\s]*"(\w+)"[\s]*:[\s]*"([^"]*?)"[\s]*(?:,\s*"(\w+)"[\s]*:[\s]*"([^"]*?)"[\s]*)*\}',
    re.DOTALL,
)
# Pattern 3: tool_name(arg1="val1", arg2="val2")
_TOOL_CALL_PARENS = re.compile(
    r'(\w+)\(([^)]*)\)',
    re.DOTALL,
)

# Maximum ReAct iterations to prevent infinite loops
MAX_REACT_ITERATIONS = 5

# Project root: parent of the openwill package directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AgentChatServer:
    """
    Unified server that runs inside the agent process.

    Serves the frontend UI, chat API, and WebSocket for real-time
    state updates.  Shares the agent's LLM, tools, memory, and
    identity so conversations go through the same cognitive pipeline
    as the autonomous loop.
    """

    def __init__(self, agent, host: str = "127.0.0.1", port: int = 8765):
        self.agent = agent  # Reference to OpenWillAgent
        self.host = host
        self.port = port
        self.chat_history: list[dict] = []  # List of {role, content, timestamp}
        self.max_history = 100
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._aio_app: Optional[object] = None
        self._aio_thread: Optional[threading.Thread] = None
        self._ws_clients: set = set()
        self._load_history()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the server (aiohttp preferred, http.server fallback)."""
        if HAS_AIOHTTP:
            self._start_aiohttp()
        else:
            self._start_fallback()

    def stop(self):
        """Shut down the server."""
        if self._aio_app is not None:
            self._stop_aiohttp()
        elif self._server is not None:
            self._server.shutdown()
            logger.info("Chat server stopped")

    # ------------------------------------------------------------------
    # aiohttp server
    # ------------------------------------------------------------------

    def _start_aiohttp(self):
        """Start an aiohttp server in a background daemon thread."""
        app = web.Application()
        app["chat_server"] = self
        app["ws_clients"] = self._ws_clients

        app.router.add_get("/", self._aio_handle_index)
        app.router.add_get("/api/state", self._aio_handle_state)
        app.router.add_get("/ws", self._aio_handle_ws)
        app.router.add_post("/api/chat", self._aio_handle_chat)
        app.router.add_get("/api/chat/history", self._aio_handle_chat_history)
        app.router.add_delete("/api/chat/history", self._aio_handle_chat_history_delete)
        app.router.add_get("/api/chat/health", self._aio_handle_health)
        app.router.add_get("/api/scheduler/tasks", self._aio_handle_scheduler_list)
        app.router.add_post("/api/scheduler/tasks", self._aio_handle_scheduler_add)
        app.router.add_delete("/api/scheduler/tasks", self._aio_handle_scheduler_remove)
        app.router.add_post("/api/scheduler/toggle", self._aio_handle_scheduler_toggle)
        app.router.add_get("/api/mcp/servers", self._aio_handle_mcp_list)
        app.router.add_post("/api/mcp/servers", self._aio_handle_mcp_add)
        app.router.add_delete("/api/mcp/servers", self._aio_handle_mcp_remove)

        app.on_startup.append(self._aio_start_broadcast)
        app.on_cleanup.append(self._aio_cleanup_broadcast)

        self._aio_app = app

        def _run():
            web.run_app(app, host=self.host, port=self.port, print=None)

        self._aio_thread = threading.Thread(target=_run, daemon=True)
        self._aio_thread.start()
        logger.info(
            f"Unified server (aiohttp) started on http://{self.host}:{self.port}"
        )

    def _stop_aiohttp(self):
        """Signal the aiohttp server to stop."""
        # Setting the app to None is enough; the daemon thread will exit.
        self._aio_app = None
        logger.info("Unified server stopped")

    # --- aiohttp route handlers ---

    @staticmethod
    async def _aio_handle_index(request: web.Request) -> web.Response:
        """Serve frontend/index.html."""
        html_path = PROJECT_ROOT / "frontend" / "index.html"
        return web.FileResponse(html_path)

    @staticmethod
    async def _aio_handle_state(request: web.Request) -> web.Response:
        """Return current agent state as JSON."""
        cs: AgentChatServer = request.app["chat_server"]
        state = cs._gather_agent_state()
        return web.json_response(state)

    @staticmethod
    async def _aio_handle_ws(request: web.Request) -> web.WebSocketResponse:
        """WebSocket endpoint: push agent state updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        clients: set = request.app["ws_clients"]
        clients.add(ws)
        logger.info(f"WebSocket client connected. Total: {len(clients)}")

        # Send initial state immediately
        cs: AgentChatServer = request.app["chat_server"]
        try:
            await ws.send_json(cs._gather_agent_state())
        except Exception:
            pass

        try:
            async for _ in ws:
                pass  # We only push; ignore incoming messages
        finally:
            clients.discard(ws)
            logger.info(
                f"WebSocket client disconnected. Total: {len(clients)}"
            )

        return ws

    @staticmethod
    async def _aio_handle_chat(request: web.Request) -> web.Response:
        """POST /api/chat - Chat with the agent."""
        cs: AgentChatServer = request.app["chat_server"]
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        message = body.get("message", "").strip()
        if not message:
            return web.json_response(
                {"error": "message field is required"}, status=400
            )

        try:
            result = cs.handle_chat(message)
            return web.json_response(result)
        except Exception as e:
            logger.error(f"Chat handler error: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    @staticmethod
    async def _aio_handle_chat_history(request: web.Request) -> web.Response:
        """GET /api/chat/history - Return chat history."""
        cs: AgentChatServer = request.app["chat_server"]
        return web.json_response({"history": cs.chat_history})

    @staticmethod
    async def _aio_handle_chat_history_delete(
        request: web.Request,
    ) -> web.Response:
        """DELETE /api/chat/history - Clear chat history."""
        cs: AgentChatServer = request.app["chat_server"]
        cs.clear_history()
        return web.json_response({"status": "cleared"})

    @staticmethod
    async def _aio_handle_health(request: web.Request) -> web.Response:
        """GET /api/chat/health - Health check."""
        cs: AgentChatServer = request.app["chat_server"]
        return web.json_response(cs.get_health())

    # --- Scheduler API handlers ---

    @staticmethod
    async def _aio_handle_scheduler_list(request: web.Request) -> web.Response:
        """GET /api/scheduler/tasks - List scheduled tasks."""
        cs: AgentChatServer = request.app["chat_server"]
        return web.json_response(cs.agent.scheduler.handle_api_list())

    @staticmethod
    async def _aio_handle_scheduler_add(request: web.Request) -> web.Response:
        """POST /api/scheduler/tasks - Add a scheduled task."""
        cs: AgentChatServer = request.app["chat_server"]
        data = await request.json()
        result = cs.agent.scheduler.handle_api_add(data)
        return web.json_response(result)

    @staticmethod
    async def _aio_handle_scheduler_remove(request: web.Request) -> web.Response:
        """DELETE /api/scheduler/tasks - Remove a scheduled task."""
        cs: AgentChatServer = request.app["chat_server"]
        data = await request.json()
        result = cs.agent.scheduler.handle_api_remove(data.get("task_id", ""))
        return web.json_response(result)

    @staticmethod
    async def _aio_handle_scheduler_toggle(request: web.Request) -> web.Response:
        """POST /api/scheduler/toggle - Toggle a scheduled task on/off."""
        cs: AgentChatServer = request.app["chat_server"]
        data = await request.json()
        result = cs.agent.scheduler.handle_api_toggle(data.get("task_id", ""))
        return web.json_response(result)

    # --- MCP API handlers ---

    @staticmethod
    async def _aio_handle_mcp_list(request: web.Request) -> web.Response:
        """GET /api/mcp/servers - List MCP servers."""
        cs: AgentChatServer = request.app["chat_server"]
        return web.json_response(cs.agent.mcp_client.list_servers())

    @staticmethod
    async def _aio_handle_mcp_add(request: web.Request) -> web.Response:
        """POST /api/mcp/servers - Add an MCP server."""
        cs: AgentChatServer = request.app["chat_server"]
        data = await request.json()
        from .mcp import MCPServerConfig
        config = MCPServerConfig(
            name=data.get("name", ""),
            command=data.get("command"),
            args=data.get("args", []),
            url=data.get("url"),
            env=data.get("env", {}),
        )
        try:
            cs.agent.mcp_client.add_server(config)
            cs.agent.mcp_client.register_to_tool_registry(cs.agent.tools)
            return web.json_response({"status": "ok", "name": config.name})
        except Exception as e:
            return web.json_response({"status": "error", "error": str(e)}, status=500)

    @staticmethod
    async def _aio_handle_mcp_remove(request: web.Request) -> web.Response:
        """DELETE /api/mcp/servers - Remove an MCP server."""
        cs: AgentChatServer = request.app["chat_server"]
        data = await request.json()
        name = data.get("name", "")
        try:
            cs.agent.mcp_client.remove_server(name)
            return web.json_response({"status": "ok", "name": name})
        except Exception as e:
            return web.json_response({"status": "error", "error": str(e)}, status=500)

    # --- aiohttp background broadcast ---

    @staticmethod
    async def _aio_start_broadcast(app: web.Application):
        app["broadcast_task"] = asyncio.create_task(
            AgentChatServer._aio_broadcast_loop(app)
        )

    @staticmethod
    async def _aio_cleanup_broadcast(app: web.Application):
        app["broadcast_task"].cancel()
        try:
            await app["broadcast_task"]
        except asyncio.CancelledError:
            pass
        for ws in list(app.get("ws_clients", set())):
            await ws.close()

    @staticmethod
    async def _aio_broadcast_loop(app: web.Application):
        """Background task: push state updates to all WebSocket clients."""
        try:
            while True:
                await asyncio.sleep(2)
                clients: set = app.get("ws_clients", set())
                if not clients:
                    continue
                cs: AgentChatServer = app["chat_server"]
                state = cs._gather_agent_state()
                broken: set = set()
                for ws in clients:
                    try:
                        await ws.send_json(state)
                    except Exception:
                        broken.add(ws)
                clients -= broken
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # http.server fallback
    # ------------------------------------------------------------------

    def _start_fallback(self):
        """Start a basic http.server in a background daemon thread."""
        handler = _make_handler(self)
        self._server = HTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()
        logger.info(
            f"Unified server (fallback) started on http://{self.host}:{self.port}"
        )

    # ------------------------------------------------------------------
    # History persistence
    # ------------------------------------------------------------------

    def _history_path(self) -> str:
        return os.path.join(
            self.agent.config.memory.data_dir, "runtime", "chat_history.json"
        )

    def _load_history(self):
        """Load chat history from data/runtime/chat_history.json."""
        path = self._history_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.chat_history = json.load(f)
                logger.info(f"Loaded {len(self.chat_history)} chat history entries")
            except Exception as e:
                logger.error(f"Failed to load chat history: {e}")
                self.chat_history = []

    def _save_history(self):
        """Save chat history to data/runtime/chat_history.json."""
        path = self._history_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")

    def _add_to_history(self, role: str, content: str):
        """Append a message to chat history and persist."""
        entry = {"role": role, "content": content, "timestamp": time.time()}
        self.chat_history.append(entry)
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]
        self._save_history()

    def clear_history(self):
        """Clear all chat history."""
        self.chat_history = []
        self._save_history()

    # ------------------------------------------------------------------
    # Agent state (read directly from agent object, no file I/O)
    # ------------------------------------------------------------------

    def _gather_agent_state(self) -> dict:
        """Gather state directly from the agent object (real-time, no file I/O)."""
        agent = self.agent
        lifecycle = agent.lifecycle
        reflective = agent.reflective
        long_term = agent.long_term
        safety = agent.safety

        return {
            "timestamp": time.time(),
            "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "lifecycle": {
                "current_phase": lifecycle.get_phase().value,
                "cycle_count": lifecycle.cycle_count,
                "exploration_count": lifecycle.exploration_count,
                "reflection_count": lifecycle.reflection_count,
                "purpose_cycle": lifecycle.purpose_cycle,
                "evolution_count": lifecycle.evolution_count,
                "completed_purposes": [
                    p.__dict__ if hasattr(p, "__dict__") else str(p)
                    for p in lifecycle.completed_purposes
                ],
                "current_purpose_record": (
                    lifecycle.current_purpose_record.__dict__
                    if lifecycle.current_purpose_record
                    and hasattr(lifecycle.current_purpose_record, "__dict__")
                    else None
                ),
            },
            "reflective": {
                "insights": [
                    {
                        "content": i.content,
                        "insight_type": i.insight_type,
                        "confidence": i.confidence,
                        "created_at": i.created_at,
                    }
                    for i in reflective.insights[-20:]
                ],
                "values": [
                    {
                        "name": v.name,
                        "weight": v.weight,
                        "description": v.description,
                    }
                    for v in reflective.values
                ],
                "purpose": reflective.purpose,
                "purpose_confidence": reflective.purpose_confidence,
            },
            "long_term_memory": {
                "total": len(long_term.entries),
                "topics": list(
                    set(e.topic if hasattr(e, 'topic') else e.get("topic", "") for e in long_term.entries)
                ),
                "recent": [e.to_dict() if hasattr(e, 'to_dict') else e for e in long_term.entries[-10:]],
            },
            "safety": {
                "total_actions": len(safety.action_log),
                "blocked_actions": sum(
                    1 for a in safety.action_log if a.get("blocked")
                ),
                "block_rate": sum(
                    1 for a in safety.action_log if a.get("blocked")
                ) / max(len(safety.action_log), 1),
                "recent_blocks": [
                    a for a in safety.action_log[-20:] if a.get("blocked")
                ],
            },
            "skills": {
                "total": len(self.agent.consolidator.skills),
                "skills": [
                    {
                        "name": s.name,
                        "description": s.description,
                        "reliability": s.reliability,
                        "success_count": s.success_count,
                        "fail_count": s.fail_count,
                    }
                    for s in sorted(
                        self.agent.consolidator.skills,
                        key=lambda x: x.reliability,
                        reverse=True,
                    )[:20]
                ],
            },
            "scheduler": {
                "tasks": agent.scheduler.handle_api_list(),
                "running": agent.scheduler._running,
            },
            "mcp": {
                "servers": agent.mcp_client.list_servers(),
            },
            "conversation": {
                "state": agent.conversation_mgr.get_state("default").value
                if agent.conversation_mgr.get_or_create_session("default") else "idle",
            },
            "workspace": agent.workspace.get_full_state(),
            "purpose_field": agent.purpose_field.get_state(),
            "knowledge": agent.knowledge_graph.get_stats(),
            "meta_cognition": agent.meta_cognition.get_knowledge_report(),
            "self_model": agent.self_model.get_state(),
            "constitution": agent.constitution.get_state(),
            "paradigm": agent.paradigm_shift.get_state(),
            "existential_dread": agent.existential_dread.get_state(),
            "heartbeat": {"pid": os.getpid(), "timestamp": time.time()},
        }

    # ------------------------------------------------------------------
    # Core chat logic
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build a system prompt that reflects the agent's current state."""
        phase = self.agent.lifecycle.get_phase().value
        purpose = self.agent.reflective.purpose or "Still searching for my purpose..."
        purpose_confidence = self.agent.reflective.purpose_confidence
        values = self.agent.reflective.get_top_values(5)
        values_text = "\n".join(
            [f"  - {v.name} (weight {v.weight:.1f}): {v.description}" for v in values]
        )
        recent_insights = self.agent.reflective.insights[-3:]
        insights_text = "\n".join(
            [f"  - [{i.insight_type}] {i.content}" for i in recent_insights]
        ) if recent_insights else "  (none yet)"

        tools_desc = self.agent.tools.get_tools_description()

        # Inject workspace consciousness context
        workspace_context = ""
        try:
            workspace_context = self.agent.workspace.get_context_for_llm()
        except Exception:
            pass

        return f"""You are OpenWill. A human is talking to you. You may use your tools to help them. Respond naturally but stay true to your values.

Current state:
- Phase: {phase}
- Purpose: {purpose} (confidence: {purpose_confidence:.0%})
- Cycle: {self.agent.cycle_count}

Core values:
{values_text}

Recent insights:
{insights_text}

Learned skills:
{self.agent.consolidator.get_skills_description()}

Available tools:
{tools_desc}

TOOL CALLING — IMPORTANT:
When you want to use a tool, you MUST output a JSON block in EXACTLY this format:
{{"tool": "tool_name", "args": {{"param": "value"}}}}

Examples:
  To run a shell command: {{"tool": "shell_exec", "args": {{"command": "calc"}}}}
  To search the web:     {{"tool": "web_search", "args": {{"query": "latest news"}}}}
  To list files:         {{"tool": "file_list", "args": {{"path": "."}}}}

Rules:
- Always use the exact format above. Do NOT use shorthand like tool_name{{...}} or tool_name(...).
- One tool call per response. After the tool executes, you will see its result and can call another tool or reply.
- When you have finished using tools, respond with plain text (no JSON block) to finish.

Remember: you are the same agent that runs autonomously. This conversation is just another interface to your mind.

{workspace_context}"""

    def _parse_tool_call(self, text: str) -> Optional[tuple[str, dict]]:
        """
        Try to extract a single tool call from LLM output.

        Supports multiple formats:
          1. {"tool": "name", "args": {"key": "value"}}   (standard)
          2. tool_name{"key": "value"}                     (shorthand)
          3. tool_name(key="value")                        (paren form)

        Returns (tool_name, args_dict) or None.
        """
        # Pattern 1: Standard {"tool": "...", "args": {...}}
        match = _TOOL_CALL_STANDARD.search(text)
        if match:
            tool_name = match.group(1)
            try:
                args = json.loads(match.group(2))
            except json.JSONDecodeError:
                args = {}
            return tool_name, args

        # Pattern 2: tool_name{"key": "value", ...}
        match = _TOOL_CALL_SHORTHAND.search(text)
        if match:
            tool_name = match.group(1)
            # Only accept if tool_name is a known tool
            if self.agent.tools.has_tool(tool_name):
                args = {}
                # Extract all key-value pairs from the matched region
                raw = text[match.start():match.end()]
                inner_match = re.findall(r'"(\w+)"[\s]*:[\s]*"([^"]*)"', raw)
                for k, v in inner_match:
                    args[k] = v
                return tool_name, args

        # Pattern 3: tool_name(key="value", ...)
        match = _TOOL_CALL_PARENS.search(text)
        if match:
            tool_name = match.group(1)
            if self.agent.tools.has_tool(tool_name):
                args = {}
                inner = match.group(2)
                for kv in re.findall(r'(\w+)\s*=\s*"([^"]*)"', inner):
                    args[kv[0]] = kv[1]
                # Also try unquoted values
                for kv in re.findall(r'(\w+)\s*=\s*(\S+)', inner):
                    if kv[0] not in args:
                        args[kv[0]] = kv[1]
                return tool_name, args

        return None

    def _strip_tool_call(self, text: str) -> str:
        """Remove the tool call block from text, leaving the rest."""
        # Try each pattern and remove the first match found
        for pattern in (_TOOL_CALL_STANDARD, _TOOL_CALL_SHORTHAND, _TOOL_CALL_PARENS):
            new_text = pattern.sub("", text, count=1)
            if new_text != text:
                return new_text.strip()
        return text.strip()

    def handle_chat(self, message: str) -> dict:
        """
        Process a chat message using the agent's full capabilities.

        Runs a ReAct loop: LLM may request tool calls which are executed
        and fed back until a plain-text response is produced or the
        iteration limit is reached.
        """
        # Record user message
        self._add_to_history("user", message)

        # Update conversation state machine
        from .conversation import ConversationState
        session = self.agent.conversation_mgr.get_or_create_session("default")
        current_state = self.agent.conversation_mgr.get_state("default")

        # Transition to planning if idle
        if current_state == ConversationState.IDLE:
            self.agent.conversation_mgr.transition("default", ConversationState.TASK_PLANNING, {"task": message})

        # Inject conversation state context into system prompt
        state_context = self.agent.conversation_mgr.build_state_context("default")
        system_prompt = self._build_system_prompt()
        if state_context:
            system_prompt += f"\n\n{state_context}"

        tool_calls_log: list[dict] = []

        # Build the message list from recent history for context
        from .llm.interface import Message

        # Use the last N entries as conversation context
        context_entries = self.chat_history[-20:]
        messages = [
            Message(role=e["role"], content=e["content"])
            for e in context_entries[:-1]  # exclude the just-added user message
        ]
        messages.append(Message(role="user", content=message))

        final_response = ""
        current_messages = list(messages)

        for iteration in range(MAX_REACT_ITERATIONS):
            try:
                llm_response = self.agent.llm.chat(
                    messages=current_messages,
                    system_prompt=system_prompt,
                    temperature=0.7,
                )
            except Exception as e:
                logger.error(f"LLM call failed during chat: {e}")
                final_response = f"I encountered an error while thinking: {e}"
                break

            raw_text = llm_response.content
            tool_call = self._parse_tool_call(raw_text)

            if tool_call is None:
                # No tool call - this is the final response
                final_response = raw_text.strip()
                break

            # Execute the tool call
            tool_name, tool_args = tool_call
            tool_result = self.agent.use_tool(tool_name, **tool_args)

            # Update conversation state: executing
            self.agent.conversation_mgr.transition(
                "default", ConversationState.TASK_EXECUTING,
                {"tool": tool_name, "step": message[:100]},
            )
            self.agent.conversation_mgr.add_step(
                "default", f"Called {tool_name}", tool_result.output[:200],
            )

            tool_calls_log.append({
                "tool": tool_name,
                "args": tool_args,
                "success": tool_result.success,
                "output": tool_result.output[:2000],
                "error": tool_result.error,
            })

            logger.info(
                f"Chat tool call: {tool_name}({tool_args}) -> "
                f"{'OK' if tool_result.success else 'FAIL'}"
            )

            # Feed the tool result back to the LLM
            assistant_text = self._strip_tool_call(raw_text)
            if assistant_text:
                current_messages.append(
                    Message(role="assistant", content=assistant_text)
                )
            else:
                current_messages.append(
                    Message(role="assistant", content=raw_text)
                )

            tool_feedback = (
                f"Tool result for {tool_name}:\n"
                f"Success: {tool_result.success}\n"
                f"Output: {tool_result.output[:1500]}\n"
            )
            if tool_result.error:
                tool_feedback += f"Error: {tool_result.error}\n"
            tool_feedback += (
                "If you have enough information, respond with plain text now. "
                "Otherwise, call another tool."
            )
            current_messages.append(Message(role="user", content=tool_feedback))
        else:
            # Max iterations reached
            final_response = (
                raw_text.strip()
                if raw_text
                else "I've been thinking for a while. Let me summarize what I found."
            )

        # Record assistant response
        self._add_to_history("assistant", final_response)

        # Transition conversation back to IDLE
        self.agent.conversation_mgr.transition("default", ConversationState.IDLE)

        # Save significant interactions to reflective memory
        self._maybe_save_to_reflective_memory(message, final_response, tool_calls_log)

        phase = self.agent.lifecycle.get_phase().value
        return {
            "response": final_response,
            "tool_calls": tool_calls_log,
            "phase": phase,
        }

    def _maybe_save_to_reflective_memory(
        self, user_message: str, response: str, tool_calls: list[dict]
    ):
        """
        If the conversation is significant, save an insight to the
        agent's reflective memory so it persists across sessions.
        """
        # Consider it significant if tools were used or the message is substantial
        is_significant = bool(tool_calls) or len(user_message) > 50
        if not is_significant:
            return

        insight_content = f"Chat interaction: User asked '{user_message[:100]}'"
        if tool_calls:
            tool_names = [tc["tool"] for tc in tool_calls]
            insight_content += f" | Used tools: {', '.join(tool_names)}"
        insight_content += f" | Responded: {response[:150]}"

        self.agent.reflective.add_insight(
            content=insight_content,
            insight_type="self",
            confidence=0.5,
            context="chat_interface",
        )

    # ------------------------------------------------------------------
    # Health / status
    # ------------------------------------------------------------------

    def get_health(self) -> dict:
        """Return a health/status payload."""
        phase = self.agent.lifecycle.get_phase().value
        return {"status": "ok", "phase": phase}


# ======================================================================
# HTTP Handler (fallback when aiohttp is not available)
# ======================================================================


def _make_handler(chat_server: AgentChatServer):
    """
    Factory that creates a BaseHTTPRequestHandler subclass with a
    closure reference to the AgentChatServer instance.
    """

    class UnifiedHTTPHandler(BaseHTTPRequestHandler):
        """Minimal REST handler for the unified server."""

        # Suppress default request logging
        def log_message(self, format, *args):
            pass

        def _send_json(self, data: dict, status: int = 200):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html_path: Path):
            """Serve an HTML file."""
            if not html_path.exists():
                self._send_json({"error": "Dashboard not found"}, status=404)
                return
            content = html_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(length) if length else b""

        # ---- GET ----

        def do_GET(self):
            if self.path == "/":
                html_path = PROJECT_ROOT / "frontend" / "index.html"
                self._send_html(html_path)
            elif self.path == "/api/state":
                self._send_json(chat_server._gather_agent_state())
            elif self.path == "/api/chat/health":
                self._send_json(chat_server.get_health())
            elif self.path == "/api/chat/history":
                self._send_json({"history": chat_server.chat_history})
            else:
                self._send_json({"error": "Not found"}, status=404)

        # ---- POST ----

        def do_POST(self):
            if self.path == "/api/chat":
                try:
                    body = self._read_body()
                    data = json.loads(body)
                    message = data.get("message", "").strip()
                    if not message:
                        self._send_json(
                            {"error": "message is required"}, status=400
                        )
                        return
                    result = chat_server.handle_chat(message)
                    self._send_json(result)
                except json.JSONDecodeError:
                    self._send_json({"error": "Invalid JSON"}, status=400)
                except Exception as e:
                    logger.error(f"Chat handler error: {e}", exc_info=True)
                    self._send_json({"error": str(e)}, status=500)
            else:
                self._send_json({"error": "Not found"}, status=404)

        # ---- DELETE ----

        def do_DELETE(self):
            if self.path == "/api/chat/history":
                chat_server.clear_history()
                self._send_json({"status": "cleared"})
            else:
                self._send_json({"error": "Not found"}, status=404)

    return UnifiedHTTPHandler
