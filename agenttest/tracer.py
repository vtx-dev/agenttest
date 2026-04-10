"""
Tracer — wraps tool functions to record calls into a Trace.

Usage:
    tracer = Tracer()

    @tracer.tool
    def send_email(to, subject, body):
        ...

    with tracer.session("welcome_email_test") as trace:
        run_my_agent(...)

    trace.save("tests/traces/welcome_email_test.trace.json")
"""
from __future__ import annotations

import functools
import json
import time
import urllib.request
import urllib.error
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

from .trace import Trace, ToolCall


def _safe_serialize(obj: Any) -> Any:
    """Best-effort JSON-serializable representation of a value."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(i) for i in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    return str(obj)


class Tracer:
    def __init__(
        self,
        traces_dir: str | Path = "tests/traces",
        cloud_url: str | None = None,
        cloud_api_key: str | None = None,
        suite: str = "default",
    ):
        self.traces_dir = Path(traces_dir)
        self.cloud_url = (cloud_url or "").rstrip("/") or None
        self.cloud_api_key = cloud_api_key or None
        self.suite = suite
        self._current: Optional[Trace] = None
        self._wrapped: list[tuple[object, str, Callable]] = []  # (obj, attr, original)

    # ── session context manager ────────────────────────────────────

    @contextmanager
    def session(self, name: str, metadata: dict | None = None):
        """
        Context manager that records all tool calls into a named Trace.

            with tracer.session("my_test") as trace:
                result = run_my_agent(...)
            trace.save(...)
        """
        trace = Trace(name=name, metadata=metadata or {})
        self._current = trace
        t0 = time.perf_counter()
        try:
            yield trace
        except Exception as e:
            trace.error = str(e)
            raise
        finally:
            trace.duration_ms = (time.perf_counter() - t0) * 1000
            self._current = None
            if self.cloud_url and self.cloud_api_key:
                self._cloud_push(trace)

    # ── tool decorator ─────────────────────────────────────────────

    def tool(self, fn: Callable) -> Callable:
        """
        Decorator that records calls to this function.

            @tracer.tool
            def search_web(query: str) -> str:
                ...
        """
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if self._current is None:
                return fn(*args, **kwargs)

            # Build args dict
            import inspect
            sig = inspect.signature(fn)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            args_dict = dict(bound.arguments)

            t0 = time.perf_counter()
            error = None
            result = None
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration = (time.perf_counter() - t0) * 1000
                call = ToolCall(
                    tool=fn.__name__,
                    args=args_dict,
                    result=result,
                    error=error,
                    duration_ms=round(duration, 2),
                )
                self._current.tool_calls.append(call)

        wrapper._agenttest_original = fn
        return wrapper

    # ── message capture ────────────────────────────────────────────

    def add_message(self, role: str, content: str, **extra):
        """Manually add a message to the current trace."""
        if self._current is not None:
            msg = {"role": role, "content": content, **extra}
            self._current.messages.append(msg)

    def capture_messages(self, messages: list[dict]):
        """Bulk-add messages (e.g. full conversation history)."""
        if self._current is not None:
            self._current.messages.extend(messages)

    def set_output(self, output: str):
        """Record the final agent output."""
        if self._current is not None:
            self._current.output = output

    # ── auto-save helper ───────────────────────────────────────────

    @contextmanager
    def record(self, name: str, metadata: dict | None = None):
        """
        Like session() but auto-saves the trace to traces_dir.

            with tracer.record("delete_file_test") as trace:
                run_agent(...)
            # trace saved to tests/traces/delete_file_test.trace.json
        """
        with self.session(name, metadata) as trace:
            yield trace
        path = self.traces_dir / f"{name}.trace.json"
        trace.save(path)
        print(f"[agenttest] trace saved → {path}")

    # ── cloud push ─────────────────────────────────────────────────

    def _cloud_push(self, trace: Trace) -> None:
        """Upload a completed trace to agenttest cloud. Non-fatal on failure."""
        from datetime import datetime, timezone
        try:
            started_at = datetime.fromisoformat(trace.created_at).timestamp()
        except Exception:
            started_at = time.time()

        payload = {
            "suite":      self.suite,
            "session":    trace.name,
            "started_at": started_at,
            "duration_s": round(trace.duration_ms / 1000, 3) if trace.duration_ms else None,
            "tool_calls": [
                {
                    "tool":     c.tool,
                    "args":     c.args,
                    "result":   _safe_serialize(c.result),
                    "error":    c.error,
                    "duration": round(c.duration_ms / 1000, 4) if c.duration_ms else None,
                }
                for c in trace.tool_calls
            ],
            "metadata": trace.metadata,
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.cloud_url}/v1/traces",
            data=body,
            headers={
                "Authorization": f"Bearer {self.cloud_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                resp = json.loads(r.read())
            print(f"[agenttest] cloud ↑ {trace.name} → {resp.get('run_url', resp.get('id', '?'))}")
        except Exception as e:
            print(f"[agenttest] cloud push failed for '{trace.name}': {e}")

    # ── MCP/OpenAI adapter helpers ─────────────────────────────────

    def wrap_openai_tools(self, tools: list[dict], executors: dict[str, Callable]) -> dict[str, Callable]:
        """
        Wrap a dict of {tool_name: callable} so calls are recorded.
        Returns the wrapped dict — use it in place of the originals.
        """
        return {name: self.tool(fn) for name, fn in executors.items()}

    def wrap_mcp_handler(self, fn: Callable) -> Callable:
        """
        Wrap an MCP tool handler (receives tool_name, args) so calls are recorded.
        """
        @functools.wraps(fn)
        def wrapper(tool_name: str, args: dict) -> Any:
            if self._current is None:
                return fn(tool_name, args)
            t0 = time.perf_counter()
            error = None
            result = None
            try:
                result = fn(tool_name, args)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                call = ToolCall(
                    tool=tool_name,
                    args=args,
                    result=result,
                    error=error,
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                )
                self._current.tool_calls.append(call)
        return wrapper


# Module-level default tracer for convenience
default_tracer = Tracer()
tool = default_tracer.tool
session = default_tracer.session
record = default_tracer.record
