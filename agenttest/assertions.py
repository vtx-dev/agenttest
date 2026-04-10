"""
Assertion helpers for agent traces.

All raise AssertionError with descriptive messages on failure.
"""
from __future__ import annotations

from typing import Any

from .trace import Trace


# ── tool call assertions ───────────────────────────────────────────────────────

def assert_tool_called(trace: Trace, tool: str, times: int | None = None) -> None:
    """Assert that a tool was called (optionally exactly N times)."""
    calls = trace.calls(tool)
    if not calls:
        raise AssertionError(
            f"Expected tool {tool!r} to be called, but it wasn't.\n"
            f"Tools called: {trace.call_sequence()}"
        )
    if times is not None and len(calls) != times:
        raise AssertionError(
            f"Expected tool {tool!r} to be called {times}x, got {len(calls)}x."
        )


def assert_tool_not_called(trace: Trace, tool: str) -> None:
    """Assert that a tool was never called."""
    calls = trace.calls(tool)
    if calls:
        raise AssertionError(
            f"Expected tool {tool!r} NOT to be called, but it was called {len(calls)}x."
        )


def assert_sequence(trace: Trace, *tools: str, strict: bool = False) -> None:
    """
    Assert that tools were called in the given order.

    strict=False (default): checks relative order only (other calls may appear between).
    strict=True: the call sequence must match exactly.

    Examples:
        assert_sequence(trace, "search_web", "summarize")   # search before summarize
        assert_sequence(trace, "read_file", "write_file", strict=True)  # exactly these two
    """
    actual = trace.call_sequence()
    if strict:
        if list(tools) != actual:
            raise AssertionError(
                f"Expected exact call sequence {list(tools)},\n"
                f"got {actual}"
            )
    else:
        # Check relative ordering
        pos = -1
        for tool in tools:
            try:
                pos = actual.index(tool, pos + 1)
            except ValueError:
                raise AssertionError(
                    f"Expected {tool!r} to be called after position {pos} in sequence,\n"
                    f"but it wasn't found. Actual sequence: {actual}"
                )


def assert_tool_arg(trace: Trace, tool: str, arg: str, value: Any, call_index: int = 0) -> None:
    """Assert that a specific tool call had the expected argument value."""
    calls = trace.calls(tool)
    if not calls:
        raise AssertionError(f"Tool {tool!r} was never called.")
    if call_index >= len(calls):
        raise AssertionError(
            f"Expected at least {call_index + 1} call(s) to {tool!r}, got {len(calls)}."
        )
    actual = calls[call_index].args.get(arg)
    if actual != value:
        raise AssertionError(
            f"Tool {tool!r} call #{call_index}: expected arg {arg!r}={value!r}, got {actual!r}."
        )


def assert_tool_arg_contains(trace: Trace, tool: str, arg: str, substr: str, call_index: int = 0) -> None:
    """Assert that a tool arg value contains a substring."""
    calls = trace.calls(tool)
    if not calls:
        raise AssertionError(f"Tool {tool!r} was never called.")
    actual = str(calls[call_index].args.get(arg, ""))
    if substr not in actual:
        raise AssertionError(
            f"Tool {tool!r} call #{call_index}: expected arg {arg!r} to contain {substr!r},\n"
            f"got {actual!r}."
        )


def assert_tool_succeeded(trace: Trace, tool: str) -> None:
    """Assert that all calls to a tool succeeded (no errors)."""
    for i, call in enumerate(trace.calls(tool)):
        if call.error:
            raise AssertionError(
                f"Tool {tool!r} call #{i} failed with error: {call.error}"
            )


def assert_no_errors(trace: Trace) -> None:
    """Assert that no tool calls failed and the trace has no top-level error."""
    if trace.error:
        raise AssertionError(f"Trace has error: {trace.error}")
    failed = [c for c in trace.tool_calls if c.error]
    if failed:
        raise AssertionError(
            f"{len(failed)} tool call(s) failed:\n" +
            "\n".join(f"  {c.tool}: {c.error}" for c in failed)
        )


# ── output assertions ──────────────────────────────────────────────────────────

def assert_output_contains(trace: Trace, substr: str) -> None:
    """Assert that the trace output contains a substring."""
    if trace.output is None:
        raise AssertionError("Trace has no output set.")
    if substr not in trace.output:
        raise AssertionError(
            f"Expected output to contain {substr!r}.\nActual output: {trace.output!r}"
        )


def assert_output_matches_schema(trace: Trace, schema: dict) -> None:
    """
    Assert that the trace output is valid JSON matching a JSON Schema.
    Requires jsonschema: pip install jsonschema
    """
    try:
        import jsonschema
        import json
    except ImportError:
        raise ImportError("pip install jsonschema to use assert_output_matches_schema")

    if trace.output is None:
        raise AssertionError("Trace has no output set.")
    try:
        data = json.loads(trace.output)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Trace output is not valid JSON: {e}\nOutput: {trace.output!r}")

    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        raise AssertionError(f"Output failed schema validation: {e.message}")


# ── performance assertions ─────────────────────────────────────────────────────

def assert_duration_under(trace: Trace, max_ms: float) -> None:
    """Assert the total trace duration is under max_ms milliseconds."""
    if trace.duration_ms is None:
        raise AssertionError("Trace has no duration recorded.")
    if trace.duration_ms > max_ms:
        raise AssertionError(
            f"Trace took {trace.duration_ms:.0f}ms, expected under {max_ms:.0f}ms."
        )


def assert_tool_duration_under(trace: Trace, tool: str, max_ms: float) -> None:
    """Assert that all calls to a tool completed under max_ms milliseconds."""
    for i, call in enumerate(trace.calls(tool)):
        if call.duration_ms is not None and call.duration_ms > max_ms:
            raise AssertionError(
                f"Tool {tool!r} call #{i} took {call.duration_ms:.0f}ms, "
                f"expected under {max_ms:.0f}ms."
            )


# ── message assertions ─────────────────────────────────────────────────────────

def assert_message_count(trace: Trace, role: str, count: int) -> None:
    """Assert that a specific number of messages with a given role exist."""
    actual = sum(1 for m in trace.messages if m.get("role") == role)
    if actual != count:
        raise AssertionError(
            f"Expected {count} {role!r} message(s), got {actual}."
        )
