"""
Replay — mock tools from a saved trace so tests run offline and fast.

Usage:
    from agenttest import Replay

    saved = Trace.load("tests/traces/my_test.trace.json")
    replay = Replay(saved)

    # Get mock versions of tools that return recorded outputs
    send_email = replay.mock("send_email")
    search_web = replay.mock("search_web")

    # Now run your agent with the mocked tools — no real calls made
    with replay.session("my_test") as trace:
        run_my_agent(send_email=send_email, search_web=search_web)

    # trace now has calls from this run; compare to saved
    replay.assert_equivalent(trace)
"""
from __future__ import annotations

from typing import Any, Callable, Iterator

from .trace import Trace, ToolCall
from .assertions import assert_sequence


class _ToolMock:
    """Iterates through recorded results in order, raising if exhausted."""

    def __init__(self, tool_name: str, recorded_calls: list[ToolCall]):
        self.tool_name = tool_name
        self._calls: list[ToolCall] = list(recorded_calls)
        self._index = 0

    def __call__(self, *args, **kwargs) -> Any:
        if self._index >= len(self._calls):
            raise ReplayError(
                f"Tool {self.tool_name!r} was called {self._index + 1}x during replay "
                f"but only {len(self._calls)} call(s) were recorded."
            )
        call = self._calls[self._index]
        self._index += 1

        if call.error:
            raise Exception(call.error)
        return call.result

    @property
    def call_count(self) -> int:
        return self._index

    def reset(self):
        self._index = 0


class ReplayError(Exception):
    pass


class Replay:
    """
    Given a saved Trace, produce mock callables that replay recorded results.
    Agents under test call the mocks; mocks return what was recorded.
    """

    def __init__(self, trace: Trace):
        self.trace = trace
        self._mocks: dict[str, _ToolMock] = {}

    def mock(self, tool_name: str) -> Callable:
        """
        Return a callable that replays the recorded results for tool_name.
        Calls are served in the order they were recorded.
        """
        calls = self.trace.calls(tool_name)
        m = _ToolMock(tool_name, calls)
        self._mocks[tool_name] = m
        return m

    def mock_all(self) -> dict[str, Callable]:
        """Return mocks for every tool that appears in the trace."""
        return {name: self.mock(name) for name in self.trace.tool_names()}

    def assert_equivalent(self, other: Trace, check_args: bool = False) -> None:
        """
        Assert that two traces called the same tools in the same order.

        check_args=True also compares argument values (useful for strict regression).
        """
        expected = self.trace.call_sequence()
        actual = other.call_sequence()
        if expected != actual:
            raise AssertionError(
                f"Call sequence mismatch.\n"
                f"  Expected: {expected}\n"
                f"  Actual:   {actual}"
            )
        if check_args:
            for i, (e, a) in enumerate(zip(self.trace.tool_calls, other.tool_calls)):
                if e.args != a.args:
                    raise AssertionError(
                        f"Tool call #{i} ({e.tool!r}) args differ.\n"
                        f"  Expected: {e.args}\n"
                        f"  Actual:   {a.args}"
                    )

    def exhausted(self) -> bool:
        """True if all mocked tools have been called exactly as many times as recorded."""
        for name, mock in self._mocks.items():
            expected = len(self.trace.calls(name))
            if mock.call_count != expected:
                return False
        return True

    def assert_exhausted(self) -> None:
        """Assert that every mocked tool was called exactly as many times as recorded."""
        errors = []
        for name, mock in self._mocks.items():
            expected = len(self.trace.calls(name))
            if mock.call_count != expected:
                errors.append(
                    f"  {name!r}: expected {expected} call(s), got {mock.call_count}"
                )
        if errors:
            raise AssertionError(
                "Not all recorded tool calls were replayed:\n" + "\n".join(errors)
            )
