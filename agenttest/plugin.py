"""
pytest plugin — auto-discovered via entry_points.

Provides:
  - `agent_trace` fixture: loads a .trace.json file matching the test name
  - `tracer` fixture: a fresh Tracer for the test
  - `--traces-dir` CLI option to override trace directory

Example:
    # tests/traces/test_send_email.trace.json must exist
    def test_send_email(agent_trace):
        assert_tool_called(agent_trace, "send_email")
        assert_no_errors(agent_trace)

    # or record during the test:
    def test_send_email_live(tracer):
        with tracer.session("live") as trace:
            result = my_agent.run("send email to alice@example.com")
        assert_tool_called(trace, "send_email")
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from .trace import Trace
from .tracer import Tracer


def pytest_addoption(parser):
    parser.addoption(
        "--traces-dir",
        default="tests/traces",
        help="Directory containing .trace.json files (default: tests/traces)",
    )


@pytest.fixture
def traces_dir(request) -> Path:
    return Path(request.config.getoption("--traces-dir"))


@pytest.fixture
def agent_trace(request, traces_dir) -> Trace:
    """
    Load the trace file matching the current test name.
    File must be at: <traces_dir>/<test_name>.trace.json

    The test name is the pytest node id's function name, with 'test_' prefix stripped.
    E.g. test_send_email → tests/traces/send_email.trace.json
    """
    # Derive trace name from test function name
    test_name = request.node.name
    # Strip test_ prefix and any [param] suffix
    trace_name = re.sub(r"^\s*test_", "", test_name)
    trace_name = re.sub(r"\[.*\]$", "", trace_name)

    path = traces_dir / f"{trace_name}.trace.json"
    if not path.exists():
        pytest.skip(f"No trace file found at {path} — run the agent first to record it.")

    return Trace.load(path)


@pytest.fixture
def tracer(tmp_path) -> Tracer:
    """Provide a fresh Tracer for the test, saving traces to tmp_path."""
    return Tracer(traces_dir=tmp_path)


@pytest.fixture
def agent_tracer(traces_dir) -> Tracer:
    """Provide a Tracer that saves to the project traces_dir."""
    return Tracer(traces_dir=traces_dir)
