"""
agenttest — pytest-native testing for AI agent workflows.

Record → Replay → Assert → Score

Quickstart:
    from agenttest import Tracer, assert_tool_called, assert_sequence

    tracer = Tracer()

    @tracer.tool
    def send_email(to, subject, body):
        return call_real_api(to, subject, body)

    with tracer.record("welcome_flow") as trace:
        my_agent.run("send a welcome email to alice@example.com")

    assert_tool_called(trace, "send_email")
    assert_sequence(trace, "search_contacts", "send_email")
"""

from .trace import Trace, ToolCall
from .tracer import Tracer
from .replay import Replay, ReplayError
from .scorer import Scorer, ScoreReport
from .assertions import (
    assert_tool_called,
    assert_tool_not_called,
    assert_sequence,
    assert_tool_arg,
    assert_tool_arg_contains,
    assert_tool_succeeded,
    assert_no_errors,
    assert_output_contains,
    assert_output_matches_schema,
    assert_duration_under,
    assert_tool_duration_under,
    assert_message_count,
)

__version__ = "0.1.0"
__all__ = [
    "Trace",
    "ToolCall",
    "Tracer",
    "Replay",
    "ReplayError",
    "Scorer",
    "ScoreReport",
    "assert_tool_called",
    "assert_tool_not_called",
    "assert_sequence",
    "assert_tool_arg",
    "assert_tool_arg_contains",
    "assert_tool_succeeded",
    "assert_no_errors",
    "assert_output_contains",
    "assert_output_matches_schema",
    "assert_duration_under",
    "assert_tool_duration_under",
    "assert_message_count",
]
