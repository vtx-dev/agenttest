"""
Tests for the simple_agent example.

Shows three testing patterns:
1. Live recording: run the agent, record trace, assert on it
2. Replay: load a saved trace, mock tools, re-run agent offline
3. Scorer: run agent N times, measure determinism
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
from agenttest import (
    Tracer, Replay, Scorer, Trace,
    assert_tool_called, assert_tool_not_called,
    assert_sequence, assert_tool_arg_contains,
    assert_no_errors, assert_output_contains,
)
from examples.simple_agent import run_research_agent, tracer as agent_tracer


# ── Pattern 1: Live recording ──────────────────────────────────────────────────

class TestResearchAgent:
    def test_basic_research_calls_tools_in_order(self):
        """Agent must: search → fetch → summarize."""
        with agent_tracer.session("test_basic") as trace:
            run_research_agent("quantum computing")

        assert_sequence(trace, "search_web", "fetch_url", "summarize")
        assert_no_errors(trace)

    def test_basic_research_does_not_send_email(self):
        """Without notify_email, no email should be sent."""
        with agent_tracer.session("test_no_email") as trace:
            run_research_agent("machine learning")

        assert_tool_not_called(trace, "send_email")

    def test_email_notification_sends_to_correct_address(self):
        """With notify_email, agent must send to that exact address."""
        with agent_tracer.session("test_with_email") as trace:
            run_research_agent("neural networks", notify_email="bob@example.com")

        assert_tool_called(trace, "send_email", times=1)
        assert_tool_arg_contains(trace, "send_email", "to", "bob@example.com")
        assert_tool_arg_contains(trace, "send_email", "subject", "neural networks")

    def test_full_sequence_with_email(self):
        """With email: search → fetch → summarize → send_email."""
        with agent_tracer.session("test_full_seq") as trace:
            run_research_agent("topic", notify_email="x@y.com")

        assert_sequence(trace, "search_web", "fetch_url", "summarize", "send_email", strict=True)

    def test_output_is_a_summary(self):
        """Output should start with 'Summary:'."""
        with agent_tracer.session("test_output") as trace:
            result = run_research_agent("AI agents")
            agent_tracer.set_output(result)

        assert_output_contains(trace, "Summary:")


# ── Pattern 2: Replay (offline, no real tool calls) ───────────────────────────

class TestReplay:
    def test_replay_from_saved_trace(self, tmp_path):
        """
        Load a saved trace, replace tools with mocks, re-run agent.
        The agent logic is tested without any real I/O.
        """
        # Record using the agent's own tracer (tools are decorated with it)
        with agent_tracer.session("replay_source") as recorded:
            run_research_agent("replay test topic")

        # Now replay it
        replay = Replay(recorded)
        mocks = replay.mock_all()

        # Patch the agent's tools with mocks
        import examples.simple_agent as agent_module
        originals = {
            "search_web": agent_module.search_web,
            "fetch_url": agent_module.fetch_url,
            "summarize": agent_module.summarize,
        }
        try:
            agent_module.search_web = mocks["search_web"]
            agent_module.fetch_url = mocks["fetch_url"]
            agent_module.summarize = mocks["summarize"]

            # Run with a fresh tracer (not the agent's tracer)
            replay_tracer = Tracer()
            with replay_tracer.session("replay_run") as replay_trace:
                result = agent_module.run_research_agent("replay test topic")
        finally:
            agent_module.search_web = originals["search_web"]
            agent_module.fetch_url = originals["fetch_url"]
            agent_module.summarize = originals["summarize"]

        # The result should match what was recorded
        assert result == recorded.calls("summarize")[0].result


# ── Pattern 3: Scorer (determinism measurement) ────────────────────────────────

class TestDeterminism:
    def test_agent_is_deterministic(self):
        """Agent should call tools in the same sequence every run."""
        scorer = Scorer(runs=3)

        def one_run() -> Trace:
            with agent_tracer.session("score_run") as t:
                run_research_agent("determinism test")
            return t

        report = scorer.score(one_run)
        print(f"\n{report}")

        assert report.sequence_consistency == 1.0, (
            f"Agent is non-deterministic: {report.unique_sequences} different sequences seen"
        )
        assert report.error_rate == 0.0
