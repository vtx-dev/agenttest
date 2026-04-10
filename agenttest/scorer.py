"""
Scorer — measure how deterministic an agent is.

Run the agent N times and report:
- Sequence consistency (did it always call the same tools in the same order?)
- Output consistency (did the final output vary?)
- Error rate
- Per-tool success rate and avg duration

Usage:
    from agenttest import Scorer

    scorer = Scorer(runs=5)

    def run_agent():
        # your agent code here; return a Trace
        with tracer.session("test") as t:
            my_agent.run("summarize this document")
        return t

    report = scorer.score(run_agent)
    print(report)
    assert report.sequence_consistency >= 0.8  # same tool sequence 80%+ of runs
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable

from .trace import Trace


@dataclass
class ScoreReport:
    runs: int
    traces: list[Trace]
    sequence_consistency: float      # 0.0–1.0, fraction of runs matching modal sequence
    error_rate: float                 # 0.0–1.0
    modal_sequence: list[str]        # most common tool call sequence
    unique_sequences: int            # number of distinct sequences seen
    output_consistency: float | None = None  # None if outputs weren't set
    per_tool: dict[str, dict] = field(default_factory=dict)

    def passed(
        self,
        min_sequence_consistency: float = 0.8,
        max_error_rate: float = 0.1,
    ) -> bool:
        return (
            self.sequence_consistency >= min_sequence_consistency
            and self.error_rate <= max_error_rate
        )

    def __str__(self) -> str:
        lines = [
            f"AgentTest Score Report ({self.runs} runs)",
            f"  Sequence consistency : {self.sequence_consistency:.0%}",
            f"  Modal sequence       : {self.modal_sequence}",
            f"  Unique sequences     : {self.unique_sequences}",
            f"  Error rate           : {self.error_rate:.0%}",
        ]
        if self.output_consistency is not None:
            lines.append(f"  Output consistency   : {self.output_consistency:.0%}")
        if self.per_tool:
            lines.append("  Per-tool stats:")
            for tool, stats in self.per_tool.items():
                lines.append(
                    f"    {tool}: called {stats['count']}x, "
                    f"success {stats['success_rate']:.0%}, "
                    f"avg {stats['avg_duration_ms']:.0f}ms"
                )
        return "\n".join(lines)


class Scorer:
    def __init__(self, runs: int = 5):
        self.runs = runs

    def score(self, agent_fn: Callable[[], Trace]) -> ScoreReport:
        """
        Run agent_fn self.runs times, collect Traces, return ScoreReport.
        agent_fn must return a Trace.
        """
        traces: list[Trace] = []
        for i in range(self.runs):
            try:
                t = agent_fn()
                traces.append(t)
            except Exception as e:
                # Create an error trace for this run
                traces.append(Trace(name=f"run_{i}", error=str(e)))

        return self._analyze(traces)

    def _analyze(self, traces: list[Trace]) -> ScoreReport:
        sequences = [tuple(t.call_sequence()) for t in traces]
        error_count = sum(1 for t in traces if t.has_error())

        # Modal sequence
        from collections import Counter
        seq_counter = Counter(sequences)
        modal_seq = list(seq_counter.most_common(1)[0][0])
        modal_count = seq_counter.most_common(1)[0][1]
        sequence_consistency = modal_count / len(traces)
        unique_sequences = len(seq_counter)

        # Output consistency
        outputs = [t.output for t in traces if t.output is not None]
        if len(outputs) >= 2:
            unique_outputs = len(set(outputs))
            output_consistency = (len(outputs) - unique_outputs + 1) / len(outputs)
        else:
            output_consistency = None

        # Per-tool stats
        tool_data: dict[str, list] = {}
        for t in traces:
            for call in t.tool_calls:
                tool_data.setdefault(call.tool, []).append(call)

        per_tool = {}
        for tool_name, calls in tool_data.items():
            durations = [c.duration_ms for c in calls if c.duration_ms is not None]
            successes = [c for c in calls if not c.error]
            per_tool[tool_name] = {
                "count": len(calls),
                "success_rate": len(successes) / len(calls),
                "avg_duration_ms": statistics.mean(durations) if durations else 0.0,
            }

        return ScoreReport(
            runs=len(traces),
            traces=traces,
            sequence_consistency=sequence_consistency,
            error_rate=error_count / len(traces),
            modal_sequence=modal_seq,
            unique_sequences=unique_sequences,
            output_consistency=output_consistency,
            per_tool=per_tool,
        )
