"""
Trace data model — the core record of a single agent run.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any]
    result: Any
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ToolCall":
        return cls(**d)


@dataclass
class Trace:
    name: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    output: Optional[str] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # ── query helpers ──────────────────────────────────────────────

    def calls(self, tool: str) -> list[ToolCall]:
        """Return all calls to a specific tool."""
        return [c for c in self.tool_calls if c.tool == tool]

    def call_sequence(self) -> list[str]:
        """Return ordered list of tool names that were called."""
        return [c.tool for c in self.tool_calls]

    def has_error(self) -> bool:
        return self.error is not None or any(c.error for c in self.tool_calls)

    def tool_names(self) -> set[str]:
        return {c.tool for c in self.tool_calls}

    def total_tool_calls(self) -> int:
        return len(self.tool_calls)

    # ── serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "Trace":
        tool_calls = [ToolCall.from_dict(c) for c in d.pop("tool_calls", [])]
        t = cls(**d)
        t.tool_calls = tool_calls
        return t

    @classmethod
    def load(cls, path: str | Path) -> "Trace":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def __repr__(self) -> str:
        return (
            f"Trace(name={self.name!r}, calls={self.call_sequence()}, "
            f"error={self.error!r})"
        )
