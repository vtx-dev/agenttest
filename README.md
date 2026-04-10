# agenttest

**pytest-native testing for AI agent workflows.**

Record agent runs as traces. Replay them offline. Assert on tool call sequences. Measure non-determinism. Works with any agent framework.

```bash
pip install agenttest
```

---

## The problem

You can write unit tests for functions. You can write integration tests for APIs. But AI agents are different — they make sequences of tool calls based on LLM decisions, and those sequences are what actually matter for correctness.

```python
# This test tells you nothing useful:
def test_agent():
    result = my_agent.run("send a welcome email to alice@example.com")
    assert "email" in result.lower()  # 🤷

# This test tells you exactly what happened:
def test_agent():
    with tracer.session("welcome_email") as trace:
        my_agent.run("send a welcome email to alice@example.com")

    assert_tool_called(trace, "send_email")
    assert_tool_arg(trace, "send_email", "to", "alice@example.com")
    assert_sequence(trace, "lookup_contacts", "send_email")
    assert_no_errors(trace)
```

---

## Quickstart

### 1. Wrap your tools

```python
from agenttest import Tracer

tracer = Tracer()

@tracer.tool
def search_web(query: str) -> list[dict]:
    return requests.get(f"https://search.example.com?q={query}").json()

@tracer.tool
def send_email(to: str, subject: str, body: str) -> str:
    return email_client.send(to, subject, body)
```

### 2. Record a session

```python
with tracer.session("welcome_email_test") as trace:
    my_agent.run("send a welcome email to alice@example.com")

# Or auto-save to tests/traces/:
with tracer.record("welcome_email_test") as trace:
    my_agent.run("send a welcome email to alice@example.com")
```

### 3. Assert on what happened

```python
from agenttest import (
    assert_tool_called, assert_tool_not_called,
    assert_sequence, assert_tool_arg, assert_no_errors,
)

assert_tool_called(trace, "send_email")
assert_tool_not_called(trace, "delete_user")
assert_sequence(trace, "search_web", "send_email")
assert_tool_arg(trace, "send_email", "to", "alice@example.com")
assert_no_errors(trace)
```

---

## Patterns

### Live recording + assertions (most common)

```python
def test_research_agent_emails_results():
    with tracer.session("research_email") as trace:
        result = my_agent.run("research quantum computing and email me")

    assert_sequence(trace, "search_web", "fetch_url", "summarize", "send_email")
    assert_tool_called(trace, "send_email", times=1)
    assert_tool_arg_contains(trace, "send_email", "subject", "quantum")
    assert_no_errors(trace)
```

### Replay (offline, no real tool calls)

Run your agent without making real API calls — tools return their recorded outputs.

```python
from agenttest import Replay, Trace

def test_agent_offline():
    saved = Trace.load("tests/traces/research_email.trace.json")
    replay = Replay(saved)
    mocks = replay.mock_all()  # {tool_name: callable}

    # Run your agent with mocked tools
    with tracer.session("replay_run") as trace:
        my_agent.run("research quantum computing", tools=mocks)

    replay.assert_equivalent(trace)  # same sequence as recorded
    replay.assert_exhausted()         # all recorded calls were replayed
```

### Pytest fixture (`agent_trace`)

If a `.trace.json` file exists for the test, it's auto-loaded:

```python
# Loads tests/traces/research_email.trace.json automatically
def test_research_email(agent_trace):
    assert_sequence(agent_trace, "search_web", "summarize", "send_email")
    assert_no_errors(agent_trace)
```

### Scorer — measure non-determinism

```python
from agenttest import Scorer

def test_agent_determinism():
    scorer = Scorer(runs=10)

    def one_run():
        with tracer.session("det_test") as t:
            my_agent.run("research quantum computing")
        return t

    report = scorer.score(one_run)
    print(report)
    # Sequence consistency : 90%
    # Modal sequence       : ['search_web', 'fetch_url', 'summarize']
    # Error rate           : 0%

    assert report.sequence_consistency >= 0.8
    assert report.error_rate == 0.0
```

---

## Assertions reference

| Assertion | Description |
|-----------|-------------|
| `assert_tool_called(trace, tool, times=None)` | Tool was called (optionally N times) |
| `assert_tool_not_called(trace, tool)` | Tool was never called |
| `assert_sequence(trace, *tools, strict=False)` | Tools called in this order |
| `assert_tool_arg(trace, tool, arg, value)` | Tool arg equals value |
| `assert_tool_arg_contains(trace, tool, arg, substr)` | Tool arg contains substring |
| `assert_tool_succeeded(trace, tool)` | All calls to tool had no errors |
| `assert_no_errors(trace)` | No tool errors and no trace-level error |
| `assert_output_contains(trace, substr)` | Final output contains substring |
| `assert_output_matches_schema(trace, schema)` | Output matches JSON Schema |
| `assert_duration_under(trace, max_ms)` | Total run under N milliseconds |
| `assert_message_count(trace, role, count)` | N messages with given role |

---

## CLI

```bash
# Pretty-print a trace
agenttest show tests/traces/my_test.trace.json

# Diff two traces
agenttest diff tests/traces/before.trace.json tests/traces/after.trace.json

# Summarize all traces in a directory
agenttest stats tests/traces/
```

---

## MCP adapter

```python
from agenttest import Tracer

tracer = Tracer()
raw_handler = lambda tool_name, args: dispatch(tool_name, args)
mcp_handler = tracer.wrap_mcp_handler(raw_handler)

with tracer.session("mcp_run") as trace:
    mcp_handler("search_web", {"query": "quantum computing"})
    mcp_handler("send_email", {"to": "alice@example.com", ...})
```

---

## Trace format

Traces are plain JSON — store them in git alongside your tests.

```json
{
  "id": "abc123",
  "name": "welcome_email_test",
  "created_at": "2026-04-10T14:00:00Z",
  "duration_ms": 1240,
  "tool_calls": [
    {
      "id": "tc_001",
      "tool": "send_email",
      "args": {"to": "alice@example.com", "subject": "Welcome!", "body": "..."},
      "result": "sent",
      "error": null,
      "duration_ms": 234,
      "timestamp": "2026-04-10T14:00:01Z"
    }
  ],
  "messages": [...],
  "output": "Email sent successfully.",
  "error": null,
  "metadata": {}
}
```

---

## Cloud: hosted trace storage + regression detection

[**agenttests.dev**](https://agenttests.dev) is the hosted layer for agenttest — trace history, run dashboard, and automatic regression detection across deploys.

```python
tracer = Tracer(
    suite="my-agent",
    cloud_url="https://agenttests.dev",
    cloud_api_key="at_...",   # get one free at agenttests.dev
)

# Every session auto-uploads. No other changes needed.
with tracer.session("my_test") as trace:
    my_agent.run(...)
```

What you get:
- **Run history** — every session stored, searchable by suite
- **Regression diff** — each run automatically compared to the previous one; sequence changes, new/dropped tools, and error rate shifts are flagged
- **Run dashboard** — browse tool call timelines, jump to diffs, filter by suite
- **Free to start** — register at [agenttests.dev](https://agenttests.dev)

agenttest itself remains MIT-licensed and fully functional without the cloud.

---

## Why not LangSmith / Langfuse?

Those are LLM observability tools. agenttest is a **testing library** focused on tool call sequences — not LLM inputs/outputs.

| | agenttest | LangSmith / Langfuse |
|--|-----------|----------------------|
| Runs in CI/CD | ✓ | Requires cloud account |
| Stored in git | ✓ (plain JSON) | Cloud only |
| Tool-call-first | ✓ | LLM-output-first |
| Offline replay | ✓ | — |
| Determinism scoring | ✓ | — |
| Regression detection | ✓ (agenttests.dev) | — |
| pytest integration | Native | Plugin |
| Core cost | Free / MIT | Paid tiers |

---

## Contributing

Issues and PRs welcome. Areas that need work:

- Async tool support (`async def` tools)
- LangChain / LangGraph adapter
- OpenAI tool-call format adapter  
- Anthropic tool-use format adapter
- HTML trace viewer
- GitHub Actions template

```bash
git clone https://github.com/vtx-dev/agenttest
cd agenttest
pip install -e ".[dev]"
pytest
```

---

MIT License
