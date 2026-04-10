"""
Example: a minimal "research assistant" agent.

Simulates an agent that:
1. Searches the web for a topic
2. Reads one of the results
3. Summarizes it
4. Optionally sends an email if the user asked for it

This file is intentionally self-contained — no real LLM or API calls.
The agent logic is just plain Python to demonstrate agenttest clearly.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agenttest import Tracer

tracer = Tracer(traces_dir="examples/traces")


# ── Tool definitions ───────────────────────────────────────────────────────────

@tracer.tool
def search_web(query: str) -> list[dict]:
    """Return fake search results."""
    return [
        {"url": "https://example.com/1", "title": f"Result for: {query}"},
        {"url": "https://example.com/2", "title": f"Another result for: {query}"},
    ]


@tracer.tool
def fetch_url(url: str) -> str:
    """Return fake page content."""
    return f"<article>Detailed content from {url}. AI agents are transforming software.</article>"


@tracer.tool
def summarize(text: str, max_words: int = 50) -> str:
    """Return a fake summary."""
    words = text.split()[:max_words]
    return "Summary: " + " ".join(words)


@tracer.tool
def send_email(to: str, subject: str, body: str) -> str:
    """Simulate sending an email."""
    return f"Email sent to {to}"


# ── Agent logic ────────────────────────────────────────────────────────────────

def run_research_agent(topic: str, notify_email: str | None = None) -> str:
    """
    A simple research agent:
    1. Search for topic
    2. Fetch first result
    3. Summarize
    4. Email if requested
    """
    results = search_web(query=topic)
    content = fetch_url(url=results[0]["url"])
    summary = summarize(text=content, max_words=30)

    if notify_email:
        send_email(
            to=notify_email,
            subject=f"Research summary: {topic}",
            body=summary,
        )
        return f"{summary}\n\n(emailed to {notify_email})"

    return summary


# ── Record traces ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Recording traces...\n")

    # Trace 1: research without email
    with tracer.record("research_no_email") as t:
        result = run_research_agent("AI agents in production")
        tracer.set_output(result)
    print(f"Output: {result}\n")

    # Trace 2: research with email notification
    with tracer.record("research_with_email") as t:
        result = run_research_agent("AI agents in production", notify_email="alice@example.com")
        tracer.set_output(result)
    print(f"Output: {result}\n")

    print("Traces saved to examples/traces/")
    print("Run: agenttest stats examples/traces/")
