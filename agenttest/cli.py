"""
agenttest CLI

Commands:
  agenttest show <trace.json>       Pretty-print a trace
  agenttest diff <a.json> <b.json>  Diff two traces
  agenttest stats <traces_dir/>     Summarize all traces in a directory
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .trace import Trace


def cmd_show(args):
    trace = Trace.load(args.file)
    print(f"Trace: {trace.name}")
    print(f"  ID         : {trace.id}")
    print(f"  Created    : {trace.created_at}")
    print(f"  Duration   : {trace.duration_ms:.0f}ms" if trace.duration_ms else "  Duration   : —")
    print(f"  Error      : {trace.error or '—'}")
    print(f"  Tool calls : {trace.total_tool_calls()}")
    print(f"  Sequence   : {trace.call_sequence()}")
    print(f"  Output     : {(trace.output or '—')[:120]}")
    if trace.tool_calls:
        print("\nTool calls:")
        for i, call in enumerate(trace.tool_calls):
            status = "✓" if not call.error else "✗"
            dur = f"{call.duration_ms:.0f}ms" if call.duration_ms else "?"
            print(f"  [{i}] {status} {call.tool}  ({dur})")
            print(f"      args   : {json.dumps(call.args)[:120]}")
            if call.error:
                print(f"      error  : {call.error}")
            else:
                result_str = str(call.result)[:120] if call.result is not None else "null"
                print(f"      result : {result_str}")
    if trace.messages:
        print(f"\nMessages ({len(trace.messages)}):")
        for m in trace.messages:
            role = m.get("role", "?")
            content = str(m.get("content", ""))[:100]
            print(f"  [{role}] {content}")


def cmd_diff(args):
    a = Trace.load(args.file_a)
    b = Trace.load(args.file_b)
    print(f"Diff: {a.name} vs {b.name}")

    seq_a = a.call_sequence()
    seq_b = b.call_sequence()
    if seq_a == seq_b:
        print(f"  Sequence : identical {seq_a}")
    else:
        print(f"  Sequence A : {seq_a}")
        print(f"  Sequence B : {seq_b}")

    only_a = a.tool_names() - b.tool_names()
    only_b = b.tool_names() - a.tool_names()
    if only_a:
        print(f"  Tools only in A : {only_a}")
    if only_b:
        print(f"  Tools only in B : {only_b}")

    if a.output != b.output:
        print(f"  Output A : {(a.output or '—')[:100]}")
        print(f"  Output B : {(b.output or '—')[:100]}")
    else:
        print(f"  Output   : identical")

    err_a = a.has_error()
    err_b = b.has_error()
    if err_a != err_b:
        print(f"  Errors   : A={'yes' if err_a else 'no'}, B={'yes' if err_b else 'no'}")


def cmd_stats(args):
    d = Path(args.dir)
    files = sorted(d.glob("*.trace.json"))
    if not files:
        print(f"No .trace.json files found in {d}")
        return

    print(f"Traces in {d}/ ({len(files)} files):\n")
    for f in files:
        t = Trace.load(f)
        status = "✗" if t.has_error() else "✓"
        dur = f"{t.duration_ms:.0f}ms" if t.duration_ms else "?"
        print(f"  {status} {t.name:<40} {dur:>8}  calls={t.total_tool_calls()}")


def main():
    parser = argparse.ArgumentParser(
        prog="agenttest",
        description="agenttest — inspect and compare agent trace files",
    )
    sub = parser.add_subparsers(dest="command")

    p_show = sub.add_parser("show", help="Pretty-print a trace file")
    p_show.add_argument("file", help="Path to .trace.json file")

    p_diff = sub.add_parser("diff", help="Diff two trace files")
    p_diff.add_argument("file_a")
    p_diff.add_argument("file_b")

    p_stats = sub.add_parser("stats", help="Summarize all traces in a directory")
    p_stats.add_argument("dir", nargs="?", default="tests/traces")

    args = parser.parse_args()
    if args.command == "show":
        cmd_show(args)
    elif args.command == "diff":
        cmd_diff(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
