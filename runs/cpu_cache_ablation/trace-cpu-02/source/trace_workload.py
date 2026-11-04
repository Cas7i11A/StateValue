"""Convert FreeInference raw session JSONL into a small closed-loop replay plan."""
import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import math
from pathlib import Path


def number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def open_trace(path):
    if path.suffix == ".zst":
        try:
            import zstandard
        except ImportError as exc:
            raise ValueError("For .zst input install zstandard, or decompress to JSONL first") from exc
        with path.open("rb") as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as text:
                    yield text
    else:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            yield stream


def convert_session(session, allow_tool_batches=False):
    if not isinstance(session, dict):
        raise ValueError("invalid_session")
    requests = session.get("requests")
    if not isinstance(requests, list) or len(requests) < 2:
        raise ValueError("too_few_requests")
    if not session.get("session_id") or not session.get("week"):
        raise ValueError("missing_session_identity")
    steps = []
    previous = None
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError("invalid_request")
        start, end = request.get("start_ms"), request.get("end_ms")
        if not number(start) or not number(end) or end < start:
            raise ValueError("invalid_timestamps")
        tools = request.get("tool_calls", [])
        if not isinstance(tools, list) or any(not isinstance(t, dict) for t in tools):
            raise ValueError("invalid_tools")
        if any(t.get("subagent") is not None for t in tools):
            raise ValueError("subagent")
        # Gap replay treats a batch as one observed wait, not independent tools.
        if len(tools) > 1 and not allow_tool_batches:
            raise ValueError("parallel_tools")
        source_step = request.get("step")
        if (not isinstance(source_step, int) or isinstance(source_step, bool)
                or source_step < 0 or (previous is not None
                                      and source_step != previous["step"] + 1)):
            raise ValueError("nonsequential_steps")
        if previous is not None:
            if start < previous["end_ms"]:
                raise ValueError("overlapping_requests")
            if (request.get("step_trigger") != "tool"
                    or request.get("turn_index") != requests[0].get("turn_index")
                    or not previous.get("tool_calls")
                    or (not allow_tool_batches and len(previous["tool_calls"]) != 1)):
                raise ValueError("not_serial_tool_chain")
            if (request.get("model"), request.get("provider")) != (
                    previous.get("model"), previous.get("provider")):
                raise ValueError("model_or_provider_change")
            steps[-1]["gap_after_s"] = (start - previous["end_ms"]) / 1000
        categories = [tool.get("category") for tool in tools]
        steps.append(dict(source_step=source_step, gap_after_s=None,
                          tool_category=("multiple" if len(tools) > 1 else
                                         categories[0] if tools else None),
                          tool_categories=categories, tool_count=len(tools)))
        previous = request
    return dict(session_id=f"{session['week']}:{session['session_id']}",
                source_start_ms=requests[0]["start_ms"], steps=steps)


def extract_segments(session, excluded_transitions):
    """Keep maximal compatible runs; never connect across a rejected transition."""
    if not isinstance(session, dict):
        raise ValueError("invalid_session")
    requests = session.get("requests")
    if not isinstance(requests, list) or len(requests) < 2:
        raise ValueError("too_few_requests")
    if not session.get("session_id") or not session.get("week"):
        raise ValueError("missing_session_identity")
    segments, run = [], []

    def finish():
        if len(run) < 2:
            return
        segment = convert_session(dict(session, requests=run), allow_tool_batches=True)
        segment["source_session_id"] = segment["session_id"]
        segment["session_id"] += f":steps-{run[0]['step']}-{run[-1]['step']}"
        segments.append(segment)

    for left, right in zip(requests, requests[1:]):
        try:
            convert_session(dict(session, requests=[left, right]), allow_tool_batches=True)
        except ValueError as exc:
            excluded_transitions[str(exc)] += 1
            finish()
            run = []
        else:
            if not run:
                run = [left]
            run.append(right)
    finish()
    return segments


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if not isinstance(plan, dict) or plan.get("schema_version") != 1 or not isinstance(plan.get("sessions"), list) or not plan["sessions"]:
        raise ValueError("Expected a nonempty replay plan with schema_version=1")
    seen = set()
    for session in plan["sessions"]:
        if not isinstance(session, dict):
            raise ValueError("Replay sessions must be objects")
        cid = session.get("session_id")
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError("Replay session IDs must be nonempty and unique")
        seen.add(cid)
        if not number(session.get("arrival_offset_s")):
            raise ValueError("Invalid arrival offset")
        steps = session.get("steps")
        if not isinstance(steps, list) or len(steps) < 2:
            raise ValueError("Each replay session needs at least two steps")
        for step in steps[:-1]:
            if not isinstance(step, dict) or not number(step.get("gap_after_s")):
                raise ValueError("Invalid gap_after_s")
        if not isinstance(steps[-1], dict) or steps[-1].get("gap_after_s") is not None:
            raise ValueError("Final step must have null gap_after_s")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+", help="Raw JSONL, JSONL.gz or JSONL.zst files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sessions", type=int, default=32)
    parser.add_argument("--selection-mode", choices=["segments", "whole-session"],
                        default="segments", help="Extract tool-chain segments (default), or strict whole sessions")
    args = parser.parse_args()
    if args.max_sessions < 1:
        parser.error("max-sessions must be positive")
    if args.output.exists():
        parser.error("Output already exists")
    sessions, seen, excluded, sources = [], set(), Counter(), []
    excluded_transitions = Counter()
    source_count = contributing_count = 0
    # Scan all supplied files so selection is chronological, not file-order based.
    for path in args.inputs:
        sources.append(dict(path=str(path), sha256=file_sha256(path)))
        with open_trace(path) as stream:
            for line in stream:
                if not line.strip():
                    continue
                source_count += 1
                try:
                    raw = json.loads(line)
                    candidates = (extract_segments(raw, excluded_transitions)
                                  if args.selection_mode == "segments" else [convert_session(raw)])
                except json.JSONDecodeError:
                    excluded["invalid_json"] += 1
                    continue
                except ValueError as exc:
                    excluded[str(exc)] += 1
                    continue
                if not candidates:
                    excluded["no_eligible_segments"] += 1
                contributed = False
                for session in candidates:
                    if session["session_id"] in seen:
                        excluded["duplicate_replay_unit"] += 1
                        continue
                    seen.add(session["session_id"])
                    sessions.append(session)
                    contributed = True
                contributing_count += contributed
    eligible = len(sessions)
    sessions.sort(key=lambda s: (s["source_start_ms"], s["session_id"]))
    sessions = sessions[:args.max_sessions]
    if not sessions:
        parser.error(f"No eligible replay units: {dict(excluded)}; rejected transitions: {dict(excluded_transitions)}")
    origin = sessions[0]["source_start_ms"]
    for session in sessions:
        session["arrival_offset_s"] = (session["source_start_ms"] - origin) / 1000
    plan = dict(schema_version=1, workload="trace_gap_closed_loop",
                source_dataset="harvardMadsys/freeinference_agentic_trace",
                source_license="CC-BY-4.0", sources=sources,
                selection=dict(mode=args.selection_mode,
                               unit="tool_chain_segment" if args.selection_mode == "segments" else "whole_session",
                               source_records=source_count, contributing_source_records=contributing_count,
                               eligible_sessions=eligible, selected_sessions=len(sessions),
                               omitted_by_limit=eligible - len(sessions), excluded=dict(excluded),
                               excluded_transitions=dict(excluded_transitions)),
                note="Synthetic contents and lengths; observed request gaps are not isolated tool runtimes.",
                sessions=sessions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), selection=plan["selection"]), indent=2))


if __name__ == "__main__":
    main()
