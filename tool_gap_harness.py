#!/usr/bin/env python3
"""StateValue P1: closed-loop inference with synthetic tool gaps (stdlib only)."""
import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * quantile
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def stream_request(url, model, messages, max_tokens, socket_timeout):
    payload = dict(model=model, messages=messages, temperature=0,
                   max_tokens=max_tokens, stream=True,
                   stream_options={"include_usage": True})
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first_text = None
    parts, usage, finish, request_id, done = [], {}, None, None, False
    with urllib.request.urlopen(req, timeout=socket_timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise RuntimeError(str(chunk["error"]))
            request_id = chunk.get("id", request_id)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    if first_text is None:
                        first_text = time.perf_counter() - start
                    parts.append(content)
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
    elapsed = time.perf_counter() - start
    if not done or finish is None:
        raise RuntimeError("Incomplete SSE response: missing [DONE] or finish_reason")
    return dict(request_id=request_id, request_latency_s=elapsed,
                ttft_s=first_text, finish_reason=finish, usage=usage,
                output_text="".join(parts))


def initial_messages(context_id, prefix_lines):
    # Context ID occurs BEFORE the padding, so contexts have distinct long prefixes.
    lines = [f"Synthetic context {context_id}. Preserve this session's notes."]
    for i in range(prefix_lines):
        lines.append(f"Note {i:04d}: the worker reads input records, checks their order, "
                     "and writes the validated result to a temporary buffer.")
    lines.append("Summarize the notes briefly. This is a synthetic inference test.")
    return [{"role": "system", "content": "You are a concise technical assistant."},
            {"role": "user", "content": "\n".join(lines)}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--gap-seconds", type=float, default=2.0)
    parser.add_argument("--prefix-lines", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--trace-plan", type=Path,
                        help="Replay plan from trace_workload.py; replaces rounds and fixed gaps")
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="Multiply trace arrival offsets and gaps by this positive factor")
    parser.add_argument("--socket-timeout", type=float, default=120.0,
                        help="Socket I/O timeout, NOT a hard task deadline")
    parser.add_argument("--output", type=Path,
                        help="New output directory; existing directories are rejected")
    args = parser.parse_args()
    if min(args.clients, args.rounds, args.prefix_lines, args.max_tokens) < 1:
        parser.error("clients, rounds, prefix-lines and max-tokens must be positive")
    if not math.isfinite(args.gap_seconds) or args.gap_seconds < 0:
        parser.error("gap-seconds must be finite and nonnegative")
    if not math.isfinite(args.socket_timeout) or args.socket_timeout <= 0:
        parser.error("socket-timeout must be finite and positive")
    if not math.isfinite(args.time_scale) or args.time_scale <= 0:
        parser.error("time-scale must be finite and positive")
    try:
        plan = None
        if args.trace_plan:
            from trace_workload import file_sha256, load_plan
            plan = load_plan(args.trace_plan)
    except (ImportError, OSError, ValueError) as exc:
        parser.error(f"Cannot read replay plan: {exc}")
    if plan is None and args.time_scale != 1:
        parser.error("time-scale requires trace-plan")
    sessions = plan["sessions"] if plan else [
        dict(session_id=f"context-{i:03d}", arrival_offset_s=0,
             steps=[dict(gap_after_s=args.gap_seconds if r + 1 < args.rounds else None,
                         tool_category="synthetic_sleep") for r in range(args.rounds)])
        for i in range(args.clients)]
    if any(not math.isfinite(value * args.time_scale)
           for s in sessions for value in [s["arrival_offset_s"]] +
           [step["gap_after_s"] for step in s["steps"][:-1]]):
        parser.error("Scaled replay times must be finite")
    args.base_url = args.base_url.rstrip("/")
    # Check health BEFORE creating output; this is not a KV transfer verification.
    with urllib.request.urlopen(args.base_url + "/health", timeout=10) as response:
        if response.status != 200:
            raise RuntimeError("Server is not healthy")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    out = args.output or Path("runs") / run_id
    out.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config["output"] = str(out)
    config["trace_plan"] = str(args.trace_plan) if args.trace_plan else None
    config.update(run_id=run_id, workload="trace_gap_closed_loop" if plan else "synthetic_fixed_gap_closed_loop",
                  python_version=platform.python_version(),
                  ttft_definition="client elapsed time to first nonempty content chunk",
                  request_seed=None, task_success_metric=False)
    if plan:
        config.update(trace_plan_sha256=file_sha256(args.trace_plan),
                      session_count=len(sessions), rounds=None, gap_seconds=None,
                      scheduling="closed-loop; clients limits in-flight requests, not live sessions")
        (out / "replay_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    start = time.perf_counter()
    lock = threading.Lock()
    request_slots = threading.Semaphore(args.clients)
    rows = []
    fields = ["context_id", "round", "ok", "request_id", "request_latency_s", "client_queue_s",
              "ttft_s", "prompt_tokens", "completion_tokens", "cached_tokens",
              "finish_reason", "error"]
    with (out / "events.jsonl").open("w") as events, (out / "requests.csv").open("w", newline="") as table:
        writer = csv.DictWriter(table, fieldnames=fields)
        writer.writeheader()

        def log(event, **data):
            item = dict(event=event, run_id=run_id, elapsed_s=time.perf_counter() - start,
                        utc=datetime.now(timezone.utc).isoformat(), **data)
            with lock:
                events.write(json.dumps(item) + "\n")
                events.flush()

        def client(session):
            cid = session["session_id"]
            arrival = session["arrival_offset_s"] * args.time_scale
            time.sleep(max(0, start + arrival - time.perf_counter()))
            messages = initial_messages(cid, args.prefix_lines)
            log("context_start", context_id=cid, planned_arrival_s=arrival,
                arrival_lateness_s=max(0, time.perf_counter() - start - arrival))
            for round_index, step in enumerate(session["steps"]):
                metadata = dict(context_id=cid, round=round_index)
                ready = time.perf_counter()
                log("request_ready", **metadata)
                request_slots.acquire()
                queue_s = time.perf_counter() - ready
                request_start = time.perf_counter()
                prompt = json.dumps(messages, sort_keys=True)
                log("request_start", **metadata,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    prompt_characters=len(prompt))
                row = dict(**metadata, ok=False, client_queue_s=queue_s)
                try:
                    result = stream_request(args.base_url, args.model, messages,
                                            args.max_tokens, args.socket_timeout)
                    usage = result["usage"]
                    details = usage.get("prompt_tokens_details") or {}
                    row.update(ok=True, request_id=result["request_id"],
                               request_latency_s=result["request_latency_s"],
                               ttft_s=result["ttft_s"],
                               prompt_tokens=usage.get("prompt_tokens"),
                               completion_tokens=usage.get("completion_tokens"),
                               cached_tokens=details.get("cached_tokens"),
                               finish_reason=result["finish_reason"], error="")
                    log("request_complete", **metadata, **result)
                    messages.append({"role": "assistant", "content": result["output_text"]})
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if isinstance(exc, urllib.error.HTTPError):
                        error += ": " + exc.read(4096).decode(errors="replace")
                    row.update(request_latency_s=time.perf_counter() - request_start,
                               error=error)
                    log("request_error", **metadata, error=error,
                        request_latency_s=row["request_latency_s"])
                finally:
                    request_slots.release()
                with lock:
                    rows.append(row)
                    writer.writerow(row)
                    table.flush()
                if not row["ok"]:
                    log("context_end", context_id=cid, completed=False)
                    return False
                if step["gap_after_s"] is not None:
                    gap = step["gap_after_s"] * args.time_scale
                    log("tool_start", **metadata, tool="synthetic_sleep",
                        source_tool_category=step.get("tool_category"),
                        source_gap_s=step["gap_after_s"], planned_gap_s=gap)
                    gap_start = time.perf_counter()
                    time.sleep(gap)
                    log("tool_end", **metadata, tool="synthetic_sleep",
                        actual_gap_s=time.perf_counter() - gap_start)
                    messages.append({"role": "user", "content":
                        f"Synthetic tool result after round {round_index}: validation completed. "
                        "Continue with a short technical summary."})
            log("context_end", context_id=cid, completed=True)
            return True

        with ThreadPoolExecutor(max_workers=len(sessions)) as executor:
            completed = list(executor.map(client, sessions))
    wall = time.perf_counter() - start
    good = [r for r in rows if r["ok"]]
    latencies = [r["request_latency_s"] for r in good]
    ttfts = [r["ttft_s"] for r in good if r["ttft_s"] is not None]
    summary = dict(workload=config["workload"], output=str(out),
                   attempted_requests=len(rows), successful_requests=len(good),
                   failed_requests=len(rows) - len(good),
                   completed_contexts=sum(completed), wall_time_s=wall,
                   successful_requests_per_s=len(good) / wall,
                   mean_successful_request_latency_s=statistics.mean(latencies) if latencies else None,
                   p95_successful_request_latency_s=percentile(latencies, .95),
                   p95_successful_request_ttft_s=percentile(ttfts, .95),
                   length_limited_requests=sum(r.get("finish_reason") == "length" for r in good),
                   note="Request throughput is not task goodput; latency excludes failed requests. "
                        "TTFT includes HTTP, queueing, prefill/restore and first-token delivery. "
                        "No separate queueing or GPU/CPU transfer timing is measured.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if all(completed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
