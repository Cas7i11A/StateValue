"""CPU-only integration checks for conversion and closed-loop HTTP replay."""
import json
import gzip
from collections import Counter
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from trace_workload import convert_session, extract_segments, load_plan, open_trace


ROOT = Path(__file__).resolve().parents[1]


def raw_session(cid="a", start=1000):
    return dict(session_id=cid, week="week-1", requests=[
        dict(step=0, turn_index=0, start_ms=start, end_ms=start + 100,
             model="m", provider="p", tool_calls=[dict(category="shell", subagent=None)]),
        dict(step=1, turn_index=0, step_trigger="tool", start_ms=start + 150,
             end_ms=start + 200, model="m", provider="p", tool_calls=[])])


class ConversionTest(unittest.TestCase):
    def test_extracts_turns_and_preserves_batch_wait(self):
        first = raw_session()
        second = raw_session(start=2000)
        for request in second["requests"]:
            request["step"] += 2
            request["turn_index"] = 1
        second["requests"][0]["step_trigger"] = "user"
        first["requests"][0]["tool_calls"].append(dict(category="read", subagent=None))
        first["requests"] += second["requests"]
        excluded = Counter()
        segments = extract_segments(first, excluded)
        self.assertEqual(len(segments), 2)
        self.assertEqual(excluded, {"not_serial_tool_chain": 1})
        self.assertEqual([s["source_start_ms"] for s in segments], [1000, 2000])
        self.assertEqual([step["source_step"] for step in segments[1]["steps"]], [2, 3])
        self.assertEqual(segments[0]["steps"][0]["tool_count"], 2)
        self.assertEqual(segments[0]["steps"][0]["gap_after_s"], .05)
        self.assertEqual(segments[0]["steps"][-1]["gap_after_s"], None)

    def test_invalid_transition_is_not_bridged(self):
        first = raw_session()
        second = raw_session(start=2000)
        for request in second["requests"]:
            request["step"] += 2
        second["requests"][0]["step_trigger"] = "tool"
        first["requests"][1]["tool_calls"] = [dict(category="shell", subagent={})]
        first["requests"] += second["requests"]
        excluded = Counter()
        segments = extract_segments(first, excluded)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["steps"][0]["source_step"], 2)
        self.assertEqual(excluded["subagent"], 2)

    def test_gzip_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl.gz"
            with gzip.open(path, "wt") as stream:
                stream.write(json.dumps(raw_session()) + "\n")
            with open_trace(path) as stream:
                self.assertEqual(convert_session(json.loads(next(stream)))["session_id"], "week-1:a")

    def test_reject_invalid_plan_before_health_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, output = Path(tmp) / "plan.json", Path(tmp) / "out"
            session = convert_session(raw_session())
            session["arrival_offset_s"] = 0
            session["steps"][0]["gap_after_s"] = -1
            plan.write_text(json.dumps(dict(schema_version=1, sessions=[session])))
            result = subprocess.run([sys.executable, str(ROOT / "tool_gap_harness.py"),
                                     "--trace-plan", str(plan), "--output", str(output)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Invalid gap_after_s", result.stderr)
            self.assertFalse(output.exists())

    def test_gap_and_exclusions(self):
        self.assertEqual(convert_session(raw_session())["steps"][0]["gap_after_s"], .05)
        for mutate, reason in [
            (lambda s: s["requests"][1].update(start_ms=1050), "overlapping_requests"),
            (lambda s: s["requests"][0]["tool_calls"][0].update(subagent={}), "subagent"),
            (lambda s: s["requests"][0]["tool_calls"].append({}), "parallel_tools"),
            (lambda s: s["requests"][1].update(step_trigger="user"), "not_serial_tool_chain"),
        ]:
            session = raw_session()
            mutate(session)
            with self.assertRaisesRegex(ValueError, reason):
                convert_session(session)

    def test_cli_selects_chronologically_and_records_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "raw.jsonl", Path(tmp) / "plan.json"
            source.write_text("\n".join(json.dumps(s) for s in [
                raw_session("late", 2000), {}, raw_session("early", 1000)]))
            result = subprocess.run([sys.executable, str(ROOT / "trace_workload.py"),
                                     str(source), "--output", str(output), "--max-sessions", "1"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = load_plan(output)
            self.assertEqual(plan["sessions"][0]["session_id"], "week-1:early:steps-0-1")
            self.assertEqual(plan["sessions"][0]["arrival_offset_s"], 0)
            self.assertEqual(plan["selection"]["excluded"], {"too_few_requests": 1})
            self.assertEqual(plan["selection"]["omitted_by_limit"], 1)


class ReplayTest(unittest.TestCase):
    def run_harness(self, trace=True, fail=False):
        lock = threading.Lock()
        state = dict(active=0, peak=0, calls=0)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with lock:
                    state["active"] += 1
                    state["calls"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                try:
                    time.sleep(.02)
                    self.send_response(500 if fail else 200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    if not fail:
                        chunk = dict(id="mock", choices=[dict(delta=dict(content="ok"),
                                                              finish_reason="stop")],
                                     usage=dict(prompt_tokens=10, completion_tokens=1))
                        self.wfile.write(("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())
                finally:
                    with lock:
                        state["active"] -= 1

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "run"
                command = [sys.executable, str(ROOT / "tool_gap_harness.py"),
                           "--base-url", f"http://127.0.0.1:{server.server_port}",
                           "--clients", "1", "--rounds", "2", "--gap-seconds", "0",
                           "--output", str(out)]
                if trace:
                    sessions = []
                    for cid, offset in [("a", 0), ("b", .01)]:
                        session = convert_session(raw_session(cid))
                        session["arrival_offset_s"] = offset
                        sessions.append(session)
                    plan = Path(tmp) / "plan.json"
                    plan.write_text(json.dumps(dict(schema_version=1, sessions=sessions)))
                    command += ["--trace-plan", str(plan), "--time-scale", "0.5"]
                result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 1 if fail else 0, result.stderr)
                summary = json.loads((out / "summary.json").read_text())
                events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
                self.assertEqual(state["peak"], 1)
                self.assertEqual(summary["completed_contexts"], 0 if fail else (2 if trace else 1))
                self.assertEqual(summary["attempted_requests"], (2 if trace else 1) * (1 if fail else 2))
                if trace:
                    self.assertTrue((out / "replay_plan.json").exists())
                if not fail:
                    for cid in {e["context_id"] for e in events}:
                        own = [e for e in events if e["context_id"] == cid]
                        completed = next(e for e in own if e["event"] == "request_complete")
                        next_ready = next(e for e in own if e["event"] == "request_ready" and e["round"] == 1)
                        self.assertGreaterEqual(next_ready["elapsed_s"] - completed["elapsed_s"], .025 if trace else 0)
                    if trace:
                        arrival = next(e for e in events if e["event"] == "context_start" and e["context_id"] == "week-1:b")
                        self.assertGreaterEqual(arrival["elapsed_s"], .005)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_trace_replay(self):
        self.run_harness()

    def test_original_smoke_mode(self):
        self.run_harness(trace=False)

    def test_failed_request_ends_session(self):
        self.run_harness(fail=True)


if __name__ == "__main__":
    unittest.main()
