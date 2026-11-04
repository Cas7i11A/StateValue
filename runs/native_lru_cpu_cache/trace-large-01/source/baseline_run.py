"""Run a cold-cache native LRU + CPU-cache experiment with owned local services."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.request

from baseline_workload import BASELINE, CPU_BYTES, load_workload
from tool_gap_harness import percentile, stream_request
from trace_workload import file_sha256

ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8000"
METRICS = ("vllm:num_preemptions_total", "vllm:prefix_cache_hits_total",
           "vllm:external_prefix_cache_hits_total", "vllm:kv_cache_usage_perc")


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def metric_values(text):
    result = {}
    for name in METRICS:
        values = []
        for line in text.splitlines():
            if re.match(re.escape(name) + r"(?:\{|\s)", line):
                value = float(line.split()[-1])
                if math.isfinite(value):
                    values.append(value)
        result[name] = sum(values) if values else None
    return result


def delta(before, after, name):
    a, b = before.get(name), after.get(name)
    return b - a if a is not None and b is not None and b >= a else None


def retrieved_tokens(text):
    values = re.findall(r"Retrieved\s+(\d+)\s+tokens", text)
    return sum(map(int, values))


def port_busy(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except ConnectionRefusedError:
        return False


class Services:
    """Own only the processes started here; never stop a preexisting service."""
    def __init__(self, out, kv_bytes, revision):
        self.out, self.kv_bytes, self.revision = out, kv_bytes, revision
        self.processes = []

    def launch(self, name):
        env = dict(os.environ, STATEVALUE_DIR=str(ROOT),
                   STATEVALUE_LOG_DIR=str(self.out / "services"),
                   STATEVALUE_KV_CACHE_BYTES=str(self.kv_bytes),
                   STATEVALUE_MODEL_REVISION=self.revision,
                   PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(["bash", str(ROOT / "statevalue.sh"), name], env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                                start_new_session=True)
        self.processes.append((name, proc))

    def wait_ready(self, name, timeout=600):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for service, proc in self.processes:
                if proc.poll() is not None:
                    raise RuntimeError(f"{service} exited with {proc.returncode}; inspect services/ logs")
            try:
                if name == "lmcache":
                    if port_busy(5555):
                        return
                else:
                    with urllib.request.urlopen(URL + "/health", timeout=2) as response:
                        if response.status == 200:
                            return
            except (OSError, TimeoutError):
                pass
            time.sleep(1)
        raise TimeoutError(f"{name} startup exceeded {timeout}s")

    def close(self):
        status = []
        for name, proc in reversed(self.processes):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=10)
            status.append(dict(name=name, pid=proc.pid, exit_code=proc.returncode))
        write_json(self.out / "service-lifecycle.json", dict(cold_start=True, stopped=utc(), services=status))

    def log(self, name):
        files = list((self.out / "services").glob(name + "-*.log"))
        if len(files) != 1:
            raise RuntimeError(f"Expected one {name} service log, got {len(files)}")
        return files[0]


class Metrics:
    def __init__(self, out):
        self.out = out
        self.stop = threading.Event()
        self.thread = None

    def snapshot(self, label):
        with urllib.request.urlopen(URL + "/metrics", timeout=5) as response:
            text = response.read().decode()
        if label:
            (self.out / f"metrics-{label}.prom").write_text(text)
        return metric_values(text)

    def start(self):
        def sample():
            with (self.out / "metrics.jsonl").open("w") as stream:
                while not self.stop.is_set():
                    item = dict(utc=utc())
                    try:
                        item["metrics"] = self.snapshot(None)
                    except Exception as exc:
                        item["error"] = str(exc)
                    stream.write(json.dumps(item) + "\n")
                    stream.flush()
                    self.stop.wait(1)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=10)


def replay(plan, out, services, clients, time_scale, socket_timeout):
    is_probe = plan["kind"] == "pressure_probe"
    slots, lock = threading.Semaphore(clients), threading.Lock()
    start = time.perf_counter()
    rows = []
    with (out / "events.jsonl").open("w") as events:
        def log(event, **data):
            with lock:
                events.write(json.dumps(dict(event=event, utc=utc(), elapsed_s=time.perf_counter() - start, **data)) + "\n")
                events.flush()

        def session_run(session):
            arrival = session["arrival_offset_s"] * time_scale
            time.sleep(max(0, start + arrival - time.perf_counter()))
            log("session_start", session_id=session["session_id"], planned_arrival_s=arrival)
            for index, step in enumerate(session["steps"]):
                row = dict(session_id=session["session_id"], step=index, label=step.get("label"),
                           prompt_sha256=step["prompt_ref"], ok=False)
                tokens = plan["prompts"][step["prompt_ref"]]
                ready = time.perf_counter()
                with slots:
                    row["client_queue_s"] = time.perf_counter() - ready
                    row["start_s"] = time.perf_counter() - start
                    log("request_start", **row, expected_prompt_tokens=len(tokens))
                    # Probe requests are serial, so this log range belongs to this request.
                    lmcache_log = services.log("lmcache") if is_probe else None
                    offset = lmcache_log.stat().st_size if lmcache_log else None
                    try:
                        result = stream_request(URL, plan["model"]["name"], [], 32, socket_timeout,
                                                prompt_token_ids=tokens, seed=plan["seed"], fixed_output=True)
                        usage = result["usage"]
                        row.update(request_latency_s=result["request_latency_s"], ttft_s=result["ttft_s"],
                                   prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                                   cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                                   finish_reason=result["finish_reason"], request_id=result["request_id"])
                        if row["prompt_tokens"] != len(tokens) or row["completion_tokens"] != 32:
                            raise RuntimeError("Server token counts differ from the frozen workload")
                        row["ok"] = True
                        log("request_complete", **row, output_text=result["output_text"])
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        log("request_error", **row)
                    if lmcache_log:
                        time.sleep(.3)
                        with lmcache_log.open("rb") as stream:
                            stream.seek(offset)
                            row["retrieved_tokens"] = retrieved_tokens(stream.read().decode(errors="replace"))
                    row["end_s"] = time.perf_counter() - start
                    with lock:
                        rows.append(row)
                if not row["ok"]:
                    return False
                if step["gap_after_s"] is not None:
                    gap = step["gap_after_s"] * time_scale
                    log("tool_start", session_id=session["session_id"], step=index, planned_gap_s=gap)
                    began = time.perf_counter()
                    time.sleep(gap)
                    log("tool_end", session_id=session["session_id"], step=index, actual_gap_s=time.perf_counter() - began)
            log("session_end", session_id=session["session_id"])
            return True

        with ThreadPoolExecutor(max_workers=len(plan["sessions"])) as executor:
            completed = list(executor.map(session_run, plan["sessions"]))
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else ["ok"]
    with (out / "requests.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    good = [row for row in rows if row["ok"]]
    return dict(attempted_requests=len(rows), successful_requests=len(good), failed_requests=len(rows) - len(good),
                expected_requests=sum(len(s["steps"]) for s in plan["sessions"]),
                completed_contexts=sum(completed), wall_time_s=time.perf_counter() - start,
                ttft_p50_s=percentile([r["ttft_s"] for r in good if r["ttft_s"] is not None], .5),
                ttft_p95_s=percentile([r["ttft_s"] for r in good if r["ttft_s"] is not None], .95),
                latency_p50_s=percentile([r["request_latency_s"] for r in good], .5),
                latency_p95_s=percentile([r["request_latency_s"] for r in good], .95),
                probe_a_repeat=next((r for r in rows if r["label"] == "A_repeat"), None))


def provenance():
    files = ("statevalue.sh", "baseline_run.py", "baseline_workload.py", "tool_gap_harness.py", "trace_workload.py")
    record = dict(source_sha256={name: file_sha256(ROOT / name) for name in files})
    for key, command in (("commit", ["git", "rev-parse", "HEAD"]),
                         ("working_tree_status", ["git", "status", "--short"])):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        record[key] = result.stdout.strip() if result.returncode == 0 else None
    return record


def acceptance(summary, expectation, kind):
    checks = dict(all_requests_successful=summary["successful_requests"] == summary["expected_requests"],
                  no_active_request_preemption=summary["preemptions_delta"] == 0)
    if kind == "pressure_probe":
        row = summary["probe_a_repeat"] or {}
        if expectation == "pressure":
            checks.update(pushers_exceed_gpu_capacity=summary["pusher_tokens"] > summary["gpu_capacity_tokens"],
                          a_repeat_restored=row.get("retrieved_tokens", 0) > 0)
        else:
            checks.update(working_set_fits_gpu=summary["probe_unique_tokens"] < summary["gpu_capacity_tokens"],
                          no_cpu_restore=summary["retrieved_tokens"] == 0,
                          gpu_prefix_reuse=bool((row.get("cached_tokens") or 0) > 0 or
                                               (summary["prefix_hit_tokens_delta"] or 0) > 0))
    elif expectation == "pressure":
        checks["cpu_restore_observed"] = summary["retrieved_tokens"] > 0
    else:
        checks["all_requests_fit_without_recycling"] = summary["cumulative_block_upper_tokens"] < summary["gpu_capacity_tokens"]
        checks["gpu_prefix_reuse"] = (summary["prefix_hit_tokens_delta"] or 0) > 0
    return dict(passed=all(checks.values()), checks=checks,
                note="Restoration under pressure is not a measurement of exact block LRU order.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--gpu-kv-bytes", type=int, required=True)
    parser.add_argument("--expect", choices=["pressure", "no-pressure"], required=True)
    parser.add_argument("--clients", type=int, default=1)
    parser.add_argument("--time-scale", type=float, default=1)
    parser.add_argument("--socket-timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = load_workload(args.plan)
    if args.gpu_kv_bytes <= 0 or args.clients < 1 or not math.isfinite(args.time_scale) or args.time_scale <= 0:
        parser.error("Require positive GPU bytes, clients and finite positive time-scale")
    if not math.isfinite(args.socket_timeout) or args.socket_timeout <= 0:
        parser.error("Require finite positive socket-timeout")
    if plan["kind"] == "pressure_probe" and (args.clients != 1 or args.time_scale != 1):
        parser.error("Probe must be sequential with clients=1 and time-scale=1")
    for port in (5555, 8000):
        if port_busy(port):
            parser.error(f"Port {port} already in use. Stop A/B services before this cold-cache experiment.")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "services").mkdir()
    shutil.copyfile(args.plan, out / "workload.json")
    (out / "source").mkdir()
    for name in provenance()["source_sha256"]:
        shutil.copyfile(ROOT / name, out / "source" / name)
    config = dict(baseline=BASELINE, workload_sha256=file_sha256(args.plan), seed=plan["seed"],
                  gpu_kv_bytes=args.gpu_kv_bytes, cpu_cache_bytes=CPU_BYTES, cpu_eviction="LRU",
                  gpu_eviction="vLLM native block-level LRU", ttl=None, predictor=None,
                  model=plan["model"], clients=args.clients, time_scale=args.time_scale,
                  output_tokens=32, ignore_eos=True, min_tokens=32, cold_start=True,
                  expectation=args.expect, started=utc(), **provenance())
    write_json(out / "config.json", config)
    services = Services(out, args.gpu_kv_bytes, plan["model"]["revision"])
    metrics = Metrics(out)
    try:
        print(f"Starting cold services: GPU KV {args.gpu_kv_bytes} bytes; logs: {out / 'services'}", flush=True)
        services.launch("lmcache")
        services.wait_ready("lmcache")
        services.launch("vllm")
        services.wait_ready("vllm")
        vllm_log = services.log("vllm").read_text(errors="replace")
        match = re.search(r"GPU KV cache size:\s*([\d,]+) tokens", vllm_log)
        if not match:
            raise RuntimeError("Cannot determine actual GPU KV token capacity from startup log")
        capacity = int(match[1].replace(",", ""))
        if plan["kind"] == "pressure_probe":
            pressure = plan["pressure"]
            if args.expect == "pressure" and pressure["pusher_unique_full_tokens"] <= capacity:
                raise RuntimeError("Probe pushers do not exceed measured GPU capacity")
            if args.expect == "no-pressure" and pressure["total_unique_prompt_tokens"] + 32 * sum(len(s["steps"]) for s in plan["sessions"]) >= capacity:
                raise RuntimeError("Probe working set does not fit control GPU capacity")
        before = metrics.snapshot("before")
        if before["vllm:num_preemptions_total"] is None:
            raise RuntimeError("Preemption metric unavailable; cannot accept sequential-pressure baseline")
        metrics.start()
        print(f"GPU capacity: {capacity} tokens. Replaying frozen workload.", flush=True)
        summary = replay(plan, out, services, args.clients, args.time_scale, args.socket_timeout)
        time.sleep(2)  # Let asynchronous metric/log updates settle before the final snapshot.
        after = metrics.snapshot("after")
        cpu_log = services.log("lmcache").read_text(errors="replace")
        summary.update(baseline=BASELINE, workload_sha256=config["workload_sha256"],
                       gpu_kv_bytes=args.gpu_kv_bytes, gpu_capacity_tokens=capacity, cpu_cache_bytes=CPU_BYTES,
                       retrieved_tokens=retrieved_tokens(cpu_log),
                       preemptions_delta=delta(before, after, "vllm:num_preemptions_total"),
                       prefix_hit_tokens_delta=delta(before, after, "vllm:prefix_cache_hits_total"),
                       eviction_count=None, transfer_bytes=None, lru_order_verified=False,
                       unmeasured=["eviction_count", "transfer_bytes", "exact_block_eviction_order"],
                       note="Client TTFT includes HTTP, queueing, restore/prefill. Latency percentiles use successful requests only.")
        summary["cumulative_block_upper_tokens"] = sum(
            ((len(plan["prompts"][step["prompt_ref"]]) + 32 + 15) // 16) * 16
            for session in plan["sessions"] for step in session["steps"])
        if plan["kind"] == "pressure_probe":
            summary.update(pusher_tokens=plan["pressure"]["pusher_unique_full_tokens"],
                           probe_unique_tokens=plan["pressure"]["total_unique_prompt_tokens"])
        summary["acceptance"] = acceptance(summary, args.expect, plan["kind"])
        write_json(out / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return 0 if summary["acceptance"]["passed"] else 1
    except BaseException as exc:
        write_json(out / "failure.json", dict(error=f"{type(exc).__name__}: {exc}", utc=utc()))
        raise
    finally:
        metrics.close()
        services.close()


if __name__ == "__main__":
    raise SystemExit(main())
