"""Baseline invariants: frozen prefixes, trace scaling and truthful measurements."""
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from baseline_run import acceptance, delta, metric_values, retrieved_tokens
from baseline_workload import BASELINE, common_prefix, load_workload, prepare_probe, prepare_trace
from baseline_compare import compare
from tool_gap_harness import stream_request


def plan():
    return dict(schema_version=2, baseline=BASELINE, seed=42, output_tokens=32,
                model=dict(kv_bytes_per_token=28672), prompts={})


class WorkloadTest(unittest.TestCase):
    def test_probe_is_identical_on_repeat_and_pushers_are_distinct(self):
        first, second = plan(), plan()
        bank = list(range(256, 4096))
        prepare_probe(first, bank)
        prepare_probe(second, bank)
        self.assertEqual(first, second)
        steps = first["sessions"][0]["steps"]
        first_tokens = first["prompts"][steps[0]["prompt_ref"]]
        repeat_tokens = first["prompts"][steps[-1]["prompt_ref"]]
        self.assertEqual(repeat_tokens[:len(first_tokens)], first_tokens)
        self.assertEqual(len(repeat_tokens) - len(first_tokens), 16)
        self.assertEqual(len({s["prompt_ref"] for s in steps[:-1]}), 4)
        self.assertGreater(first["pressure"]["pusher_unique_full_tokens"], 268435456 // 28672)
        self.assertLess(first["pressure"]["cpu_working_set_upper_bytes"], 4 * 1024**3)

    def test_trace_keeps_arrivals_gaps_and_scaled_common_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            sessions = []
            for sid, offset in (("one", 0), ("two", 10000)):
                requests = []
                for step in range(3):
                    requests.append(dict(step=step, turn_index=0, step_trigger="tool" if step else "user",
                        model="source-model", provider="p", start_ms=1000 + offset + step * 1000,
                        end_ms=1100 + offset + step * 1000,
                        block_ids=list(range(128 + step * 16)),
                        tool_calls=[dict(category="shell", subagent=None)]))
                sessions.append(dict(session_id=sid, week="w", requests=requests))
            path.write_text("\n".join(map(json.dumps, sessions)))
            work = plan()
            prepare_trace(work, [path], list(range(256, 65536)), max_requests=3)
            self.assertEqual(len(work["sessions"]), 2)
            self.assertEqual(work["sessions"][1]["arrival_offset_s"], 10)
            steps = work["sessions"][0]["steps"]
            self.assertEqual(steps[0]["gap_after_s"], .9)
            self.assertEqual(steps[-1]["gap_after_s"], None)
            self.assertEqual(steps[1]["synthetic_shared_tokens"], 128)
            self.assertEqual(len(work["prompts"][steps[1]["prompt_ref"]]), 144)
            self.assertEqual(common_prefix(work["prompts"][steps[0]["prompt_ref"]],
                                           work["prompts"][steps[1]["prompt_ref"]]), 128)
            output = Path(tmp) / "plan.json"
            output.write_text(json.dumps(work))
            load_workload(output)
            work["prompts"][steps[0]["prompt_ref"]][0] += 1
            output.write_text(json.dumps(work))
            with self.assertRaisesRegex(ValueError, "hash"):
                load_workload(output)


class MeasurementTest(unittest.TestCase):
    def test_comparison_rejects_changed_seed_and_requires_repeat_for_completion(self):
        def run(budget, restore):
            config = dict.fromkeys(("baseline", "workload_sha256", "cpu_cache_bytes", "cpu_eviction", "gpu_eviction",
                                    "model", "clients", "time_scale", "output_tokens", "ignore_eos", "min_tokens", "source_sha256"), "same")
            config.update(gpu_kv_bytes=budget, seed=42, expectation="pressure" if restore else "no-pressure")
            return dict(path="test", lifecycle=dict(cold_start=True),
                        config=config,
                        summary=dict(retrieved_tokens=restore, acceptance=dict(passed=True)))
        small, large, repeat = run(256, 4096), run(12000, 0), run(256, 4096)
        self.assertFalse(compare(small, large)["baseline_acceptance_complete"])
        self.assertTrue(compare(small, large, repeat)["baseline_acceptance_complete"])
        repeat["config"]["seed"] = 43
        self.assertFalse(compare(small, large, repeat)["comparison_passed"])

    def test_missing_metrics_are_not_zero_and_resets_are_unknown(self):
        before = metric_values('vllm:num_preemptions_total{engine="0"} 3\n')
        after = metric_values('vllm:num_preemptions_total{engine="0"} 4\n')
        self.assertEqual(delta(before, after, "vllm:num_preemptions_total"), 1)
        self.assertIsNone(before["vllm:kv_cache_usage_perc"])
        self.assertIsNone(delta(after, before, "vllm:num_preemptions_total"))
        self.assertEqual(retrieved_tokens('Retrieved 96 tokens\nPrefetch 100 retained keys\nRetrieved 16 tokens'), 112)

    def test_pressure_acceptance_requires_a_restore_and_no_preemption(self):
        summary = dict(successful_requests=5, expected_requests=5, preemptions_delta=0,
                       pusher_tokens=12288, gpu_capacity_tokens=9360,
                       probe_a_repeat=dict(retrieved_tokens=4096))
        self.assertTrue(acceptance(summary, "pressure", "pressure_probe")["passed"])
        summary["probe_a_repeat"]["retrieved_tokens"] = 0
        self.assertFalse(acceptance(summary, "pressure", "pressure_probe")["passed"])
        summary["probe_a_repeat"]["retrieved_tokens"] = 4096
        summary["preemptions_delta"] = None
        self.assertFalse(acceptance(summary, "pressure", "pressure_probe")["passed"])

    def test_raw_token_requests_force_exact_generation_length_and_seed(self):
        chunk = dict(id="test", choices=[dict(text="hello", finish_reason="length")],
                     usage=dict(prompt_tokens=3, completion_tokens=32))
        response = io.BytesIO(("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())
        with patch("urllib.request.urlopen", return_value=response) as call:
            result = stream_request("http://test", "model", [], 32, 5,
                                    prompt_token_ids=[300, 301, 302], seed=42, fixed_output=True)
        payload = json.loads(call.call_args.args[0].data)
        self.assertEqual(payload["prompt"], [300, 301, 302])
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["min_tokens"], 32)
        self.assertTrue(payload["ignore_eos"])
        self.assertEqual(result["output_text"], "hello")


class ServiceLaunchTest(unittest.TestCase):
    def test_capacity_and_revision_are_forwarded_and_invalid_capacity_rejected(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            bindir = project / ".venv/bin"
            bindir.mkdir(parents=True)
            (bindir / "activate").write_text(f'export PATH="{bindir}:$PATH"\n')
            stub = bindir / "vllm"
            stub.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
            stub.chmod(0o755)
            env = dict(os.environ, STATEVALUE_DIR=tmp, STATEVALUE_KV_CACHE_BYTES="268435456",
                       STATEVALUE_MODEL_REVISION="pinned-revision")
            result = subprocess.run(["bash", str(root / "statevalue.sh"), "vllm"], env=env,
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--kv-cache-memory-bytes\n268435456", result.stdout)
            self.assertIn("--revision\npinned-revision", result.stdout)
            self.assertIn("--tokenizer-revision\npinned-revision", result.stdout)
            env["STATEVALUE_KV_CACHE_BYTES"] = "0"
            result = subprocess.run(["bash", str(root / "statevalue.sh"), "vllm"], env=env,
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("--kv-cache-memory-bytes", result.stdout)


if __name__ == "__main__":
    unittest.main()
