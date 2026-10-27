"""Verify CPU-tier ablation matching and request-level effect reporting."""
import csv
from pathlib import Path
import tempfile
import unittest

from cpu_cache_compare import compare, paired_effect


class CPUComparisonTest(unittest.TestCase):
    def run_record(self, root, enabled):
        path = root / ("cpu" if enabled else "none")
        path.mkdir()
        fields = ["session_id", "step", "prompt_sha256", "ok", "retrieved_tokens", "ttft_s", "request_latency_s"]
        with (path / "requests.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow(dict(session_id="A", step=0, prompt_sha256="same", ok=True,
                                 retrieved_tokens=4096 if enabled else None,
                                 ttft_s=.05 if enabled else .25, request_latency_s=.45 if enabled else .65))
        config = dict(workload_sha256="same", seed=42, gpu_kv_bytes=268435456,
                      gpu_eviction="native LRU", gpu_prefix_caching=True, model={"revision": "same"},
                      clients=1, time_scale=.001, output_tokens=32, ignore_eos=True, min_tokens=32,
                      source_sha256={"runtime": "same"}, cpu_cache_mode="lmcache" if enabled else "none",
                      cpu_cache_bytes=4 * 1024**3 if enabled else 0,
                      kv_connector="LMCacheMPConnector" if enabled else None)
        summary = dict(gpu_capacity_tokens=9360, gpu_prefix_caching_enabled=True,
                       connector_disabled=not enabled, acceptance={"passed": True})
        lifecycle = dict(cold_start=True, services=[{"name": "vllm"}] + ([{"name": "lmcache"}] if enabled else []))
        return dict(path=str(path), config=config, summary=summary, lifecycle=lifecycle)

    def test_matching_and_direction_of_latency_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            cpu = self.run_record(Path(tmp), True)
            no_cpu = self.run_record(Path(tmp), False)
            report = compare(cpu, no_cpu, cpu, no_cpu)
            self.assertTrue(report["comparison_passed"])
            self.assertTrue(report["repeats_verified"])
            effect = report["paired_effect"]
            self.assertEqual(effect["cpu_restore_request_count"], 1)
            self.assertAlmostEqual(effect["cpu_restored_requests_ttft_s"]["median_no_cpu_minus_cpu"], .2)
            no_cpu["summary"]["gpu_capacity_tokens"] = 9344
            self.assertFalse(compare(cpu, no_cpu)["comparison_passed"])

    def test_rejects_wrong_service_set_and_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            cpu = self.run_record(Path(tmp), True)
            no_cpu = self.run_record(Path(tmp), False)
            no_cpu["lifecycle"]["services"].append({"name": "lmcache"})
            self.assertFalse(compare(cpu, no_cpu)["comparison_passed"])
            no_cpu["lifecycle"]["services"].pop()
            no_cpu["config"]["seed"] = 43
            self.assertFalse(compare(cpu, no_cpu)["comparison_passed"])

    def test_missing_restore_measurement_is_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            cpu = self.run_record(Path(tmp), True)
            no_cpu = self.run_record(Path(tmp), False)
            csv_path = Path(cpu["path"]) / "requests.csv"
            csv_path.write_text(csv_path.read_text().replace(",4096,", ",,"))
            self.assertIsNone(paired_effect(cpu, no_cpu)["cpu_restore_request_count"])


if __name__ == "__main__":
    unittest.main()
