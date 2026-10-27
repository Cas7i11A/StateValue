"""Check matched-budget baseline runs and emit a small reviewable comparison."""
import argparse
import json
from pathlib import Path


def read_run(path):
    return dict(path=str(path), config=json.loads((path / "config.json").read_text()),
                summary=json.loads((path / "summary.json").read_text()),
                lifecycle=json.loads((path / "service-lifecycle.json").read_text()))


def compare(small, large, repeat=None):
    keys = ("baseline", "workload_sha256", "seed", "cpu_cache_bytes", "cpu_eviction", "gpu_eviction",
            "model", "clients", "time_scale", "output_tokens", "ignore_eos", "min_tokens", "source_sha256")
    checks = {"matched_inputs_and_settings": all(k in small["config"] and k in large["config"] and
                                                small["config"][k] == large["config"][k] for k in keys),
              "different_gpu_budgets": small["config"]["gpu_kv_bytes"] < large["config"]["gpu_kv_bytes"],
              "both_cold_started": small["lifecycle"]["cold_start"] and large["lifecycle"]["cold_start"],
              "small_accepted": small["summary"]["acceptance"]["passed"],
              "large_accepted": large["summary"]["acceptance"]["passed"],
              "small_restored": small["summary"]["retrieved_tokens"] > 0,
              "more_restore_under_small_budget": small["summary"]["retrieved_tokens"] > large["summary"]["retrieved_tokens"]}
    if repeat:
        checks["repeat_matches_fixed_budget"] = all(k in repeat["config"] and small["config"].get(k) == repeat["config"][k]
                                                   for k in (*keys, "gpu_kv_bytes", "expectation"))
        checks["repeat_cold_and_accepted"] = repeat["lifecycle"]["cold_start"] and repeat["summary"]["acceptance"]["passed"]
    fields = ("successful_requests", "failed_requests", "gpu_kv_bytes", "gpu_capacity_tokens",
              "cpu_cache_bytes", "ttft_p50_s", "ttft_p95_s", "latency_p50_s", "latency_p95_s",
              "wall_time_s", "retrieved_tokens", "preemptions_delta", "eviction_count", "transfer_bytes")
    runs = [small, large] + ([repeat] if repeat else [])
    return dict(baseline="native_lru_cpu_cache", checks=checks, comparison_passed=all(checks.values()),
                repeatability_verified=bool(repeat and all(checks.values())),
                baseline_acceptance_complete=bool(repeat and all(checks.values())),
                rows=[dict(path=r["path"], **{k: r["summary"].get(k) for k in fields}) for r in runs],
                limitations=["Exact block LRU order is not verified", "Eviction count and transfer bytes are unmeasured",
                             "Synthetic token contents; scaled source lengths; client TTFT is not pure restore time"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small", type=Path, required=True)
    parser.add_argument("--large", type=Path, required=True)
    parser.add_argument("--repeat", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    report = compare(read_run(args.small), read_run(args.large), read_run(args.repeat) if args.repeat else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    return 0 if report["comparison_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
