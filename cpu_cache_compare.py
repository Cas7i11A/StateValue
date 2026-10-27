"""Compare matched GPU-LRU runs with and without the LMCache CPU tier."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from baseline_compare import read_run
from tool_gap_harness import percentile


def request_rows(run):
    with (Path(run["path"]) / "requests.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    return {(r["session_id"], r["step"], r["prompt_sha256"]): r for r in rows}


def paired_effect(cpu, no_cpu):
    left, right = request_rows(cpu), request_rows(no_cpu)
    keys = sorted(set(left) & set(right))
    restore_observed = all(r.get("retrieved_tokens") not in (None, "") for r in left.values())
    restored = [k for k in keys if float(left[k].get("retrieved_tokens") or 0) > 0]
    result = dict(cpu_requests=len(left), no_cpu_requests=len(right), matched_requests=len(keys),
                  cpu_restore_request_count=len(restored) if restore_observed else None,
                  note="Groups are identified by CPU-run restores. Closed-loop interleaving may differ; matched requests need not have identical GPU hit/miss status.")
    for label, selected in (("all", keys), ("cpu_restored_requests", restored)):
        for field in ("ttft_s", "request_latency_s"):
            pairs = [(float(left[k][field]), float(right[k][field])) for k in selected
                     if left[k].get(field) and right[k].get(field) and left[k]["ok"] == right[k]["ok"] == "True"]
            differences = [b - a for a, b in pairs]
            result[f"{label}_{field}"] = dict(
                samples=len(pairs), cpu_p50=percentile([a for a, _ in pairs], .5),
                no_cpu_p50=percentile([b for _, b in pairs], .5),
                median_no_cpu_minus_cpu=percentile(differences, .5),
                mean_no_cpu_minus_cpu=statistics.mean(differences) if differences else None)
    return result


def compare(cpu, no_cpu, cpu_repeat=None, no_cpu_repeat=None):
    keys = ("workload_sha256", "seed", "gpu_kv_bytes", "gpu_eviction", "gpu_prefix_caching",
            "model", "clients", "time_scale", "output_tokens", "ignore_eos", "min_tokens", "source_sha256")
    def match(a, b):
        return all(k in a["config"] and k in b["config"] and a["config"][k] == b["config"][k] for k in keys)
    def service_names(run):
        return sorted(s["name"] for s in run["lifecycle"]["services"])
    checks = dict(matched_workload_and_runtime=match(cpu, no_cpu),
                  same_actual_gpu_capacity=cpu["summary"]["gpu_capacity_tokens"] == no_cpu["summary"]["gpu_capacity_tokens"],
                  cpu_mode=cpu["config"].get("cpu_cache_mode") == "lmcache" and cpu["config"]["cpu_cache_bytes"] == 4 * 1024**3,
                  no_cpu_mode=no_cpu["config"].get("cpu_cache_mode") == "none" and no_cpu["config"]["cpu_cache_bytes"] == 0,
                  gpu_prefix_caching_retained=cpu["config"].get("gpu_prefix_caching") is True and no_cpu["summary"]["gpu_prefix_caching_enabled"],
                  no_cpu_connector_disabled=no_cpu["config"].get("kv_connector") is None and no_cpu["summary"]["connector_disabled"],
                  service_sets_correct=service_names(cpu) == ["lmcache", "vllm"] and service_names(no_cpu) == ["vllm"],
                  both_cold=cpu["lifecycle"]["cold_start"] and no_cpu["lifecycle"]["cold_start"],
                  both_accepted=cpu["summary"]["acceptance"]["passed"] and no_cpu["summary"]["acceptance"]["passed"])
    effect = paired_effect(cpu, no_cpu)
    checks["all_requests_matched"] = effect["matched_requests"] == effect["cpu_requests"] == effect["no_cpu_requests"]
    for label, original, repeat in (("cpu", cpu, cpu_repeat), ("no_cpu", no_cpu, no_cpu_repeat)):
        if repeat:
            checks[label + "_repeat_matches"] = (match(original, repeat)
                and original["config"]["cpu_cache_mode"] == repeat["config"]["cpu_cache_mode"]
                and original["summary"]["gpu_capacity_tokens"] == repeat["summary"]["gpu_capacity_tokens"]
                and repeat["lifecycle"]["cold_start"] and repeat["summary"]["acceptance"]["passed"])
    fields = ("baseline", "successful_requests", "failed_requests", "gpu_kv_bytes", "gpu_capacity_tokens", "cpu_cache_bytes",
              "retrieved_tokens", "prefill_kv_computed_tokens_delta", "prefix_hit_tokens_delta", "preemptions_delta",
              "ttft_p50_s", "ttft_p95_s", "latency_p50_s", "latency_p95_s", "wall_time_s")
    runs = [cpu, no_cpu] + ([cpu_repeat] if cpu_repeat else []) + ([no_cpu_repeat] if no_cpu_repeat else [])
    report = dict(comparison="gpu_lru_cpu_cache_ablation", gpu_kv_bytes=cpu["config"]["gpu_kv_bytes"], checks=checks,
        comparison_passed=all(checks.values()), repeats_verified=bool(cpu_repeat and no_cpu_repeat and all(checks.values())),
        rows=[dict(path=r["path"], **{f: r["summary"].get(f) for f in fields}) for r in runs], paired_effect=effect,
        interpretation="Positive no_cpu-minus-cpu latency differences favor the CPU-cache system; this includes connector overhead, not isolated transfer cost.")
    if cpu_repeat and no_cpu_repeat:
        report["repeat_paired_effect"] = paired_effect(cpu_repeat, no_cpu_repeat)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--no-cpu", type=Path, required=True)
    parser.add_argument("--cpu-repeat", type=Path)
    parser.add_argument("--no-cpu-repeat", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    report = compare(read_run(args.cpu), read_run(args.no_cpu),
                     read_run(args.cpu_repeat) if args.cpu_repeat else None,
                     read_run(args.no_cpu_repeat) if args.no_cpu_repeat else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    return 0 if report["comparison_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
