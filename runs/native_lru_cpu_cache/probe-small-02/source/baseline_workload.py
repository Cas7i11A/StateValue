"""Freeze deterministic token-ID workloads for native_lru_cpu_cache."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from trace_workload import extract_segments, file_sha256, open_trace

BASELINE = "native_lru_cpu_cache"
CPU_BYTES = 4 * 1024**3
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def common_prefix(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def synthetic_tokens(key, length, seed, token_bank):
    # SHA-256 rather than Python hash(): stable across processes and capacities.
    return [token_bank[int.from_bytes(hashlib.sha256(
        f"{seed}:{key}:{i}".encode()).digest()[:8], "big") % len(token_bank)]
        for i in range(length)]


def add_prompt(plan, tokens):
    key = digest(tokens)
    plan["prompts"][key] = tokens
    return key


def prepare_probe(plan, token_bank, prompt_tokens=4096, pushers=3):
    if prompt_tokens + plan["output_tokens"] + 16 > 8192:
        raise ValueError("Probe prompt plus output exceeds 8192")
    sessions = []
    first = None
    headers = set()
    for index in range(pushers + 1):
        tokens = synthetic_tokens(f"probe:{index}", prompt_tokens, plan["seed"], token_bank)
        header = tuple(tokens[:16])
        if header in headers:
            raise ValueError("Unexpected shared prefix block in probe")
        headers.add(header)
        ref = add_prompt(plan, tokens)
        if index == 0:
            first = ref
        if index == 0:
            sessions.append(dict(session_id="probe", arrival_offset_s=0, steps=[]))
        sessions[0]["steps"].append(dict(label="A_first" if index == 0 else f"pressure_{index}",
                                         prompt_ref=ref, gap_after_s=0.25))
    # A new suffix allows all of the original A to hit: vLLM caps prefix hits
    # below the final request token, which otherwise causes a tail-block miss.
    suffix = synthetic_tokens("probe:repeat-suffix", 16, plan["seed"], token_bank)
    repeat = add_prompt(plan, plan["prompts"][first] + suffix)
    sessions[0]["steps"].append(dict(label="A_repeat", prompt_ref=repeat, gap_after_s=None))
    plan.update(kind="pressure_probe", sessions=sessions,
                pressure=dict(pusher_unique_full_tokens=pushers * (prompt_tokens // 16) * 16,
                              prompt_tokens=prompt_tokens, pushers=pushers,
                              total_unique_prompt_tokens=(pushers + 1) * prompt_tokens + len(suffix),
                              a_shared_prefix_tokens=prompt_tokens, a_query_suffix_tokens=len(suffix)))
    cpu_upper = ((pushers + 1) * prompt_tokens + len(suffix) + (pushers + 2) * (plan["output_tokens"] + 16)) * plan["model"]["kv_bytes_per_token"]
    if cpu_upper >= CPU_BYTES:
        raise ValueError("Probe working set could exceed CPU cache budget")
    plan["pressure"]["cpu_working_set_upper_bytes"] = cpu_upper


def prepare_trace(plan, paths, token_bank, max_sessions=24, max_requests=5,
                  tokens_per_source_block=1, min_requests=3):
    excluded, transitions = Counter(), Counter()
    candidates, sources = [], []
    for path in paths:
        sources.append(dict(path=str(path), sha256=file_sha256(path)))
        with open_trace(path) as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    segments = extract_segments(raw, transitions)
                except (ValueError, TypeError) as exc:
                    excluded[str(exc)] += 1
                    continue
                request_by_step = {r["step"]: r for r in raw["requests"] if isinstance(r, dict) and "step" in r}
                for segment in segments:
                    steps = segment["steps"][:max_requests]
                    if len(steps) < min_requests:
                        excluded["too_short"] += 1
                        continue
                    blocks = [request_by_step[s["source_step"]].get("block_ids") for s in steps]
                    if any(not isinstance(b, list) or not b for b in blocks):
                        excluded["missing_block_ids"] += 1
                        continue
                    lengths = [len(b) * tokens_per_source_block for b in blocks]
                    if any(n < 128 or n + plan["output_tokens"] > 8192 for n in lengths):
                        excluded["scaled_length_out_of_bounds"] += 1
                        continue
                    candidates.append((segment, steps, blocks, raw["week"]))
    candidates.sort(key=lambda item: (item[0]["source_start_ms"], item[0]["session_id"]))
    sessions, used = [], set()
    for segment, steps, blocks, week in candidates:
        source_id = segment["source_session_id"]
        if source_id in used:
            continue
        used.add(source_id)
        session = dict(session_id=segment["session_id"], source_session_id=source_id,
                       source_start_ms=segment["source_start_ms"], steps=[])
        previous_tokens = previous_blocks = None
        for step, ids in zip(steps, blocks):
            tokens = []
            for block in ids:
                tokens.extend(synthetic_tokens(f"{week}:{digest(block)}", tokens_per_source_block,
                                               plan["seed"], token_bank))
            shared = common_prefix(previous_blocks, ids) if previous_blocks is not None else 0
            actual_shared = common_prefix(previous_tokens, tokens) if previous_tokens is not None else 0
            if actual_shared != shared * tokens_per_source_block:
                raise ValueError("Synthetic mapping changed prefix structure (token collision)")
            session["steps"].append(dict(source_step=step["source_step"],
                prompt_ref=add_prompt(plan, tokens), gap_after_s=step["gap_after_s"],
                source_full_block_tokens=len(ids) * 16, synthetic_prompt_tokens=len(tokens),
                source_shared_blocks=shared, synthetic_shared_tokens=actual_shared,
                source_block_ids_sha256=digest(ids)))
            previous_tokens, previous_blocks = tokens, ids
        session["steps"][-1]["gap_after_s"] = None
        sessions.append(session)
        if len(sessions) >= max_sessions:
            break
    if not sessions:
        raise ValueError(f"No eligible trace segments: {dict(excluded)}")
    origin = sessions[0]["source_start_ms"]
    for session in sessions:
        session["arrival_offset_s"] = (session["source_start_ms"] - origin) / 1000
    plan.update(kind="trace_driven_synthetic_replay", sessions=sessions, sources=sources,
                source_dataset="harvardMadsys/freeinference_agentic_trace", source_license="CC-BY-4.0",
                selection=dict(selected_sessions=len(sessions), unique_source_sessions=len(used),
                               eligible_segments=len(candidates), max_requests=max_requests,
                               min_requests=min_requests, excluded=dict(excluded),
                               excluded_transitions=dict(transitions)),
                length_mapping=dict(tokens_per_source_block=tokens_per_source_block,
                    source_tokens_per_block=16, scale=tokens_per_source_block / 16,
                    partial_source_blocks="not released; omitted",
                    overflow="exclude segment; no truncation",
                    note="One tool-chain segment per source session; original text/semantics unavailable."))


def load_workload(path):
    plan = json.loads(Path(path).read_text())
    if plan.get("schema_version") != 2 or plan.get("baseline") != BASELINE:
        raise ValueError("Expected native_lru_cpu_cache schema_version=2 plan")
    if plan.get("kind") not in ("pressure_probe", "trace_driven_synthetic_replay"):
        raise ValueError("Unknown workload kind")
    if plan.get("output_tokens") != 32 or not isinstance(plan.get("seed"), int):
        raise ValueError("Baseline requires a fixed integer seed and 32 output tokens")
    if not plan.get("sessions") or not plan.get("prompts"):
        raise ValueError("Empty workload")
    seen = set()
    for tokens_hash, tokens in plan["prompts"].items():
        if (not isinstance(tokens, list) or not tokens or len(tokens) + 32 > 8192
                or any(type(t) is not int or t < 0 for t in tokens) or digest(tokens) != tokens_hash):
            raise ValueError("Invalid prompt token IDs or hash")
    for session in plan["sessions"]:
        if session["session_id"] in seen:
            raise ValueError("Duplicate session ID")
        seen.add(session["session_id"])
        arrival = session["arrival_offset_s"]
        if not isinstance(arrival, (int, float)) or not 0 <= arrival < float("inf"):
            raise ValueError("Invalid arrival time")
        steps = session["steps"]
        if not steps or steps[-1]["gap_after_s"] is not None:
            raise ValueError("Invalid final step")
        for i, step in enumerate(steps):
            if step["prompt_ref"] not in plan["prompts"]:
                raise ValueError("Missing prompt")
            gap = step["gap_after_s"]
            if i < len(steps) - 1 and (not isinstance(gap, (int, float)) or not 0 <= gap < float("inf")):
                raise ValueError("Invalid tool gap")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "trace"):
        child = sub.add_parser(name)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--seed", type=int, default=42)
        if name == "probe":
            child.add_argument("--prompt-tokens", type=int, default=4096)
            child.add_argument("--pushers", type=int, default=3)
        else:
            child.add_argument("inputs", type=Path, nargs="+")
            child.add_argument("--max-sessions", type=int, default=24)
            child.add_argument("--max-requests", type=int, default=5)
            child.add_argument("--tokens-per-source-block", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if args.command == "probe" and (args.prompt_tokens < 256 or args.pushers < 1):
        parser.error("Probe requires prompt-tokens >= 256 and pushers >= 1")
    if args.command == "trace" and (args.max_sessions < 1 or not 3 <= args.max_requests <= 5
                                    or args.tokens_per_source_block < 1):
        parser.error("Require positive max-sessions, tokens-per-source-block and max-requests in [3,5]")
    from transformers import AutoConfig, AutoTokenizer
    from huggingface_hub import try_to_load_from_cache
    config_path = try_to_load_from_cache(MODEL, "config.json")
    revision = Path(config_path).parent.name if isinstance(config_path, str) else ""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        parser.error("Cannot resolve a cached model revision; download the model snapshot first")
    config = AutoConfig.from_pretrained(MODEL, revision=revision, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision, local_files_only=True)
    vocab = tokenizer.get_vocab()
    special = set(tokenizer.all_special_ids)
    bank = [t for t in sorted(set(vocab.values())) if t >= 256 and t not in special]
    bytes_per_token = 2 * config.num_hidden_layers * config.num_key_value_heads * (config.hidden_size // config.num_attention_heads) * 2
    plan = dict(schema_version=2, baseline=BASELINE, seed=args.seed, output_tokens=32,
                model=dict(name=MODEL, revision=revision, dtype="half", tensor_parallel_size=1,
                           kv_bytes_per_token=bytes_per_token, tokenizer_vocab_sha256=digest(vocab)),
                prompts={})
    if args.command == "probe":
        prepare_probe(plan, bank, args.prompt_tokens, args.pushers)
    else:
        prepare_trace(plan, args.inputs, bank, args.max_sessions, args.max_requests, args.tokens_per_source_block)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(plan, stream, separators=(",", ":"))
        stream.write("\n")
    load_workload(args.output)
    print(json.dumps(dict(output=str(args.output), kind=plan["kind"],
                         sessions=len(plan["sessions"]), requests=sum(len(s["steps"]) for s in plan["sessions"]),
                         model=plan["model"], selection=plan.get("selection"),
                         length_mapping=plan.get("length_mapping")), indent=2))


if __name__ == "__main__":
    main()
