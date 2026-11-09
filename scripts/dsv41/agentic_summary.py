import json
import os
import sys


def stat(d, key, field="avg"):
    v = d.get(key)
    return v.get(field) if isinstance(v, dict) else None


def counters(path):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and not line.startswith("#"):
                    name = parts[0].split("{", 1)[0]
                    if name in ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"):
                        out[name] = out.get(name, 0.0) + float(parts[-1])
    except OSError:
        pass
    return out


def window(export_path):
    try:
        with open(export_path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    start = ((d.get("summary") or {}).get("phase_time_ranges") or {}).get("profiling", {}).get("start_ns")
    out = {}
    for name in ("vllm:prefix_cache_hits", "vllm:prefix_cache_queries", "vllm:external_prefix_cache_hits"):
        total = 0.0
        for phase, sign in (("metrics", 1.0), ("warmup_metrics", -1.0)):
            for series in ((d.get(phase) or {}).get(name) or {}).get("series", []):
                if sign < 0 and not spans_warmup_of(d, name, start):
                    continue
                total += sign * float((series.get("stats") or {}).get("total", 0.0))
        out[name] = total
    if out.get("vllm:prefix_cache_queries", 0.0) <= 0:
        return None
    return out["vllm:prefix_cache_hits"], out["vllm:prefix_cache_queries"], out.get("vllm:external_prefix_cache_hits", 0.0)


def spans_warmup_of(d, name, start):
    if start is None:
        return False
    for series in ((d.get("metrics") or {}).get(name) or {}).get("series", []):
        ts = series.get("timeslices") or []
        if ts and ts[0].get("start_ns", start) < start:
            return True
    return False


def num(x, fmt="{:.1f}"):
    return "-" if x is None else fmt.format(x)


def main(argv):
    if len(argv) < 4:
        print("python3 agentic_summary.py PROFILE_EXPORT_AIPERF_JSON CONCURRENCY GPUS [METRICS_BEFORE METRICS_AFTER]")
        return 2
    path, c, gpus = argv[1], argv[2], max(1, int(argv[3]))
    with open(path) as f:
        d = json.load(f)
    total = stat(d, "total_token_throughput")
    output = stat(d, "output_token_throughput")
    itl90 = stat(d, "full_response_inter_token_latency", "p90")
    errors = d.get("error_summary") or []
    n_err = sum(int(e.get("count", 1)) if isinstance(e, dict) else 1 for e in errors)
    hit = "-"
    win = window(os.path.join(os.path.dirname(os.path.abspath(path)), "server_metrics_export.json"))
    if win is not None:
        h, q, x = win
        ext = f"; with external {(h + x) / q * 100:.1f}%" if x > 0 else ""
        win_s = f"{h / q * 100:.1f}% profiling window ({h / 1e6:.2f}M of {q / 1e6:.2f}M tokens{ext})"
    else:
        win_s = None
    whole = "-"
    if len(argv) >= 6:
        b, a = counters(argv[4]), counters(argv[5])
        h = a.get("vllm:prefix_cache_hits_total", 0.0) - b.get("vllm:prefix_cache_hits_total", 0.0)
        q = a.get("vllm:prefix_cache_queries_total", 0.0) - b.get("vllm:prefix_cache_queries_total", 0.0)
        if q > 0:
            whole = f"{h / q * 100:.1f}% ({h / 1e6:.2f}M of {q / 1e6:.2f}M tokens)"
    hit = f"{win_s}, whole leg {whole}" if win_s else f"{whole} whole leg (warmup included)"
    fields = [
        f"done {num(stat(d, 'request_count'), '{:.0f}')}",
        f"errors {n_err}",
        f"over {num(stat(d, 'benchmark_duration'), '{:.0f}')} s",
        f"total tok/s/GPU {num(None if total is None else total / gpus)}",
        f"output tok/s/GPU {num(None if output is None else output / gpus, '{:.2f}')}",
        f"interactivity (1/p90 ITL) {num(None if not itl90 else 1000.0 / itl90)} tok/s/user",
        f"TTFT p50 {num(stat(d, 'time_to_first_token', 'p50'), '{:.0f}')} ms p90 {num(stat(d, 'time_to_first_token', 'p90'), '{:.0f}')} ms",
        f"ISL {num(stat(d, 'input_sequence_length'), '{:.0f}')} OSL {num(stat(d, 'output_sequence_length'), '{:.0f}')}",
        f"prefix hit {hit}",
        f"theoretical {num(stat(d, 'theoretical_prefix_cache_hit'))}%",
    ]
    print(f"agentic c {c}: " + ", ".join(fields))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
