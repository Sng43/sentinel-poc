"""Latency benchmark for the /predict inference path.

Times the work behind a clinical alert (LightGBM predict + SHAP TreeExplainer +
alert assembly) over many requests and reports the distribution. Two modes so you
can compare hardware/software environments:

    uv run python bench_latency.py                 # in-process (compute only), 200 reqs
    uv run python bench_latency.py 500             # in-process, custom count
    uv run python bench_latency.py --url URL        # over HTTP against a running server
    uv run python bench_latency.py 300 --url https://<user>-<space>.hf.space

In-process uses Starlette's TestClient (no network); --url uses httpx against a live
server (local uvicorn or the deployed Hugging Face Space), so it includes the real
HTTP stack. Prints a summary table and writes outputs/reports/latency_benchmark.json.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

N = 200
URL = None
_args = sys.argv[1:]
for i, a in enumerate(_args):
    if a == "--url":
        URL = _args[i + 1]
    elif a.isdigit():
        N = int(a)


def _percentile(sorted_ms: list[float], p: float) -> float:
    """Nearest-rank percentile (p in [0,100])."""
    if not sorted_ms:
        return 0.0
    k = max(0, min(len(sorted_ms) - 1, round(p / 100 * (len(sorted_ms) - 1))))
    return sorted_ms[k]


def _make_client():
    """A client exposing .get(path)/.post(path, json=) — same interface either way."""
    if URL:
        import httpx
        return httpx.Client(base_url=URL.rstrip("/"), timeout=30), f"HTTP {URL}"
    from fastapi.testclient import TestClient
    from backend.app import app
    return TestClient(app), "in-process (no network)"


def main() -> None:
    client, mode = _make_client()
    body = client.get("/sample").json()          # a real 198-feature patient-hour

    for _ in range(5):                            # warm up caches / connection
        client.post("/predict", json=body)

    latencies_ms: list[float] = []
    for _ in range(N):
        t0 = time.perf_counter()
        r = client.post("/predict", json=body)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200

    latencies_ms.sort()
    total_s = sum(latencies_ms) / 1000
    result = {
        "mode": mode,
        "machine": f"{platform.system()} {platform.machine()} · Python {platform.python_version()}",
        "iterations": N,
        "mean_ms": round(sum(latencies_ms) / N, 2),
        "p50_ms": round(_percentile(latencies_ms, 50), 2),
        "p95_ms": round(_percentile(latencies_ms, 95), 2),
        "p99_ms": round(_percentile(latencies_ms, 99), 2),
        "min_ms": round(latencies_ms[0], 2),
        "max_ms": round(latencies_ms[-1], 2),
        "throughput_req_per_s": round(N / total_s, 1),
    }

    print("\n=== /predict latency benchmark ===")
    for k, v in result.items():
        print(f"{k:>22}: {v}")

    out = Path(__file__).parent / "outputs" / "reports" / "latency_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
