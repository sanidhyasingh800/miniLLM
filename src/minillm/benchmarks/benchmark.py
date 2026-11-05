# this module will directly be used to input the results of a run (in the form of a InferenceResponse)
# and will compute the expected metrics and writeback to a csv file
# essentially computes and stores the metric for a single run into a single row into the given csv
import numpy as np
import csv
from pathlib import Path
from minillm.schemas import InferenceResponse

def calculate_metrics(
    response: InferenceResponse,
    engine: str,
    trial: int,
    prompt_tokens: int,
    output_tokens: int,
    peak_gpu_memory_mb: float = 0.0
) -> dict:

    responses = response.requests
    t_init = response.t_init
    t_end = response.t_end

    total_latency_ms = (t_end - t_init) / 1e6
    total_latency_s = total_latency_ms / 1e3

    num_requests = len(responses)
    total_tokens = sum(req.n for req in responses)

    # Validate benchmark results
    assert num_requests > 0
    assert total_latency_s > 0
    assert all(req.n == output_tokens for req in responses)
    assert all(
        t_init <= req.t0 <= req.t1 <= req.t2 <= t_end
        for req in responses
    )

    # Throughput
    output_throughput = total_tokens / total_latency_s
    request_throughput = num_requests / total_latency_s

    # Per-request metrics (milliseconds)
    ttft_list = [
        (req.t1 - t_init) / 1e6
        for req in responses
    ]

    queue_delay_list = [
        (req.t0 - t_init) / 1e6
        for req in responses
    ]

    tpot_list = [
        (req.t2 - req.t1) / (req.n - 1) / 1e6
        for req in responses
    ]

    latency_list = [
        (req.t2 - t_init) / 1e6
        for req in responses
    ]

    return {
        "engine": engine,
        "trial": trial,
        "num_requests": num_requests,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_ms": total_latency_ms,
        "throughput_tok_s": output_throughput,
        "throughput_req_s": request_throughput,
        "mean_ttft_ms": float(np.mean(ttft_list)),
        "p50_ttft_ms": float(np.percentile(ttft_list, 50)),
        "p95_ttft_ms": float(np.percentile(ttft_list, 95)),
        "mean_tpot_ms": float(np.mean(tpot_list)),
        "mean_queue_ms": float(np.mean(queue_delay_list)),
        "mean_request_latency_ms": float(np.mean(latency_list)),
        "p95_request_latency_ms": float(np.percentile(latency_list, 95)),
        "peak_gpu_memory_mb": peak_gpu_memory_mb
    }


def save_result(metrics: dict, filename: str):
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    filepath = data_dir / filename
    write_header = not filepath.exists() or filepath.stat().st_size == 0

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(metrics)