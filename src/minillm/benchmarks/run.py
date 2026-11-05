# this is the main orchestrator for running our decided upon runs (see documentations)
# we should be able to provide the engine and the output csv file we want and it will do the rest
# First it initializes the loads up the model on the device (default to cpu if gpu not avail)
# Build all experiment configs (N, P, O)
# execute each config
#   a single cold run on the model + 3 trials per each config
# save each trial to csv
import argparse
import torch
from pathlib import Path

from dataclasses import dataclass
from itertools import product

from .workloads import get_workload
from .benchmark import calculate_metrics, save_result
from minillm.engines import v0Engine
from minillm.setup import load_model

# pythons way of making a struct ig
@dataclass(frozen = True)
class Run:
    n: int # number of requests
    p: int # input size
    o: int # output size

def get_configs(stress:bool = False)->list[Run]:

    #Experiment 1: Request Scaling:
    configs = {Run(n, 32, 32) for n in [1,2,4,8,16,32,64,128,256]}
    #Experiment 2: Outut Scaling
    configs.update( Run(1, 64, o) for o in [16,32,64,128,256,512])
    configs.update( Run(8, 64, o) for o in [16,32,64,128,256,512])
    #Experiment 3: Input Scaling: 
    configs.update( Run(1, p, 32) for p in [16, 64, 128])
    configs.update( Run(8, p, 32) for p in [16, 64, 128])
    #Experiment 4: Cross product of scaling requests and outputs
    configs.update( Run(n, 64, o) for n, o in product([1, 8, 32, 128],[16, 64, 256]))
    # Experiment 5: highly scaled requests
    if stress:
        configs.update(Run(n, 32, 32) for n in [512, 1024, 4096, 8192])

    return sorted(configs, key = lambda run: (run.n, run.p, run.o))


def run_benchmarks(engine, engine_name, tokenizer, filename, stress = False):
    configs = get_configs(stress)
    for config in configs:
        n,p,o = config.n, config.p, config.o
        print(f"Running: N={n}, P={p}, O={o}")
        warmup_requests = get_workload(tokenizer, 1, p)
        engine.inference(warmup_requests, max_new_tokens=o)
        for trial in range(1,4):
            # Reset memory tracking before inference
            if engine.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(engine.device)
            requests =  get_workload(tokenizer, n, p)
            response = engine.inference(requests, max_new_tokens = o)
            peak_mb = 0.0

            if engine.device.type == "cuda":
                torch.cuda.synchronize(engine.device)
                peak_mb = (torch.cuda.max_memory_allocated(engine.device) / 1e6)

            metrics = calculate_metrics(response, engine_name, trial, p, o,peak_gpu_memory_mb=peak_mb)
            save_result(metrics, filename)

# argparse, thanks gpt
# provide the engine version, the output csv file name and stress or not
def main():
    parser = argparse.ArgumentParser(
        description="MiniLLM Inference Benchmarker"
    )

    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--stress", action="store_true")

    args = parser.parse_args()

    # Load model once
    model, tokenizer, device = load_model()

    # Select engine
    engines = {
        "v0": v0Engine,
    }

    if args.engine not in engines:
        raise ValueError(f"Unknown engine: {args.engine}")

    engine = engines[args.engine](model, tokenizer, device)

    # Prepare output CSV
    filename = Path(args.output).name
    output_path = Path(__file__).parent / "data" / filename
    output_path.unlink(missing_ok=True)

    print(f"Engine: {args.engine}")
    print(f"Device: {device}")
    print(f"Output: {output_path}")

    run_benchmarks(
        engine=engine,
        engine_name=args.engine,
        tokenizer=tokenizer,
        filename=filename,
        stress=args.stress
    )


if __name__ == "__main__":
    main()