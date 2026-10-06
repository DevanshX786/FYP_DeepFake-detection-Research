import argparse
import os
import subprocess
import sys
import time
import psutil
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logging import get_logger


def get_gpu_metrics():
    """Retrieve GPU VRAM and Utilization metrics via torch or nvidia-smi."""
    vram_mb = 0.0
    gpu_util = 0.0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    
    # Query nvidia-smi if available
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, timeout=2
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = res.stdout.strip().split("\n")[0].split(",")
            vram_mb = float(parts[0].strip())
            gpu_util = float(parts[1].strip())
    except Exception:
        pass
    
    return vram_mb, gpu_util


def run_concurrency_benchmark(epochs: int = 1, batch_size: int = 8):
    logger = get_logger("concurrency_benchmark", log_file="experiments/concurrency_benchmark.log")
    logger.info("==================================================================")
    logger.info("      CONTROLLED 2-EXPERIMENT CONCURRENCY BENCHMARK (GPU)         ")
    logger.info("==================================================================")

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    logger.info(f"Target GPU: {device_name} ({total_gpu_vram:.2f} GB VRAM)")
    logger.info(f"Test Setup: Concurrent execution of EXP-1 (ResNet-BiLSTM) and EXP-2 (RGB ConvNeXt)")
    logger.info(f"Parameters: {epochs} epoch(s), batch_size={batch_size}, num_workers=0")

    # Command 1: EXP-1
    cmd1 = [
        sys.executable, "scripts/run_single_experiment.py",
        "--exp_id", "EXP-1",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", "0.0001",
        "--seed", "42",
        "--num_workers", "0",
        "--exp_dir", "experiments/benchmark_exp_1",
    ]

    # Command 2: EXP-2
    cmd2 = [
        sys.executable, "scripts/run_single_experiment.py",
        "--exp_id", "EXP-2",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", "0.0001",
        "--seed", "42",
        "--num_workers", "0",
        "--exp_dir", "experiments/benchmark_exp_2",
    ]

    t0 = time.time()
    p1 = subprocess.Popen(cmd1)
    p2 = subprocess.Popen(cmd2)

    logger.info(f"Launched Process 1 (PID {p1.pid}: EXP-1) and Process 2 (PID {p2.pid}: EXP-2)...")

    peak_vram_mb = 0.0
    peak_gpu_util = 0.0
    peak_cpu_util = 0.0
    peak_ram_gb = 0.0

    samples_count = 0
    sum_gpu_util = 0.0
    sum_vram = 0.0

    while p1.poll() is None or p2.poll() is None:
        vram_mb, gpu_util = get_gpu_metrics()
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_gb = psutil.virtual_memory().used / (1024**3)

        peak_vram_mb = max(peak_vram_mb, vram_mb)
        peak_gpu_util = max(peak_gpu_util, gpu_util)
        peak_cpu_util = max(peak_cpu_util, cpu_pct)
        peak_ram_gb = max(peak_ram_gb, ram_gb)

        if vram_mb > 0:
            sum_vram += vram_mb
            sum_gpu_util += gpu_util
            samples_count += 1

        time.sleep(2)

    ret1 = p1.wait()
    ret2 = p2.wait()
    wall_time = time.time() - t0

    avg_vram_mb = sum_vram / samples_count if samples_count > 0 else 0
    avg_gpu_util = sum_gpu_util / samples_count if samples_count > 0 else 0

    # 3500 train samples per experiment * 2 experiments = 7000 samples total
    total_samples = 3500 * 2
    agg_throughput = total_samples / wall_time if wall_time > 0 else 0

    logger.info("==================================================================")
    logger.info("               BENCHMARK RESULTS & METRICS                        ")
    logger.info("==================================================================")
    logger.info(f" Process 1 (EXP-1) Exit Code: {ret1}")
    logger.info(f" Process 2 (EXP-2) Exit Code: {ret2}")
    logger.info(f" Wall-clock Time (1 Epoch):  {wall_time:.2f} s ({wall_time/60:.2f} min)")
    logger.info(f" Peak GPU VRAM:              {peak_vram_mb:.2f} MB ({peak_vram_mb/1024:.2f} GB / {total_gpu_vram:.2f} GB)")
    logger.info(f" Avg GPU VRAM:               {avg_vram_mb:.2f} MB")
    logger.info(f" Peak GPU Utilization:       {peak_gpu_util:.1f} %")
    logger.info(f" Avg GPU Utilization:        {avg_gpu_util:.1f} %")
    logger.info(f" Peak CPU Utilization:       {peak_cpu_util:.1f} %")
    logger.info(f" Peak System RAM:            {peak_ram_gb:.2f} GB")
    logger.info(f" Aggregate Throughput:       {agg_throughput:.2f} videos/sec (~{agg_throughput*16:.1f} frames/sec)")
    logger.info("==================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 2-experiment concurrency benchmark.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    run_concurrency_benchmark(epochs=args.epochs, batch_size=args.batch_size)
