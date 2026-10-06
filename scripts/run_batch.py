import argparse
import subprocess
import sys
import time

def run_batch(exp_ids, epochs=30, batch_size=8, lr=1e-4, seed=42, num_workers=4):
    print("==================================================================")
    print(f"            EXECUTING CONCURRENT BATCH: {exp_ids}                 ")
    print("==================================================================")
    
    processes = []
    for exp_id in exp_ids:
        cmd = [
            sys.executable, "scripts/run_single_experiment.py",
            "--exp_id", exp_id,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--lr", str(lr),
            "--seed", str(seed),
            "--num_workers", str(num_workers),
            "--exp_dir", f"experiments/{exp_id.lower().replace('-', '_')}",
        ]
        p = subprocess.Popen(cmd)
        processes.append((exp_id, p))
        print(f"Launched {exp_id} (PID: {p.pid}) -> experiments/{exp_id.lower().replace('-', '_')}")
    
    t0 = time.time()
    exit_codes = {}
    for exp_id, p in processes:
        ret = p.wait()
        exit_codes[exp_id] = ret
        print(f"Experiment {exp_id} finished with Exit Code: {ret}")
    
    total_time = time.time() - t0
    print("==================================================================")
    print(f" Batch {exp_ids} Completed in {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f" Exit Codes: {exit_codes}")
    print("==================================================================")
    
    all_success = all(code == 0 for code in exit_codes.values())
    if not all_success:
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a concurrent batch of experiments.")
    parser.add_argument("--exp_ids", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    run_batch(
        exp_ids=args.exp_ids,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
    )
