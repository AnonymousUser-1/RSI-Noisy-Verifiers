#!/usr/bin/env python3
"""Run one job per GPU with a resumable queue and an allocation-time ceiling."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from rsi.common import digest, read_json, write_json


def launch(manifest_path, gpu_ids, execute, gpu_hours):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if len(set(gpu_ids)) != len(gpu_ids) or not gpu_ids:
        raise ValueError("GPU IDs must be unique and nonempty")
    if not execute:
        for job in manifest["jobs"]:
            print(job["id"], " ".join(job["argv"]))
        print("Dry run only. Add --execute after profiling; ceiling:", gpu_hours, "allocated GPU-hours")
        return
    if gpu_hours <= 0:
        raise ValueError("gpu-hours must be positive")
    root = Path(__file__).resolve().parent
    state_path = manifest_path.with_name("launch_state.json")
    state = read_json(state_path) if state_path.exists() else {"manifest_hash": digest(manifest), "done": [], "gpu_seconds": 0}
    if state["manifest_hash"] != digest(manifest):
        raise ValueError("Manifest changed since this queue was started")
    pending = [j for j in manifest["jobs"] if j["id"] not in state["done"]]
    active, failed = {}, []
    previous = time.monotonic()
    try:
        while pending or active:
            now = time.monotonic()
            state["gpu_seconds"] += (now-previous)*len(active)
            previous = now
            if state["gpu_seconds"] >= gpu_hours*3600:
                raise TimeoutError("Allocated GPU-hour ceiling reached; completed rounds remain resumable")
            for gpu in list(active):
                proc, job, log = active[gpu]
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                if code == 0:
                    state["done"].append(job["id"])
                else:
                    failed.append(job["id"])
                    print("FAILED", job["id"], "exit", code, flush=True)
            # Stop dispatching after a failure, but let other in-flight jobs finish.
            if failed:
                pending = []
            for gpu in gpu_ids:
                if gpu not in active and pending:
                    job = pending.pop(0)
                    logs = manifest_path.parent / "logs"
                    logs.mkdir(exist_ok=True)
                    log = (logs / (job["id"] + ".log")).open("a")
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1",
                               TOKENIZERS_PARALLELISM="false")
                    proc = subprocess.Popen([sys.executable] + job["argv"], cwd=root, env=env,
                                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    active[gpu] = (proc, job, log)
                    print("GPU", gpu, "started", job["id"], flush=True)
            write_json(state_path, state)
            if active:
                time.sleep(2)
        if failed:
            raise RuntimeError("Failed jobs (see logs): " + ", ".join(failed))
    finally:
        for proc, job, log in active.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            log.close()
        write_json(state_path, state)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--gpu-hours", type=float, default=40, help="Cumulative ceiling for THIS queue, including prior attempts")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    launch(a.manifest, a.gpus.split(","), a.execute, a.gpu_hours)
