#!/usr/bin/env python3
"""
inbox_watcher.py — poll worker/inbox for new job JSON files and run Blender headless
Usage: python inbox_watcher.py --inbox worker/inbox --blender "C:/Path/to/blender.exe" --worker worker/blender_executor.py
"""

import argparse
import time
import subprocess
from pathlib import Path
import shutil
import os

def call_blender_job(blender_exe: str, worker_script: str, job_path: Path):
    cmd = [
        blender_exe,
        "--background",
        "--python", str(worker_script),
        "--",
        "--action-file", str(job_path),
        "--render-scene"
    ]

    print("Running:", " ".join(map(str, cmd)))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Blender failed:", e)

def find_blender(provided: str | None) -> str:
    if provided:
        return provided
    # env override
    env = os.environ.get("BLENDER_EXE")
    if env:
        return env
    # PATH lookup
    exe = shutil.which("blender") or shutil.which("blender.exe")
    if exe:
        return exe
    # common fallback
    return r"C:\Program Files\Blender Foundation\Blender\blender.exe"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inbox", default="worker/inbox")
    p.add_argument("--poll-interval", type=float, default=2.0)
    p.add_argument("--blender", default=None, help="Path to blender executable (or set BLENDER_EXE env)")
    p.add_argument("--worker", default="worker/blender_executor.py")
    args = p.parse_args()

    blender_exe = find_blender(args.blender)
    worker_script = Path(args.worker).resolve()

    inbox = Path(args.inbox)
    inbox.mkdir(parents=True, exist_ok=True)
    processing = inbox.parent / "processing"
    processing.mkdir(parents=True, exist_ok=True)
    done = inbox.parent / "done"
    done.mkdir(parents=True, exist_ok=True)

    print("Watching", inbox, " — Blender:", blender_exe)
    try:
        while True:
            files = sorted([p for p in inbox.glob("*.json") if p.is_file()])
            for f in files:
                print("Found job:", f.name)
                target = processing / f.name
                try:
                    shutil.move(str(f), str(target))
                except Exception:
                    target = f  # if move fails, use original
                call_blender_job(blender_exe, worker_script, target)
                # after successful run, move to done
                dest_done = done / target.name
                try:
                    shutil.move(str(target), str(dest_done))
                except Exception:
                    print("Warning: could not move to done:", target)
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("Inbox watcher exiting on user interrupt.")

if __name__ == "__main__":
    main()
