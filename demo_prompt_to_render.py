# demo_prompt_to_render.py
import subprocess
import os
from pathlib import Path
import shutil

PROMPT = input("Enter your prompt: ")

INFER = [
    "python", str(Path("worker/infer.py").resolve()), PROMPT,
    "--model-dir", str(Path("Models/json_gen_model").resolve()),
    "--out-inbox", str(Path("worker/inbox").resolve())
]

print("\n→ Generating action JSON from prompt...")
subprocess.run(INFER, check=True)

# Get newest job file
inbox = Path("worker/inbox")
job = max(inbox.glob("job_*.json"), key=os.path.getctime)

print(f"→ Executing Blender with job {job.name} ...")

# discover blender: BLENDER_EXE env, common location or PATH
blender_exe = os.environ.get("BLENDER_EXE")
if not blender_exe:
    blender_exe = shutil.which("blender") or shutil.which("blender.exe")
if not blender_exe:
    # fallback to common hardcoded path (still override via BLENDER_EXE env)
    blender_exe = r"C:\Program Files\Blender Foundation\Blender\blender.exe"

blender_script = Path("worker/blender_executor.py").resolve()
job_path = Path(job).resolve()

cmd = [
    str(blender_exe),
    "--background",
    "--python", str(blender_script),
    "--",
    "--action-file", str(job_path),
    "--render-scene"
]

print("Running Blender:", " ".join(map(str, cmd)))
subprocess.run(cmd, check=True)

# Find newest render
outbox = Path("worker/outbox")
render = max(outbox.glob("render_*.png"), key=os.path.getctime)
print(f"\n Render complete → {render}")
