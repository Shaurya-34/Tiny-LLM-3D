#!/usr/bin/env python3
"""
infer.py — produce a canonical job JSON from a natural-language prompt
(using a local transformers causal LM) and push it to worker/inbox/.

Features:
- Loads a local fine-tuned HF-style model directory on Windows safely.
- Tries model generation first; falls back to deterministic rule-based parser.
- Color and primitive alias maps for robust fallback generation.
- Validates job payload against worker/action_schema.json when present.
- Optionally calls Blender to run the job.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
import re
from pathlib import Path
from typing import Any, Dict, Optional, List
from contextlib import nullcontext

# transformers / torch imports (optional runtime requirement if you will use model)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    import torch  # type: ignore
except Exception:
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    torch = None  # type: ignore

# create a safe no_grad context: uses torch.no_grad when torch available, otherwise no-op
_no_grad_ctx = (torch.no_grad if torch is not None else nullcontext)

# ----------------------------
# Color / Primitive Alias Maps
# ----------------------------
COLOR_ALIASES: Dict[str, List[float]] = {
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
    "yellow": [1.0, 1.0, 0.0],
    "purple": [0.5, 0.0, 0.5],
    "orange": [1.0, 0.5, 0.0],
    "silver": [0.75, 0.75, 0.75],
    "gold": [1.0, 0.84, 0.0],
    "white": [1.0, 1.0, 1.0],
    "black": [0.0, 0.0, 0.0],
}

PRIMITIVE_ALIASES: Dict[str, str] = {
    "cube": "CUBE",
    "box": "CUBE",
    "block": "CUBE",
    "sphere": "SPHERE",
    "ball": "SPHERE",
    "cone": "CONE",
    "cylinder": "CYLINDER",
    "torus": "TORUS",
    "donut": "TORUS",
    "plane": "PLANE",
    "ground": "PLANE",
}

# ----------------------------
# Prompt builder
# ----------------------------
def build_prompt(intent: str, retrieved: list) -> str:
    return "USER_INTENT: " + intent + "\nRETRIEVED: " + json.dumps(retrieved) + "\nOUTPUT:"


def normalize_legacy_payload(payload: Dict) -> Dict:
    """
    Convert legacy/alternate action shapes into canonical {"version":"1.0","actions":[...]} entries.
    - add_object -> create_primitive
    - object_type/type -> primitive (uppercased)
    - id -> name
    - color string -> RGB using COLOR_ALIASES (if available)
    """
    if not isinstance(payload, dict):
        return payload
    actions = payload.get("actions")
    if not isinstance(actions, list):
        return payload

    normalized_actions: List[Dict[str, Any]] = []
    for a in actions:
        if not isinstance(a, dict):
            normalized_actions.append(a)
            continue

        act_name = a.get("action") or a.get("type")
        # legacy add_object -> create_primitive
        if act_name == "add_object":
            primitive = a.get("object_type") or a.get("type") or a.get("primitive") or "CUBE"
            if isinstance(primitive, str):
                primitive = primitive.upper()
            name = a.get("id") or a.get("name") or f"{str(primitive).lower()}_{uuid.uuid4().hex[:6]}"
            # position: try common keys
            position = a.get("position") or a.get("location")
            # color: convert named color to rgb if possible
            color = a.get("color")
            if isinstance(color, str):
                color = COLOR_ALIASES.get(color.lower(), color)
            # build canonical action
            new: Dict[str, Any] = {
                "action": "create_primitive",
                "type": "create_primitive",
                "primitive": primitive,
                "name": name,
            }
            if position is not None:
                new["position"] = position
            if color is not None:
                new["color"] = color
            if "animation" in a:
                new["animation"] = a["animation"]
            # copy any other safe-known fields if present
            for k in ("rotation", "scale"):
                if k in a:
                    new[k] = a[k]
            normalized_actions.append(new)
        else:
            # Already canonical or other action — but also map shorthand fields if useful:
            # e.g., support legacy 'id' -> 'name' for other actions
            if "id" in a and "name" not in a:
                a["name"] = a["id"]
            # if color provided as name convert it
            if "color" in a and isinstance(a["color"], str):
                a["color"] = COLOR_ALIASES.get(a["color"].lower(), a["color"])
            normalized_actions.append(a)

    payload["actions"] = normalized_actions
    return payload


# ----------------------------
# Utilities
# ----------------------------
def extract_first_json(s: str) -> str:
    """Find the first balanced JSON object in a string and return it (as text)."""
    start = s.find("{")
    if start == -1:
        raise ValueError("No JSON object found in output.")
    depth = 0
    in_string = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_string = False
        else:
            if ch == "\"":
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    raise ValueError("Incomplete JSON object in model output.")


def atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
    os.replace(str(tmp), str(path))


def normalize_model_out(obj: Any) -> Dict:
    """Ensure job payload is {"version":"1.0","actions":[...]}."""
    if isinstance(obj, dict) and "actions" in obj and isinstance(obj["actions"], list):
        return obj
    if isinstance(obj, dict):
        return {"version": "1.0", "actions": [obj]}
    if isinstance(obj, list):
        return {"version": "1.0", "actions": obj}
    raise ValueError("Extracted JSON must be an object or an array of actions.")


# ----------------------------
# Model loader (Windows-friendly)
# ----------------------------
def load_model_and_tokenizer(model_dir: str, device) -> Any:
    """
    Load tokenizer & model from a local folder. If transformers/torch are not
    installed or the model doesn't load, raise an exception for the caller to
    handle (caller may fallback to rule-based).
    """
    if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
        raise RuntimeError("transformers/torch not available in this Python environment.")

    p = Path(model_dir)
    if p.exists():
        # Use concrete filesystem path (transformers handles absolute path fine)
        model_path = str(p.resolve())
    else:
        model_path = model_dir  # maybe a HF repo id (not typical for local-only)

    print(f"→ Loading model and tokenizer from local path: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    except Exception as e:
        print(f"Failed to load model/tokenizer from {model_path}: {e}")
        raise

    # move model to device and prepare tokens
    model.to(device)
    model.eval()

    # ensure eos/pad tokens exist
    if getattr(tokenizer, "eos_token", None) is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


# ----------------------------
# Rule-based fallback
# ----------------------------
def rule_based_payload_from_prompt(prompt: str) -> Dict:
    """Construct a conservative payload from a prompt deterministically."""
    pl: Dict[str, Any] = {"version": "1.0", "actions": []}
    low = prompt.lower()

    # delete all / clear intent
    if any(kw in low for kw in ("delete all", "clear scene", "remove all", "wipe", "delete everything")):
        pl["actions"].append({"action": "delete_all"})

    # primitive detection
    primitive = "CUBE"
    for word, prim in PRIMITIVE_ALIASES.items():
        if re.search(rf"\b{re.escape(word)}\b", low):
            primitive = prim
            break

    # color detection
    color = None
    for name, rgb in COLOR_ALIASES.items():
        if re.search(rf"\b{re.escape(name)}\b", low):
            color = rgb
            break
    if color is None:
        color = [1.0, 1.0, 1.0]

    # parse simple tuple "(x,y,z)" or "at x,y,z"
    pos = None
    m = re.search(r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)", prompt)
    if m:
        pos = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
    else:
        m2 = re.search(r"at\s+(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", low)
        if m2:
            pos = [float(m2.group(1)), float(m2.group(2)), float(m2.group(3))]
    if pos is None:
        pos = [0.0, 0.0, 0.0]

    create = {
        "action": "create_primitive",
        "type": "create_primitive",
        "primitive": primitive,
        "name": f"{primitive.lower()}_{uuid.uuid4().hex[:6]}",
        "position": pos,
        "color": color,
    }

    if "spin" in low or "rotate" in low:
        create["animation"] = {"name": "spin_slow", "speed": 1.57}

    pl["actions"].append(create)
    return pl


# ----------------------------
# Model inference
# ----------------------------
def generate_json_from_prompt(tokenizer, model, prompt: str, device, max_new_tokens: int = 256) -> Dict:
    """
    Tokenize prompt, run model.generate (no grad if torch present), extract first JSON object from output.
    """
    # prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    # move tensors to device if torch is present and device is a torch.device
    if torch is not None and hasattr(inputs, "to"):
        try:
            inputs = inputs.to(device)
        except Exception:
            # if .to fails, continue with CPU tensors (best-effort)
            pass

    # use the safe no-grad context stored in _no_grad_ctx
    with _no_grad_ctx():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic; set True for creative sampling
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
        )

    # decode only the newly generated tokens
    start_idx = inputs["input_ids"].shape[-1]
    gen_txt = tokenizer.decode(out[0][start_idx:], skip_special_tokens=False)

    # extract first JSON and parse
    jtxt = extract_first_json(gen_txt)
    parsed = json.loads(jtxt)
    return parsed


# ----------------------------
# Blender integration
# ----------------------------
def call_blender(
    blender_exe: str,
    worker_script: str,
    job_path: str,
    render_scene: bool = True,
    blender_args: Optional[List[str]] = None,
) -> None:
    """
    Spawn a headless Blender process to run the worker script on a job file.
    blender_args (optional): extra strings to append to Blender command.
    """
    # ensure blender_args is a plain list (avoid Optional[List] extend warning)
    extra: List[str] = list(blender_args) if blender_args else []

    cmd: List[str] = [
        str(blender_exe),
        "--background",
        "--python",
        str(worker_script),
        "--",
        "--action-file",
        str(job_path),
    ]
    if render_scene:
        cmd.append("--render-scene")

    # only extend if extra contains items; this makes static type-checkers happy
    if extra:
        cmd.extend(extra)

    print("Calling Blender:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("prompt", help="Natural language prompt")
    p.add_argument("--model-dir", required=True, help="Path to fine-tuned model directory (local)")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p.add_argument("--retrieved", default="[]", help="JSON list of retrieved snippets (optional)")
    p.add_argument("--out-inbox", default="worker/inbox", help="Worker inbox directory to write job JSON into")
    p.add_argument("--basename", default=None, help="Optional basename for job file")
    p.add_argument("--run-blender", action="store_true", help="Run Blender after writing job (requires blender_exe & worker script)")
    p.add_argument("--blender-exe", default=r"C:\SteamLibrary\steamapps\common\Blender\blender.exe", help="Blender executable")
    p.add_argument("--worker-script", default="worker/blender_executor.py", help="Worker script path (relative or absolute)")
    p.add_argument("--dry-run", action="store_true", help="Do not call blender even if --run-blender set")
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args()

    # device selection
    if args.device:
        device_str = args.device
    else:
        device_str = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    device = torch.device(device_str) if torch is not None else device_str
    print("Using device:", device)

    # attempt to load model & tokenizer; if fails, we'll fall back to rule-based
    tokenizer = None
    model = None
    try:
        tokenizer, model = load_model_and_tokenizer(args.model_dir, device)
    except Exception as e:
        print("Model load failed or not available; will use rule-based fallback. Reason:", e)

    # build prompt
    try:
        retrieved = json.loads(args.retrieved)
    except Exception:
        retrieved = []
    prompt_text = build_prompt(args.prompt, retrieved)

    # try model generation first (if model available)
    parsed = None
    if tokenizer is not None and model is not None:
        try:
            parsed = generate_json_from_prompt(tokenizer, model, prompt_text, device, max_new_tokens=args.max_new_tokens)
        except Exception as e:
            print("Generation failed, falling back to rules. Reason:", e)

    if parsed is None:
        print("Using rule-based fallback for prompt.")
        payload = rule_based_payload_from_prompt(args.prompt)
    else:
        try:
            payload = normalize_model_out(parsed)
        except Exception as e:
            print("Model output parse/normalize error:", e, file=sys.stderr)
            print("Using rule-based fallback for prompt.")
            payload = rule_based_payload_from_prompt(args.prompt)

    # Normalize legacy shapes (important) before schema validation
    payload = normalize_legacy_payload(payload)

    # Validate against local schema if present
    schema_path = Path(__file__).resolve().parent / "action_schema.json"
    if schema_path.exists():
        try:
            import jsonschema  # type: ignore

            with open(schema_path, "r", encoding="utf-8-sig") as f:
                schema = json.load(f)
            # try full payload validation first
            try:
                jsonschema.validate(instance=payload, schema=schema)
                print("validated against schema")
            except Exception:
                # try per-action validation (items schema)
                items_schema = schema.get("properties", {}).get("actions", {}).get("items")
                if items_schema:
                    per_ok = True
                    errs: List[str] = []
                    for i, a in enumerate(payload.get("actions", [])):
                        try:
                            jsonschema.validate(instance=a, schema=items_schema)
                        except Exception as e:
                            per_ok = False
                            errs.append(f"action[{i}] {e}")
                    if per_ok:
                        print("validated against schema (per-action validation)")
                    else:
                        raise ValueError("Schema validation failed: " + "; ".join(errs))
                else:
                    raise
        except ModuleNotFoundError:
            print("jsonschema not installed, skipping schema validation")
        except Exception as e:
            print("Schema validation failed:", e, file=sys.stderr)
            raise

    # write job to inbox
    inbox = Path(args.out_inbox)
    inbox.mkdir(parents=True, exist_ok=True)
    name = args.basename or f"job_{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}.json"
    job_path = inbox / name
    atomic_write(job_path, json.dumps(payload, ensure_ascii=False, indent=2))
    print("Wrote job to:", job_path)

    # optionally call blender
    if args.run_blender and not args.dry_run:
        blender_exe = Path(args.blender_exe)
        worker_script = Path(args.worker_script)
        if not blender_exe.exists():
            print("Blender executable not found:", blender_exe, file=sys.stderr)
            sys.exit(2)
        if not worker_script.exists():
            print("Worker script not found:", worker_script, file=sys.stderr)
            sys.exit(3)
        try:
            call_blender(str(blender_exe), str(worker_script), str(job_path), render_scene=True)
        except subprocess.CalledProcessError as e:
            print("Blender call failed:", e, file=sys.stderr)
            sys.exit(e.returncode)


if __name__ == "__main__":
    main()
