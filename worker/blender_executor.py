#!/usr/bin/env python3
"""
blender_executor.py — worker that validates an action JSON and, when run inside Blender,
executes actions and optionally renders the scene. Supports a dry-run mode when bpy is not available.
"""

import sys
import os
import json
import argparse
import shutil
import time
import traceback
from pathlib import Path
from datetime import datetime

# make sure local worker and schema directories are on sys.path so imports work
_sys_root = Path(__file__).resolve().parent
if str(_sys_root) not in sys.path:
    sys.path.insert(0, str(_sys_root))
_schema_dir = _sys_root / "schema"
if str(_schema_dir) not in sys.path:
    sys.path.insert(0, str(_schema_dir))

# Try to import bpy (present only inside Blender)
HAS_BPY = True
try:
    import bpy  # type: ignore
except Exception:
    HAS_BPY = False

# Try to import jsonschema for validation (optional)
try:
    import jsonschema
except Exception:
    jsonschema = None

# Try to import safe_executor (project-local)
SE = None
try:
    from worker.schema import safe_executor as se  # type: ignore
    SE = se
except Exception:
    try:
        import safe_executor as se  # type: ignore
        SE = se
    except Exception:
        SE = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--action-file", required=True)
    p.add_argument("--collection", default="AI_Sandbox")
    p.add_argument("--render-scene", action="store_true")
    p.add_argument("--render-width", type=int, default=512)
    p.add_argument("--render-height", type=int, default=512)
    p.add_argument("--dry-run", action="store_true", help="If set, skip any bpy calls even if bpy is available")
    return p.parse_args()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def validate_with_schema(obj, schema_path):
    if not jsonschema:
        return (True, "jsonschema not installed, skipping schema validation")
    try:
        with open(schema_path, "r", encoding="utf-8-sig") as f:
            schema = json.load(f)
    except Exception as e:
        return (False, f"failed to read schema: {e}")

    # Try whole payload validation first
    try:
        jsonschema.validate(instance=obj, schema=schema)
        return (True, "validated against schema")
    except Exception as full_err:
        full_err_txt = str(full_err)

    # If payload contains actions try per-action validation
    try:
        actions = obj.get("actions") if isinstance(obj, dict) else None
        if isinstance(actions, list) and actions:
            per_errors = []
            all_ok = True
            for i, a in enumerate(actions):
                try:
                    jsonschema.validate(instance=a, schema=schema)
                except Exception as e:
                    all_ok = False
                    per_errors.append(f"action[{i}] error: {e}")
            if all_ok:
                return (True, "validated against schema (per-action validation)")
            return (False, "schema validation failed for payload and per-action: " + "; ".join(per_errors))
    except Exception:
        pass

    return (False, f"schema validation failed: {full_err_txt}")


def clear_scene(keep_camera: bool = True, keep_lights: bool = False):
    """
    Robustly clear the current Blender scene:
      - remove all objects except (optionally) cameras and lights
      - remove non-root collections
      - purge orphan meshes/materials/images

    Safe no-op when bpy is not available (dry runs / outside Blender).
    """
    if not HAS_BPY:
        return

    # Remove objects (but optionally keep camera / lights)
    # Use bpy.data.objects.remove to avoid context-dependent operators.
    for obj in list(bpy.data.objects):
        try:
            if keep_camera and obj.type == "CAMERA":
                continue
            if keep_lights and obj.type == "LIGHT":
                continue
            # unlink and remove object
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            # best-effort removal; continue even if some removals fail
            pass

    # Remove any collections except the master 'Scene Collection' (name might vary by locale)
    # Use a conservative test: keep any collection that matches the active scene.collection
    try:
        root_col = bpy.context.scene.collection if hasattr(bpy.context.scene, "collection") else None
    except Exception:
        root_col = None

    for col in list(bpy.data.collections):
        # keep root collection
        if root_col is not None and col == root_col:
            continue
        # some installs call it "Scene Collection"; keep it
        if root_col is None and col.name == "Scene Collection":
            continue
        try:
            bpy.data.collections.remove(col)
        except Exception:
            pass

    # Purge orphan datablocks (meshes, materials, images) to avoid dangling memory and leftover
    for block_list in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        try:
            for block in list(block_list):
                try:
                    if getattr(block, "users", 1) == 0:
                        block_list.remove(block)
                except Exception:
                    pass
        except Exception:
            pass


def safe_clear_collection(name: str):
    """
    Remove objects from a specific collection and remove the collection itself.
    Kept for backward compatibility / targeted clearing.
    """
    if not HAS_BPY:
        return
    try:
        col = bpy.data.collections.get(name)
        if not col:
            return
        for obj in list(col.objects):
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception:
                pass
        try:
            bpy.data.collections.remove(col)
        except Exception:
            pass
    except Exception:
        # swallow errors to avoid crashing worker
        pass


def prepare_render_settings(width: int, height: int, filepath: str):
    scene = bpy.context.scene
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.filepath = str(Path(filepath).resolve())
    scene.render.image_settings.file_format = 'PNG'
    # pick a safe engine for multiple Blender versions
    for candidate in ("BLENDER_EEVEE_NEXT", "CYCLES", "BLENDER_WORKBENCH"):
        try:
            scene.render.engine = candidate
            if scene.render.engine == candidate:
                break
        except Exception:
            pass
    # final fallback
    try:
        if scene.render.engine not in ("BLENDER_EEVEE_NEXT", "CYCLES", "BLENDER_WORKBENCH"):
            scene.render.engine = "CYCLES"
    except Exception:
        pass


def _fallback_execute_actions(actions, collection_name="AI_Sandbox", dry_run=True):
    """
    Local fallback executor used only when a project-provided safe_executor is missing
    or fails. Returns a list of messages describing what was observed.
    """
    msgs = []
    for a in (actions if isinstance(actions, list) else [actions]):
        # be defensive if action is not a mapping
        if not isinstance(a, dict):
            msgs.append(f"Fallback executor saw non-dict action: {repr(a)}")
            continue
        act = a.get("action") or a.get("type") or a.get("action_type") or None
        msgs.append(f"Fallback executor saw action: {act}")
    return msgs


def attempt_render_and_copy(render_tmp: Path, final_outbox_dir: Path):
    if not HAS_BPY:
        return (False, f"bpy not available; cannot render to {render_tmp}")
    # ensure camera exists
    cam = next((o for o in bpy.data.objects if o.type == "CAMERA"), None)
    if not cam:
        try:
            bpy.ops.object.camera_add(location=(0, -5, 2), rotation=(1.2, 0, 0))
            cam = bpy.context.active_object
        except Exception:
            pass
    if cam:
        bpy.context.scene.camera = cam

    prepare_render_settings(bpy.context.scene.render.resolution_x or 512,
                            bpy.context.scene.render.resolution_y or 512,
                            str(render_tmp))

    # make sure target filepath is exact
    bpy.context.scene.render.filepath = str(render_tmp.resolve())

    bpy.ops.render.render(write_still=True)

    if not render_tmp.exists() or render_tmp.stat().st_size == 0:
        reported = bpy.context.scene.render.filepath
        return (False, f"Render file not found at expected path {render_tmp}; scene.render.filepath={reported}")

    name_stamp = datetime.utcnow().strftime('%Y%m%d%H%M%S') + f"_{os.getpid()}"
    final_out = final_outbox_dir / f"render_{name_stamp}.png"
    partial = final_out.with_suffix(final_out.suffix + ".partial")
    shutil.copy2(str(render_tmp.resolve()), str(partial))
    os.replace(str(partial), str(final_out))
    try:
        os.chmod(str(final_out), 0o644)
    except Exception:
        pass
    try:
        if render_tmp.exists():
            render_tmp.unlink()
    except Exception:
        pass
    return (True, str(final_out.resolve()))


def print_json(obj):
    print(json.dumps(obj, ensure_ascii=False))
    sys.stdout.flush()


def _normalize_legacy_payload(raw: dict) -> dict:
    """
    Convert legacy single-action payloads into the modern {"actions": [...]} shape
    and normalize common legacy fields.

    Recognizes forms like:
      { "action": "add_object", "params": { "type":"cube", "name": "...", ... } }
    or already-correct payloads with "actions".

    Returns a normalized payload dict.
    """
    if not isinstance(raw, dict):
        return raw

    # already normalized
    if "actions" in raw and isinstance(raw["actions"], list):
        return raw

    # legacy single-action with 'action' + 'params'
    if "action" in raw and isinstance(raw.get("params"), dict):
        # ensure act_name is always a str for type-checkers
        act_name = str(raw.get("action") or "")
        params = dict(raw.get("params") or {})

        legacy_map: dict[str, str] = {
            "add_object": "create_primitive",
            "set_transform": "transform",
            "move": "move_object",
            "scale": "scale_object",
            "rotate": "rotate_object",
            "delete": "delete_object",
        }

        new_type = legacy_map.get(act_name, act_name)

        if new_type == "create_primitive" or bool(params.get("type")):
            primitive = params.pop("type", None) or params.get("primitive") or "CUBE"
            # normalize primitive string if applicable
            if isinstance(primitive, str):
                primitive_val = primitive.upper()
            else:
                primitive_val = primitive

            name = params.get("name") or params.get("id")
            action_obj = {
                "type": "create_primitive",
                "primitive": primitive_val,
                "name": name or params.get("name") or params.get("id"),
                # copy common fields
                "position": params.get("location") or params.get("position"),
                "rotation": params.get("rotation"),
                "scale": params.get("scale"),
                "color": params.get("color") or params.get("colour"),
            }
            # include any leftover params (preserve values)
            for k, v in params.items():
                if k not in {
                    "location",
                    "position",
                    "rotation",
                    "scale",
                    "color",
                    "colour",
                    "name",
                    "id",
                    "type",
                    "primitive",
                }:
                    action_obj[k] = v
        else:
            # general mapping: drop params up one level and set type
            action_obj = {"type": new_type}
            # flatten params into action
            for k, v in params.items():
                action_obj[k] = v

        normalized = {"version": "1.0", "actions": [action_obj]}
        return normalized

    # fallback: can't normalize — return original (let schema/validator fail)
    return raw


def main():
    args = parse_args()

    try:
        print_json({"debug": "worker_start", "argv": sys.argv})
    except Exception:
        sys.stdout.write("DEBUG_START\n")
        sys.stdout.flush()

    action_path = Path(args.action_file).resolve()
    if not action_path.exists():
        print_json({"success": False, "messages": [], "render": None, "error": f"action file not found: {action_path}"})
        return

    try:
        raw = load_json(action_path)
        action = _normalize_legacy_payload(raw)
    except Exception as e:
        tb = traceback.format_exc()
        print_json({"success": False, "messages": [], "render": None, "error": f"failed to read action file: {e}", "traceback": tb})
        return

    # ensure 'type' -> 'action' compatibility for old jobs
    if isinstance(action, dict) and isinstance(action.get("actions"), list):
        for a in action["actions"]:
            if "action" not in a and "type" in a:
                a["action"] = a["type"]

    msgs = []

    # Use local schema file (relative to this script) instead of a hard-coded absolute path
    schema_file = Path(__file__).resolve().parent / "action_schema.json"
    if schema_file.exists():
        ok, m = validate_with_schema(action, str(schema_file))
        msgs.append(m)
        if not ok:
            print_json({"success": False, "messages": msgs, "render": None, "error": "schema validation failed"})
            return
    else:
        msgs.append("no action_schema.json found; skipping schema validation")

    # Clear the entire scene before executing actions (delete default cube etc.)
    # Keep camera by default so render camera still exists. If you'd rather destroy camera too,
    # call clear_scene(keep_camera=False, keep_lights=False).
    if HAS_BPY and not args.dry_run:
        try:
            clear_scene(keep_camera=True, keep_lights=False)
            msgs.append("scene cleared before execution")
        except Exception:
            msgs.append("warning: clear_scene raised an exception")

    try:
        if isinstance(action, dict) and isinstance(action.get("actions"), list):
            actions_to_run = action["actions"]
        else:
            actions_to_run = action
        # pass dry_run flag: if user requested --dry-run, or bpy isn't present, set dry_run=True
        effective_dry = bool(args.dry_run) or (not HAS_BPY)

        # Prefer project-provided safe_executor.execute_actions when available.
        exec_msgs = None
        if SE is not None:
            exec_fn = getattr(SE, "execute_actions", None)
            if callable(exec_fn):
                try:
                    exec_msgs = exec_fn(actions_to_run, collection_name=args.collection, dry_run=effective_dry)
                except Exception as e:
                    tb = traceback.format_exc()
                    msgs.append(f"project safe_executor.execute_actions raised exception: {e}")
                    msgs.append(tb)
                    exec_msgs = None
            else:
                # try alternative name
                alt = getattr(SE, "run_actions", None)
                if callable(alt):
                    try:
                        exec_msgs = alt(actions_to_run, collection_name=args.collection, dry_run=effective_dry)
                    except Exception as e:
                        tb = traceback.format_exc()
                        msgs.append(f"project safe_executor.run_actions raised exception: {e}")
                        msgs.append(tb)
                        exec_msgs = None

        # Fallback to local minimal executor if project one was missing or errored
        if exec_msgs is None:
            exec_msgs = _fallback_execute_actions(actions_to_run, collection_name=args.collection, dry_run=effective_dry)

        if isinstance(exec_msgs, list):
            msgs.extend(exec_msgs)
        else:
            msgs.append(str(exec_msgs))
    except Exception as e:
        tb = traceback.format_exc()
        print_json({"success": False, "messages": msgs, "render": None, "error": f"execution error: {e}", "traceback": tb})
        return

    render_path = None
    if args.render_scene:
        base = Path(__file__).resolve().parent
        final_outbox_dir = base / "outbox"
        tmp_dir = final_outbox_dir / "tmp_renders"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        final_outbox_dir.mkdir(parents=True, exist_ok=True)
        render_tmp = tmp_dir / f"render_tmp_{int(time.time())}.png"

        if not HAS_BPY or args.dry_run:
            msgs.append("dry-run or bpy unavailable; skipping actual render")
        else:
            try:
                ok, res = attempt_render_and_copy(render_tmp, final_outbox_dir)
            except Exception as e:
                tb = traceback.format_exc()
                print_json({"success": False, "messages": msgs, "render": None, "error": f"render attempt raised exception: {e}", "traceback": tb})
                return
            if not ok:
                print_json({"success": False, "messages": msgs, "render": None, "error": res})
                return
            render_path = res
            msgs.append(f"render written to outbox: {render_path}")

    print_json({
        "success": True,
        "messages": msgs + (["Render complete"] if args.render_scene and render_path else ["Execution only"]),
        "render": str(render_path) if render_path else None,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })


if __name__ == "__main__":
    main()
