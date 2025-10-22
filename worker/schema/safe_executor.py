# safe_executor.py
"""
safe_executor — a small, sandboxed interpreter for canonical action payloads.
Intended to be imported by blender_executor.py. Uses blender_wrapper for the
actual Blender or dry-run implementations.

Actions supported (canonical keys expected; permissive about extra fields):
 - create_primitive / type=create_primitive
    fields: primitive, name, position/position, rotation, scale, color, collection
 - move_object / transform with position
 - rotate_object / transform with rotation
 - scale_object / transform with scale
 - delete_object (by name)
 - delete_all
 - apply_material_to_object (material spec)
 - add_modifier_to_object
 - apply_modifier_stack
 - import_image_as_plane
 - transform (general: position/rotation/scale)
"""

from typing import List, Dict, Any, Optional, Union
import logging
import uuid
import traceback

# Try to import local blender_wrapper; fallback to a minimal stub if missing.
try:
    # safe_executor is in worker/schema/ — expect blender_wrapper at worker/blender_wrapper.py
    from worker import blender_wrapper as bw  # type: ignore
except Exception:
    try:
        import blender_wrapper as bw  # type: ignore
    except Exception:
        bw = None  # type: ignore

logger = logging.getLogger("safe_executor")
if not logger.hasHandlers():
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[safe_executor] %(levelname)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def _normalize_action(a: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize some legacy fields and provide defaults."""
    act = dict(a)
    # support both 'type' and 'action' keys
    if "action" not in act and "type" in act:
        act["action"] = act["type"]
    # normalize position aliases
    if "position" not in act and "location" in act:
        act["position"] = act["location"]
    # ensure names
    if act.get("action") == "create_primitive":
        if "name" not in act:
            prefix = (act.get("primitive") or "obj").lower()
            act["name"] = f"{prefix}_{uuid.uuid4().hex[:6]}"
        # canonicalize primitive
        if "primitive" in act and isinstance(act["primitive"], str):
            act["primitive"] = str(act["primitive"]).upper()
    return act


def _normalize_legacy_payload(raw: dict) -> dict:
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

    # fallback: can't normalize — return original
    return raw


def _call_bw(func_name: str, action: Dict[str, Any], collection_name: str, dry_run: bool) -> str:
    """
    Try to call bw.<func_name>(action, collection_name=..., dry_run=...) if available.
    Otherwise return a dry-run message.
    """
    # If blender_wrapper is not present, simulate
    if bw is None:
        return f"DRY_RUN: {func_name} not available — would run {action.get('action') or action.get('type')} with {action.get('name') or action.get('primitive') or ''}"

    func = getattr(bw, func_name, None)
    if not callable(func):
        # try a more generic execute_action function if available
        generic = getattr(bw, "execute_action", None)
        if callable(generic):
            try:
                generic(action, collection_name=collection_name, dry_run=dry_run)
                return f"CALLED: bw.execute_action -> {action.get('action') or action.get('type')}"
            except Exception as e:
                logger.exception("exception in bw.execute_action")
                return f"ERROR calling bw.execute_action: {e}"
        # function not found: dry-run text
        return f"DRY_RUN: bw.{func_name} missing — would run {action.get('action') or action.get('type')}"
    # call the function — wrappers are expected to accept (action, collection_name=..., dry_run=...)
    try:
        # attempt calling with named args; fallback to single-arg
        try:
            res = func(action, collection_name=collection_name, dry_run=dry_run)
        except TypeError:
            res = func(action)
        # normalize result to string for simple API
        if isinstance(res, str):
            return res
        return f"CALLED: bw.{func_name} -> {repr(res)}"
    except Exception as e:
        logger.exception("exception while calling bw.%s", func_name)
        return f"ERROR calling bw.{func_name}: {e}"


def execute_actions(
    actions: Union[List[Dict[str, Any]], Dict[str, Any]],
    collection_name: str = "AI_Sandbox",
    dry_run: bool = True,
) -> List[str]:
    """
    Execute a list (or legacy single payload dict) of actions.

    Args:
        actions: either a list of action dicts, or a dict containing {"actions": [...]}
                 or a legacy single-action payload (see _normalize_legacy_payload).
        collection_name: name of the Blender collection / sandbox to use.
        dry_run: if True, do not call Blender operations (or wrappers should respect dry_run).

    Returns:
        List[str] — result strings for each processed action (success / dry-run / error).
    """
    results: List[str] = []

    # support receiving the entire payload dict (legacy support)
    payload = actions
    if isinstance(actions, dict):
        payload = _normalize_legacy_payload(actions)

    # Normalize into a concrete list of actions (always produce a list)
    acts: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        maybe_actions = payload.get("actions")
        if isinstance(maybe_actions, list):
            # ensure each entry is a dict
            acts = [a if isinstance(a, dict) else dict(a) for a in maybe_actions]
        else:
            # treat payload itself as a single action dict
            acts = [payload]
    elif isinstance(payload, list):
        acts = [a if isinstance(a, dict) else dict(a) for a in payload]
    else:
        msg = "Invalid payload: expected list or dict with 'actions'"
        logger.error(msg)
        raise ValueError(msg)

    for raw in acts:
        try:
            act = _normalize_action(raw if isinstance(raw, dict) else dict(raw))
            action_name = (act.get("action") or act.get("type") or "").lower()
            logger.info("processing action: %s", action_name)

            # dispatch — map canonical action name to bw function names
            if action_name in {"create_primitive", "add_object"}:
                res = _call_bw("create_primitive", act, collection_name, dry_run)
            elif action_name in {"move_object", "move", "set_location"}:
                res = _call_bw("move_object", act, collection_name, dry_run)
            elif action_name in {"rotate_object", "rotate"}:
                res = _call_bw("rotate_object", act, collection_name, dry_run)
            elif action_name in {"scale_object", "scale"}:
                res = _call_bw("scale_object", act, collection_name, dry_run)
            elif action_name in {"transform", "set_transform"}:
                res = _call_bw("set_transform", act, collection_name, dry_run)
            elif action_name in {"delete_object", "delete"}:
                res = _call_bw("delete_object", act, collection_name, dry_run)
            elif action_name in {"delete_all", "clear_collection"}:
                res = _call_bw("delete_all", act, collection_name, dry_run)
            elif action_name in {"apply_material_to_object", "apply_material"}:
                res = _call_bw("apply_material_to_object", act, collection_name, dry_run)
            elif action_name in {"add_modifier_to_object", "add_modifier"}:
                res = _call_bw("add_modifier_to_object", act, collection_name, dry_run)
            elif action_name in {"apply_modifier_stack", "apply_modifiers"}:
                res = _call_bw("apply_modifier_stack", act, collection_name, dry_run)
            elif action_name in {"import_image_as_plane", "import_image"}:
                res = _call_bw("import_image_as_plane", act, collection_name, dry_run)
            else:
                # unknown action: try generic entrypoint on bw, else dry-run
                if bw is not None and hasattr(bw, "execute_action"):
                    res = _call_bw("execute_action", act, collection_name, dry_run)
                else:
                    res = f"DRY_RUN: unknown action '{action_name}' — payload: {act}"
                    logger.warning(res)

            results.append(res)
        except Exception as e:
            tb = traceback.format_exc()
            logger.exception("Exception while processing action: %s", e)
            results.append(f"ERROR: exception processing action {raw!r}: {e}\n{tb}")

    return results



# If run as a script, provide a tiny CLI for quick dry-run testing.
if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python safe_executor.py <actions.json> [--collection NAME] [--live]")
        sys.exit(1)

    path = sys.argv[1]
    collection = "AI_Sandbox"
    live = False
    if "--collection" in sys.argv:
        try:
            collection = sys.argv[sys.argv.index("--collection") + 1]
        except Exception:
            pass
    if "--live" in sys.argv:
        live = True

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    out = execute_actions(payload, collection_name=collection, dry_run=not live)
    print(json.dumps(out, indent=2))
