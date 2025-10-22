"""
blender_wrapper.py
Lightweight wrapper that exposes a small, testable API used by safe_executor.
- Works in two modes:
    * dry_run=True -> no bpy usage, logs actions
    * dry_run=False -> attempts to use bpy APIs (must run inside Blender)
- Designed to be safe to import in regular Python (bpy import is optional)
"""

from pathlib import Path
import logging
import os
import time
import shutil
import uuid
from typing import Optional, List, Dict, Any

logger = logging.getLogger("blender_wrapper")
if not logger.hasHandlers():
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[blender_wrapper] %(levelname)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# Try to import bpy — if not present we operate in dry-run only
_HAS_BPY = True
try:
    import bpy  # type: ignore
except Exception:
    _HAS_BPY = False

# Map user modifier names to Blender types (extend as needed)
_MODIFIER_TYPE_MAP = {
    "SUBSURF": "SUBSURF",
    "BEVEL": "BEVEL",
    "BOOLEAN": "BOOLEAN",
    "ARRAY": "ARRAY",
    "MIRROR": "MIRROR",
    "SOLIDIFY": "SOLIDIFY",
}

def ensure_camera_and_light():
    """Ensure camera and light exist, positioned to capture the full scene."""
    if not _HAS_BPY:
        return
    import bpy
    # --- Camera setup ---
    cam = bpy.data.objects.get("Camera")
    if not cam:
        bpy.ops.object.camera_add(location=(0, -10, 5), rotation=(1.1, 0, 0))
        cam = bpy.context.active_object
        cam.name = "Camera"
    else:
        cam.location = (0, -10, 5)
        cam.rotation_euler = (1.1, 0, 0)

    bpy.context.scene.camera = cam
    cam.data.lens = 35  # widen slightly (smaller = zoom out)

    # --- Light setup ---
    light = bpy.data.objects.get("Main_Light")
    if not light:
        bpy.ops.object.light_add(type='SUN', location=(6, -6, 10))
        light = bpy.context.active_object
        light.name = "Main_Light"
        light.data.energy = 5.0
    else:
        light.location = (6, -6, 10)
        light.data.energy = 5.0

    # --- Ambient world lighting ---
    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[1].default_value = 1.0

def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

# ------------- utility helpers -------------
def ensure_collection(name: str, dry_run: bool = True):
    """
    Ensure a named collection exists and return it (or None in dry-run).
    """
    if dry_run or not _HAS_BPY:
        logger.debug("ensure_collection (dry_run): %s", name)
        return None
    col = bpy.data.collections.get(name)
    if not col:
        col = bpy.data.collections.new(name)
        try:
            bpy.context.scene.collection.children.link(col)
        except Exception:
            # some contexts may not allow linking; best-effort
            pass
    return col

def get_object_by_name(name: str):
    if not _HAS_BPY:
        logger.debug("get_object_by_name (dry_run): %s", name)
        return None
    return bpy.data.objects.get(name)

# ------------- primary actions -------------
def create_primitive(*args, **kwargs) -> Optional[str]:
    """
    Flexible signature:
      create_primitive(action_dict, collection_name=..., dry_run=...)
      or create_primitive(primitive='CUBE', name='obj', location=[...], ..., dry_run=...)
    Returns object name on success, or None.
    """
    # normalize caller styles
    if args and isinstance(args[0], dict):
        action = args[0]
        primitive = (action.get("primitive") or action.get("type") or "CUBE").upper()
        name = action.get("name") or action.get("id") or f"{primitive}_{uuid.uuid4().hex[:6]}"
        location = action.get("position") or action.get("location") or action.get("loc") or [0.0, 0.0, 0.0]
        rotation = action.get("rotation")
        scale = action.get("scale")
        color = action.get("color") or action.get("colour")
        collection = action.get("collection") or kwargs.get("collection")
        dry_run = kwargs.get("dry_run", action.get("dry_run", True))
    else:
        primitive = (kwargs.get("primitive") or kwargs.get("type") or "CUBE").upper()
        name = kwargs.get("name") or f"{primitive}_{uuid.uuid4().hex[:6]}"
        location = kwargs.get("location") or kwargs.get("position") or [0.0, 0.0, 0.0]
        rotation = kwargs.get("rotation")
        scale = kwargs.get("scale")
        color = kwargs.get("color") or kwargs.get("colour")
        collection = kwargs.get("collection")
        dry_run = kwargs.get("dry_run", True)

    location = location or [0.0, 0.0, 0.0]
    rotation = rotation or [0.0, 0.0, 0.0]
    scale = scale or [1.0, 1.0, 1.0]

    logger.info("create_primitive: %s name=%s loc=%s scale=%s dry_run=%s", primitive, name, location, scale, dry_run)
    if dry_run or not _HAS_BPY:
        return name or f"{primitive}_{uuid.uuid4().hex[:6]}"

    # ensure lighting and camera before creation
    try:
        ensure_camera_and_light()
    except Exception:
        logger.warning("Failed to ensure camera/light setup")

    # real bpy creation
    try:
        bpy.ops.object.select_all(action="DESELECT")
        if primitive == "CUBE":
            bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
        elif primitive in {"SPHERE", "UV_SPHERE", "UVSPHERE"}:
            bpy.ops.mesh.primitive_uv_sphere_add(location=location, rotation=rotation)
        elif primitive == "PLANE":
            bpy.ops.mesh.primitive_plane_add(location=location, rotation=rotation)
        elif primitive == "CYLINDER":
            bpy.ops.mesh.primitive_cylinder_add(location=location, rotation=rotation)
        elif primitive == "CONE":
            bpy.ops.mesh.primitive_cone_add(location=location, rotation=rotation)
        elif primitive == "TORUS":
            bpy.ops.mesh.primitive_torus_add(location=location, rotation=rotation)
        else:
            logger.warning("Unknown primitive '%s' — default to cube", primitive)
            bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)

        obj = bpy.context.active_object
        if obj is None:
            logger.error("create_primitive: unable to obtain created object")
            return None

        obj.name = name or obj.name
        try:
            obj.scale = scale
        except Exception:
            pass

        if collection:
            col = bpy.data.collections.get(collection) or bpy.data.collections.new(collection)
            if obj.name not in [o.name for o in col.objects]:
                try:
                    col.objects.link(obj)
                except Exception:
                    pass

        if color is not None:
            try:
                mat = bpy.data.materials.get(f"mat_{obj.name}") or bpy.data.materials.new(f"mat_{obj.name}")
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf and isinstance(color, (list, tuple)) and len(color) >= 3:
                    bsdf.inputs["Base Color"].default_value = (
                        float(color[0]), float(color[1]), float(color[2]), 1.0)
                if obj.data and hasattr(obj.data, "materials"):
                    if obj.data.materials:
                        obj.data.materials[0] = mat
                    else:
                        obj.data.materials.append(mat)
            except Exception:
                logger.exception("Failed to apply color/material to %s", obj.name)

        return obj.name
    except Exception:
        logger.exception("create_primitive failed")
        return None

def move_object(*args, **kwargs):
    """
    move_object(name_or_action, location=..., dry_run=...)
    Accepts either action dict or explicit args.
    """
    if args and isinstance(args[0], dict):
        action = args[0]
        name = action.get("name") or action.get("id")
        location = action.get("position") or action.get("location")
        dry_run = kwargs.get("dry_run", action.get("dry_run", True))
    else:
        name = kwargs.get("name") or (args[0] if args else None)
        location = kwargs.get("location") or (args[1] if len(args) > 1 else None)
        dry_run = kwargs.get("dry_run", True)

    logger.info("move_object: %s -> %s dry_run=%s", name, location, dry_run)
    if dry_run or not _HAS_BPY:
        return
    if not name:
        logger.warning("move_object: missing name")
        return
    obj = bpy.data.objects.get(name)
    if not obj:
        logger.warning("move_object: object not found: %s", name)
        return
    try:
        obj.location = location
    except Exception:
        logger.exception("move_object failed for %s", name)

def rotate_object(*args, **kwargs):
    if args and isinstance(args[0], dict):
        action = args[0]
        name = action.get("name") or action.get("id")
        rotation = action.get("rotation")
        dry_run = kwargs.get("dry_run", action.get("dry_run", True))
    else:
        name = kwargs.get("name") or (args[0] if args else None)
        rotation = kwargs.get("rotation") or (args[1] if len(args) > 1 else None)
        dry_run = kwargs.get("dry_run", True)

    logger.info("rotate_object: %s -> %s dry_run=%s", name, rotation, dry_run)
    if dry_run or not _HAS_BPY:
        return
    if not name:
        logger.warning("rotate_object: missing name")
        return
    obj = bpy.data.objects.get(name)
    if not obj:
        logger.warning("rotate_object: object not found: %s", name)
        return
    try:
        obj.rotation_euler = rotation
    except Exception:
        logger.exception("rotate_object failed for %s", name)

def scale_object(*args, **kwargs):
    if args and isinstance(args[0], dict):
        action = args[0]
        name = action.get("name") or action.get("id")
        scale = action.get("scale")
        dry_run = kwargs.get("dry_run", action.get("dry_run", True))
    else:
        name = kwargs.get("name") or (args[0] if args else None)
        scale = kwargs.get("scale") or (args[1] if len(args) > 1 else None)
        dry_run = kwargs.get("dry_run", True)

    logger.info("scale_object: %s -> %s dry_run=%s", name, scale, dry_run)
    if dry_run or not _HAS_BPY:
        return
    if not name:
        logger.warning("scale_object: missing name")
        return
    obj = bpy.data.objects.get(name)
    if not obj:
        logger.warning("scale_object: object not found: %s", name)
        return
    try:
        obj.scale = scale
    except Exception:
        logger.exception("scale_object failed for %s", name)

def delete_object(*args, **kwargs):
    if args and isinstance(args[0], dict):
        action = args[0]
        name = action.get("name") or action.get("id")
        dry_run = kwargs.get("dry_run", action.get("dry_run", True))
    else:
        name = kwargs.get("name") or (args[0] if args else None)
        dry_run = kwargs.get("dry_run", True)

    logger.info("delete_object: %s dry_run=%s", name, dry_run)
    if dry_run or not _HAS_BPY:
        return
    if not name:
        logger.warning("delete_object: missing name")
        return
    obj = bpy.data.objects.get(name)
    if not obj:
        logger.warning("delete_object: object not found: %s", name)
        return
    try:
        bpy.data.objects.remove(obj, do_unlink=True)
    except Exception:
        logger.exception("delete_object failed for %s", name)

def delete_all(*args, **kwargs):
    # signature tolerant: delete_all(action_dict?, dry_run=...)
    if args and isinstance(args[0], dict):
        action = args[0]
        dry_run = kwargs.get("dry_run", action.get("dry_run", True))
    else:
        dry_run = kwargs.get("dry_run", True)
    logger.info("delete_all: dry_run=%s", dry_run)
    if dry_run or not _HAS_BPY:
        return
    try:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
    except Exception:
        logger.exception("delete_all failed")

def add_modifier_to_object(*args, **kwargs):
    """
    Flexible signature:
      add_modifier_to_object(action_dict, dry_run=...)
      or add_modifier_to_object(obj_name, modifier_type, params, dry_run=...)
    """
    if args and isinstance(args[0], dict):
        action = args[0]
        obj_name = action.get("name") or action.get("object") or action.get("obj_name")
        modifier_type = action.get("modifier") or action.get("modifier_type") or action.get("type")
        params = action.get("params", {}) or {}
        dry_run = kwargs.get("dry_run", action.get("dry_run", True))
    else:
        obj_name = kwargs.get("obj_name") or (args[0] if args else None)
        modifier_type = kwargs.get("modifier_type") or (args[1] if len(args) > 1 else None)
        params = kwargs.get("params", {}) or (args[2] if len(args) > 2 else {})
        dry_run = kwargs.get("dry_run", True)

    params = params or {}
    if modifier_type is None or not str(modifier_type).strip():
        logger.warning("add_modifier_to_object: skipping (missing or invalid modifier_type) for %s", obj_name)
        return

    modifier_type = str(modifier_type).upper()
    logger.info("add_modifier_to_object: %s %s params=%s dry_run=%s", obj_name, modifier_type, params, dry_run)
    if dry_run or not _HAS_BPY:
        return

    obj = bpy.data.objects.get(obj_name)
    if not obj:
        logger.warning("add_modifier_to_object: object not found: %s", obj_name)
        return

    mtype = _MODIFIER_TYPE_MAP.get(modifier_type, modifier_type)
    try:
        mod = obj.modifiers.new(name=f"MOD_{mtype}", type=mtype)
        for k, v in params.items():
            try:
                if hasattr(mod, k):
                    setattr(mod, k, v)
            except Exception:
                pass
    except Exception:
        logger.exception("add_modifier_to_object failed for %s type=%s", obj_name, mtype)

def apply_modifier_stack(*args, **kwargs):
    """
    apply_modifier_stack(action_dict or (obj_name, stack, dry_run))
    """
    if args and isinstance(args[0], dict):
        action = args[0]
        obj_name = action.get("name") or action.get("object") or action.get("obj_name")
        stack = action.get("stack") or action.get("modifiers") or action.get("apply") or []
        dry_run = kwargs.get("dry_run", action.get("dry_run", True))
    else:
        obj_name = kwargs.get("obj_name") or (args[0] if args else None)
        stack = kwargs.get("stack") or (args[1] if len(args) > 1 else [])
        dry_run = kwargs.get("dry_run", True)

    logger.info("apply_modifier_stack: %s stack_len=%d dry_run=%s", obj_name, len(stack or []), dry_run)
    if dry_run or not _HAS_BPY:
        return
    for mod in stack:
        mtype = mod.get("type") or mod.get("modifier") or mod.get("modifier_type")
        params = mod.get("params", {}) or {}
        if mtype is None or (isinstance(mtype, str) and not mtype.strip()):
            logger.warning("apply_modifier_stack: skipping modifier with missing type for object %s: %r", obj_name, mod)
            continue
        try:
            add_modifier_to_object({"name": obj_name, "modifier": str(mtype), "params": params}, dry_run=False)
        except Exception:
            logger.exception("apply_modifier_stack: failed to add modifier %s to %s", mtype, obj_name)

def apply_material_to_object(*args, **kwargs):
    """
    apply_material_to_object(action_dict or (obj_name, material_spec, dry_run))
    """
    if args and isinstance(args[0], dict):
        action = args[0]
        obj_name = action.get("name") or action.get("object") or action.get("obj_name")
        material_spec = action.get("material") or action.get("material_spec") or action
        dry_run = kwargs.get("dry_run", action.get("dry_run", True))
    else:
        obj_name = kwargs.get("obj_name") or (args[0] if args else None)
        material_spec = kwargs.get("material_spec") or (args[1] if len(args) > 1 else {})
        dry_run = kwargs.get("dry_run", True)

    logger.info("apply_material_to_object: %s spec=%s dry_run=%s", obj_name, material_spec, dry_run)
    if dry_run or not _HAS_BPY:
        return
    obj = bpy.data.objects.get(obj_name)
    if not obj:
        logger.warning("apply_material_to_object: object not found: %s", obj_name)
        return
    try:
        mat_name = material_spec.get("name") or f"mat_{obj_name}"
        mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            color = material_spec.get("base_color") or material_spec.get("color")
            if color and isinstance(color, (list, tuple)) and len(color) >= 3:
                try:
                    bsdf.inputs["Base Color"].default_value = (float(color[0]), float(color[1]), float(color[2]), 1.0)
                except Exception:
                    pass
        if obj.data and hasattr(obj.data, "materials"):
            if obj.data.materials:
                obj.data.materials[0] = mat
            else:
                obj.data.materials.append(mat)
    except Exception:
        logger.exception("apply_material_to_object failed for %s", obj_name)

def import_image_as_plane(*args, **kwargs) -> Optional[str]:
    """
    import_image_as_plane(action_dict or (image_path, name, location, scale, collection, dry_run))
    Returns the created object's name (or None).
    """
    if args and isinstance(args[0], dict):
        action = args[0]
        image_path = action.get("path") or action.get("image_path") or action.get("filepath")
        name = action.get("name") or action.get("id") or "ImagePlane"
        location = action.get("position") or action.get("location") or [0.0, 0.0, 0.0]
        scale = action.get("scale") or [1.0, 1.0, 1.0]
        collection = action.get("collection")
        dry_run = kwargs.get("dry_run", action.get("dry_run", True))
    else:
        image_path = kwargs.get("image_path") or (args[0] if args else None)
        name = kwargs.get("name") or (args[1] if len(args) > 1 else "ImagePlane")
        location = kwargs.get("location") or (args[2] if len(args) > 2 else [0.0, 0.0, 0.0])
        scale = kwargs.get("scale") or (args[3] if len(args) > 3 else [1.0, 1.0, 1.0])
        collection = kwargs.get("collection")
        dry_run = kwargs.get("dry_run", True)

    logger.info("import_image_as_plane: %s -> %s dry_run=%s", image_path, name, dry_run)
    location = location or [0.0, 0.0, 0.0]
    scale = scale or [1.0, 1.0, 1.0]
    if dry_run or not _HAS_BPY:
        if not image_path or not Path(str(image_path)).exists():
            logger.warning("import_image_as_plane (dry_run): image not found: %s", image_path)
        return name
    try:
        try:
            # prefer the import_images_as_planes addon operator if available
            bpy.ops.import_image.to_plane(filepath=str(image_path))
            obj = bpy.context.active_object
            if obj:
                obj.name = name
                obj.location = location
                obj.scale = scale
                if collection:
                    col = bpy.data.collections.get(collection) or bpy.data.collections.new(collection)
                    if obj.name not in [o.name for o in col.objects]:
                        try:
                            col.objects.link(obj)
                        except Exception:
                            pass
                return obj.name
        except Exception:
            # fallback: create a plane and assign image texture
            bpy.ops.mesh.primitive_plane_add(location=location)
            obj = bpy.context.active_object
            if obj is None:
                return None
            obj.name = name
            img = None
            try:
                img = bpy.data.images.load(str(image_path), check_existing=True)
            except Exception:
                logger.exception("Failed to load image %s", image_path)
            if img:
                mat = bpy.data.materials.new(name=f"mat_{name}")
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                tex_node = nodes.new("ShaderNodeTexImage")
                tex_node.image = img
                bsdf = nodes.get("Principled BSDF")
                if bsdf:
                    try:
                        links.new(tex_node.outputs.get("Color"), bsdf.inputs.get("Base Color"))
                    except Exception:
                        pass
                if obj.data and hasattr(obj.data, "materials"):
                    if obj.data.materials:
                        obj.data.materials[0] = mat
                    else:
                        obj.data.materials.append(mat)
            obj.scale = scale
            return obj.name
    except Exception:
        logger.exception("import_image_as_plane failed")
        return None

# ------------- generic entrypoints -------------
def set_transform(*args, **kwargs):
    """
    Convenience: accept an action dict with position/rotation/scale and apply to named object.
    """
    if args and isinstance(args[0], dict):
        action = args[0]
        name = action.get("name") or action.get("object") or action.get("id")
        pos = action.get("position") or action.get("location")
        rot = action.get("rotation")
        scale = action.get("scale")
        dry_run = kwargs.get("dry_run", action.get("dry_run", True))
    else:
        name = kwargs.get("name") or (args[0] if args else None)
        pos = kwargs.get("position")
        rot = kwargs.get("rotation")
        scale = kwargs.get("scale")
        dry_run = kwargs.get("dry_run", True)

    if pos is not None:
        move_object({"name": name, "position": pos}, dry_run=dry_run)
    if rot is not None:
        rotate_object({"name": name, "rotation": rot}, dry_run=dry_run)
    if scale is not None:
        scale_object({"name": name, "scale": scale}, dry_run=dry_run)

def execute_action(action: Dict[str, Any], collection_name: Optional[str] = None, dry_run: bool = True) -> Any:
    """
    Generic entrypoint that safe_executor can call. Accepts an action dict with 'action' or 'type'
    and dispatches to the more specific functions above.
    Returns a short result (string/object) where applicable.
    """
    if not isinstance(action, dict):
        logger.warning("execute_action: expected dict, got %r", type(action))
        return None

    act_name = (action.get("action") or action.get("type") or "").lower()
    # include collection_name into action if provided
    if collection_name:
        action = dict(action)
        action["collection"] = action.get("collection") or collection_name

    # normalize typical synonyms
    if act_name in {"create_primitive", "add_object", "add"}:
        return create_primitive(action, dry_run=dry_run)
    if act_name in {"move_object", "move", "set_location", "translate"}:
        return move_object(action, dry_run=dry_run)
    if act_name in {"rotate_object", "rotate"}:
        return rotate_object(action, dry_run=dry_run)
    if act_name in {"scale_object", "scale"}:
        return scale_object(action, dry_run=dry_run)
    if act_name in {"transform", "set_transform"}:
        return set_transform(action, dry_run=dry_run)
    if act_name in {"delete_object", "delete"}:
        return delete_object(action, dry_run=dry_run)
    if act_name in {"delete_all", "clear_collection"}:
        return delete_all(action, dry_run=dry_run)
    if act_name in {"apply_material_to_object", "apply_material", "material"}:
        return apply_material_to_object(action, dry_run=dry_run)
    if act_name in {"add_modifier_to_object", "add_modifier"}:
        return add_modifier_to_object(action, dry_run=dry_run)
    if act_name in {"apply_modifier_stack", "apply_modifiers"}:
        return apply_modifier_stack(action, dry_run=dry_run)
    if act_name in {"import_image_as_plane", "import_image", "import_image_plane"}:
        return import_image_as_plane(action, dry_run=dry_run)

    # unknown action: try to be helpful
    logger.warning("execute_action: unknown action '%s' — returning raw", act_name)
    return action

# ------------- small IO helper -------------
def atomic_copy_to_outbox(src: str, outdir: str):
    """Copy file to outbox atomically (partial -> rename)."""
    srcp = Path(src)
    outdirp = Path(outdir)
    _ensure_dir(outdirp)
    if not srcp.exists():
        raise FileNotFoundError(src)
    stamp = time.strftime("%Y%m%d%H%M%S")
    final = outdirp / f"render_{stamp}_{os.getpid()}{srcp.suffix}"
    partial = final.with_suffix(final.suffix + ".partial")
    shutil.copy2(str(srcp), str(partial))
    os.replace(str(partial), str(final))
    try:
        os.chmod(str(final), 0o644)
    except Exception:
        pass
    return str(final)

def list_outbox(outdir: str) -> List[str]:
    p = Path(outdir)
    if not p.exists():
        return []
    return [str(x) for x in sorted(p.glob("render_*.png"))]

# End of file
