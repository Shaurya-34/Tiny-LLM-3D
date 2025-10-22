"""
worker package for Tiny-LLM-3D

Handles:
- Blender executor (runs JSON-defined actions)
- Safe executor wrapper (sandbox logic)
- Blender wrapper utilities (bpy helpers)
- Inbox watcher and schema validation
"""

import importlib
import logging
import sys
from pathlib import Path
from typing import Optional, Dict

# ------------------------------
# Package Metadata
# ------------------------------
__version__ = "1.0.0"
__author__ = "Tiny-LLM-3D Team"

# ------------------------------
# Logging Setup
# ------------------------------
logger = logging.getLogger("ai_copilot")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ------------------------------
# Dependency Guards
# ------------------------------
try:
    import bpy  # type: ignore
    _HAS_BPY = True
except Exception:
    _HAS_BPY = False

# ------------------------------
# Path Helpers
# ------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
INBOX_DIR = PACKAGE_ROOT / "inbox"
OUTBOX_DIR = PACKAGE_ROOT / "outbox"
SCHEMA_PATH = PACKAGE_ROOT / "action_schema.json"

for d in (INBOX_DIR, OUTBOX_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------
# Lazy imports / Exports
# ------------------------------
# We'll attempt to import the common modules and expose them as attributes.
# If a module is missing we set the name to None so callers can detect.
_blender_wrapper = None
_safe_executor = None
_blender_executor = None

def _try_import(name: str):
    try:
        return importlib.import_module(f".{name}", package=__name__)
    except Exception:
        logger.debug("worker.%s not importable", name, exc_info=False)
        return None

_blender_wrapper = _try_import("blender_wrapper")
_safe_executor = _try_import("safe_executor")
_blender_executor = _try_import("blender_executor")

# Expose public names
blender_wrapper = _blender_wrapper
safe_executor = _safe_executor
blender_executor = _blender_executor

__all__ = ["blender_wrapper", "safe_executor", "blender_executor", "is_blender_available", "get_worker_paths", "reload_modules"]

# ------------------------------
# Utility
# ------------------------------
def is_blender_available() -> bool:
    """Return True if bpy (Blender Python API) is importable."""
    return _HAS_BPY

def get_worker_paths() -> Dict[str, str]:
    """Convenient access to core worker directories."""
    return {
        "root": str(PACKAGE_ROOT),
        "inbox": str(INBOX_DIR),
        "outbox": str(OUTBOX_DIR),
        "schema": str(SCHEMA_PATH),
    }

def reload_modules() -> None:
    """
    Re-import core modules (useful in REPL/testing). Keeps attributes updated.
    """
    global blender_wrapper, safe_executor, blender_executor
    blender_wrapper = _try_import("blender_wrapper")
    safe_executor = _try_import("safe_executor")
    blender_executor = _try_import("blender_executor")
    logger.info("worker modules reloaded: blender_wrapper=%s safe_executor=%s blender_executor=%s",
                "present" if blender_wrapper else "missing",
                "present" if safe_executor else "missing",
                "present" if blender_executor else "missing")

logger.debug("worker package initialized (bpy=%s)", _HAS_BPY)
