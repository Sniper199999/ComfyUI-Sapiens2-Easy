from .sapiens2_nodes.folders import register_model_folders
from .sapiens2_nodes.model_loading import _ensure_sapiens_importable

register_model_folders()

try:
    _ensure_sapiens_importable("")
except Exception:
    pass

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

