from .advanced import Sapiens2NormalAdvanced, Sapiens2PoseAdvanced, Sapiens2SegmentationAdvanced
from .easy import (
    Sapiens2ModelLoader,
    Sapiens2ModelLoaderManual,
    Sapiens2ModelUnload,
    Sapiens2Normal,
    Sapiens2Pointmap,
    Sapiens2Pose,
    Sapiens2Segmentation,
)
from .pointmap_advanced import Sapiens2PointmapMeshAdvanced
from .retarget import Sapiens2PoseRetarget, Sapiens2PoseRenderConfig
from .tpose import Sapiens2PoseToTPose
from .transition import Sapiens2PoseTransitionAnimation


NODE_CLASS_MAPPINGS = {
    "Sapiens2ModelLoader": Sapiens2ModelLoader,
    "Sapiens2ModelLoaderManual": Sapiens2ModelLoaderManual,
    "Sapiens2ModelUnload": Sapiens2ModelUnload,
    "Sapiens2Segmentation": Sapiens2Segmentation,
    "Sapiens2SegmentationAdvanced": Sapiens2SegmentationAdvanced,
    "Sapiens2Normal": Sapiens2Normal,
    "Sapiens2NormalAdvanced": Sapiens2NormalAdvanced,
    "Sapiens2Pointmap": Sapiens2Pointmap,
    "Sapiens2PointmapMeshAdvanced": Sapiens2PointmapMeshAdvanced,
    "Sapiens2Pose": Sapiens2Pose,
    "Sapiens2PoseAdvanced": Sapiens2PoseAdvanced,
    "Sapiens2PoseRetarget": Sapiens2PoseRetarget,
    "Sapiens2PoseToTPose": Sapiens2PoseToTPose,
    "Sapiens2PoseRenderConfig": Sapiens2PoseRenderConfig,
    "Sapiens2PoseTransitionAnimation": Sapiens2PoseTransitionAnimation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Sapiens2ModelLoader": "Sapiens2 Model Loader",
    "Sapiens2ModelLoaderManual": "Sapiens2 Manual Model Loader",
    "Sapiens2ModelUnload": "Sapiens2 Unload Model",
    "Sapiens2Segmentation": "Sapiens2 Segmentation",
    "Sapiens2SegmentationAdvanced": "Sapiens2 Segmentation Advanced",
    "Sapiens2Normal": "Sapiens2 Normal",
    "Sapiens2NormalAdvanced": "Sapiens2 Normal Advanced",
    "Sapiens2Pointmap": "Sapiens2 Pointmap",
    "Sapiens2PointmapMeshAdvanced": "Sapiens2 Pointmap Mesh Advanced",
    "Sapiens2Pose": "Sapiens2 Pose",
    "Sapiens2PoseAdvanced": "Sapiens2 Pose Advanced",
    "Sapiens2PoseRetarget": "Sapiens2 Pose Retarget",
    "Sapiens2PoseToTPose": "Sapiens2 Pose To T-Pose",
    "Sapiens2PoseRenderConfig": "Sapiens2 Pose Render Config",
    "Sapiens2PoseTransitionAnimation": "Sapiens2 Pose Transition Animation",
}


