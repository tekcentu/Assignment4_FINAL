"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.35.0"
__what_is_new__ = (
    "Precast Handling Stages: check one frame member through lifting / stock / "
    "truck stages · Reactions, sling tensions, V/M (shared diagram helpers) · "
    "Temporary tool — does not change the model"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
