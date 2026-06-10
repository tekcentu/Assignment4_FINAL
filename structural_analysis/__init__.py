"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.31.1"
__what_is_new__ = (
    "Assign / Clear rigid offsets commands (selection or all frames) · "
    "Auto-offsets from physical joint overlap with safe-span guard · "
    "Creation dialog stays simple; L_total / L_flex live in Element Details"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
