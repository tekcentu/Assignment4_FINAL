"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.25.0"
__what_is_new__ = (
    "Modal Mass Source Settings (Run menu) · "
    "Joint masses storage + file I/O · "
    "New GUI models default self-weight to DEAD"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
