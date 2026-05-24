"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.6.0"
__what_is_new__ = (
    "3D Y-up/Z-up viewer · live section preview · grid fills window"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
