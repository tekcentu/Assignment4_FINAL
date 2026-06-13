"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.30.9"
__what_is_new__ = (
    "Origin X/Y arrows are a fixed on-screen size — no longer overshoot on zoom · "
    "Shown generated grid puts its line coordinates on the axes (constant values) · "
    "Default/generated grids independent; snap behavior still unchanged"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
