"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.30.8"
__what_is_new__ = (
    "Default and generated grids are independent visual layers (toggle each) · "
    "View → Grid: Generate from nodes, Clear generated grid (undoable) · "
    "Snap behavior unchanged — display toggles never silently affect clicks"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
