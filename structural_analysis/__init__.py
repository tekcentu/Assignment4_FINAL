"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.41.0"
__what_is_new__ = (
    "Case selector now filters canvas load glyphs (works before solve) · "
    "Combination shows factored effective loads · "
    "SUM_ALL shows every load"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
