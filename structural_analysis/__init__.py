"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.18.0"
__what_is_new__ = (
    "Load case manager · "
    "case-by-case static run with SUM_ALL view · "
    "per-case result switching from the toolbar"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
