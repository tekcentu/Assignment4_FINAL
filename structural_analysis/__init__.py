"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.9.1"
__what_is_new__ = (
    "assembled joint masses · consistent-mass diagnostics"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
