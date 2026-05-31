"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.22.0"
__what_is_new__ = (
    "Pre-solve validation + canvas highlighting · "
    "truss free-end mechanism detection · "
    "Solve All skips empty load cases"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
