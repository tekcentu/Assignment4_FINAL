"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.30.4"
__what_is_new__ = (
    "Fix frame V/M diagram sign under distributed loads · "
    "Shear now crosses zero correctly under UDL · "
    "dM/dx = V preserved (solver math unchanged)"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
