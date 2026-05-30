"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.15.0"
__what_is_new__ = (
    "UDL/PointLoad axial components (wx, px) · "
    "local vs global load direction · "
    "N diagrams reflect axial member loads"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
