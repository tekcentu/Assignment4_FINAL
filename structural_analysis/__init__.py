"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.30.2"
__what_is_new__ = (
    "Modal results say 'Structure' not 'Component' · "
    "Per-structure explanatory note in modal dialog · "
    "Internal code names unchanged"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
