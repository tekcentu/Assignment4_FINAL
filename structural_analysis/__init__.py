"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.39.0"
__what_is_new__ = (
    "Precast: sling angle relabelled 'from horizontal (T/H only)' + tooltip · "
    "Auto angle from hook height + lift-point spacing · "
    "Low-angle (< 45°) soft warning"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
