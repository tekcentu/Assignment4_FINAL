"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.40.1"
__what_is_new__ = (
    "Units V1: N/V/M hover + member-end-force readouts now convert · "
    "Load arrows stay explicitly labelled kN / kN·m / kN/m (never misleading) · "
    "Inputs/coords still kN-m (display-only)"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
