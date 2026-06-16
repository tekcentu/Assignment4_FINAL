"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.38.0"
__what_is_new__ = (
    "Precast: per-stage DAF + enable/disable with grey-out · "
    "Richer stage sketch (UDL band, reaction arrows + values, sling T/H) · "
    "OK/WARNING status chips and high-DAF soft warning"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
