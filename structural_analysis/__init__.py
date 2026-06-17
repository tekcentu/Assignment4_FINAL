"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.40.2"
__what_is_new__ = (
    "File → Export station results CSV (21 stations/member, SAP-compare) · "
    "Columns scaled to active Units V1 preset · "
    "Truss rows emit N only"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
