"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.34.0"
__what_is_new__ = (
    "Fix: default reference grid stays visible when a structural grid is shown · "
    "Fix: grid A/B/1/2 labels now gated by the toggle (off by default) · "
    "Grid lines + plain coordinates always show; letters only when enabled"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
