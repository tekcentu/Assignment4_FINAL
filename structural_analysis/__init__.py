"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.40.9"
__what_is_new__ = (
    "Fix: N/V/M diagrams + station export use case-filtered / factored "
    "member loads (matches the displayed combination) · "
    "Display: shear/moment diagrams use a canonical display direction "
    "(left→right or bottom→top) so node-order doesn't affect the visual · "
    "Shear side is world-anchored: +V above/right, −V below/left (SAP-like) · "
    "Top-right release badge: stacked rows so menu titles never get crowded · "
    "Numerical sample_internal_force, station export, hover read-outs unchanged"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
