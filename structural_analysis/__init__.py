"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.40.8"
__what_is_new__ = (
    "Fix: N/V/M diagrams + station export use case-filtered / factored "
    "member loads (matches the displayed combination) · "
    "Display: shear diagram side is world-anchored & sign-based — "
    "horizontal +V above / −V below, vertical +V right / −V left (SAP-like) · "
    "Top-right release badge: stacked rows so menu titles never get crowded · "
    "Numerical V values, station export, hover read-outs, moment unchanged"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
