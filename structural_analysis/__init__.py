"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.40.7"
__what_is_new__ = (
    "Fix: N/V/M diagrams + station export use case-filtered / factored "
    "member loads (matches the displayed combination) · "
    "Display: single-sign shear members lobe OUTWARD from the structure "
    "centroid (portal columns mirror); sign-changing members keep the "
    "textbook axis convention (+V above / −V below) · "
    "Top-right release badge: stacked rows so menu titles never get crowded · "
    "Numerical V values, station export, hover read-outs unchanged"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
