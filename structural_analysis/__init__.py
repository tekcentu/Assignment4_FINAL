"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.33.0"
__what_is_new__ = (
    "Section dimension labels (b / h / tf / tw) now share one drawing helper · "
    "Labels pinned to the panel edges with leaders — no more clipped 'h =' on squares · "
    "Element-Details thumbnail and Add-Section preview can no longer drift"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
