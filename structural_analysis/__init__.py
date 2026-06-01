"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.22.4"
__what_is_new__ = (
    "Faster canvas redraws (batched element/node/selection artists) · "
    "dense-view auto-hides IDs with a halo for legibility · "
    "labeled grid culls to the visible viewport"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
