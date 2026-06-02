"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.23.0"
__what_is_new__ = (
    "Tabbed Element Detail Dialog — Properties, Results, Load Assignments · "
    "Results tab has its own case/combo selector (raw-id userData) · "
    "Loads tab supports per-row Add/Edit/Delete (all undoable)"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
