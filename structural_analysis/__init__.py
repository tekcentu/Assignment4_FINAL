"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.42.0"
__what_is_new__ = (
    "Final-submission simplification: modal analysis now uses LUMPED / "
    "row-sum mass only — Consistent mass removed from the GUI and from "
    "the public solve_modal default · "
    "Passing the legacy 'consistent' value still works (DeprecationWarning, "
    "maps to lumped) — saved files keep loading · "
    "Massless rotational DOFs handled by Guyan condensation, no artificial mass"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
