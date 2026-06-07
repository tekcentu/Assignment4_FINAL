"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.27.1"
__what_is_new__ = (
    "Selection menu: filter by type/section/material · "
    "Select Similar (right-click) · "
    "Named groups: create/select/manage (Selection → Groups) · "
    "Group element IDs update correctly after Renumber Elements"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
