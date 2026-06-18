"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.41.0"
__what_is_new__ = (
    "Building Wizard: separate beam / column section selectors · "
    "Clearer 'Building Wizard' button label · "
    "Validates both sections before generating"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
