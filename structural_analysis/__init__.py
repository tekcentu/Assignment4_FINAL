"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.40.3"
__what_is_new__ = (
    "Display: shear-diagram lobes mirror outward for portal columns · "
    "Numerical V values, station export, hover read-outs unchanged · "
    "Single-member / collinear models keep the +y_local convention"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
