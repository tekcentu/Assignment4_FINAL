"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.41.0"
__what_is_new__ = (
    "Mechanics fix: a UDL on a frame member with rigid end offsets now "
    "loads the FULL member (ΣR = w·L_total), with rigid-zone load "
    "transferred to the joints (force + moment); inclined members verified "
    "by global equilibrium · "
    "Diagrams integrate the UDL from x=0 so flexible-span values stay "
    "consistent with the full-length load · "
    "Shear/moment display: canonical direction + world-anchored side "
    "(node-order invariant, +V above/right) · "
    "Numerical reactions for point/thermal/self-weight paths unchanged"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
