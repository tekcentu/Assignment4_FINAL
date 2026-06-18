"""2D Structural Analysis Package — CE 4011 Assignment 3."""

__version__ = "0.40.6"
__what_is_new__ = (
    "Fix: N/V/M diagrams + station export use case-filtered / factored "
    "member loads (matches the displayed combination) · "
    "Display: every shear lobe (+V and −V) extends OUTWARD from the "
    "structure centroid; sign is shown by colour only · "
    "Top-right release badge: stacked rows (one clause per row) so the "
    "File / Edit / View menu titles never get crowded · "
    "Numerical V values, station export, hover read-outs unchanged"
)

from .model import (
    StructuralModel, Node, Material, Support, NodalLoad,
    UniformDistributedLoad, PointLoad, AnalysisResult,
)
from .element import Element2D, FrameElement2D, TrussElement2D
from .main import run_analysis, run_from_file
