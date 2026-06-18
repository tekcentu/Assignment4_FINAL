"""Qt-free helpers that decide which load glyphs the canvas should draw.

This is a *display/filtering* layer only — it never touches the solver,
the assembled system, member-force recovery, or any numeric result. It
answers a single question for the canvas:

    "Given the active case/combination selection, which nodal and member
     loads should be drawn, and at what (possibly factored) magnitude?"

The canvas calls these for both the load-arrow auto-scale pass and the
actual draw pass, so the on-screen arrow lengths reflect exactly the set
of loads that is visible.

Display modes
-------------
``ACTIVE_CASE`` (default)
    Show only the loads belonging to the active selection:

    * a plain load case  → loads whose ``load_case`` equals the active case
    * a load combination → the referenced cases' loads, each **scaled by
      its combination factor** (factored effective loads). The original
      ``load_case`` tag is preserved on the scaled copy so the canvas'
      combination highlighting still treats them as constituent loads.
    * ``SUM_ALL``        → every load (same as ``ALL``)

``ALL``
    The legacy view — every load glyph at its stored magnitude.

``HIDE``
    Draw no load glyphs at all.

The functions return plain model dataclasses (frozen), producing
``dataclasses.replace`` copies only when a combination factor must be
applied. Original model objects are never mutated.
"""

from __future__ import annotations

import dataclasses

from ..model import (
    FrameTemperatureLoad,
    NodalLoad,
    PointLoad,
    StructuralModel,
    TrussTemperatureLoad,
    UniformDistributedLoad,
)
from ..multi_case_result import SUM_ALL_KEY

# Display-mode tokens (kept as bare strings so callers/tests don't need an
# enum import; the canvas stores one of these in ``_load_display_mode``).
ACTIVE_CASE = "active_case"
ALL = "all"
HIDE = "hide"

DEFAULT_MODE = ACTIVE_CASE


def _scale_load(ld, factor: float):
    """Return a magnitude-scaled copy of a load, preserving its case tag.

    Only the magnitude fields are scaled; geometry (node id, position
    ``a``, coord system) and the ``load_case`` tag are untouched so the
    factored copy still draws in the right place and still highlights as
    a constituent of the active combination.
    """
    if factor == 1.0:
        return ld
    if isinstance(ld, NodalLoad):
        return dataclasses.replace(
            ld, fx=ld.fx * factor, fy=ld.fy * factor, mz=ld.mz * factor,
        )
    if isinstance(ld, UniformDistributedLoad):
        # gravity loads keep wx == 0 (0 * factor == 0) so __post_init__ holds.
        return dataclasses.replace(
            ld, wy=ld.wy * factor, wx=ld.wx * factor,
        )
    if isinstance(ld, PointLoad):
        return dataclasses.replace(
            ld, py=ld.py * factor, px=ld.px * factor,
        )
    if isinstance(ld, TrussTemperatureLoad):
        return dataclasses.replace(ld, delta_T=ld.delta_T * factor)
    if isinstance(ld, FrameTemperatureLoad):
        return dataclasses.replace(
            ld, t_top=ld.t_top * factor, t_bottom=ld.t_bottom * factor,
        )
    return ld


def _case_of(ld) -> str:
    return getattr(ld, "load_case", "DEFAULT")


def _filter_for_selection(loads, active_case, load_combinations, mode):
    """Shared core for nodal/member filtering. ``loads`` is any iterable of
    load objects carrying a ``load_case`` attribute."""
    if mode == HIDE:
        return []
    if mode == ALL or active_case == SUM_ALL_KEY:
        return list(loads)
    # ACTIVE_CASE mode.
    combos = load_combinations or {}
    if active_case in combos:
        terms = combos[active_case].terms
        out = []
        for ld in loads:
            factor = terms.get(_case_of(ld))
            if factor:  # skip cases not referenced (None) or zero factor
                out.append(_scale_load(ld, factor))
        return out
    return [ld for ld in loads if _case_of(ld) == active_case]


def visible_nodal_loads(
    model: StructuralModel,
    active_case: str,
    load_combinations: dict | None,
    mode: str = DEFAULT_MODE,
) -> list:
    """Nodal loads to draw for the active selection (see module docstring)."""
    return _filter_for_selection(
        model.nodal_loads, active_case, load_combinations, mode,
    )


def visible_member_loads(
    elem,
    active_case: str,
    load_combinations: dict | None,
    mode: str = DEFAULT_MODE,
) -> list:
    """Member loads on ``elem`` to draw for the active selection."""
    return _filter_for_selection(
        getattr(elem, "member_loads", []), active_case, load_combinations, mode,
    )
