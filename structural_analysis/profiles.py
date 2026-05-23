"""Material templates and section-shape property calculators.

Stage 1 of the section-profile system. This module is **the single
source of truth** for:

- material presets (``MATERIAL_TEMPLATES``) consumed by the
  MaterialDialog "Template" combobox;
- section-shape names (``SECTION_SHAPES``);
- pure-function calculators that turn raw dimensions (b, h, tf, tw)
  into the A / I / depth / width / J that go onto a :class:`Section`;
- the 2D cross-section outline (``section_outline``) used by the 3D
  viewer to extrude each element into a prism.

The calculators return a dict whose keys match Section field names so
the dialog can splat them straight into ``dataclasses.replace`` /
``Section(...)``. ``J`` (torsion constant) is computed only for
I-sections — rectangles and squares leave J=0, which is consistent
with the 2D solver ignoring J.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Section


MATERIAL_TEMPLATES: dict[str, dict[str, float | str]] = {
    "Steel_S275": dict(
        name="Steel_S275", E=2.10e8, alpha=1.2e-5, density=7850.0, nu=0.30,
    ),
    "Concrete_C30": dict(
        name="Concrete_C30", E=3.30e7, alpha=1.0e-5, density=2500.0, nu=0.20,
    ),
}

SECTION_SHAPES: tuple[str, ...] = ("manual", "rectangle", "square", "i_section")


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")


def rectangle_properties(b: float, h: float) -> dict[str, float]:
    """Cross-section properties for a solid rectangle b × h.

    Returns a dict with keys A, I, depth, width, J that can be applied
    to a :class:`Section`. ``J`` is left at 0 — see module docstring.
    """
    _require_positive("b", b)
    _require_positive("h", h)
    return dict(A=b * h, I=b * h ** 3 / 12.0, depth=h, width=b, J=0.0)


def square_properties(h: float) -> dict[str, float]:
    """Solid square of side ``h``."""
    return rectangle_properties(h, h)


def i_section_properties(
    h: float, b: float, tf: float, tw: float
) -> dict[str, float]:
    """Doubly-symmetric I-section.

    h is the overall depth, b the flange width, tf the flange
    thickness, tw the web thickness. J uses the standard thin-walled
    open-section approximation::

        J ≈ ( 2·b·tf³  +  (h - 2·tf)·tw³ ) / 3

    Raises:
        ValueError: if any dimension is non-positive, if ``h ≤ 2 tf``
            (no web left), or if ``tw > b`` (web wider than flange).
    """
    for name, val in (("h", h), ("b", b), ("tf", tf), ("tw", tw)):
        _require_positive(name, val)
    if h <= 2.0 * tf:
        raise ValueError(
            f"I-section depth h must be greater than 2·tf "
            f"(got h={h!r}, tf={tf!r})."
        )
    if tw > b:
        raise ValueError(
            f"Web thickness tw cannot exceed flange width b "
            f"(got tw={tw!r}, b={b!r})."
        )

    hw = h - 2.0 * tf  # web height
    A = 2.0 * b * tf + tw * hw
    I = (b * h ** 3 - (b - tw) * hw ** 3) / 12.0
    J = (2.0 * b * tf ** 3 + hw * tw ** 3) / 3.0
    return dict(A=A, I=I, depth=h, width=b, J=J)


def properties_for_shape(shape_type: str, **dims: float) -> dict[str, float]:
    """Dispatch helper: look up the calculator by shape name.

    Raises ValueError on unknown shape or on ``shape_type="manual"`` —
    manual sections have no derived properties.
    """
    if shape_type == "rectangle":
        return rectangle_properties(b=dims["b"], h=dims["h"])
    if shape_type == "square":
        return square_properties(h=dims["h"])
    if shape_type == "i_section":
        return i_section_properties(
            h=dims["h"], b=dims["b"], tf=dims["tf"], tw=dims["tw"],
        )
    if shape_type == "manual":
        raise ValueError(
            "shape_type='manual' has no derived properties — read "
            "A/I/depth/width directly from the dialog instead."
        )
    raise ValueError(
        f"Unknown shape_type {shape_type!r}; expected one of {SECTION_SHAPES}."
    )


# ── outline polygons used by the 3D viewer ────────────────────


def section_outline(
    section: "Section", *, fallback_size: float = 0.1,
) -> list[tuple[float, float]]:
    """Return ``(y, z)`` vertices tracing the section's cross-section
    outline once in the element's local frame.

    Convention: depth (h) lies along local y (the in-plane axis),
    width (b) lies along local z (out of the 2D plane).

    For ``shape_type="manual"`` no real geometry is known; the viewer
    falls back to a square area-equivalent prism with side ``√A``. If
    A is also zero, the caller's ``fallback_size`` is used so the
    element is still visible.
    """
    shape = section.shape_type
    if shape == "rectangle":
        b, h = section.b, section.h
    elif shape == "square":
        b = h = section.h
    elif shape == "i_section":
        b, h, tf, tw = section.b, section.h, section.tf, section.tw
        hy = h / 2.0
        bz = b / 2.0
        twz = tw / 2.0
        wy = hy - tf  # web top/bottom y
        # CCW starting at top-right corner of top flange.
        return [
            ( hy,  bz),
            ( hy, -bz),
            ( wy, -bz),
            ( wy, -twz),
            (-wy, -twz),
            (-wy, -bz),
            (-hy, -bz),
            (-hy,  bz),
            (-wy,  bz),
            (-wy,  twz),
            ( wy,  twz),
            ( wy,  bz),
        ]
    elif shape == "manual":
        if section.A > 0.0:
            side = math.sqrt(section.A)
        else:
            side = fallback_size
        b = h = side
    else:
        raise ValueError(
            f"Unknown shape_type {shape!r}; expected one of {SECTION_SHAPES}."
        )

    hy = h / 2.0
    bz = b / 2.0
    return [( hy,  bz), ( hy, -bz), (-hy, -bz), (-hy,  bz)]
