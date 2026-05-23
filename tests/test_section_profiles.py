"""Tests for material templates, shape calculators, and the Material.G
derived property.

I/O round-trip tests for the new Material/Section fields live further
down in this file (added in Commit 2 once the parser/writer wire them
through)."""

from __future__ import annotations

import math

import pytest

from structural_analysis.model import Material, Section
from structural_analysis.profiles import (
    MATERIAL_TEMPLATES,
    SECTION_SHAPES,
    i_section_properties,
    properties_for_shape,
    rectangle_properties,
    section_outline,
    square_properties,
)


# ── Material.G ────────────────────────────────────────────────


def test_material_G_typical_steel():
    m = Material(id=1, E=2.10e8, nu=0.30)
    assert m.G == pytest.approx(2.10e8 / (2 * 1.30))


def test_material_G_at_nu_zero_is_half_E():
    # G = E / (2*(1+nu)); at nu=0 this is E/2 — not a special case.
    m = Material(id=1, E=2.10e8, nu=0.0)
    assert m.G == pytest.approx(1.05e8)


def test_material_default_nu_is_zero():
    m = Material(id=1, E=1.0)
    assert m.nu == 0.0
    assert m.template == ""


# ── shape calculators ──────────────────────────────────────────


def test_rectangle_properties_basic():
    p = rectangle_properties(b=0.3, h=0.5)
    assert p["A"] == pytest.approx(0.15)
    assert p["I"] == pytest.approx(0.3 * 0.5 ** 3 / 12.0)
    assert p["depth"] == 0.5
    assert p["width"] == 0.3
    assert p["J"] == 0.0


def test_square_properties_matches_rectangle():
    assert square_properties(0.4) == rectangle_properties(0.4, 0.4)


def test_rectangle_rejects_nonpositive():
    with pytest.raises(ValueError):
        rectangle_properties(0.0, 0.5)
    with pytest.raises(ValueError):
        rectangle_properties(0.3, -0.5)


def test_i_section_properties_ipe200_like():
    # IPE-200-ish: h=0.200, b=0.100, tf=0.0085, tw=0.0056
    p = i_section_properties(h=0.200, b=0.100, tf=0.0085, tw=0.0056)
    # Hand calc: A = 2*b*tf + tw*(h-2*tf)
    hw = 0.200 - 2 * 0.0085
    A_expected = 2 * 0.100 * 0.0085 + 0.0056 * hw
    I_expected = (0.100 * 0.200 ** 3 - (0.100 - 0.0056) * hw ** 3) / 12.0
    J_expected = (2 * 0.100 * 0.0085 ** 3 + hw * 0.0056 ** 3) / 3.0
    assert p["A"] == pytest.approx(A_expected, rel=1e-12)
    assert p["I"] == pytest.approx(I_expected, rel=1e-12)
    assert p["J"] == pytest.approx(J_expected, rel=1e-12)
    assert p["depth"] == 0.200
    assert p["width"] == 0.100


def test_i_section_rejects_thick_flange():
    # h <= 2*tf leaves no web
    with pytest.raises(ValueError, match="depth h must be greater than"):
        i_section_properties(h=0.02, b=0.1, tf=0.02, tw=0.005)


def test_i_section_rejects_web_wider_than_flange():
    with pytest.raises(ValueError, match="Web thickness tw cannot exceed"):
        i_section_properties(h=0.2, b=0.05, tf=0.01, tw=0.06)


def test_properties_for_shape_dispatch():
    assert properties_for_shape("rectangle", b=0.3, h=0.5) == \
        rectangle_properties(0.3, 0.5)
    assert properties_for_shape("square", h=0.4) == square_properties(0.4)
    assert properties_for_shape(
        "i_section", h=0.2, b=0.1, tf=0.0085, tw=0.0056,
    ) == i_section_properties(h=0.2, b=0.1, tf=0.0085, tw=0.0056)


def test_properties_for_shape_rejects_manual():
    with pytest.raises(ValueError, match="manual"):
        properties_for_shape("manual")


def test_properties_for_shape_unknown():
    with pytest.raises(ValueError, match="Unknown shape_type"):
        properties_for_shape("circle", r=0.1)


# ── templates ──────────────────────────────────────────────────


def test_material_templates_have_expected_keys():
    assert "Steel_S275" in MATERIAL_TEMPLATES
    assert "Concrete_C30" in MATERIAL_TEMPLATES
    for name, preset in MATERIAL_TEMPLATES.items():
        for key in ("name", "E", "alpha", "density", "nu"):
            assert key in preset, f"{name} missing {key}"


def test_section_shapes_includes_manual():
    assert "manual" in SECTION_SHAPES


# ── I/O round-trip ──────────────────────────────────────────────


def _build_model_with_new_fields():
    from structural_analysis.model import StructuralModel, Node
    from structural_analysis.element import FrameElement2D
    m = StructuralModel()
    m.title = "Profile-roundtrip"
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 1.0, 0.0)
    m.materials[1] = Material(
        id=1, name="Steel_S275", E=2.10e8, alpha=1.2e-5,
        density=7850.0, nu=0.30, template="Steel_S275",
    )
    m.sections[1] = Section(
        id=1, name="Rect300x500", material_id=1,
        A=0.15, I=3.125e-3, depth=0.5, width=0.3,
        J=0.0, shape_type="rectangle", b=0.3, h=0.5,
    )
    m.sections[2] = Section(
        id=2, name="IPE", material_id=1,
        A=2.85e-3, I=1.943e-5, depth=0.200, width=0.100,
        J=6.98e-8, shape_type="i_section",
        b=0.100, h=0.200, tf=0.0085, tw=0.0056,
    )
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2, A=0.15, I=3.125e-3, E=2.10e8,
        section_id=1,
    ))
    return m


def test_round_trip_txt_with_new_fields(tmp_path):
    from structural_analysis.file_io import read_input_file
    from structural_analysis.gui_common.file_writer import write_input_file

    model = _build_model_with_new_fields()
    path = tmp_path / "roundtrip.txt"
    write_input_file(model, str(path))
    reloaded = read_input_file(str(path))

    m1 = reloaded.materials[1]
    assert m1.nu == pytest.approx(0.30)
    assert m1.template == "Steel_S275"
    s1 = reloaded.sections[1]
    assert s1.shape_type == "rectangle"
    assert s1.b == pytest.approx(0.3)
    assert s1.h == pytest.approx(0.5)
    assert s1.width == pytest.approx(0.3)
    assert s1.J == 0.0
    s2 = reloaded.sections[2]
    assert s2.shape_type == "i_section"
    assert s2.tf == pytest.approx(0.0085)
    assert s2.tw == pytest.approx(0.0056)
    assert s2.J == pytest.approx(6.98e-8)


def test_round_trip_txt_legacy_unchanged_when_no_new_fields(tmp_path):
    """A model with all-default new fields must serialize without any
    trailing key=value tokens."""
    from structural_analysis.model import StructuralModel, Node
    from structural_analysis.element import FrameElement2D
    from structural_analysis.gui_common.file_writer import write_input_file

    m = StructuralModel()
    m.title = "Plain"
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 1.0, 0.0)
    m.materials[1] = Material(id=1, name="Steel", E=2.10e8, alpha=1.2e-5)
    m.sections[1] = Section(id=1, name="Rect", material_id=1,
                             A=0.15, I=3.125e-3, depth=0.5)
    m.elements.append(FrameElement2D(
        id=1, node_i=1, node_j=2, A=0.15, I=3.125e-3, E=2.10e8,
        section_id=1,
    ))
    p = tmp_path / "plain.txt"
    write_input_file(m, str(p))
    text = p.read_text()
    # No "key=value" tokens should appear on MATERIALS/SECTIONS rows
    # when all new fields are at default.
    for line in text.splitlines():
        if line.startswith(("1  2.1e+08", "1  1  0.15")):
            assert "=" not in line, f"unexpected kwarg in: {line!r}"


def test_unknown_kwarg_raises(tmp_path):
    from structural_analysis.file_io import read_input_file

    p = tmp_path / "bad.txt"
    p.write_text(
        "TITLE\nT\n\n"
        "NODES 2\n1 0 0\n2 1 0\n\n"
        "MATERIALS 1\n1 2.1e8 1.2e-5 0 Steel bogus=42\n\n"
        "SECTIONS 1\n1 1 0.1 1e-4 0.2 Name\n\n"
        "ELEMENTS 1\n1 1 2 1 FRAME\n"
    )
    with pytest.raises(ValueError, match="bogus"):
        read_input_file(str(p))


def test_round_trip_spa_json_with_new_fields(tmp_path):
    """The .spa.json format embeds the .txt under model_txt — verify
    new fields survive that path too."""
    from structural_analysis.gui_qt.project_io import (
        Project, save_project_json, load_project_json,
    )

    model = _build_model_with_new_fields()
    path = tmp_path / "project.spa.json"
    save_project_json(Project(model=model, title=model.title), str(path))
    reloaded_project = load_project_json(str(path))
    reloaded = reloaded_project.model

    m1 = reloaded.materials[1]
    assert m1.nu == pytest.approx(0.30)
    assert m1.template == "Steel_S275"
    s1 = reloaded.sections[1]
    assert s1.shape_type == "rectangle"
    assert s1.b == pytest.approx(0.3) and s1.h == pytest.approx(0.5)


# ── section_outline (3D viewer feed) ───────────────────────────


def test_section_outline_rectangle_has_4_vertices():
    s = Section(id=1, shape_type="rectangle", b=0.3, h=0.5)
    pts = section_outline(s)
    assert len(pts) == 4
    ys = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    # depth along y, width along z
    assert max(ys) - min(ys) == pytest.approx(0.5)
    assert max(zs) - min(zs) == pytest.approx(0.3)


def test_section_outline_i_section_has_12_vertices():
    s = Section(
        id=1, shape_type="i_section",
        b=0.100, h=0.200, tf=0.0085, tw=0.0056,
    )
    pts = section_outline(s)
    # Standard I outline traces all 12 corners.
    assert len(pts) == 12
    ys = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    assert max(ys) - min(ys) == pytest.approx(0.200)
    assert max(zs) - min(zs) == pytest.approx(0.100)


def test_section_outline_manual_uses_sqrt_A():
    s = Section(id=1, shape_type="manual", A=0.16)
    pts = section_outline(s)
    assert len(pts) == 4
    side_y = max(p[0] for p in pts) - min(p[0] for p in pts)
    side_z = max(p[1] for p in pts) - min(p[1] for p in pts)
    # √0.16 = 0.4 m square area-equivalent.
    assert side_y == pytest.approx(0.4)
    assert side_z == pytest.approx(0.4)


def test_section_outline_manual_with_zero_area_uses_fallback():
    s = Section(id=1, shape_type="manual", A=0.0)
    pts = section_outline(s, fallback_size=0.05)
    assert len(pts) == 4
    side = max(p[0] for p in pts) - min(p[0] for p in pts)
    assert side == pytest.approx(0.05)


def test_section_outline_unknown_shape_raises():
    s = Section(id=1, shape_type="hexagon", A=0.1)
    with pytest.raises(ValueError, match="Unknown shape_type"):
        section_outline(s)


def test_round_trip_example_file_unchanged(tmp_path):
    """Save-and-reload one of the shipped example files: all default
    fields must remain absent."""
    import pathlib
    from structural_analysis.file_io import read_input_file
    from structural_analysis.gui_common.file_writer import write_input_file

    src = pathlib.Path(__file__).parent.parent / "inputs"
    candidates = sorted(src.glob("example_*.txt"))
    assert candidates, "no example_*.txt files in inputs/"
    example = candidates[0]
    m1 = read_input_file(str(example))
    out = tmp_path / "saved.txt"
    write_input_file(m1, str(out))
    text = out.read_text()
    # Plain example files have no new-field data — none of the new
    # kwargs should appear.
    for token in ("nu=", "template=", "shape=", "width=", "J=",
                  "b=", "h=", "tf=", "tw="):
        assert token not in text, (
            f"saving an untouched example introduced {token!r}: {text}"
        )
    # And the reload should still work.
    read_input_file(str(out))
