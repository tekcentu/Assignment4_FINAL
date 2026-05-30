"""Unit tests for the pure load-summary formatter used by the
inspector's loads table and the multi-select status bar."""

from __future__ import annotations

from structural_analysis.element import FrameElement2D, TrussElement2D
from structural_analysis.gui_common.commands import (
    AddMemberCmd,
    AddNodeCmd,
)
from structural_analysis.gui_qt.load_summary import (
    ElementLoadRow,
    format_element_loads,
    format_selection_load_counts,
    summarize_selection_loads,
)
from structural_analysis.model import (
    FrameTemperatureLoad,
    Material,
    PointLoad,
    Section,
    StructuralModel,
    TrussTemperatureLoad,
    UniformDistributedLoad,
)


def _model_with_one_frame(length: float = 6.0) -> tuple[StructuralModel, int]:
    m = StructuralModel(title="t")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    AddMemberCmd(
        x_i=0.0, y_i=0.0, x_j=length, y_j=0.0,
        kind="frame", section_id=1,
    ).do(m)
    return m, m.elements[0].id


def _model_with_one_truss(length: float = 4.0) -> tuple[StructuralModel, int]:
    m = StructuralModel(title="t")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    AddMemberCmd(
        x_i=0.0, y_i=0.0, x_j=length, y_j=0.0,
        kind="truss", section_id=1,
    ).do(m)
    return m, m.elements[0].id


def test_format_no_loads_returns_empty_list():
    m, _ = _model_with_one_frame()
    rows = format_element_loads(m, m.elements[0])
    assert rows == []


def test_format_udl_row_shows_local_y():
    m, _ = _model_with_one_frame()
    m.elements[0].member_loads.append(UniformDistributedLoad(wy=-10.0))
    rows = format_element_loads(m, m.elements[0])
    assert len(rows) == 1
    r = rows[0]
    assert r.index == 0
    assert r.kind == "UDL"
    assert "wy = -10" in r.magnitude
    assert "local" in r.position.lower() or "local" in r.meaning.lower()


def test_format_point_load_shows_a_distance():
    m, _ = _model_with_one_frame(length=6.0)
    m.elements[0].member_loads.append(PointLoad(py=-20.0, a=2.5))
    rows = format_element_loads(m, m.elements[0])
    assert rows[0].kind == "PointLoad"
    assert "py = -20" in rows[0].magnitude
    assert "a = 2.5" in rows[0].position
    assert "split" not in rows[0].position.lower()


def test_format_point_load_at_a_equals_L_labels_split_endpoint():
    m, _ = _model_with_one_frame(length=6.0)
    m.elements[0].member_loads.append(PointLoad(py=-20.0, a=6.0))
    rows = format_element_loads(m, m.elements[0])
    assert "j-end" in rows[0].position or "split" in rows[0].position.lower()


def test_format_point_load_at_a_equals_zero_labels_i_end():
    m, _ = _model_with_one_frame(length=6.0)
    m.elements[0].member_loads.append(PointLoad(py=-20.0, a=0.0))
    rows = format_element_loads(m, m.elements[0])
    assert "i-end" in rows[0].position


def test_format_frame_thermal_uniform_says_axial():
    m, _ = _model_with_one_frame()
    m.elements[0].member_loads.append(
        FrameTemperatureLoad(t_top=25.0, t_bottom=25.0)
    )
    rows = format_element_loads(m, m.elements[0])
    assert rows[0].kind == "Thermal"
    assert "frame" in rows[0].type_label.lower()
    assert "axial" in rows[0].meaning.lower()


def test_format_frame_thermal_gradient_says_bending():
    m, _ = _model_with_one_frame()
    m.elements[0].member_loads.append(
        FrameTemperatureLoad(t_top=10.0, t_bottom=-10.0)
    )
    rows = format_element_loads(m, m.elements[0])
    assert "bending" in rows[0].meaning.lower()


def test_format_truss_thermal_says_uniform():
    m, _ = _model_with_one_truss()
    m.elements[0].member_loads.append(TrussTemperatureLoad(delta_T=15.0))
    rows = format_element_loads(m, m.elements[0])
    assert rows[0].kind == "Thermal"
    assert "truss" in rows[0].type_label.lower()
    assert "ΔT = 15" in rows[0].magnitude
    assert "axial" in rows[0].meaning.lower()


def test_repeated_thermal_yields_two_rows_not_one():
    """Adding the same thermal load twice MUST surface as two distinct
    rows — the spec requires duplicates be visible, not hidden."""
    m, _ = _model_with_one_frame()
    m.elements[0].member_loads.append(
        FrameTemperatureLoad(t_top=10.0, t_bottom=10.0)
    )
    m.elements[0].member_loads.append(
        FrameTemperatureLoad(t_top=10.0, t_bottom=10.0)
    )
    rows = format_element_loads(m, m.elements[0])
    assert len(rows) == 2
    assert rows[0].index == 0 and rows[1].index == 1


def test_format_mixed_loads_preserves_storage_order():
    m, _ = _model_with_one_frame(length=6.0)
    elem = m.elements[0]
    elem.member_loads.append(UniformDistributedLoad(wy=-10.0))
    elem.member_loads.append(PointLoad(py=-20.0, a=3.0))
    elem.member_loads.append(FrameTemperatureLoad(t_top=10.0, t_bottom=30.0))
    rows = format_element_loads(m, elem)
    assert [r.kind for r in rows] == ["UDL", "PointLoad", "Thermal"]
    assert [r.index for r in rows] == [0, 1, 2]


def test_summarize_selection_groups_counts_by_kind():
    m = StructuralModel(title="t")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    AddMemberCmd(
        x_i=0.0, y_i=0.0, x_j=4.0, y_j=0.0,
        kind="frame", section_id=1,
    ).do(m)
    AddMemberCmd(
        x_i=4.0, y_i=0.0, x_j=8.0, y_j=0.0,
        kind="frame", section_id=1,
    ).do(m)
    a, b = m.elements[0], m.elements[1]
    a.member_loads.append(UniformDistributedLoad(wy=-5.0))
    a.member_loads.append(PointLoad(py=-2.0, a=1.0))
    b.member_loads.append(UniformDistributedLoad(wy=-5.0))
    b.member_loads.append(FrameTemperatureLoad(t_top=5.0, t_bottom=5.0))
    counts = summarize_selection_loads(m, [a.id, b.id])
    assert counts == {"UDL": 2, "PointLoad": 1, "Thermal": 1}


def test_summarize_selection_ignores_unselected_elements():
    m = StructuralModel(title="t")
    m.materials[1] = Material(id=1, name="Steel", E=2.1e8, density=7850.0)
    m.sections[1] = Section(
        id=1, name="S", material_id=1, A=0.01, I=1e-4, depth=0.3,
    )
    AddMemberCmd(
        x_i=0.0, y_i=0.0, x_j=4.0, y_j=0.0,
        kind="frame", section_id=1,
    ).do(m)
    AddMemberCmd(
        x_i=4.0, y_i=0.0, x_j=8.0, y_j=0.0,
        kind="frame", section_id=1,
    ).do(m)
    a, b = m.elements[0], m.elements[1]
    a.member_loads.append(UniformDistributedLoad(wy=-5.0))
    b.member_loads.append(UniformDistributedLoad(wy=-5.0))
    counts = summarize_selection_loads(m, [a.id])
    assert counts == {"UDL": 1, "PointLoad": 0, "Thermal": 0}


def test_format_selection_load_counts_skips_zero_categories():
    text = format_selection_load_counts(
        {"UDL": 2, "PointLoad": 0, "Thermal": 1}
    )
    assert text == "2 UDL · 1 Thermal"


def test_format_selection_load_counts_empty_returns_empty_string():
    text = format_selection_load_counts(
        {"UDL": 0, "PointLoad": 0, "Thermal": 0}
    )
    assert text == ""


def test_element_load_row_is_frozen_dataclass():
    row = ElementLoadRow(
        index=0, kind="UDL", type_label="UDL",
        magnitude="wy = -10 kN/m",
        position="Full length, local axes",
        meaning="Transverse line load (local axes)",
    )
    assert row.index == 0
    assert row.kind == "UDL"


# ── PR #25 — coord_system label + axial component formatting ────────


def test_format_udl_with_only_wy_keeps_compact_form():
    m, _ = _model_with_one_frame()
    m.elements[0].member_loads.append(UniformDistributedLoad(wy=-10.0))
    rows = format_element_loads(m, m.elements[0])
    assert "wy = -10" in rows[0].magnitude
    # wx component is zero — should NOT appear in the magnitude string.
    assert "wx" not in rows[0].magnitude


def test_format_udl_with_both_wx_and_wy_shows_two_component_form():
    m, _ = _model_with_one_frame()
    m.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-10.0, wx=4.0)
    )
    rows = format_element_loads(m, m.elements[0])
    assert "wx" in rows[0].magnitude and "wy" in rows[0].magnitude
    assert "4" in rows[0].magnitude
    assert "-10" in rows[0].magnitude


def test_format_local_udl_position_says_local_axes():
    m, _ = _model_with_one_frame()
    m.elements[0].member_loads.append(UniformDistributedLoad(wy=-10.0))
    rows = format_element_loads(m, m.elements[0])
    assert "local" in rows[0].position.lower()


def test_format_global_udl_position_says_global_axes():
    m, _ = _model_with_one_frame()
    m.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-10.0, coord_system="global")
    )
    rows = format_element_loads(m, m.elements[0])
    assert "global" in rows[0].position.lower()


def test_format_global_pointload_labels_global_frame():
    """In global mode, components surface as pX / pY (uppercase axes)
    rather than the local px / py — v0.16.0 semantic distinction."""
    m, _ = _model_with_one_frame()
    m.elements[0].member_loads.append(
        PointLoad(py=-20.0, a=3.0, px=5.0, coord_system="global")
    )
    rows = format_element_loads(m, m.elements[0])
    assert "global" in rows[0].position.lower()
    assert "pX" in rows[0].magnitude and "pY" in rows[0].magnitude


# ── PR #26 — gravity label + qX/qY semantic naming + local-eq note ──


def test_format_global_udl_uses_qX_qY_naming():
    """v0.16: global mode says 'qY' (uppercase Y), not 'wy', to make
    the distinction from local-axis components obvious in the load list."""
    m, _ = _model_with_one_frame()
    m.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-10.0, coord_system="global")
    )
    rows = format_element_loads(m, m.elements[0])
    # Compact form when only one global component is non-zero.
    assert "qY" in rows[0].magnitude
    assert "wy" not in rows[0].magnitude


def test_format_gravity_udl_shows_magnitude_and_label():
    """Gravity loads use a single 'magnitude' string and the position
    column says 'Gravity (global -Y)'."""
    m, _ = _model_with_one_frame()
    m.elements[0].member_loads.append(
        UniformDistributedLoad(wy=10.0, coord_system="gravity")
    )
    rows = format_element_loads(m, m.elements[0])
    assert "magnitude" in rows[0].magnitude.lower()
    assert "gravity" in rows[0].position.lower()
    assert "-y" in rows[0].position.lower()


def test_format_local_load_has_no_local_eq_note():
    """Local loads should NOT carry a 'local eq:' annotation — they ARE
    in local coords already."""
    m, _ = _model_with_one_frame()
    m.elements[0].member_loads.append(UniformDistributedLoad(wy=-10.0))
    rows = format_element_loads(m, m.elements[0])
    assert "local eq" not in rows[0].meaning.lower()


def test_format_global_load_includes_local_eq_note():
    """Global / Gravity loads should show the projected (wx_l, wy_l)
    in the meaning column so the user can verify the conversion."""
    m, _ = _model_with_one_frame()
    m.elements[0].member_loads.append(
        UniformDistributedLoad(wy=-10.0, coord_system="global")
    )
    rows = format_element_loads(m, m.elements[0])
    assert "local eq" in rows[0].meaning.lower()


def test_format_gravity_load_includes_local_eq_note():
    m, _ = _model_with_one_frame()
    m.elements[0].member_loads.append(
        UniformDistributedLoad(wy=10.0, coord_system="gravity")
    )
    rows = format_element_loads(m, m.elements[0])
    assert "local eq" in rows[0].meaning.lower()
