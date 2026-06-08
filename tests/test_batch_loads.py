"""Tests for BatchAddMemberLoadsCmd and BatchAddNodalLoadsCmd (PR #41)."""

import pytest

from structural_analysis.element import FrameElement2D, TrussElement2D
from structural_analysis.gui_common.commands import (
    BatchAddMemberLoadsCmd,
    BatchAddNodalLoadsCmd,
)
from structural_analysis.model import (
    FrameTemperatureLoad,
    Material,
    NodalLoad,
    Node,
    PointLoad,
    Section,
    StructuralModel,
    TrussTemperatureLoad,
    UniformDistributedLoad,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _frame_model(n_elements: int = 3, lengths: list[float] | None = None):
    """A horizontal chain of frame elements with configurable lengths."""
    if lengths is None:
        lengths = [1.0] * n_elements
    assert len(lengths) == n_elements
    m = StructuralModel()
    m.materials[1] = Material(id=1, E=2.1e8, alpha=1e-5, density=7850.0)
    m.sections[1] = Section(id=1, material_id=1, A=0.01, I=1e-4, depth=0.3)
    x = 0.0
    m.nodes[1] = Node(1, 0.0, 0.0)
    for k, L in enumerate(lengths):
        x += L
        m.nodes[k + 2] = Node(k + 2, x, 0.0)
        m.elements.append(
            FrameElement2D(
                id=k + 1, node_i=k + 1, node_j=k + 2,
                E=2.1e8, A=0.01, I=1e-4, section_id=1,
            )
        )
    return m


def _mixed_model():
    """Two frames + one truss in a single model so we can test mixed batches."""
    m = StructuralModel()
    m.materials[1] = Material(id=1, E=2.1e8, alpha=1e-5, density=7850.0)
    m.sections[1] = Section(id=1, material_id=1, A=0.01, I=1e-4, depth=0.3)
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 1.0, 0.0)
    m.nodes[3] = Node(3, 2.0, 0.0)
    m.nodes[4] = Node(4, 3.0, 0.0)
    m.elements.append(
        FrameElement2D(id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, section_id=1)
    )
    m.elements.append(
        FrameElement2D(id=2, node_i=2, node_j=3, E=2.1e8, A=0.01, I=1e-4, section_id=1)
    )
    m.elements.append(
        TrussElement2D(id=3, node_i=3, node_j=4, E=2.1e8, A=0.01, section_id=1)
    )
    return m


# ── BatchAddMemberLoadsCmd ───────────────────────────────────────────────


class TestBatchMemberLoads:
    def test_udl_assigns_one_load_per_frame_element(self):
        m = _frame_model(3)
        loads = [
            (eid, UniformDistributedLoad(wy=-5.0, wx=0.0))
            for eid in (1, 2, 3)
        ]
        cmd = BatchAddMemberLoadsCmd(loads=loads)
        cmd.do(m)
        for e in m.elements:
            assert len(e.member_loads) == 1
            assert isinstance(e.member_loads[0], UniformDistributedLoad)
            assert e.member_loads[0].wy == -5.0

    def test_point_load_relative_position_per_element_length(self):
        """ratio=0.5 must produce a = 0.5 * L for each element regardless
        of length — the GUI layer is what does the conversion; this test
        just confirms the command stores whatever a the caller supplied
        (different per element)."""
        m = _frame_model(2, lengths=[2.0, 6.0])
        ratio = 0.5
        loads = []
        for elem in m.elements:
            ni = m.nodes[elem.node_i]
            nj = m.nodes[elem.node_j]
            L = ((nj.x - ni.x) ** 2 + (nj.y - ni.y) ** 2) ** 0.5
            loads.append((elem.id, PointLoad(py=-10.0, a=ratio * L)))
        cmd = BatchAddMemberLoadsCmd(loads=loads)
        cmd.do(m)
        assert m.elements[0].member_loads[0].a == pytest.approx(1.0)
        assert m.elements[1].member_loads[0].a == pytest.approx(3.0)

    def test_thermal_uniform_works_on_frames(self):
        m = _frame_model(2)
        loads = [
            (eid, FrameTemperatureLoad(t_top=20.0, t_bottom=20.0))
            for eid in (1, 2)
        ]
        BatchAddMemberLoadsCmd(loads=loads).do(m)
        for e in m.elements:
            assert isinstance(e.member_loads[0], FrameTemperatureLoad)

    def test_thermal_gradient_on_truss_selection_blocked(self):
        """FrameTemperatureLoad with non-uniform top/bottom on a truss
        must raise BEFORE any element is mutated."""
        m = _mixed_model()
        # Truss elem id 3.
        loads = [(3, FrameTemperatureLoad(t_top=30.0, t_bottom=10.0))]
        cmd = BatchAddMemberLoadsCmd(loads=loads)
        with pytest.raises(ValueError, match="truss"):
            cmd.do(m)
        # No loads anywhere.
        for e in m.elements:
            assert e.member_loads == []

    def test_mixed_frame_truss_blocked_atomically(self):
        """If one element in the batch is incompatible, NO element gets
        the load (validate-then-mutate)."""
        m = _mixed_model()
        # Apply a UDL to all three — truss can't hold it. But our actual
        # block is FrameTemperatureLoad on truss; UDL on truss is not
        # caught at the command layer (the dialog blocks it). So use
        # the type-mismatch path to prove atomicity.
        loads = [
            (1, FrameTemperatureLoad(t_top=10.0, t_bottom=10.0)),
            (2, FrameTemperatureLoad(t_top=10.0, t_bottom=10.0)),
            (3, FrameTemperatureLoad(t_top=10.0, t_bottom=10.0)),
        ]
        cmd = BatchAddMemberLoadsCmd(loads=loads)
        with pytest.raises(ValueError):
            cmd.do(m)
        for e in m.elements:
            assert e.member_loads == []

    def test_missing_element_id_blocks_atomically(self):
        m = _frame_model(2)
        loads = [
            (1, UniformDistributedLoad(wy=-5.0)),
            (99, UniformDistributedLoad(wy=-5.0)),
        ]
        cmd = BatchAddMemberLoadsCmd(loads=loads)
        with pytest.raises(ValueError, match="99"):
            cmd.do(m)
        for e in m.elements:
            assert e.member_loads == []

    def test_load_case_preserved(self):
        m = _frame_model(2)
        loads = [
            (1, UniformDistributedLoad(wy=-1.0, load_case="LIVE")),
            (2, UniformDistributedLoad(wy=-1.0, load_case="LIVE")),
        ]
        BatchAddMemberLoadsCmd(loads=loads).do(m)
        for e in m.elements:
            assert e.member_loads[0].load_case == "LIVE"

    def test_undo_removes_all_batch_loads(self):
        m = _frame_model(3)
        loads = [(eid, UniformDistributedLoad(wy=-5.0)) for eid in (1, 2, 3)]
        cmd = BatchAddMemberLoadsCmd(loads=loads)
        cmd.do(m)
        cmd.undo(m)
        for e in m.elements:
            assert e.member_loads == []

    def test_redo_restores_all_loads(self):
        m = _frame_model(3)
        loads = [(eid, UniformDistributedLoad(wy=-5.0)) for eid in (1, 2, 3)]
        cmd = BatchAddMemberLoadsCmd(loads=loads)
        cmd.do(m)
        cmd.undo(m)
        cmd.do(m)  # redo
        for e in m.elements:
            assert len(e.member_loads) == 1

    def test_undo_preserves_preexisting_identical_load(self):
        """Regression: undo must remove only the batch-added row, even
        when an equal load already existed on the element before the
        batch ran (identity-based remove would have removed the wrong
        one)."""
        m = _frame_model(2)
        pre = UniformDistributedLoad(wy=-5.0)
        m.elements[0].member_loads.append(pre)
        m.elements[1].member_loads.append(UniformDistributedLoad(wy=-5.0))
        # Now batch-add an EQUAL load to each — same wy, same defaults.
        loads = [(eid, UniformDistributedLoad(wy=-5.0)) for eid in (1, 2)]
        cmd = BatchAddMemberLoadsCmd(loads=loads)
        cmd.do(m)
        cmd.undo(m)
        # Each element should still have exactly one load left — the
        # pre-existing one — not zero (which would mean undo took the
        # wrong row).
        assert len(m.elements[0].member_loads) == 1
        assert len(m.elements[1].member_loads) == 1
        # And it must be the SAME object reference the user had before.
        assert m.elements[0].member_loads[0] is pre


# ── BatchAddNodalLoadsCmd ────────────────────────────────────────────────


class TestBatchNodalLoads:
    def test_appends_one_load_per_node(self):
        m = _frame_model(3)
        # 4 nodes; load on nodes 2, 3, 4.
        loads = [
            NodalLoad(node_id=nid, fx=10.0, fy=-5.0, mz=0.0)
            for nid in (2, 3, 4)
        ]
        BatchAddNodalLoadsCmd(loads=loads).do(m)
        assert len(m.nodal_loads) == 3
        assert {ld.node_id for ld in m.nodal_loads} == {2, 3, 4}

    def test_preserves_existing_nodal_loads(self):
        m = _frame_model(2)
        pre = NodalLoad(node_id=2, fx=100.0, fy=0.0, mz=0.0)
        m.nodal_loads.append(pre)
        loads = [NodalLoad(node_id=nid, fx=10.0, fy=0.0, mz=0.0) for nid in (2, 3)]
        BatchAddNodalLoadsCmd(loads=loads).do(m)
        # Pre-existing load still there.
        assert pre in m.nodal_loads
        assert len(m.nodal_loads) == 3

    def test_load_case_preserved(self):
        m = _frame_model(2)
        loads = [
            NodalLoad(node_id=nid, fx=5.0, fy=0.0, mz=0.0, load_case="WIND")
            for nid in (1, 2)
        ]
        BatchAddNodalLoadsCmd(loads=loads).do(m)
        for ld in m.nodal_loads:
            assert ld.load_case == "WIND"

    def test_missing_node_blocks_atomically(self):
        m = _frame_model(2)
        loads = [
            NodalLoad(node_id=1, fx=5.0, fy=0.0, mz=0.0),
            NodalLoad(node_id=99, fx=5.0, fy=0.0, mz=0.0),
        ]
        cmd = BatchAddNodalLoadsCmd(loads=loads)
        with pytest.raises(ValueError, match="99"):
            cmd.do(m)
        assert m.nodal_loads == []

    def test_undo_removes_added_rows(self):
        m = _frame_model(2)
        loads = [NodalLoad(node_id=nid, fx=5.0, fy=0.0, mz=0.0) for nid in (1, 2, 3)]
        cmd = BatchAddNodalLoadsCmd(loads=loads)
        cmd.do(m)
        cmd.undo(m)
        assert m.nodal_loads == []

    def test_redo_restores_rows(self):
        m = _frame_model(2)
        loads = [NodalLoad(node_id=nid, fx=5.0, fy=0.0, mz=0.0) for nid in (1, 2, 3)]
        cmd = BatchAddNodalLoadsCmd(loads=loads)
        cmd.do(m)
        cmd.undo(m)
        cmd.do(m)
        assert len(m.nodal_loads) == 3

    def test_undo_preserves_preexisting_identical_load(self):
        """Regression: undo must remove only the batch-added rows, not
        a pre-existing identical row."""
        m = _frame_model(2)
        pre = NodalLoad(node_id=2, fx=5.0, fy=0.0, mz=0.0)
        m.nodal_loads.append(pre)
        # Batch-add equal loads.
        loads = [NodalLoad(node_id=nid, fx=5.0, fy=0.0, mz=0.0) for nid in (1, 2, 3)]
        cmd = BatchAddNodalLoadsCmd(loads=loads)
        cmd.do(m)
        cmd.undo(m)
        # Only the pre-existing one remains.
        assert m.nodal_loads == [pre]
        assert m.nodal_loads[0] is pre
