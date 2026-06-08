"""Tests for JointMass and ModalMassSource dataclasses (PR #40)."""

import pytest

from structural_analysis.model import (
    JointMass,
    ModalMassSource,
    StructuralModel,
    Node,
    Material,
    Section,
)
from structural_analysis.element import FrameElement2D


# ── JointMass ──────────────────────────────────────────────────


class TestJointMass:
    def test_default_zeros_valid(self):
        jm = JointMass(node_id=1)
        assert jm.mx == 0.0
        assert jm.my == 0.0

    def test_positive_values_valid(self):
        jm = JointMass(node_id=5, mx=1000.0, my=500.0)
        assert jm.mx == 1000.0
        assert jm.my == 500.0

    def test_storage_round_trip(self):
        model = StructuralModel()
        model.nodes[1] = Node(1, 0.0, 0.0)
        model.nodes[2] = Node(2, 1.0, 0.0)
        jm1 = JointMass(node_id=1, mx=500.0, my=500.0)
        jm2 = JointMass(node_id=2, mx=1200.0, my=1200.0)
        model.joint_masses[1] = jm1
        model.joint_masses[2] = jm2
        assert model.joint_masses[1].mx == 500.0
        assert model.joint_masses[2].my == 1200.0
        del model.joint_masses[1]
        assert 1 not in model.joint_masses
        assert 2 in model.joint_masses

    def test_rejects_negative_mx(self):
        with pytest.raises(ValueError, match="mx"):
            JointMass(node_id=1, mx=-1.0)

    def test_rejects_negative_my(self):
        with pytest.raises(ValueError, match="my"):
            JointMass(node_id=1, my=-0.001)

    def test_rejects_nan_mx(self):
        import math
        with pytest.raises(ValueError, match="mx"):
            JointMass(node_id=1, mx=math.nan)

    def test_rejects_inf_my(self):
        import math
        with pytest.raises(ValueError, match="my"):
            JointMass(node_id=1, my=math.inf)

    def test_zero_is_valid(self):
        jm = JointMass(node_id=3, mx=0.0, my=0.0)
        assert jm.mx == 0.0


# ── ModalMassSource ────────────────────────────────────────────


class TestModalMassSource:
    def test_default_is_default(self):
        src = ModalMassSource()
        assert src.include_self_mass is True
        assert src.include_joint_masses is True
        assert src.include_load_cases is False
        assert src.load_case_factors == {}
        assert src.is_default()

    def test_non_default_self_mass_off(self):
        src = ModalMassSource(include_self_mass=False)
        assert not src.is_default()

    def test_non_default_with_factors(self):
        src = ModalMassSource(
            include_load_cases=True,
            load_case_factors={"DEAD": 1.0, "LIVE": 0.3},
        )
        assert not src.is_default()
        assert src.load_case_factors["DEAD"] == 1.0

    def test_rejects_negative_factor(self):
        with pytest.raises(ValueError, match="load_case_factors"):
            ModalMassSource(
                include_load_cases=True,
                load_case_factors={"LIVE": -0.1},
            )

    def test_rejects_nan_factor(self):
        import math
        with pytest.raises(ValueError, match="load_case_factors"):
            ModalMassSource(
                include_load_cases=True,
                load_case_factors={"DEAD": math.nan},
            )

    def test_rejects_inf_factor(self):
        import math
        with pytest.raises(ValueError, match="load_case_factors"):
            ModalMassSource(
                include_load_cases=True,
                load_case_factors={"DEAD": math.inf},
            )

    def test_zero_factor_allowed(self):
        # Zero factor is technically a no-op but must not raise.
        src = ModalMassSource(
            include_load_cases=True,
            load_case_factors={"LIVE": 0.0},
        )
        assert src.load_case_factors["LIVE"] == 0.0

    def test_model_default_factory(self):
        model = StructuralModel()
        assert isinstance(model.modal_mass_source, ModalMassSource)
        assert model.modal_mass_source.is_default()

    def test_model_joint_masses_default_empty(self):
        model = StructuralModel()
        assert model.joint_masses == {}


# ── DeleteNodeCmd joint-mass cascade ─────────────────────────────────────


def _two_node_model():
    """Minimal model: two nodes, one frame element, joint mass on node 2."""
    m = StructuralModel()
    m.materials[1] = Material(id=1, E=2.1e8, alpha=1e-5, density=7850.0)
    m.sections[1] = Section(id=1, material_id=1, A=0.01, I=1e-4, depth=0.3)
    m.nodes[1] = Node(1, 0.0, 0.0)
    m.nodes[2] = Node(2, 1.0, 0.0)
    m.elements.append(
        FrameElement2D(id=1, node_i=1, node_j=2, E=2.1e8, A=0.01, I=1e-4, section_id=1)
    )
    m.joint_masses[2] = JointMass(node_id=2, mx=500.0, my=500.0)
    return m


def test_delete_node_removes_joint_mass():
    """DeleteNodeCmd.do() must remove the joint mass from the deleted node."""
    from structural_analysis.gui_common.commands import DeleteNodeCmd
    m = _two_node_model()
    cmd = DeleteNodeCmd(node_id=2)
    cmd.do(m)
    assert 2 not in m.joint_masses


def test_delete_node_undo_restores_joint_mass():
    """DeleteNodeCmd.undo() must restore the saved joint mass."""
    from structural_analysis.gui_common.commands import DeleteNodeCmd
    m = _two_node_model()
    cmd = DeleteNodeCmd(node_id=2)
    cmd.do(m)
    cmd.undo(m)
    assert 2 in m.joint_masses
    assert m.joint_masses[2].mx == 500.0
    assert m.joint_masses[2].my == 500.0


def test_delete_node_without_joint_mass_does_not_crash():
    """Deleting a node that has no joint mass must not raise."""
    from structural_analysis.gui_common.commands import DeleteNodeCmd
    m = _two_node_model()
    # node 1 has no joint mass
    cmd = DeleteNodeCmd(node_id=1)
    cmd.do(m)
    assert 1 not in m.joint_masses
