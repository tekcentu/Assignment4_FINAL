"""Multi-case static-analysis result wrapper (v0.18 — PR-A).

Holds one :class:`AnalysisResult` per solved load case plus the
``active_case`` pointer the GUI uses to pick which case's diagrams,
displacements, and reactions are currently displayed.

The solver itself is unchanged in this PR — ``run_multi_case_analysis``
(in ``main.py``) loops over enabled cases and stores each result in the
``cases`` dict. SUM_ALL is a derived view computed on demand from the
already-solved cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from .model import AnalysisResult, StructuralModel


# Sentinel string the GUI uses to select the SUM_ALL view in toolbar
# combos / dialog dropdowns. Never written to disk and never a key in
# ``MultiCaseAnalysisResult.cases``.
SUM_ALL_KEY: str = "SUM_ALL"


@dataclass
class MultiCaseAnalysisResult:
    """One :class:`AnalysisResult` per solved case + an active pointer.

    ``cases`` only contains cases that actually solved (status ``"ok"``);
    failed solves are surfaced in :attr:`failed_cases` so the GUI can
    show an error badge. SUM_ALL is intentionally not stored in
    ``cases`` — it's a synthesised view returned by :meth:`sum_all` and
    is only available when every *enabled* case solved successfully
    (per the explicit user redirect in the PR-A approval).
    """

    cases: dict[str, AnalysisResult] = field(default_factory=dict)
    active_case: str = "DEFAULT"

    # Cases the multi-run attempted but that failed. Keyed by case
    # name → short message ("Singular K", "Assembly error: …", etc.).
    # Used by the GUI to show a per-case status badge; SUM_ALL is
    # blocked while this dict is non-empty (for enabled cases).
    failed_cases: dict[str, str] = field(default_factory=dict)

    # Names of cases the multi-run was asked to solve (regardless of
    # success). Used to decide SUM_ALL availability: SUM_ALL needs
    # every requested case to be in ``cases`` (i.e. no failed solves
    # AND no skipped cases).
    requested_cases: list[str] = field(default_factory=list)

    SUM_ALL_KEY: ClassVar[str] = SUM_ALL_KEY

    _sum_all_cache: AnalysisResult | None = field(
        default=None, init=False, repr=False, compare=False,
    )

    # ── lookup ──────────────────────────────────────────────────

    def get(self, case_name: str) -> AnalysisResult | None:
        """Return the stored result for ``case_name`` (or the SUM_ALL view).

        ``None`` when the case has no solved result (typical pre-solve
        state, a disabled case, or a case that failed).
        """
        if case_name == SUM_ALL_KEY:
            return self.sum_all()
        return self.cases.get(case_name)

    @property
    def active_result(self) -> AnalysisResult | None:
        """Convenience: result for the currently active case."""
        return self.get(self.active_case)

    @property
    def status(self) -> str:
        """``"ok"`` iff every solved case is ``"ok"`` AND there are no
        failed cases AND at least one case solved."""
        if not self.cases:
            return "error"
        if self.failed_cases:
            return "error"
        return (
            "ok" if all(r.status == "ok" for r in self.cases.values())
            else "error"
        )

    # ── SUM_ALL ────────────────────────────────────────────────

    def sum_all_available(self) -> bool:
        """SUM_ALL needs every *requested* case to have solved.

        Per the PR-A approval (item 8): do not silently skip failed or
        unsolved enabled cases. Disabled cases are excluded from the
        request set by the caller, so their absence here is correct."""
        if not self.requested_cases:
            return False
        if self.failed_cases:
            return False
        return all(name in self.cases for name in self.requested_cases)

    def sum_all(self) -> AnalysisResult | None:
        """Linear superposition of every solved case (D, reactions,
        member_results.f_local / d_local / d_global).

        Returns ``None`` when :meth:`sum_all_available` is False so the
        caller can render a placeholder rather than a partial sum that
        would mislead the user. Cached after first call — invalidated
        only when a new ``MultiCaseAnalysisResult`` is constructed."""
        if not self.sum_all_available():
            return None
        if self._sum_all_cache is not None:
            return self._sum_all_cache
        # SUM_ALL is exactly a unit-coefficient combination of every
        # solved case — share the combine kernel with user-defined
        # combinations (PR #29) so the superposition math lives in
        # one place.
        pairs = [(r, 1.0) for r in self.cases.values()]
        self._sum_all_cache = _combine_results(
            pairs,
            label=(
                f"[SUM_ALL] derived linear superposition of "
                f"{len(self.cases)} case(s): " + ", ".join(sorted(self.cases))
            ),
        )
        return self._sum_all_cache

    # ── coefficient combinations (PR #29) ──────────────────────

    def missing_cases_for(self, terms: dict[str, float]) -> list[str]:
        """Return the referenced case names that have no solved result.

        A combination is only computable when every case it references
        is present in :attr:`cases` (solved OK). Disabled / failed /
        never-requested cases all surface here."""
        return sorted(
            name for name in terms if name not in self.cases
        )

    def combination_available(self, terms: dict[str, float]) -> bool:
        """True iff every case referenced by ``terms`` has a solved
        result so the combination can be computed."""
        if not terms:
            return False
        return not self.missing_cases_for(terms)

    def combination(
        self, terms: dict[str, float], *, name: str = "COMBINATION",
    ) -> AnalysisResult | None:
        """Compute the coefficient-weighted linear combination of solved
        cases described by ``terms`` (``{case_name: coefficient}``).

        Returns ``None`` when any referenced case is unsolved so the
        caller can render a clear placeholder rather than a partial /
        misleading result. Not cached on the wrapper (the wrapper does
        not own the combination definitions) — the host caches the
        active combination result if it wants to."""
        if not self.combination_available(terms):
            return None
        pairs = [
            (self.cases[case_name], float(coeff))
            for case_name, coeff in terms.items()
        ]
        label = (
            f"[{name}] derived combination: "
            + " + ".join(
                f"{coeff:g}·{case_name}"
                for case_name, coeff in terms.items()
            )
        )
        return _combine_results(pairs, label=label)

    # ── helpers ────────────────────────────────────────────────

    def available_case_names(self, *, include_sum_all: bool = True) -> list[str]:
        """Return the case names suitable for a toolbar combo.

        Per the PR-A approval, real cases come first (sorted by name),
        with SUM_ALL appended last — but only when it's actually
        available (every requested case solved)."""
        names = sorted(self.cases.keys())
        if include_sum_all and self.sum_all_available() and len(self.cases) >= 2:
            names.append(SUM_ALL_KEY)
        return names


def _combine_results(
    pairs: list[tuple[AnalysisResult, float]],
    *,
    label: str,
) -> AnalysisResult | None:
    """Scale each ``AnalysisResult`` by its coefficient and sum the
    responses into a single derived ``AnalysisResult``.

    Shared kernel for both SUM_ALL (all coefficients 1.0) and
    user-defined coefficient combinations (PR #29). Linear elastic
    superposition is exact, so combining solved cases this way matches
    what re-solving with the summed load vector would produce.

    Combined fields: ``D``, ``reactions``, and per-element
    ``member_results`` (``f_local`` / ``d_local`` / ``d_global``). Sum
    buckets are lazily sized via ``np.zeros_like`` so heterogeneous DOF
    widths (frame 6 vs truss 4) never crash. Structural metadata
    (E_map, num_eq, G_vectors, K, elem_data) is copied from the first
    contributing case — these are case-independent for a fixed model
    topology. ``F`` is None: no single load vector covers the
    superposition.

    Returns ``None`` for an empty ``pairs`` list (nothing to combine)."""
    if not pairs:
        return None
    first = pairs[0][0]
    member_keys = first.member_results.keys()
    D_sum = None
    reactions_sum: dict[int, dict[str, float]] = {}
    f_local_sum: dict[int, np.ndarray] = {}
    d_local_sum: dict[int, np.ndarray] = {}
    d_global_sum: dict[int, np.ndarray] = {}
    for r, coeff in pairs:
        c = float(coeff)
        if r.D is not None:
            scaled_D = c * np.asarray(r.D)
            D_sum = scaled_D if D_sum is None else D_sum + scaled_D
        for nid, comp in r.reactions.items():
            bucket = reactions_sum.setdefault(nid, {})
            for k, v in comp.items():
                bucket[k] = bucket.get(k, 0.0) + c * v
        for eid in member_keys:
            mr = r.member_results.get(eid, {})
            if "f_local" in mr:
                val = c * np.asarray(mr["f_local"])
                f_local_sum[eid] = (
                    val if eid not in f_local_sum
                    else f_local_sum[eid] + val
                )
            if "d_local" in mr:
                val = c * np.asarray(mr["d_local"])
                d_local_sum[eid] = (
                    val if eid not in d_local_sum
                    else d_local_sum[eid] + val
                )
            if "d_global" in mr:
                val = c * np.asarray(mr["d_global"])
                d_global_sum[eid] = (
                    val if eid not in d_global_sum
                    else d_global_sum[eid] + val
                )
    combined_member_results: dict[int, dict] = {}
    for eid in member_keys:
        entry: dict = {}
        if eid in f_local_sum:
            entry["f_local"] = f_local_sum[eid]
        if eid in d_local_sum:
            entry["d_local"] = d_local_sum[eid]
        if eid in d_global_sum:
            entry["d_global"] = d_global_sum[eid]
        combined_member_results[eid] = entry
    # Residual norms aren't superposable; surface the max scaled value
    # so a well-conditioned combined view reports a small number.
    max_res = max(
        (abs(float(coeff)) * (getattr(r, "residual", 0.0) or 0.0))
        for r, coeff in pairs
    )
    max_eqres = max(
        (abs(float(coeff)) * (getattr(r, "eq_residual", 0.0) or 0.0))
        for r, coeff in pairs
    )
    return AnalysisResult(
        status="ok",
        title=first.title,
        warnings=[label],
        E_map=first.E_map,
        num_eq=first.num_eq,
        G_vectors=first.G_vectors,
        K=first.K,
        F=None,
        D=D_sum,
        residual=max_res,
        member_results=combined_member_results,
        reactions=reactions_sum,
        eq_residual=max_eqres,
        elem_data=first.elem_data,
    )


def make_active_case_safe(
    multi: MultiCaseAnalysisResult | None,
    desired: str,
    fallback: str = "DEFAULT",
) -> str:
    """Choose an ``active_case`` that exists in ``multi.cases`` (or
    SUM_ALL when available).

    Used by the GUI when a model is loaded, a case is deleted, or the
    multi-result is rebuilt after a solve — we want to land on
    ``desired`` if it's still valid, otherwise fall back to DEFAULT (or
    the first available case if even DEFAULT is missing)."""
    if multi is None:
        return desired
    if desired in multi.cases:
        return desired
    if desired == SUM_ALL_KEY and multi.sum_all_available():
        return desired
    if fallback in multi.cases:
        return fallback
    if multi.cases:
        return next(iter(sorted(multi.cases.keys())))
    return desired


__all__ = [
    "SUM_ALL_KEY",
    "MultiCaseAnalysisResult",
    "make_active_case_safe",
    "_combine_results",
]
