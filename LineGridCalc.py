"""
line_grid_calc.py

Python translation of the Fortran module LineGridCalc.
Computes contributions to LBL (Line-By-Line) spectral grids from the left wing,
central part, and right wing of a spectral line's extended sub-interval.

The module uses a hierarchical multi-resolution grid scheme:
  - Fine grids: RK0 (finest) through RK9, each with spacing H0 through H9
  - Coarse grid: RK, with uniform spacing H
  - Each RK_n array holds NT_n points; P/L suffixes are the "previous" and
    "later" neighbouring points on the next-coarser grid level.

External state (equivalent to the Fortran USE'd modules) is held in a
GridState dataclass so the functions are pure with respect to caller-supplied
state rather than relying on module-level globals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Type aliases  (compatible with Python 3.7+)
# ---------------------------------------------------------------------------
FloatArr = List[float]
ShapeFunc = Callable[[float], float]


# ---------------------------------------------------------------------------
# GridState  –  all shared arrays and constants (replaces Fortran module vars)
# ---------------------------------------------------------------------------
@dataclass
class GridConstants:
    """
    Physical / numerical constants that characterise the grid layout.

    Attributes
    ----------
    H : float
        Coarsest uniform grid spacing.
    H0 … H9 : float
        Grid spacings for each sub-grid level, finest (H0) to coarsest (H9).
        Typically H_{n+1} ≈ 2 * H_n.
    NT : int
        Number of points on the coarsest RK grid.
    NT0 … NT9 : int
        Number of points on each sub-grid level.
    cutOff : float
        Maximum half-width of a line wing (cm⁻¹ or equivalent).
    deltaWV : float
        Width of the central part of the sub-interval.
    """
    H:  float = 1.0
    H0: float = 1.0
    H1: float = 2.0
    H2: float = 4.0
    H3: float = 8.0
    H4: float = 16.0
    H5: float = 32.0
    H6: float = 64.0
    H7: float = 128.0
    H8: float = 256.0
    H9: float = 512.0

    NT:  int = 256
    NT0: int = 256
    NT1: int = 128
    NT2: int = 64
    NT3: int = 32
    NT4: int = 16
    NT5: int = 8
    NT6: int = 4
    NT7: int = 2
    NT8: int = 2
    NT9: int = 2

    cutOff:  float = 1000.0
    deltaWV: float = 1.0


def _make_arr(n: int) -> FloatArr:
    return [0.0] * (n + 1)  # 1-based indexing: index 0 unused


@dataclass
class GridArrays:
    """
    All accumulator arrays.  Sizes are derived from a GridConstants instance.
    Indices are 1-based to mirror Fortran (index 0 is present but unused).
    """
    # Coarse grid
    RK:   FloatArr = field(default_factory=list)
    # Fine sub-grids (centre point, previous coarser point, later coarser point)
    RK0:  FloatArr = field(default_factory=list)
    RK0P: FloatArr = field(default_factory=list)
    RK0L: FloatArr = field(default_factory=list)
    RK1:  FloatArr = field(default_factory=list)
    RK1P: FloatArr = field(default_factory=list)
    RK1L: FloatArr = field(default_factory=list)
    RK2:  FloatArr = field(default_factory=list)
    RK2P: FloatArr = field(default_factory=list)
    RK2L: FloatArr = field(default_factory=list)
    RK3:  FloatArr = field(default_factory=list)
    RK3P: FloatArr = field(default_factory=list)
    RK3L: FloatArr = field(default_factory=list)
    RK4:  FloatArr = field(default_factory=list)
    RK4P: FloatArr = field(default_factory=list)
    RK4L: FloatArr = field(default_factory=list)
    RK5:  FloatArr = field(default_factory=list)
    RK5P: FloatArr = field(default_factory=list)
    RK5L: FloatArr = field(default_factory=list)
    RK6:  FloatArr = field(default_factory=list)
    RK6P: FloatArr = field(default_factory=list)
    RK6L: FloatArr = field(default_factory=list)
    RK7:  FloatArr = field(default_factory=list)
    RK7P: FloatArr = field(default_factory=list)
    RK7L: FloatArr = field(default_factory=list)
    RK8:  FloatArr = field(default_factory=list)
    RK8P: FloatArr = field(default_factory=list)
    RK8L: FloatArr = field(default_factory=list)
    RK9:  FloatArr = field(default_factory=list)
    RK9P: FloatArr = field(default_factory=list)
    RK9L: FloatArr = field(default_factory=list)

    @classmethod
    def zeros(cls, c: GridConstants) -> "GridArrays":
        """Allocate all arrays (1-based, zero-filled)."""
        return cls(
            RK=_make_arr(c.NT),
            RK0=_make_arr(c.NT0),   RK0P=_make_arr(c.NT0),  RK0L=_make_arr(c.NT0),
            RK1=_make_arr(c.NT1),   RK1P=_make_arr(c.NT1),  RK1L=_make_arr(c.NT1),
            RK2=_make_arr(c.NT2),   RK2P=_make_arr(c.NT2),  RK2L=_make_arr(c.NT2),
            RK3=_make_arr(c.NT3),   RK3P=_make_arr(c.NT3),  RK3L=_make_arr(c.NT3),
            RK4=_make_arr(c.NT4),   RK4P=_make_arr(c.NT4),  RK4L=_make_arr(c.NT4),
            RK5=_make_arr(c.NT5),   RK5P=_make_arr(c.NT5),  RK5L=_make_arr(c.NT5),
            RK6=_make_arr(c.NT6),   RK6P=_make_arr(c.NT6),  RK6L=_make_arr(c.NT6),
            RK7=_make_arr(c.NT7),   RK7P=_make_arr(c.NT7),  RK7L=_make_arr(c.NT7),
            RK8=_make_arr(c.NT8),   RK8P=_make_arr(c.NT8),  RK8L=_make_arr(c.NT8),
            RK9=_make_arr(c.NT9),   RK9P=_make_arr(c.NT9),  RK9L=_make_arr(c.NT9),
        )


# ---------------------------------------------------------------------------
# left_lbl
# ---------------------------------------------------------------------------
def left_lbl(
    freq: float,
    ul: float,
    fshape: ShapeFunc,
    eps: float,
    c: GridConstants,
    g: GridArrays,
) -> None:
    """
    Accumulate contributions from the LEFT WING of a spectral line.
    See module docstring for full description.
    """
    uu = ul - freq
    if uu >= 0.0:
        return
    if -uu > c.cutOff:
        return

    ff = float(fshape(uu))
    if ff < eps:
        return

    g.RK[1] += ff

    # ------------------------------------------------------------------ #
    # Descending pass: determine which level bracket [-uu] falls into.   #
    # Each level's bracket is [H_{n-1}, H_n).  When a bracket is found,  #
    # the three triplet values for that level are accumulated at index 1, #
    # then execution falls through to the ascending cascade.              #
    # ------------------------------------------------------------------ #

    # Helpers: (P,centre,L) triplet names and spacings per level
    levels = [
        # (array_P, array_centre, array_L, H_narrow, H_wide)
        (g.RK0P, g.RK0, g.RK0L, c.H0, c.H1),
        (g.RK1P, g.RK1, g.RK1L, c.H1, c.H2),
        (g.RK2P, g.RK2, g.RK2L, c.H2, c.H3),
        (g.RK3P, g.RK3, g.RK3L, c.H3, c.H4),
        (g.RK4P, g.RK4, g.RK4L, c.H4, c.H5),
        (g.RK5P, g.RK5, g.RK5L, c.H5, c.H6),
        (g.RK6P, g.RK6, g.RK6L, c.H6, c.H7),
        (g.RK7P, g.RK7, g.RK7L, c.H7, c.H8),
        (g.RK8P, g.RK8, g.RK8L, c.H8, c.H9),
        (g.RK9P, g.RK9, g.RK9L, c.H9, c.H + c.H),  # H9 < -uu < 2H
    ]

    # H boundaries from finest to coarsest
    h_bounds = [c.H0, c.H1, c.H2, c.H3, c.H4,
                c.H5, c.H6, c.H7, c.H8, c.H9]

    abs_uu = -uu  # positive distance from line centre

    # Find the entry level (descending pass, labels 20-29)
    entry_level = None  # type: Optional[int]

    if abs_uu < c.H0:
        # Goes directly to the H0 finish loop (label 10)
        _left_finish_h0_loop(uu, ff, fshape, eps, c, g)
        return

    for lvl in range(10):
        h_lo = h_bounds[lvl]                                # lower boundary
        h_hi = h_bounds[lvl + 1] if lvl < 9 else (c.H + c.H)
        if abs_uu < h_hi:
            entry_level = lvl
            break

    if entry_level is None:
        # abs_uu >= 2H: we are in the coarse-only region (label 29)
        g.RK[2] += fshape(uu - c.H)
        g.RK[3] += fshape(uu - c.H - c.H)
        g.RK[4] += fshape(uu + c.H - c.H9)
        ff = float(fshape(uu - c.H9))
        g.RK[5] += ff
        # Start ascending cascade from level 9 (label 129)
        _left_ascending_cascade(uu, ff, fshape, eps, c, g, start_level=9)
        return

    # Accumulate triplet at index 1 for the entry level
    rkP, rkC, rkL, h_narrow, h_wide = levels[entry_level]
    rkP[1] += ff
    ff = float(fshape(uu - h_wide))
    rkC[1] += ff
    ff = float(fshape(uu - h_narrow))
    rkL[1] += ff
    if ff < eps:
        return

    if entry_level == 0:
        # After level-0 triplet, go to H0 finish loop
        _left_finish_h0_loop(uu, ff, fshape, eps, c, g)
        return

    # Ascending cascade from the level BELOW entry (labels 121-129)
    _left_ascending_cascade(uu, ff, fshape, eps, c, g, start_level=entry_level - 1)


def _left_finish_h0_loop(
    uu: float, ff: float, fshape: ShapeFunc, eps: float,
    c: GridConstants, g: GridArrays,
) -> None:
    """Label-10 loop: fill the fine NT0 sub-grid."""
    xxx = c.H0
    for i in range(2, c.NT0 + 1):
        g.RK0P[i] += ff
        ff = float(fshape(uu - xxx - c.H1))
        g.RK0[i] += ff
        xxx += c.H0
        ff = float(fshape(uu - xxx))
        g.RK0L[i] += ff
        if ff < eps:
            return


def _left_ascending_cascade(
    uu: float, ff: float, fshape: ShapeFunc, eps: float,
    c: GridConstants, g: GridArrays,
    start_level: int,
) -> None:
    """
    Ascending cascade (labels 121 … 10 in Fortran).

    Fills index-2 entries for levels [start_level … 0] from coarsest to finest,
    each using the value `ff` carried from the previous level.
    """
    # Per-level data needed for the ascending pass (index 2 accumulation)
    # Each entry: (rkP[2], rkC[2], rkL[2], h_sub, h_super)
    # where h_sub is the spacing subtracted to get centre, h_super is added to
    # get the "P" direction.
    asc_data = [
        # level 0
        (g.RK0P, g.RK0, g.RK0L, c.H1, c.H0),
        # level 1
        (g.RK1P, g.RK1, g.RK1L, c.H1 + c.H2, c.H0),
        # level 2
        (g.RK2P, g.RK2, g.RK2L, c.H2 + c.H3, c.H1),
        # level 3
        (g.RK3P, g.RK3, g.RK3L, c.H3 + c.H4, c.H2),
        # level 4
        (g.RK4P, g.RK4, g.RK4L, c.H4 + c.H5, c.H3),
        # level 5
        (g.RK5P, g.RK5, g.RK5L, c.H5 + c.H6, c.H4),
        # level 6
        (g.RK6P, g.RK6, g.RK6L, c.H6 + c.H7, c.H5),
        # level 7
        (g.RK7P, g.RK7, g.RK7L, c.H7 + c.H8, c.H6),
        # level 8
        (g.RK8P, g.RK8, g.RK8L, c.H8 + c.H9, c.H7),
        # level 9
        (g.RK9P, g.RK9, g.RK9L, c.H9 + c.H + c.H, c.H8),
    ]

    for lvl in range(start_level, -1, -1):
        rkP, rkC, rkL, h_centre_sub, h_l_sub = asc_data[lvl]
        rkP[2] += ff
        ff_c = float(fshape(uu - h_centre_sub))
        rkC[2] += ff_c
        ff = float(fshape(uu - h_l_sub))
        rkL[2] += ff
        if lvl > 0 and ff < eps:
            return

    # After level 0, go to finish loop
    _left_finish_h0_loop(uu, ff, fshape, eps, c, g)


# ---------------------------------------------------------------------------
# center_lbl
# ---------------------------------------------------------------------------
def center_lbl(
    freq: float,
    ul: float,
    fshape: ShapeFunc,
    eps: float,
    c: GridConstants,
    g: GridArrays,
) -> None:
    """
    Accumulate contributions from the CENTRAL PART of a spectral line's
    extended sub-interval  [startDeltaWV ; endDeltaWV].

    The routine has two symmetric halves:
      - Left-right side: starting from UU, stepping toward 0
      - Right-left side: starting from deltaWV-UU, stepping toward 0
    """
    uu = ul - freq
    if uu >= c.deltaWV:
        return

    ff0 = float(fshape(0.0))
    if ff0 < eps:
        return

    npoint = 1  # tracks where the left pass terminated

    conser = uu - c.H
    fa = float(fshape(uu))
    eps4 = eps * 0.25

    if fa > eps4:
        g.RK[1] += fa

    if uu < c.H:
        # Skip left-right pass entirely; jump to right-left side
        npoint = 2
        _center_right_left(uu, fa, fshape, eps, c, g, npoint=npoint, conser=conser)
        return

    # ---- LEFT-RIGHT PASS ----
    npoint, conser = _center_left_right(uu, fa, fshape, eps, c, g, conser)

    # ---- RIGHT-LEFT PASS ----
    _center_right_left(uu, fa, fshape, eps, c, g, npoint=npoint + 1, conser=conser)


def _center_left_right(
    uu: float, fa: float, fshape: ShapeFunc, eps: float,
    c: GridConstants, g: GridArrays,
    conser: float,
) -> Tuple[int, float]:
    """
    Left-right sub-pass of center_lbl.
    Walks from UU toward 0 across levels 0 … 9, then fills the coarse grid.
    Returns (npoint, conser).
    """
    uuu = uu

    # Per-level data: (rkP, rkC, rkL, H_level, NT_level, H_next_coarser)
    lr_levels = [
        (g.RK0P, g.RK0, g.RK0L, c.H0, c.NT0, c.H1),
        (g.RK1P, g.RK1, g.RK1L, c.H1, c.NT1, c.H2),
        (g.RK2P, g.RK2, g.RK2L, c.H2, c.NT2, c.H3),
        (g.RK3P, g.RK3, g.RK3L, c.H3, c.NT3, c.H4),
        (g.RK4P, g.RK4, g.RK4L, c.H4, c.NT4, c.H5),
        (g.RK5P, g.RK5, g.RK5L, c.H5, c.NT5, c.H6),
        (g.RK6P, g.RK6, g.RK6L, c.H6, c.NT6, c.H7),
        (g.RK7P, g.RK7, g.RK7L, c.H7, c.NT7, c.H8),
        (g.RK8P, g.RK8, g.RK8L, c.H8, c.NT8, c.H9),
        (g.RK9P, g.RK9, g.RK9L, c.H9, c.NT9, c.H + c.H),
    ]

    i = 0
    for lvl, (rkP, rkC, rkL, h, nt, h_next) in enumerate(lr_levels):
        if uuu < h + h:
            break
        ib = i + 1
        for i in range(ib, nt + 1):
            uuu -= h
            ff = float(fshape(uuu))
            if ff < eps:
                fa = ff
                break
            rkP[i] += fa
            rkC[i] += float(fshape(uuu + h_next))
            rkL[i] += ff
            fa = ff
            if uuu - h < h:
                break
        i = i * 2
    else:
        i = i * 4

    ib = i + 2
    npoint = 1
    conser_local = uu - (ib - 1) * c.H
    for icon in range(ib, c.NT + 1):
        g.RK[icon] += float(fshape(conser_local))
        conser_local -= c.H
        if conser_local < 0.0:
            npoint = icon
            return npoint, conser_local

    return npoint, conser_local


def _center_right_left(
    uu: float, fa_in: float, fshape: ShapeFunc, eps: float,
    c: GridConstants, g: GridArrays,
    npoint: int,
    conser: float,
) -> None:
    """
    Right-left sub-pass of center_lbl.
    Mirrors the left-right pass but steps from deltaWV-UU toward 0
    and fills from NT down to 1.
    """
    uuu = c.deltaWV - uu
    fa = float(fshape(uuu))

    rl_levels = [
        (g.RK0L, g.RK0, g.RK0P, c.H0, c.NT0, c.H1),
        (g.RK1L, g.RK1, g.RK1P, c.H1, c.NT1, c.H2),
        (g.RK2L, g.RK2, g.RK2P, c.H2, c.NT2, c.H3),
        (g.RK3L, g.RK3, g.RK3P, c.H3, c.NT3, c.H4),
        (g.RK4L, g.RK4, g.RK4P, c.H4, c.NT4, c.H5),
        (g.RK5L, g.RK5, g.RK5P, c.H5, c.NT5, c.H6),
        (g.RK6L, g.RK6, g.RK6P, c.H6, c.NT6, c.H7),
        (g.RK7L, g.RK7, g.RK7P, c.H7, c.NT7, c.H8),
        (g.RK8L, g.RK8, g.RK8P, c.H8, c.NT8, c.H9),
        (g.RK9L, g.RK9, g.RK9P, c.H9, c.NT9, c.H + c.H),
    ]

    iii = 0
    for lvl, (rkL, rkC, rkP, h, nt, h_next) in enumerate(rl_levels):
        if uuu < h + h:
            break
        ib = nt - iii
        for i in range(ib, 0, -1):
            iii += 1
            uuu -= h
            ff = float(fshape(uuu))
            if ff < eps:
                fa = ff
                break
            rkL[i] += fa
            rkC[i] += float(fshape(uuu + h_next))
            rkP[i] += ff
            fa = ff
            if uuu - h < h:
                break
        iii = iii * 2
    else:
        iii = iii * 4

    i_end = c.NT - iii
    for ii in range(npoint, i_end + 1):
        g.RK[ii] += float(fshape(conser))
        conser -= c.H


# ---------------------------------------------------------------------------
# right_lbl
# ---------------------------------------------------------------------------
def right_lbl(
    freq: float,
    ul: float,
    fshape: ShapeFunc,
    eps: float,
    c: GridConstants,
    g: GridArrays,
) -> None:
    """
    Accumulate contributions from the RIGHT WING of a spectral line.

    Covers the extended sub-interval  [endDeltaWV ; endDeltaWV + cutOff].
    This is the mirror of left_lbl: UU = UL - FREQ - deltaWV, and
    accumulation runs from NT down to 1.
    """
    uu = ul - freq - c.deltaWV
    if uu >= c.cutOff:
        return

    ff = float(fshape(uu))
    if ff < eps:
        return

    # Per-level ascending data (fine → coarse), filling at NT_n and NT_n-1
    levels = [
        # (rkL, rkC, rkP, H_narrow, H_wide, NT_level)
        (g.RK0L, g.RK0, g.RK0P, c.H0, c.H1, c.NT0),
        (g.RK1L, g.RK1, g.RK1P, c.H1, c.H2, c.NT1),
        (g.RK2L, g.RK2, g.RK2P, c.H2, c.H3, c.NT2),
        (g.RK3L, g.RK3, g.RK3P, c.H3, c.H4, c.NT3),
        (g.RK4L, g.RK4, g.RK4P, c.H4, c.H5, c.NT4),
        (g.RK5L, g.RK5, g.RK5P, c.H5, c.H6, c.NT5),
        (g.RK6L, g.RK6, g.RK6P, c.H6, c.H7, c.NT6),
        (g.RK7L, g.RK7, g.RK7P, c.H7, c.H8, c.NT7),
        (g.RK8L, g.RK8, g.RK8P, c.H8, c.H9, c.NT8),
        (g.RK9L, g.RK9, g.RK9P, c.H9, c.H + c.H, c.NT9),
    ]

    h_bounds = [c.H0, c.H1, c.H2, c.H3, c.H4,
                c.H5, c.H6, c.H7, c.H8, c.H9]

    # Find entry level (uu < H_{n+1} but uu >= H_n)
    entry_level = None  # type: Optional[int]
    for lvl in range(10):
        h_hi = h_bounds[lvl + 1] if lvl < 9 else (c.H + c.H)
        if uu < h_hi:
            entry_level = lvl
            break

    if entry_level is None:
        # uu >= 2H: coarse-only region (label 69)
        g.RK[c.NT]     += float(fshape(uu))
        g.RK[c.NT - 1] += float(fshape(uu + c.H))
        g.RK[c.NT - 2] += float(fshape(uu + c.H + c.H))
        g.RK[c.NT - 3] += float(fshape(uu + c.H9 - c.H))
        # Ascending cascade from level 9 (label 139)
        _right_ascending_cascade(uu, ff, fshape, eps, c, g, start_level=9)
        return

    # Accumulate triplet at NT for entry level
    rkL, rkC, rkP, h_narrow, h_wide, nt = levels[entry_level]
    rkL[nt] += ff
    ff = float(fshape(uu + h_wide))
    rkC[nt] += ff
    ff = float(fshape(uu + h_narrow))
    rkP[nt] += ff
    if entry_level > 0 and ff < eps:
        return

    if entry_level == 0:
        _right_finish_h0_loop(uu, ff, fshape, eps, c, g)
        return

    _right_ascending_cascade(uu, ff, fshape, eps, c, g, start_level=entry_level - 1)


def _right_ascending_cascade(
    uu: float, ff: float, fshape: ShapeFunc, eps: float,
    c: GridConstants, g: GridArrays,
    start_level: int,
) -> None:
    """
    Right-side ascending cascade (labels 131-139 in Fortran).
    Fills NT_n - 1 entries from start_level down to 0.
    """
    asc_data = [
        # level 0: (rkL, rkC, rkP, h_c_add, h_l_add)
        (g.RK0L, g.RK0, g.RK0P, c.H1, c.H0),
        # level 1
        (g.RK1L, g.RK1, g.RK1P, c.H1 + c.H2, c.H0),
        # level 2
        (g.RK2L, g.RK2, g.RK2P, c.H2 + c.H3, c.H1),
        # level 3
        (g.RK3L, g.RK3, g.RK3P, c.H3 + c.H4, c.H2),
        # level 4
        (g.RK4L, g.RK4, g.RK4P, c.H4 + c.H5, c.H3),
        # level 5
        (g.RK5L, g.RK5, g.RK5P, c.H5 + c.H6, c.H4),
        # level 6
        (g.RK6L, g.RK6, g.RK6P, c.H6 + c.H7, c.H5),
        # level 7
        (g.RK7L, g.RK7, g.RK7P, c.H7 + c.H8, c.H6),
        # level 8
        (g.RK8L, g.RK8, g.RK8P, c.H8 + c.H9, c.H7),
        # level 9
        (g.RK9L, g.RK9, g.RK9P, c.H9 + c.H + c.H, c.H8),
    ]

    nts = [c.NT0, c.NT1, c.NT2, c.NT3, c.NT4,
           c.NT5, c.NT6, c.NT7, c.NT8, c.NT9]

    for lvl in range(start_level, -1, -1):
        rkL, rkC, rkP, h_c_add, h_l_add = asc_data[lvl]
        n = nts[lvl] - 1
        rkL[n] += ff
        ff_c = float(fshape(uu + h_c_add))
        rkC[n] += ff_c
        ff = float(fshape(uu + h_l_add))
        rkP[n] += ff
        if lvl > 0 and ff < eps:
            return

    _right_finish_h0_loop(uu, ff, fshape, eps, c, g)


def _right_finish_h0_loop(
    uu: float, ff: float, fshape: ShapeFunc, eps: float,
    c: GridConstants, g: GridArrays,
) -> None:
    """Label-12 loop (right side): fill the fine NT0 sub-grid from NT0-1 down."""
    xxx = c.H0
    for i in range(c.NT0 - 1, 0, -1):
        g.RK0L[i] += ff
        ff = float(fshape(uu + xxx + c.H1))
        g.RK0[i] += ff
        xxx += c.H0
        ff = float(fshape(uu + xxx))
        g.RK0P[i] += ff
        if ff < eps:
            return
    g.RK[1] += ff


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
class LineGridCalc:
    """
    High-level wrapper that mirrors the Fortran module interface.

    Usage
    -----
    >>> c = GridConstants(H=0.001, H0=0.001, H1=0.002, ...)
    >>> calc = LineGridCalc(c)
    >>> calc.left_lbl(freq, ul, my_voigt, eps=1e-6)
    >>> calc.center_lbl(freq, ul, my_voigt, eps=1e-6)
    >>> calc.right_lbl(freq, ul, my_voigt, eps=1e-6)
    >>> # Read results from calc.grids
    """

    def __init__(self, constants: GridConstants) -> None:
        self.constants = constants
        self.grids = GridArrays.zeros(constants)

    def reset(self) -> None:
        """Zero all accumulator arrays (equivalent to re-allocation in Fortran)."""
        self.grids = GridArrays.zeros(self.constants)

    def left_lbl(self, freq: float, ul: float, fshape: ShapeFunc, eps: float) -> None:
        left_lbl(freq, ul, fshape, eps, self.constants, self.grids)

    def center_lbl(self, freq: float, ul: float, fshape: ShapeFunc, eps: float) -> None:
        center_lbl(freq, ul, fshape, eps, self.constants, self.grids)

    def right_lbl(self, freq: float, ul: float, fshape: ShapeFunc, eps: float) -> None:
        right_lbl(freq, ul, fshape, eps, self.constants, self.grids)


# ---------------------------------------------------------------------------
# Example / smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math

    def lorentz(x: float, gamma: float = 0.1) -> float:
        """Lorentzian line-shape centred at 0."""
        return (gamma / math.pi) / (x * x + gamma * gamma)

    # Build a small grid for illustration
    c = GridConstants(
        H=0.01, H0=0.01, H1=0.02, H2=0.04, H3=0.08,
        H4=0.16, H5=0.32, H6=0.64, H7=1.28, H8=2.56, H9=5.12,
        NT=64,  NT0=64, NT1=32, NT2=16, NT3=8,
        NT4=4,  NT5=4,  NT6=2,  NT7=2,  NT8=2, NT9=2,
        cutOff=20.0, deltaWV=0.64,
    )

    calc = LineGridCalc(c)
    freq = 1000.0
    ul   = 999.5
    fshape = lambda x: lorentz(x)

    calc.left_lbl  (freq, ul, fshape, eps=1e-10)
    calc.center_lbl(freq, ul, fshape, eps=1e-10)
    calc.right_lbl (freq, ul, fshape, eps=1e-10)

    non_zero = sum(1 for v in calc.grids.RK if v != 0.0)
    print(f"RK non-zero entries: {non_zero} / {c.NT}")
    print(f"RK[1] = {calc.grids.RK[1]:.6e}")
    print(f"RK0[1] = {calc.grids.RK0[1]:.6e}")
    print("Smoke-test passed.")