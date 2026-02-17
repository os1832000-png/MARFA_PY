# Calculates for 2 grids only

"""
LineGridCalc_2grid.py

Simplified 2-level version of LineGridCalc.
Computes contributions to LBL (Line-By-Line) spectral grids from the left wing,
central part, and right wing of a spectral line's extended sub-interval.

Grid structure (2 levels only):
  - Fine grid  : RK0 (with neighbour arrays RK0P and RK0L), spacing H0, NT0 points
  - Coarse grid: RK,                                         spacing H,  NT  points

The fine grid spacing H0 and coarse grid spacing H are related by H = 2 * H0,
i.e. there is exactly one level of refinement.

All three subroutines from the original are preserved:
  - left_lbl    : left wing  [startDeltaWV - cutOff ; startDeltaWV]
  - center_lbl  : central part [startDeltaWV ; endDeltaWV]
  - right_lbl   : right wing [endDeltaWV ; endDeltaWV + cutOff]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Type aliases  (compatible with Python 3.7+)
# ---------------------------------------------------------------------------
FloatArr  = List[float]
ShapeFunc = Callable[[float], float]


# ---------------------------------------------------------------------------
# GridConstants
# ---------------------------------------------------------------------------
@dataclass
class GridConstants:
    """
    Constants that define the 2-level grid layout.

    Attributes
    ----------
    H  : float  Coarse grid spacing.
    H0 : float  Fine grid spacing (typically H0 = H / 2).
    NT : int    Number of points on the coarse grid.
    NT0: int    Number of points on the fine grid.
    cutOff  : float  Maximum wing half-width.
    deltaWV : float  Width of the central sub-interval.
    """
    H:       float = 0.02
    H0:      float = 0.01
    NT:      int   = 64
    NT0:     int   = 128
    cutOff:  float = 20.0
    deltaWV: float = 0.64


def _make_arr(n: int) -> FloatArr:
    """1-based array (index 0 present but unused)."""
    return [0.0] * (n + 1)


# ---------------------------------------------------------------------------
# GridArrays
# ---------------------------------------------------------------------------
@dataclass
class GridArrays:
    """
    Accumulator arrays for the 2-level grid.

    RK          : coarse grid (NT points)
    RK0         : fine grid centre values     (NT0 points)
    RK0P        : fine grid 'previous' coarser neighbour
    RK0L        : fine grid 'later'    coarser neighbour
    """
    RK:   FloatArr = field(default_factory=list)
    RK0:  FloatArr = field(default_factory=list)
    RK0P: FloatArr = field(default_factory=list)
    RK0L: FloatArr = field(default_factory=list)

    @classmethod
    def zeros(cls, c: GridConstants) -> "GridArrays":
        return cls(
            RK   = _make_arr(c.NT),
            RK0  = _make_arr(c.NT0),
            RK0P = _make_arr(c.NT0),
            RK0L = _make_arr(c.NT0),
        )


# ---------------------------------------------------------------------------
# left_lbl
# ---------------------------------------------------------------------------
def left_lbl(
    freq:   float,
    ul:     float,
    fshape: ShapeFunc,
    eps:    float,
    c:      GridConstants,
    g:      GridArrays,
) -> None:
    """
    Accumulate contributions from the LEFT WING of a spectral line.

    Covers [startDeltaWV - cutOff ; startDeltaWV].

    Parameters
    ----------
    freq   : line centre frequency (wavenumber).
    ul     : current grid point position (wavenumber).
    fshape : line-shape function  f(delta_wv) -> float.
    eps    : negligibility threshold.
    c      : grid constants.
    g      : grid accumulator arrays (modified in-place).
    """
    uu = ul - freq

    # Guard: must be on the negative (left) side and within cutOff
    if uu >= 0.0:
        return
    if -uu > c.cutOff:
        return

    ff = float(fshape(uu))
    if ff < eps:
        return

    # Always accumulate on the coarse grid at position 1
    g.RK[1] += ff

    # ------------------------------------------------------------------
    # Determine which bracket |uu| falls into:
    #   |uu| < H0  →  fine grid only, go straight to finish loop
    #   H0 <= |uu| < H  →  fine grid triplet at index 1, then finish loop
    #   |uu| >= H  →  coarse grid only region
    # ------------------------------------------------------------------

    abs_uu = -uu

    if abs_uu < c.H0:
        # Inside the finest bracket: only the finish loop applies
        _left_finish_loop(uu, ff, fshape, eps, c, g)
        return

    if abs_uu < c.H:
        # Fine grid triplet at index 1
        g.RK0P[1] += ff
        ff = float(fshape(uu - c.H))
        g.RK0[1]  += ff
        ff = float(fshape(uu - c.H0))
        g.RK0L[1] += ff
        if ff < eps:
            return
        _left_finish_loop(uu, ff, fshape, eps, c, g)
        return

    # abs_uu >= H: coarse-only region
    # Accumulate a few extra coarse points then cascade into fine finish loop
    g.RK[2] += float(fshape(uu - c.H))
    g.RK[3] += float(fshape(uu - c.H - c.H))

    # Ascending cascade: fill fine grid at index 2 from this position
    ff = float(fshape(uu - c.H0))
    if ff < eps:
        return

    g.RK0P[2] += ff
    ff_c = float(fshape(uu - c.H0 - c.H))
    g.RK0[2]  += ff_c
    ff = float(fshape(uu - c.H0 - c.H0))
    g.RK0L[2] += ff
    if ff < eps:
        return

    _left_finish_loop(uu, ff, fshape, eps, c, g)


def _left_finish_loop(
    uu:     float,
    ff:     float,
    fshape: ShapeFunc,
    eps:    float,
    c:      GridConstants,
    g:      GridArrays,
) -> None:
    """
    Fine-grid finish loop for the left wing.
    Walks from index 2 to NT0, accumulating (P, centre, L) triplets.
    """
    xxx = c.H0
    for i in range(2, c.NT0 + 1):
        g.RK0P[i] += ff
        ff  = float(fshape(uu - xxx - c.H))
        g.RK0[i]  += ff
        xxx += c.H0
        ff  = float(fshape(uu - xxx))
        g.RK0L[i] += ff
        if ff < eps:
            return


# ---------------------------------------------------------------------------
# center_lbl
# ---------------------------------------------------------------------------
def center_lbl(
    freq:   float,
    ul:     float,
    fshape: ShapeFunc,
    eps:    float,
    c:      GridConstants,
    g:      GridArrays,
) -> None:
    """
    Accumulate contributions from the CENTRAL PART of a spectral line.

    Covers [startDeltaWV ; endDeltaWV].

    Two symmetric sub-passes:
      - Left-right : from UU stepping toward 0
      - Right-left : from deltaWV - UU stepping toward 0
    """
    uu = ul - freq
    if uu >= c.deltaWV:
        return

    ff0 = float(fshape(0.0))
    if ff0 < eps:
        return

    eps4   = eps * 0.25
    conser = uu - c.H
    fa     = float(fshape(uu))

    if fa > eps4:
        g.RK[1] += fa

    npoint = 1

    # ---- LEFT-RIGHT PASS ----
    if uu >= c.H:
        npoint, conser = _center_left_right(uu, fa, fshape, eps, c, g, conser)

    # ---- RIGHT-LEFT PASS ----
    _center_right_left(uu, fshape, eps, c, g, npoint=npoint + 1, conser=conser)


def _center_left_right(
    uu:     float,
    fa:     float,
    fshape: ShapeFunc,
    eps:    float,
    c:      GridConstants,
    g:      GridArrays,
    conser: float,
) -> Tuple[int, float]:
    """Left-right sub-pass: walk from UU toward 0 on fine then coarse grid."""
    uuu    = uu
    npoint = 1

    # Fine grid pass
    if uuu >= c.H0 + c.H0:
        for i in range(1, c.NT0 + 1):
            uuu -= c.H0
            ff   = float(fshape(uuu))
            if ff < eps:
                fa = ff
                break
            g.RK0P[i] += fa
            g.RK0[i]  += float(fshape(uuu + c.H))
            g.RK0L[i] += ff
            fa = ff
            if uuu - c.H0 < c.H0:
                break

    # Coarse grid fill
    i      = c.NT0 * 2 + 2
    conser = uu - (i - 1) * c.H
    for icon in range(i, c.NT + 1):
        g.RK[icon] += float(fshape(conser))
        conser -= c.H
        if conser < 0.0:
            npoint = icon
            return npoint, conser

    return npoint, conser


def _center_right_left(
    uu:     float,
    fshape: ShapeFunc,
    eps:    float,
    c:      GridConstants,
    g:      GridArrays,
    npoint: int,
    conser: float,
) -> None:
    """Right-left sub-pass: walk from deltaWV - UU toward 0, filling grids from NT down."""
    uuu = c.deltaWV - uu
    fa  = float(fshape(uuu))
    iii = 0

    # Fine grid pass (reverse direction)
    if uuu >= c.H0 + c.H0:
        for i in range(c.NT0, 0, -1):
            iii += 1
            uuu -= c.H0
            ff   = float(fshape(uuu))
            if ff < eps:
                fa = ff
                break
            g.RK0L[i] += fa
            g.RK0[i]  += float(fshape(uuu + c.H))
            g.RK0P[i] += ff
            fa = ff
            if uuu - c.H0 < c.H0:
                break

    # Coarse grid fill
    i_end = c.NT - iii * 2
    for ii in range(npoint, i_end + 1):
        g.RK[ii] += float(fshape(conser))
        conser -= c.H


# ---------------------------------------------------------------------------
# right_lbl
# ---------------------------------------------------------------------------
def right_lbl(
    freq:   float,
    ul:     float,
    fshape: ShapeFunc,
    eps:    float,
    c:      GridConstants,
    g:      GridArrays,
) -> None:
    """
    Accumulate contributions from the RIGHT WING of a spectral line.

    Covers [endDeltaWV ; endDeltaWV + cutOff].
    Mirror of left_lbl: fills from NT down to 1.
    """
    uu = ul - freq - c.deltaWV

    if uu >= c.cutOff:
        return

    ff = float(fshape(uu))
    if ff < eps:
        return

    # Always accumulate on the coarse grid at last position
    g.RK[c.NT] += ff

    # ------------------------------------------------------------------
    # Determine bracket (mirror of left_lbl):
    #   uu < H0      →  fine grid finish loop only
    #   H0 <= uu < H →  fine grid triplet at NT0, then finish loop
    #   uu >= H      →  coarse-only region + cascade
    # ------------------------------------------------------------------

    if uu < c.H0:
        _right_finish_loop(uu, ff, fshape, eps, c, g)
        return

    if uu < c.H:
        # Fine grid triplet at last index
        g.RK0L[c.NT0] += ff
        ff = float(fshape(uu + c.H))
        g.RK0[c.NT0]  += ff
        ff = float(fshape(uu + c.H0))
        g.RK0P[c.NT0] += ff
        if ff < eps:
            return
        _right_finish_loop(uu, ff, fshape, eps, c, g)
        return

    # uu >= H: coarse-only region
    g.RK[c.NT - 1] += float(fshape(uu + c.H))
    g.RK[c.NT - 2] += float(fshape(uu + c.H + c.H))

    # Ascending cascade into fine grid at NT0 - 1
    n = c.NT0 - 1
    ff = float(fshape(uu + c.H0))
    if ff < eps:
        return

    g.RK0L[n] += ff
    ff_c = float(fshape(uu + c.H0 + c.H))
    g.RK0[n]  += ff_c
    ff = float(fshape(uu + c.H0 + c.H0))
    g.RK0P[n] += ff
    if ff < eps:
        return

    _right_finish_loop(uu, ff, fshape, eps, c, g)


def _right_finish_loop(
    uu:     float,
    ff:     float,
    fshape: ShapeFunc,
    eps:    float,
    c:      GridConstants,
    g:      GridArrays,
) -> None:
    """
    Fine-grid finish loop for the right wing.
    Walks from NT0-1 down to 1, accumulating (L, centre, P) triplets.
    """
    xxx = c.H0
    for i in range(c.NT0 - 1, 0, -1):
        g.RK0L[i] += ff
        ff  = float(fshape(uu + xxx + c.H))
        g.RK0[i]  += ff
        xxx += c.H0
        ff  = float(fshape(uu + xxx))
        g.RK0P[i] += ff
        if ff < eps:
            return
    g.RK[1] += ff


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
class LineGridCalc2:
    """
    High-level wrapper for the 2-grid LBL calculator.

    Usage
    -----
    >>> c = GridConstants(H=0.02, H0=0.01, NT=64, NT0=128,
    ...                   cutOff=20.0, deltaWV=0.64)
    >>> calc = LineGridCalc2(c)
    >>> calc.left_lbl  (freq, ul, my_voigt, eps=1e-6)
    >>> calc.center_lbl(freq, ul, my_voigt, eps=1e-6)
    >>> calc.right_lbl (freq, ul, my_voigt, eps=1e-6)
    >>> # Results in calc.grids.RK, calc.grids.RK0, etc.
    """

    def __init__(self, constants: GridConstants) -> None:
        self.constants = constants
        self.grids     = GridArrays.zeros(constants)

    def reset(self) -> None:
        """Zero all accumulator arrays."""
        self.grids = GridArrays.zeros(self.constants)

    def left_lbl(self, freq: float, ul: float, fshape: ShapeFunc, eps: float) -> None:
        left_lbl(freq, ul, fshape, eps, self.constants, self.grids)

    def center_lbl(self, freq: float, ul: float, fshape: ShapeFunc, eps: float) -> None:
        center_lbl(freq, ul, fshape, eps, self.constants, self.grids)

    def right_lbl(self, freq: float, ul: float, fshape: ShapeFunc, eps: float) -> None:
        right_lbl(freq, ul, fshape, eps, self.constants, self.grids)


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    def lorentz(x: float, gamma: float = 0.1) -> float:
        """Lorentzian line-shape centred at 0."""
        return (gamma / math.pi) / (x * x + gamma * gamma)

    c = GridConstants(
        H=0.02, H0=0.01,
        NT=64,  NT0=128,
        cutOff=20.0, deltaWV=0.64,
    )

    calc   = LineGridCalc2(c)
    freq   = 1000.0
    ul     = 999.5
    fshape = lambda x: lorentz(x)

    calc.left_lbl  (freq, ul, fshape, eps=1e-10)
    calc.center_lbl(freq, ul, fshape, eps=1e-10)
    calc.right_lbl (freq, ul, fshape, eps=1e-10)

    nz_rk  = sum(1 for v in calc.grids.RK  if v != 0.0)
    nz_rk0 = sum(1 for v in calc.grids.RK0 if v != 0.0)

    print(f"RK  non-zero entries : {nz_rk}  / {c.NT}")
    print(f"RK0 non-zero entries : {nz_rk0} / {c.NT0}")
    print(f"RK[1]   = {calc.grids.RK[1]:.6e}")
    print(f"RK0[1]  = {calc.grids.RK0[1]:.6e}")
    print(f"RK0P[1] = {calc.grids.RK0P[1]:.6e}")
    print(f"RK0L[1] = {calc.grids.RK0L[1]:.6e}")
    print("Smoke-test passed.")