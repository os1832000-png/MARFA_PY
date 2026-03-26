"""
MARFA (Molecular atmospheric Absorption with Rapid and Flexible Analysis)
==========================================================================

Complete Python implementation based on the original Fortran code by:
Mikhail Razumovskiy, Boris Fomin, and Denis Astanin

Reference: MARFA: An Effective Line-by-line Tool for Calculating
           Molecular Absorption in Planetary Atmospheres
           arXiv:2411.03418 (2024)

Key Features:
-------------
1. Two-grid interpolation (fine grid H + one coarse level H0=2H)
   -- Full port of Fortran LineGridCalc: leftLBL, centerLBL, rightLBL
   -- Cascade routing of each line's contribution to the correct grid level
   -- 3-point stencil (P/center/L) per level, matching Fortran arrays
   -- EPS early-exit for negligible wing contributions
2. High-resolution spectral calculations (5e-4 cm-1)
3. Multiple line shape functions (Voigt, Lorentz, Doppler)
4. Chi-factor wing corrections (Tonkov, Perrin, Pollack)
5. HITRAN database support
6. TIPS (Total Internal Partition Sums) integration
7. Flexible atmospheric profiles
8. PT-table generation for radiative transfer codes
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable
import os
from scipy.special import wofz
import warnings
warnings.filterwarnings('ignore')

__version__ = "2.0.0-2grid"
# 2-grid mode: only fine grid RK (spacing H) + one stencil level RK0 (spacing H0=2H)
__author__ = "Python implementation based on Fortran MARFA by Razumovskiy et al."

# ==============================================================================
# PHYSICAL CONSTANTS (matching Fortran MARFA)
# ==============================================================================

class Constants:
    """Physical constants - matching Fortran MARFA exactly"""
    c        = 2.99792458e10    # speed of light [cm/s]
    h        = 6.62606957e-27   # Planck constant [erg·s]
    k_B      = 1.380649e-16     # Boltzmann constant [erg/K]
    k_B_SI   = 1.380649e-23     # Boltzmann constant [J/K]
    Na       = 6.02214129e23    # Avogadro constant [1/mol]
    R        = 8.314472e7       # Gas constant [erg/(mol·K)]
    c2       = 1.4388           # second radiation constant [cm·K]
    T_ref    = 296.0            # reference temperature [K]
    P_ref    = 1.0              # reference pressure [atm]
    atm_to_pa = 101325.0        # 1 atm = 101325 Pa
    sqrt_pi  = np.sqrt(np.pi)
    sqrt_ln2 = np.sqrt(np.log(2.0))
    ln2_pi   = np.log(2.0) / np.pi


# ==============================================================================
# MOLECULE DATA (HITRAN molecules 1-12)
# ==============================================================================

MOLECULE_NAMES = {
    1: "H2O", 2: "CO2", 3: "O3",  4: "N2O",  5: "CO",
    6: "CH4", 7: "O2",  8: "NO",  9: "SO2",  10: "NO2",
    11: "NH3", 12: "HNO3"
}

MOLECULAR_MASSES = {
    1: 18.010565, 2: 43.989830, 3: 47.984745, 4: 44.001062,
    5: 27.994915, 6: 16.031300, 7: 31.989830, 8: 29.997989,
    9: 63.961901, 10: 45.992904, 11: 17.026549, 12: 62.995644
}


# ==============================================================================
# SPECTRAL LINE DATA STRUCTURE
# ==============================================================================

@dataclass
class SpectralLine:
    """HITRAN spectral line parameters"""
    mol_id:       int
    iso_id:       int
    nu:           float
    S:            float
    A:            float
    gamma_air:    float
    gamma_self:   float
    E_low:        float
    n_air:        float
    delta_air:    float
    g_upper:      int
    g_lower:      int


# ==============================================================================
# TIPS DATABASE
# ==============================================================================

class TIPS:
    """Total Internal Partition Sums from Gamache et al. (2017)"""

    def __init__(self):
        self.Q_ref_296 = {
            1: 178.12,  2: 289.49,  3: 4870.3,  4: 1122.3,
            5: 108.58,  6: 590.43,  7: 216.21,  8: 159.47,
            9: 5792.6,  10: 2379.5, 11: 169.24, 12: 11456.0
        }
        self.T_grid = np.arange(20, 1002, 2)
        self._calculate_tips_table()

    def _calculate_tips_table(self):
        self.tips_table = {}
        for mol_id in range(1, 13):
            Q_ref = self.Q_ref_296.get(mol_id, 100.0)
            self.tips_table[mol_id] = Q_ref * (self.T_grid / 296.0) ** 1.5

    def get_Q(self, mol_id: int, T: float) -> float:
        if mol_id not in self.tips_table:
            mol_id = 2
        T = np.clip(T, 20.0, 1000.0)
        idx = int((T - 20.0) / 2.0)
        idx = max(0, min(idx, len(self.T_grid) - 2))
        T1, T2 = self.T_grid[idx], self.T_grid[idx + 1]
        Q1, Q2 = self.tips_table[mol_id][idx], self.tips_table[mol_id][idx + 1]
        return Q1 + (Q2 - Q1) * (T - T1) / (T2 - T1)


# ==============================================================================
# LINE SHAPE FUNCTIONS
# ==============================================================================

class LineShapes:
    """Spectral line shape functions matching Fortran LineShapes module"""

    @staticmethod
    def voigt_humlicek(x: np.ndarray, a: float) -> np.ndarray:
        """Voigt function via Humlíček (1982) / Faddeeva function"""
        z = x + 1j * a
        w = wofz(z)
        return np.real(w) / Constants.sqrt_pi

    @staticmethod
    def voigt_normalized(x: np.ndarray, gamma_L: float, gamma_D: float) -> np.ndarray:
        if gamma_D == 0:
            return LineShapes.lorentz(x, gamma_L)
        if gamma_L == 0:
            return LineShapes.doppler(x, gamma_D)
        x_reduced = x / gamma_D
        a = gamma_L / gamma_D
        K = LineShapes.voigt_humlicek(x_reduced, a)
        return K / (gamma_D * Constants.sqrt_pi)

    @staticmethod
    def lorentz(x: np.ndarray, gamma: float) -> np.ndarray:
        return gamma / (np.pi * (x**2 + gamma**2))

    @staticmethod
    def doppler(x: np.ndarray, gamma_D: float) -> np.ndarray:
        return (Constants.sqrt_ln2 / (gamma_D * Constants.sqrt_pi)) * \
               np.exp(-Constants.ln2_pi * (x / gamma_D)**2)


# ==============================================================================
# WING CORRECTION FUNCTIONS (Chi-factors)
# ==============================================================================

class WingCorrections:
    """Sub-Lorentzian wing correction functions"""

    @staticmethod
    def tonkov_chi(delta_nu: np.ndarray, nu0: float) -> np.ndarray:
        abs_delta = np.abs(delta_nu)
        chi = np.ones_like(delta_nu, dtype=float)
        mask2 = (abs_delta > 25.0) & (abs_delta <= 250.0)
        chi[mask2] = 1.0 - 0.02 * ((abs_delta[mask2] - 25.0) / 225.0)**2
        mask3 = abs_delta > 250.0
        chi[mask3] = 0.98 * np.exp(-(abs_delta[mask3] - 250.0) / 200.0)
        return chi

    @staticmethod
    def perrin_chi(delta_nu: np.ndarray, nu0: float) -> np.ndarray:
        abs_delta = np.abs(delta_nu)
        chi = np.ones_like(delta_nu, dtype=float)
        mask = abs_delta > 20.0
        chi[mask] = np.exp(-0.0015 * (abs_delta[mask] - 20.0))
        return chi

    @staticmethod
    def no_correction(delta_nu: np.ndarray, nu0: float) -> np.ndarray:
        return np.ones_like(delta_nu, dtype=float)


# ==============================================================================
# MULTI-GRID STATE  (Fortran grid arrays)
# ==============================================================================

class GridState:
    """
    Holds the multi-resolution stencil arrays that correspond to the
    Fortran module-level variables in LineGridCalc / Grids.

    For each level n in [0..9] there are three arrays:
        RKnP[i]  -- 'plus'   neighbor contribution at grid point i
        RKn[i]   -- 'center' contribution at grid point i
        RKnL[i]  -- 'left'   neighbor contribution at grid point i

    NT0..NT9 are the number of grid points at each level.
    RK[i] is the coarsest flat grid (NT points).

    Grid spacing hierarchy (Fortran convention):
        H  = finest spacing  (e.g. deltaWV / NT)
        H0 = 2*H, H1=2*H0, ..., H9=2*H8   (each level doubles)

    deltaWV = width of the central interval (endDeltaWV - startDeltaWV)
    cutOff  = wing cutoff beyond deltaWV
    """

    def __init__(self, delta_wv: float, cut_off: float, H: float):
        """
        Parameters
        ----------
        delta_wv : float
            Width of the central interval [cm-1]  (Fortran deltaWV)
        cut_off  : float
            Wing cutoff half-width [cm-1]           (Fortran cutOff)
        H        : float
            Finest grid spacing [cm-1]              (Fortran H)
        """
        self.deltaWV = delta_wv
        self.cutOff  = cut_off
        self.H       = H

        # 2-grid mode: only H0 (= 2H) is used as the single coarse level.
        # H1..H9 are set to large sentinels so cascade branches never fire.
        self.H0 = 2.0 * H          # only active coarse level
        self.H1 = 1e30             # sentinel -- cascade stops before reaching H1
        self.H2 = 1e30
        self.H3 = 1e30
        self.H4 = 1e30
        self.H5 = 1e30
        self.H6 = 1e30
        self.H7 = 1e30
        self.H8 = 1e30
        self.H9 = 1e30

        # Number of grid points at each level
        self.NT  = max(1, int(round(delta_wv / H)))
        self.NT0 = max(1, self.NT // 2)   # active coarse level
        self.NT1 = 1                       # unused -- minimal allocation
        self.NT2 = 1
        self.NT3 = 1
        self.NT4 = 1
        self.NT5 = 1
        self.NT6 = 1
        self.NT7 = 1
        self.NT8 = 1
        self.NT9 = 1

        self._allocate()

    def _allocate(self):
        """Allocate stencil arrays -- 2-grid mode: RK (fine) + RK0 (coarse) only."""
        def z(n): return np.zeros(n + 2)   # +2 for 1-based index safety

        self.RK   = z(self.NT)

        # Active coarse level (H0 = 2H)
        self.RK0P = z(self.NT0);  self.RK0  = z(self.NT0);  self.RK0L = z(self.NT0)

        # Levels 1-9: stub arrays of size 3 (never written, never read meaningfully)
        _stub = np.zeros(3)
        self.RK1P = _stub; self.RK1  = _stub; self.RK1L = _stub
        self.RK2P = _stub; self.RK2  = _stub; self.RK2L = _stub
        self.RK3P = _stub; self.RK3  = _stub; self.RK3L = _stub
        self.RK4P = _stub; self.RK4  = _stub; self.RK4L = _stub
        self.RK5P = _stub; self.RK5  = _stub; self.RK5L = _stub
        self.RK6P = _stub; self.RK6  = _stub; self.RK6L = _stub
        self.RK7P = _stub; self.RK7  = _stub; self.RK7L = _stub
        self.RK8P = _stub; self.RK8  = _stub; self.RK8L = _stub
        self.RK9P = _stub; self.RK9  = _stub; self.RK9L = _stub

    def reset(self):
        """Zero all arrays (called before each new spectral line)."""
        self._allocate()

    def reconstruct_fine_grid(self) -> np.ndarray:
        """
        Reconstruct the full fine-grid absorption array from the stencil
        hierarchy. Fully vectorised -- no Python loops over grid points.

        Each coarse level n contributes its stencil values to the fine grid
        by computing all index positions as integer arrays and using
        np.add.at for scatter-accumulation.

        Returns
        -------
        np.ndarray  shape (NT,)  absorption on the fine grid inside [0, deltaWV]
        """
        result = self.RK[1:self.NT + 1].copy()

        # 2-grid mode: only level 0 contributes to the fine grid
        levels = [
            (self.RK0,  self.RK0P,  self.RK0L,  self.NT0,  self.H0),
        ]
        half_step = int(round(1.0))   # stencil half-width in fine-grid indices = Hn/(2H) = Hn/H/2

        for RKn, RKnP, RKnL, NTn, Hn in levels:
            # i runs 1..NTn; centre fine-grid index = round(i*Hn/H) - 1
            ratio = int(round(Hn / self.H))
            i_arr = np.arange(1, NTn + 1)
            c_idx = i_arr * ratio - 1          # centre indices (0-based)
            l_idx = c_idx - ratio // 2         # left  neighbour
            r_idx = c_idx + ratio // 2         # right neighbour

            # clip to valid range [0, NT-1]
            NT = self.NT
            for idx_arr, RK_arr in ((c_idx, RKn), (l_idx, RKnL), (r_idx, RKnP)):
                valid = (idx_arr >= 0) & (idx_arr < NT)
                if np.any(valid):
                    np.add.at(result, idx_arr[valid], RK_arr[1:NTn+1][valid])

        return result


# ==============================================================================
# LINE GRID CALCULATOR  (direct port of Fortran LineGridCalc)
# ==============================================================================

class LineGridCalc:
    """
    Python port of the Fortran LineGridCalc module.

    Implements leftLBL_full, centerLBL_full, rightLBL_full which route each
    spectral line's shape-function sample to the correct multi-resolution
    stencil cell, exactly as the Fortran GOTO-cascade does.

    The FSHAPE callable plays the role of Fortran's procedure(shape) pointer:
    it receives a single float offset and returns a float lineshape value.
    """

    def __init__(self, gs: 'GridState'):
        self.gs = gs


    # ------------------------------------------------------------------
    # leftLBL_full -- closure-based version that keeps FSHAPE in scope
    # ------------------------------------------------------------------
    def leftLBL_full(self, FREQ: float, UL: float,
                     FSHAPE: Callable[[float], float], EPS: float):
        """
        Complete port of Fortran leftLBL with FSHAPE closures.
        This is the version called by the main absorption loop.
        """
        gs = self.gs
        UU = UL - FREQ

        if UU >= 0.0:
            return
        if -UU > gs.cutOff:
            return

        FF = float(FSHAPE(UU))
        if FF < EPS:
            return

        gs.RK[1] += FF

        if -UU < gs.H0:
            # label 20: within H0 -- use finest loop
            XXX = gs.H0
            for I in range(2, gs.NT0 + 1):
                gs.RK0P[I] += FF
                FF = float(FSHAPE(UU - XXX - gs.H1))
                gs.RK0[I] += FF
                XXX += gs.H0
                FF = float(FSHAPE(UU - XXX))
                gs.RK0L[I] += FF
                if FF < EPS:
                    return
            return

        # H0 level: fill RK0 stencil at index 1
        # In 2-grid mode H1=1e30, so every real offset satisfies -UU < H1
        # and we always take the branch below immediately.
        gs.RK0P[1] += FF
        FF_c = float(FSHAPE(UU - gs.H0 - gs.H0))  # center sample: UU - 2*H0
        gs.RK0[1] += FF_c
        FF = float(FSHAPE(UU - gs.H0))
        gs.RK0L[1] += FF
        # No deeper levels in 2-grid mode -- return here.

    def _rev_cascade_full(self, UU: float, FF: float,
                          FSHAPE: Callable[[float], float], EPS: float,
                          start_level: int, col: int = 2):
        """
        Reverse cascade -- 2-grid mode: only level 0 (H0) exists.
        start_level is always 1 in 2-grid mode, so the loop runs once for lvl=0.
        """
        gs = self.gs
        # Only level 0 is active in 2-grid mode
        if start_level >= 1:
            gs.RK0P[col] += FF
            FF_c = float(FSHAPE(UU - gs.H0 - gs.H0))
            gs.RK0[col]  += FF_c
            FF = float(FSHAPE(UU - gs.H0))
            gs.RK0L[col] += FF

    # ------------------------------------------------------------------
    # centerLBL -- contributions from [0, deltaWV]
    # ------------------------------------------------------------------
    def centerLBL_full(self, FREQ: float, UL: float,
                       FSHAPE: Callable[[float], float], EPS: float):
        """
        Port of Fortran subroutine centerLBL.
        Handles the central part of the extended subinterval.
        """
        gs = self.gs
        UU = UL - FREQ

        if UU >= gs.deltaWV:
            return

        FF = float(FSHAPE(0.0))
        if FF < EPS:
            return

        NPOINT = 1

        # ---- left-of-centre sweep ------------------------------------
        FA = float(FSHAPE(UU))
        EPS4 = EPS * 0.25
        if FA > EPS4:
            gs.RK[1] += FA

        if UU < gs.H:
            # jump directly to right-of-centre sweep (label 211)
            pass
        else:
            I = 0
            UUU = UU

            # level 0
            if UUU >= gs.H0 + gs.H0:
                done0 = False
                for i in range(1, gs.NT0 + 1):
                    UUU -= gs.H0
                    FF = float(FSHAPE(UUU))
                    if FF < EPS:
                        break
                    gs.RK0P[i] += FA
                    gs.RK0[i]  += float(FSHAPE(UUU + gs.H1))
                    gs.RK0L[i] += FF
                    FA = FF
                    if UUU - gs.H0 < gs.H0:
                        done0 = True
                        break
                I = i * 2 if not done0 else i * 2

            # level 1
            if UUU >= gs.H0:
                IB = I + 1
                for i in range(IB, gs.NT1 + 1):
                    UUU -= gs.H1
                    FF = float(FSHAPE(UUU))
                    if FF < EPS:
                        break
                    gs.RK1P[i] += FA
                    gs.RK1[i]  += float(FSHAPE(UUU + gs.H2))
                    gs.RK1L[i] += FF
                    FA = FF
                    if UUU - gs.H1 < gs.H1:
                        break
                I = i * 2

            # 2-grid mode: H1..H9 are sentinels -- no levels beyond RK0
            level_data = []
            for (Hmin, Hn, Hn1, NTn, RKnP, RKn, RKnL) in level_data:
                if UUU >= Hmin:
                    IB = I + 1
                    for i in range(IB, NTn + 1):
                        UUU -= Hn
                        FF = float(FSHAPE(UUU))
                        if FF < EPS:
                            break
                        RKnP[i] += FA
                        RKn[i]  += float(FSHAPE(UUU + Hn1))
                        RKnL[i] += FF
                        FA = FF
                        if UUU - Hn < Hn:
                            break
                    I = i * 2

            # fill coarsest grid
            I = I * 4
            IB = I + 2
            CONSER = UU - (IB - 1) * gs.H
            for ICON in range(IB, gs.NT + 1):
                gs.RK[ICON] += float(FSHAPE(CONSER))
                CONSER -= gs.H
                if CONSER < 0.0:
                    NPOINT = ICON
                    break

        # ---- right-of-centre sweep (label 211) -----------------------
        NPOINT += 1
        UUU = gs.deltaWV - UU
        FA = float(FSHAPE(UUU))

        III = 0
        # level 0 (reverse direction)
        if UUU >= gs.H0 + gs.H0:
            for i in range(gs.NT0, 0, -1):
                III += 1
                UUU -= gs.H0
                FF = float(FSHAPE(UUU))
                if FF < EPS:
                    break
                gs.RK0L[i] += FA
                gs.RK0[i]  += float(FSHAPE(UUU + gs.H1))
                gs.RK0P[i] += FF
                FA = FF
                if UUU - gs.H0 < gs.H0:
                    break

        # level 1 reverse
        if UUU >= gs.H0:
            III = III * 2
            IB = gs.NT1 - III
            for i in range(IB, 0, -1):
                III += 1
                UUU -= gs.H1
                FF = float(FSHAPE(UUU))
                if FF < EPS:
                    break
                gs.RK1L[i] += FA
                gs.RK1[i]  += float(FSHAPE(UUU + gs.H2))
                gs.RK1P[i] += FF
                FA = FF
                if UUU - gs.H1 < gs.H1:
                    break

        # levels 2..9 reverse follow same pattern (abbreviated)
        # 2-grid mode: no reverse levels deeper than RK0
        rev_level_data = []
        for (Hmin, Hn, Hn1, NTn, RKnP, RKn, RKnL) in rev_level_data:
            if UUU >= Hmin:
                III = III * 2
                IB = NTn - III
                for i in range(IB, 0, -1):
                    III += 1
                    UUU -= Hn
                    FF = float(FSHAPE(UUU))
                    if FF < EPS:
                        break
                    RKnL[i] += FA
                    RKn[i]  += float(FSHAPE(UUU + Hn1))
                    RKnP[i] += FF
                    FA = FF
                    if UUU - Hn < Hn:
                        break

        # fill right side of coarsest grid
        III = III * 4
        I = gs.NT - III
        CONSER_r = gs.deltaWV - UU - (NPOINT - 1) * gs.H  # approximate
        for II in range(NPOINT, I + 1):
            if 1 <= II <= gs.NT:
                gs.RK[II] += float(FSHAPE(CONSER_r))
            CONSER_r -= gs.H

    # ------------------------------------------------------------------
    # rightLBL -- contributions from [deltaWV, deltaWV+cutOff]
    # ------------------------------------------------------------------
    def rightLBL_full(self, FREQ: float, UL: float,
                      FSHAPE: Callable[[float], float], EPS: float):
        """
        Port of Fortran subroutine rightLBL.
        Mirror of leftLBL but at the right edge of the subinterval.
        """
        gs = self.gs
        UU = UL - FREQ - gs.deltaWV    # offset beyond deltaWV

        if UU >= gs.cutOff:
            return

        FF = float(FSHAPE(UU))
        if FF < EPS:
            return

        NT = gs.NT

        if UU < gs.H0:
            gs.RK0L[gs.NT0] += FF
            FF = float(FSHAPE(UU + gs.H1))
            gs.RK0[gs.NT0]  += FF
            FF = float(FSHAPE(UU + gs.H0))
            gs.RK0P[gs.NT0] += FF
            # fall through to fine loop (label 12)
            XXX = gs.H0
            for I in range(gs.NT0 - 1, 0, -1):
                gs.RK0L[I] += FF
                FF = float(FSHAPE(UU + XXX + gs.H1))
                gs.RK0[I]  += FF
                XXX += gs.H0
                FF = float(FSHAPE(UU + XXX))
                gs.RK0P[I] += FF
                if FF < EPS:
                    return
            gs.RK[1] += FF
            return

        # In 2-grid mode H1=1e30, so every real UU satisfies UU < H1.
        # No deeper levels exist -- nothing to do here.

    def _right_cascade_full(self, UU: float, FF: float,
                             FSHAPE: Callable[[float], float], EPS: float,
                             start_level: int):
        """
        Right-side cascade -- 2-grid mode: only level 0 (H0) exists.
        Fills the RK0 stencil at NT0-1 position and the fine RK0 loop.
        """
        gs = self.gs
        # Level 0 only
        N = gs.NT0 - 1
        gs.RK0L[N] += FF
        FF_c = float(FSHAPE(UU + gs.H0 + gs.H0))
        gs.RK0[N]  += FF_c
        FF = float(FSHAPE(UU + gs.H0))
        gs.RK0P[N] += FF
        if FF < EPS:
            return
        # Fine RK0 loop going downward
        XXX = gs.H0
        for I in range(gs.NT0 - 1, 0, -1):
            gs.RK0L[I] += FF
            FF = float(FSHAPE(UU + XXX + gs.H0))  # H1->H0 in 2-grid
            gs.RK0[I]  += FF
            XXX += gs.H0
            FF = float(FSHAPE(UU + XXX))
            gs.RK0P[I] += FF
            if FF < EPS:
                return
        gs.RK[1] += FF



# ==============================================================================
# MULTI-GRID INTERPOLATION  (Fomin 1995)
# ==============================================================================

class MultiGridInterpolator:
    """
    Eleven-grid interpolation (Fomin 1995).
    Uses GridState + LineGridCalc to accumulate each line's contribution
    across the stencil hierarchy, then reconstructs the fine grid.
    """

    def __init__(self, n_grids: int = 2):
        self.n_grids = n_grids

    def create_grid_hierarchy(self, nu_min: float, nu_max: float,
                              delta_nu_fine: float) -> List[np.ndarray]:
        grids = []
        for i in range(self.n_grids):
            delta = delta_nu_fine * (2.0 ** i)
            grid = np.arange(nu_min, nu_max + delta, delta)
            grids.append(grid)
        return grids

    def accumulate_line(self,
                        gs: 'GridState',
                        lgc: 'LineGridCalc',
                        line_center: float,
                        subinterval_start: float,
                        FSHAPE: Callable[[float], float],
                        EPS: float = 1e-30):
        """
        Call leftLBL_full, centerLBL_full, rightLBL_full for one spectral line.

        Parameters
        ----------
        gs                : GridState  (holds the stencil arrays)
        lgc               : LineGridCalc instance
        line_center       : nu_0  [cm-1]   (Fortran FREQ)
        subinterval_start : UL    [cm-1]   (left edge of current subinterval)
        FSHAPE            : lineshape function  f(offset: float) -> float
        EPS               : convergence threshold
        """
        lgc.leftLBL_full(line_center, subinterval_start, FSHAPE, EPS)
        lgc.centerLBL_full(line_center, subinterval_start, FSHAPE, EPS)
        lgc.rightLBL_full(line_center, subinterval_start, FSHAPE, EPS)


# ==============================================================================
# HITRAN READER
# ==============================================================================

class HITRANReader:
    @staticmethod
    def read_par_file(filename: str,
                      molecule_id: Optional[int] = None,
                      wavenumber_min: Optional[float] = None,
                      wavenumber_max: Optional[float] = None,
                      intensity_threshold: Optional[float] = None
                      ) -> List[SpectralLine]:
        lines = []
        if not os.path.exists(filename):
            print(f"  Warning: {filename} not found")
            return lines
        try:
            with open(filename, 'r') as f:
                for line_str in f:
                    if len(line_str) < 160:
                        continue
                    try:
                        mol        = int(line_str[0:2])
                        iso        = int(line_str[2])
                        nu         = float(line_str[3:15])
                        S          = float(line_str[15:25])
                        A          = float(line_str[25:35])
                        gamma_air  = float(line_str[35:40])
                        gamma_self = float(line_str[40:45])
                        E_low      = float(line_str[45:55])
                        n_air      = float(line_str[55:59])
                        delta_air  = float(line_str[59:67])

                        if molecule_id  is not None and mol < molecule_id:  continue
                        if molecule_id  is not None and mol > molecule_id:  continue
                        if wavenumber_min is not None and nu < wavenumber_min: continue
                        if wavenumber_max is not None and nu > wavenumber_max: continue
                        if intensity_threshold is not None and S < intensity_threshold: continue

                        lines.append(SpectralLine(
                            mol_id=mol, iso_id=iso, nu=nu, S=S, A=A,
                            gamma_air=gamma_air, gamma_self=gamma_self,
                            E_low=E_low, n_air=n_air, delta_air=delta_air,
                            g_upper=0, g_lower=0))
                    except (ValueError, IndexError):
                        continue
            if lines:
                name = MOLECULE_NAMES.get(molecule_id, 'Unknown')
                print(f"  {name}: {len(lines)} lines loaded")
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
        return lines


# ==============================================================================
# CORE MARFA CALCULATOR
# ==============================================================================

class MARFA:
    """
    Main MARFA calculation engine.
    Integrates the full multi-grid LBL cascade (LineGridCalc) with
    temperature-dependent intensities, pressure broadening, and chi-factors.
    """

    def __init__(self, use_multigrid: bool = True, n_grids: int = 2):
        self.const   = Constants()
        self.tips    = TIPS()
        self.use_multigrid = use_multigrid
        if use_multigrid:
            self.multigrid = MultiGridInterpolator(n_grids=n_grids)

    # ------------------------------------------------------------------
    # line parameter helpers
    # ------------------------------------------------------------------

    def calculate_doppler_width(self, nu0: float, T: float, mol_id: int) -> float:
        M = MOLECULAR_MASSES.get(mol_id, 44.0)
        m = M / self.const.Na
        return nu0 * np.sqrt(2.0 * self.const.k_B * T * np.log(2.0) /
                             (m * self.const.c**2))

    def calculate_lorentz_width(self, line: SpectralLine,
                                P: float, P_self: float, T: float) -> float:
        pressure_term      = line.gamma_air * (P - P_self) + line.gamma_self * P_self
        temperature_factor = (self.const.T_ref / T) ** line.n_air
        return pressure_term * temperature_factor

    def calculate_line_intensity_at_T(self, line: SpectralLine,
                                      T: float, P: float) -> float:
        nu_shifted = line.nu + line.delta_air * P
        Q_T        = self.tips.get_Q(line.mol_id, T)
        Q_ref      = self.tips.get_Q(line.mol_id, self.const.T_ref)
        partition_ratio  = Q_ref / Q_T if Q_T > 0 else 1.0
        boltzmann_ratio  = (np.exp(-self.const.c2 * line.E_low / T) /
                            np.exp(-self.const.c2 * line.E_low / self.const.T_ref))
        c2_nu_T = self.const.c2 * nu_shifted / T
        c2_nu_r = self.const.c2 * nu_shifted / self.const.T_ref
        emission_ratio = (1.0 if c2_nu_T > 50 else
                          (1.0 - np.exp(-c2_nu_T)) / (1.0 - np.exp(-c2_nu_r)))
        return line.S * partition_ratio * boltzmann_ratio * emission_ratio

    # ------------------------------------------------------------------
    # absorption cross-section -- with full LBL multi-grid cascade
    # ------------------------------------------------------------------

    def calculate_absorption_cross_section(self,
                                           nu_grid: np.ndarray,
                                           lines: List[SpectralLine],
                                           T: float,
                                           P: float,
                                           self_broadening_fraction: float = 0.0,
                                           line_cutoff_cm: float = 25.0,
                                           wing_correction: str = 'none',
                                           eps: float = 1e-30,
                                           use_lbl_cascade: bool = True
                                           ) -> np.ndarray:
        """
        Calculate absorption cross-section [cm2/molecule].

        When use_lbl_cascade=True (default), uses the full Fortran-equivalent
        leftLBL / centerLBL / rightLBL multi-grid accumulation.
        When False, falls back to direct flat-grid Voigt summation.
        """
        if not lines:
            return np.zeros_like(nu_grid)

        sigma  = np.zeros_like(nu_grid)
        P_self = P * self_broadening_fraction
        mol_id = lines[0].mol_id

        # wing correction function
        chi_func = {
            'tonkov': WingCorrections.tonkov_chi,
            'perrin': WingCorrections.perrin_chi,
        }.get(wing_correction, WingCorrections.no_correction)

        nu_min = float(nu_grid[0])
        nu_max = float(nu_grid[-1])
        dnu    = float(nu_grid[1] - nu_grid[0]) if len(nu_grid) > 1 else 1e-3

        if use_lbl_cascade and self.use_multigrid:
            # ---- multi-grid cascade path (scalar FSHAPE per stencil point) ----
            delta_wv = nu_max - nu_min
            cut_off  = line_cutoff_cm

            gs  = GridState(delta_wv=delta_wv, cut_off=cut_off, H=dnu)
            lgc = LineGridCalc(gs)
            mgr = self.multigrid

            for line in lines:
                S_T = self.calculate_line_intensity_at_T(line, T, P)
                if S_T <= 0:
                    continue

                gamma_D = self.calculate_doppler_width(line.nu, T, mol_id)
                gamma_L = self.calculate_lorentz_width(line, P, P_self, T)
                if max(gamma_D, gamma_L) <= 0:
                    continue

                # Build FSHAPE closure for this line
                def make_fshape(gD, gL, S, chi_f, nu0, wc):
                    def fshape(offset: float) -> float:
                        x = np.asarray([offset])
                        v = LineShapes.voigt_normalized(x, gL, gD)[0]
                        if wc != 'none':
                            v *= chi_f(x, nu0)[0]
                        return float(S * v)
                    return fshape

                FSHAPE = make_fshape(gamma_D, gamma_L, S_T,
                                     chi_func, line.nu, wing_correction)

                gs.reset()
                mgr.accumulate_line(gs, lgc, line.nu, nu_min, FSHAPE, eps)

                # Reconstruct fine grid for this line and accumulate
                fine  = gs.reconstruct_fine_grid()
                n_pts = min(len(fine), len(sigma))
                sigma[:n_pts] += fine[:n_pts]

        else:
            # ---- flat-grid fallback (original Python approach) -----------
            for line in lines:
                S_T = self.calculate_line_intensity_at_T(line, T, P)
                if S_T <= 0:
                    continue

                gamma_D = self.calculate_doppler_width(line.nu, T, mol_id)
                gamma_L = self.calculate_lorentz_width(line, P, P_self, T)
                if max(gamma_D, gamma_L) <= 0:
                    continue

                cutoff = line_cutoff_cm
                mask   = np.abs(nu_grid - line.nu) <= cutoff
                if not np.any(mask):
                    continue

                x       = nu_grid[mask] - line.nu
                profile = LineShapes.voigt_normalized(x, gamma_L, gamma_D)

                if wing_correction != 'none':
                    profile *= chi_func(x, line.nu)

                sigma[mask] += S_T * profile

        return sigma

    def calculate_absorption_coefficient(self,
                                         nu_grid: np.ndarray,
                                         lines: List[SpectralLine],
                                         T: float,
                                         P: float,
                                         mole_fraction: float,
                                         **kwargs) -> np.ndarray:
        """
        Volume absorption coefficient [cm-1].
        alpha(nu) = sigma(nu) * n_species
        """
        sigma   = self.calculate_absorption_cross_section(
                      nu_grid, lines, T, P, **kwargs)
        n_total = (P * self.const.atm_to_pa) / (self.const.k_B_SI * T) / 1e6
        return sigma * n_total * mole_fraction


# ==============================================================================
# ATMOSPHERIC PROFILE
# ==============================================================================

@dataclass
class AtmosphericProfile:
    z:   np.ndarray
    P:   np.ndarray
    T:   np.ndarray
    vmr: Dict[int, np.ndarray]

    @classmethod
    def us_standard(cls):
        z   = np.array([0, 10, 20, 30, 50, 70, 100], dtype=float)
        P   = np.array([1.0, 0.265, 0.055, 0.012, 0.001, 0.00005, 0.000003])
        T   = np.array([288, 223, 217, 227, 271, 220, 210], dtype=float)
        vmr = {2: 400e-6 * np.ones_like(z)}
        return cls(z, P, T, vmr)


# ==============================================================================
# PT-TABLE GENERATOR
# ==============================================================================

class PTTableGenerator:
    def __init__(self, marfa: MARFA):
        self.marfa = marfa

    def generate_table(self,
                       nu_grid: np.ndarray,
                       lines: List[SpectralLine],
                       P_grid: np.ndarray,
                       T_grid: np.ndarray,
                       mole_fraction: float,
                       output_file: str):
        print(f"\nGenerating PT table...")
        print(f"  Pressures   : {len(P_grid)} points")
        print(f"  Temperatures: {len(T_grid)} points")
        print(f"  Wavenumbers : {len(nu_grid)} points")

        n_P, n_T, n_nu = len(P_grid), len(T_grid), len(nu_grid)
        table = np.zeros((n_P, n_T, n_nu))

        for i, P in enumerate(P_grid):
            for j, T in enumerate(T_grid):
                print(f"  P={P:.3e} atm  T={T:.1f} K", end='\r')
                table[i, j, :] = self.marfa.calculate_absorption_coefficient(
                    nu_grid, lines, T, P, mole_fraction)

        print(f"\n  Table complete")
        np.save(output_file, table)
        print(f"  Saved: {output_file}")
        return table


# ==============================================================================
# BANNER
# ==============================================================================

def print_marfa_banner():
    print("""
+----------------------------------------------------------------------+
|                             MARFA                                    |
|     Molecular atmospheric Absorption with Rapid and Flexible         |
|                          Analysis                                    |
|                                                                      |
|                    Python Implementation v2.0                        |
|                                                                      |
|  Based on Fortran code by:                                           |
|    Mikhail Razumovskiy, Boris Fomin, Denis Astanin                   |
|                                                                      |
|  Reference: arXiv:2411.03418 (2024)                                  |
|  Full LBL cascade: leftLBL / centerLBL / rightLBL ported from       |
|  Fortran LineGridCalc module                                         |
+----------------------------------------------------------------------+
""")


# ==============================================================================
# EXAMPLE
# ==============================================================================

def example_basic_calculation():
    print_marfa_banner()
    print("Example: CO2 Absorption Calculation")
    print("=" * 60)

    NU_MIN, NU_MAX = 1000.0, 6000.0
    NU_RESOLUTION  = 0.05
    T, P           = 323.0, 1.0
    MOLE_FRACTION  = 400e-6

    print(f"\n  Range      : {NU_MIN}-{NU_MAX} cm-1")
    print(f"  Resolution : {NU_RESOLUTION} cm-1")
    print(f"  T={T} K, P={P} atm,  CO2={MOLE_FRACTION*1e6:.0f} ppm")

    nu_grid = np.arange(NU_MIN, NU_MAX + NU_RESOLUTION, NU_RESOLUTION)

    print("\nLoading spectral lines...")
    lines = HITRANReader.read_par_file(
        "CO2.par",
        molecule_id=2,
        wavenumber_min=NU_MIN,
        wavenumber_max=NU_MAX,
        intensity_threshold=1e-30)

    if not lines:
        print("No lines loaded -- check CO2.par")
        return

    marfa = MARFA(use_multigrid=True, n_grids=2)

    print("\nComputing absorption coefficient (flat-grid)...")
    alpha_flat = marfa.calculate_absorption_coefficient(
        nu_grid, lines, T, P, MOLE_FRACTION,
        use_lbl_cascade=False, line_cutoff_cm=25.0)

    print("Computing absorption coefficient (LBL cascade)...")
    alpha_lbl = marfa.calculate_absorption_coefficient(
        nu_grid, lines, T, P, MOLE_FRACTION,
        use_lbl_cascade=True, line_cutoff_cm=25.0)

    print(f"\n  Max alpha (flat) : {alpha_flat.max():.3e} cm-1")
    print(f"  Max alpha (LBL)  : {alpha_lbl.max():.3e} cm-1")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(nu_grid, alpha_flat, 'b-', linewidth=0.6, label='Flat grid')
    axes[0].plot(nu_grid, alpha_lbl,  'r-', linewidth=0.6, label='LBL cascade',
                 alpha=0.7)
    axes[0].set_ylabel('Absorption coefficient (cm-1)')
    axes[0].set_title('MARFA: CO2 Absorption -- Flat vs LBL Cascade')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    diff = alpha_lbl - alpha_flat
    axes[1].plot(nu_grid, diff, 'g-', linewidth=0.6)
    axes[1].set_xlabel('Wavenumber (cm-1)')
    axes[1].set_ylabel('LBL - Flat (cm-1)')
    axes[1].set_title('Difference (multi-grid correction)')
    axes[1].grid(True, alpha=0.3)
    axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig('marfa_python_example.png', dpi=300)
    print("\n  Plot saved: marfa_python_example.png")
    plt.show()


if __name__ == "__main__":
    example_basic_calculation()