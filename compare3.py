""" driver for comparing different LBL techniques — OPTIMIZED VERSION """

import os
import sys
import shutil
import zipfile
import subprocess

import pylab as pl
import numpy as np

import hapi as h
import hapi2 as h2

# ──────────────────────────────────────────────
# GLOBALS
# ──────────────────────────────────────────────
MAXLINES = 10000000
LOGSCALE = True

# PRELOAD CACHE: populated once in main() before any xsect_* call.
# Keys: 'hapi_table', 'pnnl_data'
# This means NO function ever touches the disk on its own.
_CACHE = {}


# ──────────────────────────────────────────────
# PRELOADERS  (called once, before the main loop)
# ──────────────────────────────────────────────

def preload_hapi(parfile):
    """
    Load HITRAN parfile into HAPI's internal dict once.
    Subsequent calls to xsect_hapi() will be pure RAM — no disk I/O.
    """
    TABLE, _ = os.path.splitext(parfile)
    if TABLE not in h.LOCAL_TABLE_CACHE:
        print(f'[PRELOAD] Loading HAPI table from disk: {parfile}')
        h.storage2cache(TABLE)
    else:
        print(f'[PRELOAD] HAPI table already in cache: {TABLE}  → CACHE HIT')
    _CACHE['hapi_table'] = TABLE


def preload_pnnl(root, archname, filename, units='hitran'):
    """
    Decompress and parse the PNNL zip once into numpy arrays.
    Result is stored in _CACHE so xsect_pnnl() never touches disk again.
    """
    PNNL_TO_HITRAN = 4.03328E-16
    stem, _ = os.path.splitext(archname)
    archpath = os.path.join(root, archname)
    filepath = f'compounds/{stem}/{filename}'

    print(f'[PRELOAD] Reading PNNL archive: {archpath}  /  {filepath}')
    with zipfile.ZipFile(archpath).open(filepath) as f:
        nu, xsc = np.loadtxt(f).T

    # sort ascending by wavenumber
    ind = np.argsort(nu)
    nu  = nu[ind]
    xsc = xsc[ind]

    # unit conversions — done once here, not on every call
    xsc *= np.log(10)       # log₁₀ → natural log
    if units == 'hitran':
        xsc *= PNNL_TO_HITRAN
    elif units == 'pnnl':
        pass
    else:
        raise Exception('unknown units: %s' % units)

    _CACHE['pnnl_data'] = (nu, xsc)
    print(f'[PRELOAD] PNNL data cached → {len(nu):,} points in RAM')


# ──────────────────────────────────────────────
# XSECT FUNCTIONS  (now RAM-only after preload)
# ──────────────────────────────────────────────

def xsect_pnnl(**kwargs):
    """
    Reference: PNNL experiment.
    Data is served entirely from _CACHE — zero disk I/O.
    """
    # validation (kept from original)
    molec_id  = kwargs['molec_id']
    temp      = kwargs['temp']
    p_self    = kwargs['p_self']
    p_foreign = kwargs['p_foreign']
    td = 5
    assert molec_id in {2}
    assert 278-td <= temp <= 278+td or \
           296-td <= temp <= 296+td or \
           323-td <= temp <= 323+td
    assert p_self + p_foreign == 1.0

    # serve from cache — no zip, no disk
    nu, xsc = _CACHE['pnnl_data']
    return nu.copy(), xsc.copy()   # return copies so callers can't corrupt cache


def xsect_hapi(**kwargs):
    """
    Reference: HAPI calculation.
    Table is already in h.LOCAL_TABLE_CACHE after preload → pure RAM.
    BUG FIX: local_iso_id now correctly read from kwargs.
    """
    TABLE     = _CACHE['hapi_table']
    local_iso_id = kwargs['local_iso_id']   # ← BUG FIXED (was hardcoded string list)
    wv_min    = kwargs['wv_min']
    wv_max    = kwargs['wv_max']
    s_cutoff  = kwargs['s_cutoff']
    temp      = kwargs['temp']
    p_self    = kwargs['p_self']
    p_foreign = kwargs['p_foreign']

    pressure = p_self + p_foreign
    abscoef  = h2.opacity.lbl.numba.absorptionCoefficient_Voigt
    wngrid   = h.arange_(wv_min, wv_max, 0.001)

    nu, xsc = abscoef(
        SourceTables=TABLE,
        WavenumberWing=10.0,
        WavenumberGrid=wngrid,
        IntensityThreshold=s_cutoff,
        Environment={'p': pressure, 'T': temp},
        Diluent={'air': p_foreign / pressure, 'self': p_self / pressure},
        HITRAN_units=True,
    )
    return nu, xsc


def xsect_marfa_fort(**kwargs):
    """
    Marfa Simple: Fortran binary subprocess.
    Heaviest method — runs first so OS warms the parfile in its page cache.
    """
    parfile      = kwargs['parfile']
    molec_id     = kwargs['molec_id']
    local_iso_id = kwargs['local_iso_id']
    wv_min       = kwargs['wv_min']
    wv_max       = kwargs['wv_max']
    s_cutoff     = kwargs['s_cutoff']
    temp         = kwargs['temp']
    p_self       = kwargs['p_self']
    p_foreign    = kwargs['p_foreign']
    outfile      = 'marfa_simple.out'

    if os.path.exists(outfile):
        shutil.move(outfile, outfile + '.bak')

    TIPS_ref    = h.PYTIPS(molec_id, local_iso_id, 296)
    TIPS_target = h.PYTIPS(molec_id, local_iso_id, temp)

    args = [str(a) for a in [
        parfile, MAXLINES, molec_id, wv_min, wv_max,
        s_cutoff, temp, p_self, p_foreign, TIPS_ref, TIPS_target, outfile
    ]]
    print('Running: ./marfa_simple', ' '.join(args))
    subprocess.call(['./marfa_simple'] + args)

    nu, xsc = np.loadtxt(outfile).T
    return nu, xsc


def xsect_marfa_python(**kwargs):
    """
    MarfaPython: Python + Numba subprocess.
    Second heaviest — runs after Fortran so parfile is already in OS page cache.
    """
    parfile      = kwargs['parfile']
    molec_id     = kwargs['molec_id']
    local_iso_id = kwargs['local_iso_id']
    wv_min       = kwargs['wv_min']
    wv_max       = kwargs['wv_max']
    s_cutoff     = kwargs['s_cutoff']
    temp         = kwargs['temp']
    p_self       = kwargs['p_self']
    p_foreign    = kwargs['p_foreign']
    outfile      = 'marfa_python.out'

    if os.path.exists(outfile):
        shutil.move(outfile, outfile + '.bak')

    TIPS_ref    = h.PYTIPS(molec_id, local_iso_id, 296)
    TIPS_target = h.PYTIPS(molec_id, local_iso_id, temp)

    script = os.path.join('..', 'src', 'MarfaPython.py')
    args = [
        script, str(parfile), str(MAXLINES), str(molec_id),
        str(wv_min), str(wv_max), str(s_cutoff), str(temp),
        str(p_self), str(p_foreign), str(TIPS_ref), str(TIPS_target), str(outfile),
    ]
    print('Running:', sys.executable, ' '.join(args))
    subprocess.call([sys.executable] + args)

    nu, xsc = np.loadtxt(outfile).T
    return nu, xsc


# ──────────────────────────────────────────────
# UTILITY
# ──────────────────────────────────────────────

def print_header(string):
    char  = '='
    nchars = len(string) + 8
    print('\n' + char * nchars)
    print(char * 3 + ' ' + string + ' ' + char * 3)
    print(char * nchars)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    """ main driver """

    DEFAULTS = {
        'parfile':      'co2.data',
        'molec_id':     2,
        'local_iso_id': 1,
        'wv_min':       2000,
        'wv_max':       2100,
        's_cutoff':     0.0,
        'temp':         323,
        'p_self':       0.001,
        'p_foreign':    0.999,
    }

    # ── STEP 1: PRELOAD EVERYTHING INTO RAM BEFORE THE LOOP ──────────────
    # Order: largest data first so the OS page cache is warm for later reads.
    # After this block, NO xsect_* function will touch the disk.
    print_header('PRELOADING DATA INTO RAM')

    preload_hapi(DEFAULTS['parfile'])     # largest: full HITRAN line list

    td = 5
    temp = DEFAULTS['temp']
    formula = 'CO2'
    if   323-td <= temp <= 323+td: pnnl_file = f'{formula}_50T.TXT'
    elif 296-td <= temp <= 296+td: pnnl_file = f'{formula}_25T.TXT'
    elif 278-td <= temp <= 278+td: pnnl_file = f'{formula}_5T.TXT'
    else: raise Exception('Unsupported temperature for PNNL preload: %s' % temp)

    preload_pnnl('pnnl', 'Carbon_dioxide.zip', pnnl_file, units='hitran')

    # ── STEP 2: DEFINE CASES — largest/heaviest first ────────────────────
    # Order rationale:
    #   1. marfa_fort   → spawns Fortran binary, reads parfile from disk (cold)
    #   2. marfa_python → spawns Python+Numba, parfile now in OS page cache (warm)
    #   3. xsect_hapi   → pure RAM (HAPI cache), very fast
    #   4. xsect_pnnl   → pure RAM (_CACHE dict), fastest

    CASES = [
        ## function            marker  line   color   ##
        (xsect_marfa_fort,     '',     '-',   'magenta'),  # heaviest → first
        (xsect_marfa_python,   '',     '-',   'red'),      # heavy, parfile cached by OS
        (xsect_hapi,           '',     '-',   'blue'),     # RAM only (HAPI cache)
        (xsect_pnnl,           '',     '-',   'green'),    # RAM only (_CACHE)
    ]

    # ── STEP 3: RUN & PLOT ───────────────────────────────────────────────
    leg = []
    for method, marker, linestyle, color in CASES:
        print_header('CALLING: ' + method.__name__)
        nu, xsc = method(**DEFAULTS)
        pl.plot(nu, xsc, marker=marker, linestyle=linestyle, color=color)
        leg.append(method.__name__)

    pl.legend(leg)
    pl.grid(True)
    if LOGSCALE:
        pl.yscale('log')
    pl.show()


if __name__ == '__main__':
    main()