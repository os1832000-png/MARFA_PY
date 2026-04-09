""" driver for comparing different LBL techniques """

import os
import sys
import json
import shutil
import zipfile 
import subprocess

import matplotlib as mpl
#mpl.use('Agg')
import pylab as pl
import numpy as np

import hapi as h
import hapi2 as h2

#from scipy.interpolate import Akima1DInterpolator

MAXLINES = 10000000
LOGSCALE = True

def extract_pnnl_from_zip(root,archname,filename,units='hitran'):
    """ 
    Extract PNNL archive. Parameters:
       root: root folder for PNNL archives
       archname: name of archive file (e.g. Carbon_dioxide.zip)
       filename: name of spectra file (e.g. CO2_25T.TXT)
    """
    PNNL_TO_HITRAN = 4.03328E-16
    stem,_ = os.path.splitext(archname)
    archpath = os.path.join(root,archname)
    filepath = f'compounds/{stem}/{filename}'
    print('Opening archive:',archpath)
    print('Reading file:',filepath)
    with zipfile.ZipFile(archpath).open(filepath) as f:
        nu,xsc = np.loadtxt(f).T
        #data = sort_2d_array_by_rows(data)
        ind = np.argsort(nu) # sort in ascendig order by nu
        nu = nu[ind]
        xsc = xsc[ind]
        xsc *= np.log(10) # convert from 10-based to naperian
    if units=='hitran':
        xsc *= PNNL_TO_HITRAN
    elif units=='pnnl':
        pass
    else:
        raise Exception('unknown units: %s'%units)
    return nu,xsc

def xsect_pnnl(**kwargs):
    """
    Reference: PNNL experiment
    """
    molec_id = kwargs['molec_id']
    temp = kwargs['temp']
    p_self = kwargs['p_self']
    p_foreign = kwargs['p_foreign']
    
    td = 5# temperature delta
    
    # check validity of arguments
    assert molec_id in {2}
    assert 278-td<=temp<=278+td or 296-td<=temp<=296+td or 323-td<=temp<=323+td
    assert p_self+p_foreign == 1.0
    
    # deduce PNNL path
    root = 'pnnl'
    if molec_id==2:
        archname = 'Carbon_dioxide.zip'; formula = 'CO2'
    else:
        raise Exception('Not supported molecule ID:',molec_id)
    
    # deduce PNNL filenames
    if 278-td<=temp<=278+td:
        filename = f'{formula}_5T.TXT'
    elif 296-td<=temp<=296+td:
        filename = f'{formula}_25T.TXT'
    elif 323-td<=temp<=323+td:
        filename = f'{formula}_50T.TXT'
    else:
        raise Exception('Not supported temperature:',temp)
    
    nu,xsc = extract_pnnl_from_zip(root,archname,filename,units='hitran')
    
    return nu,xsc

def xsect_hapi(**kwargs):    
    """
    Reference: HAPI calculation
    """
    parfile = kwargs['parfile']
    #maxlines = kwargs['maxlines']
    maxlines = MAXLINES
    molec_id = kwargs['molec_id']
    local_iso_id = ['local_iso_id']
    wv_min = kwargs['wv_min']
    wv_max = kwargs['wv_max']
    s_cutoff = kwargs['s_cutoff']
    temp = kwargs['temp']
    p_self = kwargs['p_self']
    p_foreign = kwargs['p_foreign']
    #TIPS_ref = kwargs['TIPS_ref']
    #TIPS_target = kwargs['TIPS_target']
    #outfile = kwargs['outfile']
    
    HAPI_DEBUG = None # don't do debugging
    #HAPI_DEBUG = [] # do debugging, record result into HAPI_DEBUG list
    
    TABLE,_ = os.path.splitext(parfile)
    h.storage2cache(TABLE)
    
    pressure = p_self+p_foreign
    
    #abscoef = h.absorptionCoefficient_Voigt
    abscoef = h2.opacity.lbl.numba.absorptionCoefficient_Voigt
    
    wngrid = h.arange_(wv_min,wv_max,0.001)
    
    nu,xsc = abscoef(
        #Components=[(molec_id,local_iso_id),],
        SourceTables=TABLE,
        #WavenumberRange=[wv_min,wv_max],
        #WavenumberStep=0.001,
        WavenumberWing=10.0,
        WavenumberGrid=wngrid,
        IntensityThreshold=s_cutoff,
        Environment={'p':pressure,'T':temp},
        Diluent={'air':p_foreign/pressure,'self':p_self/pressure},
        #DEBUG=HAPI_DEBUG,
        HITRAN_units=True,
    )
    
    return nu,xsc

def xsect_marfa_fort(**kwargs):
    """
    Marfa Simple: simplified version of original Marfa Fortran code
    """
    parfile = kwargs['parfile']
    #maxlines = kwargs['maxlines']
    maxlines = MAXLINES
    molec_id = kwargs['molec_id']
    local_iso_id = kwargs['local_iso_id']
    wv_min = kwargs['wv_min']
    wv_max = kwargs['wv_max']
    s_cutoff = kwargs['s_cutoff']
    temp = kwargs['temp']
    p_self = kwargs['p_self']
    p_foreign = kwargs['p_foreign']
    #TIPS_ref = kwargs['TIPS_ref']
    #TIPS_target = kwargs['TIPS_target']
    #outfile = kwargs['outfile']
    outfile = 'marfa_simple.out'
    
    if os.path.exists(outfile):
        print('Backing up file',outfile)
        shutil.move(outfile,outfile+'.bak')
    
    TIPS_ref = h.PYTIPS(molec_id,local_iso_id,296)
    TIPS_target = h.PYTIPS(molec_id,local_iso_id,temp)

    #$ ./marfa_simple
    #Usage: marfa_simple <parfile> <maxlines> <molec_id> <wv_min> <wv_max> <cutoff> <temp> <p_self> <p_foreign> <TIPS_ref> <TIPS_target> <outfile>
    command = './marfa_simple'
    args = [str(arg) for arg in [parfile,maxlines,molec_id,wv_min,wv_max,s_cutoff,temp,p_self,p_foreign,TIPS_ref,TIPS_target,outfile]]
    print('Running: %s %s'%(command,' '.join(args)))
    subprocess.call([command]+args)
    nu,xsc = np.loadtxt(outfile).T
    
    return nu,xsc

def xsect_marfa_python(**kwargs):
    """
    MarfaPython: simplified Python + Numba 3-grid implementation.
    """
    parfile = kwargs['parfile']
    maxlines = MAXLINES
    molec_id = kwargs['molec_id']
    local_iso_id = kwargs['local_iso_id']
    wv_min = kwargs['wv_min']
    wv_max = kwargs['wv_max']
    s_cutoff = kwargs['s_cutoff']
    temp = kwargs['temp']
    p_self = kwargs['p_self']
    p_foreign = kwargs['p_foreign']
    outfile = 'marfa_python.out'

    if os.path.exists(outfile):
        print('Backing up file',outfile)
        shutil.move(outfile,outfile+'.bak')

    TIPS_ref = h.PYTIPS(molec_id,local_iso_id,296)
    TIPS_target = h.PYTIPS(molec_id,local_iso_id,temp)

    command = sys.executable
    script = os.path.join('..','src','MarfaPython.py')
    args = [
        script, str(parfile), str(maxlines), str(molec_id), str(wv_min), str(wv_max),
        str(s_cutoff), str(temp), str(p_self), str(p_foreign), str(TIPS_ref), str(TIPS_target), str(outfile),
    ]
    print('Running: %s %s'%(command,' '.join(args)))
    subprocess.call([command]+args)
    nu,xsc = np.loadtxt(outfile).T
    return nu,xsc

def print_header(string): # header pretty print
    char = '='
    nchars = len(string)+8
    print('')
    print(char*nchars)
    print(char*3+' '+string+' '+char*3)
    print(char*nchars)

def main():
    """ main driver """
    
    DEFAULTS = {
        'parfile': 'co2.data',
        'molec_id': 2,
        'local_iso_id': 1,
        'wv_min': 2000,
        'wv_max': 2100,
        's_cutoff': 0.0,
        'temp': 323,
        'p_self': 0.001,
        'p_foreign': 0.999,
    }
    
    CASES = [
        ## function         marker     line     color ##
        (xsect_marfa_python, '',     '-',    'red'), #Exchanged it with the 4th case
        (xsect_hapi,           '',     '-',    'blue' ),
        #(xsect_marfa_fort,   '',     '-',    'magenta'), 
        (xsect_pnnl,           '',     '-',   'green'),
    ]

    leg = []
    for method,marker,linestyle,color in CASES:
        print_header('CALLING: '+method.__name__)
        nu,xsc = method(**DEFAULTS)
        pl.plot(nu,xsc,marker=marker,linestyle=linestyle,color=color)
        leg.append(method.__name__)
        
    pl.legend(leg)
    pl.grid(True)
    
    if LOGSCALE: pl.yscale('log')
    
    pl.show()
    
if __name__=='__main__':
    main()
