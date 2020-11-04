#!/usr/bin/env python3
"""
This function tests whether Epstein Zin yields the same as CRRA when the IES and RRA are the same.
Doesn't give exactly the same results (presumably due to differences in interpolation with powers) but the policy functions are close.
"""
# PYTHON_PREAMBLE_START_STANDARD:{{{

# Christopher David Cotton (c)
# http://www.cdcotton.com

# modules needed for preamble
import importlib
import os
from pathlib import Path
import sys

# Get full real filename
__fullrealfile__ = os.path.abspath(__file__)

# Function to get git directory containing this file
def getprojectdir(filename):
    curlevel = filename
    while curlevel is not '/':
        curlevel = os.path.dirname(curlevel)
        if os.path.exists(curlevel + '/.git/'):
            return(curlevel + '/')
    return(None)

# Directory of project
__projectdir__ = Path(getprojectdir(__fullrealfile__))

# Function to call functions from files by their absolute path.
# Imports modules if they've not already been imported
# First argument is filename, second is function name, third is dictionary containing loaded modules.
modulesdict = {}
def importattr(modulefilename, func, modulesdict = modulesdict):
    # get modulefilename as string to prevent problems in <= python3.5 with pathlib -> os
    modulefilename = str(modulefilename)
    # if function in this file
    if modulefilename == __fullrealfile__:
        return(eval(func))
    else:
        # add file to moduledict if not there already
        if modulefilename not in modulesdict:
            # check filename exists
            if not os.path.isfile(modulefilename):
                raise Exception('Module not exists: ' + modulefilename + '. Function: ' + func + '. Filename called from: ' + __fullrealfile__ + '.')
            # add directory to path
            sys.path.append(os.path.dirname(modulefilename))
            # actually add module to moduledict
            modulesdict[modulefilename] = importlib.import_module(''.join(os.path.basename(modulefilename).split('.')[: -1]))

        # get the actual function from the file and return it
        return(getattr(modulesdict[modulefilename], func))

# PYTHON_PREAMBLE_END:}}}

import numpy as np

# Defaults:{{{1
endogstatevec = np.exp(np.linspace(np.log(0.01), np.log(50), 50))
exogstatevec = [0.01, 0.1]
transmissionarray = np.array([[0.9, 0.1], [0.4, 0.6]])
BETA = 0.9
R = 1.098
rra = 2
invies = 2

def example_epsteinzin_singleiteration(crra = False):
    """
    Basic idea of example considered here:

    crra is True allows me to compare with CRRA case
    """


    def boundfunction(endogstate_now, exogstate):
        s1low = None
        # must save strictly less than all assets today
        s1high = endogstate_now * R + exogstate - 1e-4
        return(s1low, s1high)


    if crra is True:
        def inputfunction(betaEV, s1val, s2val, s1val_new):
            c = s1val * R + s2val - s1val_new
            return(c**(1-rra)/(1-rra) + betaEV(s1val_new))
    else:
        def inputfunction(BETA, Vfunc, nextperiodprobs, s1val, s2val, s1val_new):
            c = s1val * R + s2val - s1val_new
            V_overexog = Vfunc(s1val_new)

            V = importattr(__projectdir__ / Path('submodules/vfi-general/epsteinzin_func.py'), 'vf_epsteinzin')(rra, invies, c, BETA, V_overexog, nextperiodprobs)

            return(V)

    if crra is True:
        functiontype = 'value-betaEV'
    else:
        functiontype = 'value-full'

    V, pol = importattr(__projectdir__ / Path('submodules/vfi-general/vfi_1endogstate_func.py'), 'solvevfi_1endogstate_continuous')(inputfunction, endogstatevec, exogstatevec, transmissionarray, BETA, printinfo = True, boundfunction = boundfunction, functiontype = functiontype, interpmethod = 'numpy')

    return(V, pol)


def compare():
    V1, pol1 = example_epsteinzin_singleiteration(crra = False)
    V2, pol2 = example_epsteinzin_singleiteration(crra = True)

    if np.max(np.abs(pol1 - pol2)) > 1e-8:
        print(pol1)
        print('\n\n')
        print(pol2)
        raise ValueError('Different policy functions.')
    else:
        print('Same')


# Full:{{{1
compare()
