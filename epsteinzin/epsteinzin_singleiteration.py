#!/usr/bin/env python3
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
BETA = 0.95
R = 1.048

def example_epsteinzin_singleiteration(crra = False):
    """
    Basic idea of example considered here:

    crra is True allows me to compare with CRRA case
    """
    
    rra = 2
    invies = 2

    # two states: Assets today, Income today
    endogstates_now = [1.0, 2.0]
    exogstates_now = [0.0]

    # need a decent number of endogenous future states - otherwise answers can be quite different
    endogstates_future = np.linspace(0.1, 1)

    # exogstates_future = [0.0, 0.2]
    # transmissionarray = np.array([[0.5, 0.5]])

    exogstates_future = [0.0]
    transmissionarray = np.array([[1]])

    # Vprime:
    Vprime = np.empty([len(endogstates_future), len(exogstates_future)])
    for endogstatei in range(len(endogstates_future)):
        a_p = endogstates_future[endogstatei]
        for exogstatei in range(len(exogstates_future)):
            income_p = exogstates_future[exogstatei]

            c_p = a_p + income_p

            if crra is True:
                Vprime[endogstatei, exogstatei] = c_p ** (1 - rra) / (1 - rra)
            else:
                Vprime[endogstatei, exogstatei] = (c_p ** (1 - invies)  ) ** (1/(1 - invies))
            
    if crra is True:
        def inputfunction(s1val, s2val, s1val_new):
            c = s1val + s2val - s1val_new
            return(c**(1-rra)/(1-rra))
    else:
        def inputfunction(BETA, Vfunc, nextperiodprobs, s1val, s2val, s1val_new):
            c = s1val + s2val - s1val_new
            V_overexog = Vfunc(s1val_new)

            V = importattr(__projectdir__ / Path('submodules/vfi-general/epsteinzin_func.py'), 'vf_epsteinzin')(rra, invies, c, BETA, V_overexog, nextperiodprobs)

            return(V)

    def boundfunction(endogstate_now, exogstate):
        s1low = None
        # must save strictly less than all assets today
        s1high = endogstate_now * R + exogstate - 1e-4
        return(s1low, s1high)

    if crra is True:
        functiontype = 'reward'
    else:
        functiontype = 'value-full'

    V, pol = importattr(__projectdir__ / Path('submodules/vfi-general/vf_solveback_1endogstate_func.py'), 'vf_1endogstate_continuous_oneiteration')(inputfunction, Vprime, endogstates_now, endogstates_future, exogstates_now, exogstates_future, transmissionarray, BETA, functiontype = functiontype, boundfunction = boundfunction)

    print('pol:')
    print(pol)
    print('V:')
    print(V)

# Run:{{{1
print('CRRA:')
example_epsteinzin_singleiteration(crra = True)
print('\n NOT CRRA:')
example_epsteinzin_singleiteration(crra = False)
