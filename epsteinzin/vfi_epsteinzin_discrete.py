#!/usr/bin/env python3
"""
Should be exactly the same since we're not doing any interpolation in discrete time.
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
endogstatevec = np.exp(np.linspace(np.log(0.01), np.log(50), 500))
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
    ns1 = len(endogstatevec)
    ns2 = len(exogstatevec)

    rewardarray = np.empty([ns1, ns2, ns1])
    for s1 in range(ns1):
        for s2 in range(ns2):
            for s1prime in range(ns1):
                c = endogstatevec[s1] * R + exogstatevec[s2] - endogstatevec[s1prime]
                if crra is True:
                    # in the crra case we don't need an input function
                    # rewardarray is just u(c) like normal
                    if c > 0:
                        rewardarray[s1, s2, s1prime] = c ** (1 - rra) / (1 - rra)
                    else:
                        rewardarray[s1, s2, s1prime] = -1e8
                else:
                    # in this case we just set the rewardarray to be c
                    # we'll then input this c into the epstein zin value function later
                    rewardarray[s1, s2, s1prime] = c

    if crra is True:
        Vfunc = None
    else:
        def Vfunc(BETAval, Vval, nextperiodprobs, c):
            if c > 0:
                V = importattr(__projectdir__ / Path('submodules/vfi-general/epsteinzin_func.py'), 'vf_epsteinzin')(rra, invies, c, BETAval, Vval, nextperiodprobs)
            else:
                V = -1e8
            return(V)

    V, pol = importattr(__projectdir__ / Path('submodules/vfi-general/vfi_1endogstate_func.py'), 'solvevfi_1endogstate_discrete')(rewardarray, transmissionarray, BETA, Vfunc = Vfunc, printinfo = True)


    return(V, pol)


def compare():
    V1, pol1 = example_epsteinzin_singleiteration(crra = False)
    V2, pol2 = example_epsteinzin_singleiteration(crra = True)

    print(list(np.column_stack([pol1, pol2])))
    if np.max(np.abs(pol1 - pol2)) > 1e-8:
        raise ValueError('Different policy functions.')
    else:
        print('Same')


# Full:{{{1
compare()
