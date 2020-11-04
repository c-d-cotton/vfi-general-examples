#!/usr/bin/env python3
"""
Solve for a consumption-savings problem over a discrete state space
Solve for the stationary distribution using two different methods: transmissionstar, separatedstates

Need careful choice of endogstatevec:
- Need decent number of states for this to work well. ns1 = 200 gives quite different answer to ns1 = 2000
- Need to use log scale rather than linear scale to emphasize the points close to the bound where there will be large differences in utility (if use linear scale get completely different results even with ns1 = 2000)
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
ns1 = 2000
endogstatevec = np.exp(np.linspace(np.log(0.01), np.log(100), ns1))
exogstatevec = [0.01, 0.1]
transmissionarray = np.array([[0.9, 0.1], [0.4, 0.6]])
BETA = 0.95
R = 1.048

def full():
    """
    Compute VFI for consumption-savings problem.
    """
    ns1 = len(endogstatevec)
    ns2 = len(exogstatevec)

    rewardarray = np.empty([ns1, ns2, ns1])
    for s1 in range(ns1):
        for s2 in range(ns2):
            for s1prime in range(ns1):
                
                C = endogstatevec[s1] * R + exogstatevec[s2] - endogstatevec[s1prime]

                if C > 0:
                    rewardarray[s1, s2, s1prime] = np.log(C)
                else:
                    rewardarray[s1, s2, s1prime] = -1e8

    V, pol = importattr(__projectdir__ / Path('submodules/vfi-general/vfi_1endogstate_func.py'), 'solvevfi_1endogstate_discrete')(rewardarray, transmissionarray, beta = BETA, printinfo = True)
    print(list(V))
    print(list(pol))

    # Solving for transmission array via transmissionstar array
    transmissionstararray = importattr(__projectdir__ / Path('submodules/vfi-general/vfi_1endogstate_func.py'), 'gentransmissionstararray_1endogstate_discrete')(transmissionarray, pol)
    fullstatedist, endogstatedist = importattr(__projectdir__ / Path('submodules/vfi-general/vfi_1endogstate_func.py'), 'getstationarydist_1endogstate_full')(transmissionstararray, ns1)
    print('Mean Savings via Full Transmission Array')
    print(list(endogstatedist))
    print(np.sum(endogstatedist * endogstatevec))

    # Solving for transmission array via quick method
    polprobs = importattr(__projectdir__ / Path('submodules/vfi-general/vfi_1endogstate_func.py'), 'getpolprobs_1endogstate_discrete')(pol)
    fullstatedist, endogstatedist = importattr(__projectdir__ / Path('submodules/vfi-general/vfi_1endogstate_func.py'), 'getstationarydist_1endogstate_direct')(transmissionarray, polprobs)
    print('Mean Savings via Quick Method')
    print(list(endogstatedist))
    print(np.sum(endogstatedist * endogstatevec))



# Run:{{{1
full()
