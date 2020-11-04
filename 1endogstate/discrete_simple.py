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

def oneendogstate_discrete_example():
    # set x = 2 for probability to be [50, 50]
    # set x = 5 for probability to be [0.33, 0.67]
    x = 5

    rewardarray = np.empty([2,2,2])
    # last period holiday, sunny today
    rewardarray[0,0] = [0,1]
    # last period holiday, rainy today
    rewardarray[0,1] = [0,1]
    # last period work, sunny today
    rewardarray[1,0] = [x,0]
    # last period work, rainy today
    rewardarray[1,1] = [0.5,0]

    ns1 = np.shape(rewardarray)[0]
    ns2 = np.shape(rewardarray)[1]

    transmissionarray = np.array([[0.6,0.4], [0.45,0.55]])

    V, pol = importattr(__projectdir__ / Path('submodules/vfi-general/vfi_1endogstate_func.py'), 'solvevfi_1endogstate_discrete')(rewardarray, transmissionarray, beta = 0.95, printinfo = False)

    print('policy function')
    print(pol)
    print('value function')
    print(V)

    transmissionstararray = importattr(__projectdir__ / Path('submodules/vfi-general/vfi_1endogstate_func.py'), 'gentransmissionstararray_1endogstate_discrete')(transmissionarray, pol)
    print('transmission star array')
    print(transmissionstararray)

    fullstatedist, endogstatedist = importattr(__projectdir__ / Path('submodules/vfi-general/vfi_1endogstate_func.py'), 'getstationarydist_1endogstate_full')(transmissionstararray, ns1)
    print('probs of holiday and work')
    print(endogstatedist)

oneendogstate_discrete_example()
