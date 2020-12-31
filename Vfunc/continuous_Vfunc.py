#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/../')

import numpy as np

# Defaults:{{{1
endogstatevec = np.exp(np.linspace(np.log(0.01), np.log(50), 50))
exogstatevec = [0.01, 0.1]
transmissionarray = np.array([[0.9, 0.1], [0.4, 0.6]])
BETA = 0.95
R = 1.048

def vfull(functiontype):
    """
    Verifying that when I specify, functiontype == 'reward/value-betaEV/value-full' that we get the correct outcomes
    Do this by verifying that these yield the same results when they are specified to be the same
    """

    def rewardfunction(endogstate_now, exogstate, endogstate_future):
        C = endogstate_now * R + exogstate - endogstate_future
        return(np.log(C))

    if functiontype == 'reward':
        def inputfunction(endogstate_now, exogstate, endogstate_future):
            return(rewardfunction(endogstate_now, exogstate, endogstate_future))

    elif functiontype == 'value-betaEV':
        def inputfunction(betaEVfunc, endogstate_now, exogstate, endogstate_future):
            return(rewardfunction(endogstate_now, exogstate, endogstate_future) + betaEVfunc(endogstate_future))
    elif functiontype == 'value-full':
        def inputfunction(betaval, Vfunc, nextperiodprobs, endogstate_now, exogstate, endogstate_future):
            return(rewardfunction(endogstate_now, exogstate, endogstate_future) + BETA * Vfunc(endogstate_future).dot(nextperiodprobs))
    else:
        raise ValueError('functiontype incorrect')

    def boundfunction(endogstate_now, exogstate):
        s1low = None
        # must save strictly less than all assets today
        s1high = endogstate_now * R + exogstate - 1e-4
        return(s1low, s1high)
        

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vfi_1endogstate_func import solvevfi_1endogstate_continuous
    V, pol = solvevfi_1endogstate_continuous(inputfunction, endogstatevec, exogstatevec, transmissionarray, BETA, printinfo = True, boundfunction = boundfunction, functiontype = functiontype)

    return(V, pol)


def compare():
    V1, pol1 = vfull('reward')
    V2, pol2 = vfull('value-betaEV')
    V3, pol3 = vfull('value-full')

    if np.max(np.abs(V1 - V2)) > 1e-5 or np.max(np.abs(V1 - V3)) > 1e-5:
        print(V1)
        print(V2)
        print(V3)
        print(V1 - V2)
        print(V1 - V3)
        raise ValueError('Yield different results.')
    else:
        print('Same')

# Full:{{{1
compare()
