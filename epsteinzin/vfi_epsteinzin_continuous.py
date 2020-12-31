#!/usr/bin/env python3
"""
This function tests whether Epstein Zin yields the same as CRRA when the IES and RRA are the same.
Doesn't give exactly the same results (presumably due to differences in interpolation with powers) but the policy functions are close.
"""
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/../')

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

            sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
            from epsteinzin_func import vf_epsteinzin
            V = vf_epsteinzin(rra, invies, c, BETA, V_overexog, nextperiodprobs)

            return(V)

    if crra is True:
        functiontype = 'value-betaEV'
    else:
        functiontype = 'value-full'

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vfi_1endogstate_func import solvevfi_1endogstate_continuous
    V, pol = solvevfi_1endogstate_continuous(inputfunction, endogstatevec, exogstatevec, transmissionarray, BETA, printinfo = True, boundfunction = boundfunction, functiontype = functiontype, interpmethod = 'numpy')

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
