#!/usr/bin/env python3
"""
Should be exactly the same since we're not doing any interpolation in discrete time.
"""
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/../')

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
                sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
                from epsteinzin_func import vf_epsteinzin
                V = vf_epsteinzin(rra, invies, c, BETAval, Vval, nextperiodprobs)
            else:
                V = -1e8
            return(V)

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vfi_1endogstate_func import solvevfi_1endogstate_discrete
    V, pol = solvevfi_1endogstate_discrete(rewardarray, transmissionarray, BETA, Vfunc = Vfunc, printinfo = True)


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
