#!/usr/bin/env python3
"""
Solve for a consumption-savings problem over a discrete state space
Solve for the stationary distribution using two different methods: transmissionstar, separatedstates

Need careful choice of endogstatevec:
- Need decent number of states for this to work well. ns1 = 200 gives quite different answer to ns1 = 2000
- Need to use log scale rather than linear scale to emphasize the points close to the bound where there will be large differences in utility (if use linear scale get completely different results even with ns1 = 2000)
"""
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/../')

import numpy as np
import pandas as pd

sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
from vfi_1endogstate_func import *

# Defaults:{{{1
ns1 = 2000
endogstatevec = np.exp(np.linspace(np.log(0.01), np.log(100), ns1))
exogstatevec = [0.5, 1.5]
transmissionarray = np.array([[0.9, 0.1], [0.1, 0.9]])

# Parameters:{{{1
BETA = 0.8
A = 1
ALPHA = 0.3
DELTA = 0.1

# get Lbar
sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/')))
from markov_func import getstationarydist
Ldist = getstationarydist(transmissionarray)
Lmean = np.sum(Ldist * exogstatevec)

def getKd(R):
    K = (A / (R - 1 + DELTA)) ** (1 / (1-ALPHA)) * Lmean
    return(K)


def getKs(R):
    """
    This returns the aggregate 
    """

    W = A * (A / (R - 1 + DELTA)) ** (ALPHA / (1-ALPHA))

    ns1 = len(endogstatevec)
    ns2 = len(exogstatevec)

    rewardarray = np.empty([ns1, ns2, ns1])
    for s1 in range(ns1):
        for s2 in range(ns2):
            for s1prime in range(ns1):
                
                C = endogstatevec[s1] * R + W * exogstatevec[s2] - endogstatevec[s1prime]

                if C > 0:
                    rewardarray[s1, s2, s1prime] = np.log(C)
                else:
                    rewardarray[s1, s2, s1prime] = -1e8

    # set relatively imprecise criterion for speed
    V, pol = solvevfi_1endogstate_discrete(rewardarray, transmissionarray, beta = BETA, printinfo = False, crit = 1e-3)
    # print(list(V))
    # print(list(pol))

    # Solving for transmission array via quick method
    polprobs = getpolprobs_1endogstate_discrete(pol)
    fullstatedist, endogstatedist = getstationarydist_1endogstate_direct(transmissionarray, polprobs)
    meanK = np.sum(endogstatedist * endogstatevec)

    return(meanK)


def getsolution():
    """
    The steady state is about R = 1.233 and K = 4.8
    """
    Rval = np.linspace(1 / BETA - 0.05, 1 / BETA - 0.005, 10)
    Kdlist = []
    Kslist = []
    for R in Rval:
        Kdlist.append(getKd(R))
        Kslist.append(getKs(R))

    df = pd.DataFrame({'R': Rval, 'Kd': Kdlist, 'Ks': Kslist})
    print('Solution where Kd and Ks intersect:')
    # not calculating precisely in the interest of time
    print(df)

# Run:{{{1
if __name__ == "__main__":
    getsolution()

