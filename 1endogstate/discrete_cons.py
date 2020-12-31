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

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vfi_1endogstate_func import solvevfi_1endogstate_discrete
    V, pol = solvevfi_1endogstate_discrete(rewardarray, transmissionarray, beta = BETA, printinfo = True)
    print(list(V))
    print(list(pol))

    # Solving for transmission array via transmissionstar array
    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vfi_1endogstate_func import gentransmissionstararray_1endogstate_discrete
    transmissionstararray = gentransmissionstararray_1endogstate_discrete(transmissionarray, pol)
    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vfi_1endogstate_func import getstationarydist_1endogstate_full
    fullstatedist, endogstatedist = getstationarydist_1endogstate_full(transmissionstararray, ns1)
    print('Mean Savings via Full Transmission Array')
    print(list(endogstatedist))
    print(np.sum(endogstatedist * endogstatevec))

    # Solving for transmission array via quick method
    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vfi_1endogstate_func import getpolprobs_1endogstate_discrete
    polprobs = getpolprobs_1endogstate_discrete(pol)
    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vfi_1endogstate_func import getstationarydist_1endogstate_direct
    fullstatedist, endogstatedist = getstationarydist_1endogstate_direct(transmissionarray, polprobs)
    print('Mean Savings via Quick Method')
    print(list(endogstatedist))
    print(np.sum(endogstatedist * endogstatevec))



# Run:{{{1
full()
