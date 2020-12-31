#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/../')

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

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vfi_1endogstate_func import solvevfi_1endogstate_discrete
    V, pol = solvevfi_1endogstate_discrete(rewardarray, transmissionarray, beta = 0.95, printinfo = False)

    print('policy function')
    print(pol)
    print('value function')
    print(V)

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vfi_1endogstate_func import gentransmissionstararray_1endogstate_discrete
    transmissionstararray = gentransmissionstararray_1endogstate_discrete(transmissionarray, pol)
    print('transmission star array')
    print(transmissionstararray)

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vfi_1endogstate_func import getstationarydist_1endogstate_full
    fullstatedist, endogstatedist = getstationarydist_1endogstate_full(transmissionstararray, ns1)
    print('probs of holiday and work')
    print(endogstatedist)

oneendogstate_discrete_example()
