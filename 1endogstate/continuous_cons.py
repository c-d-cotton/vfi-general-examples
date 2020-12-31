#!/usr/bin/env python3
"""
Computes VFI for consumption-savings problem both with and without specifying bounds (specifying bounds is easier)
Next, compute the stationary probability distribution using two methods: 1. by computing the full transmission array 2. a quicker method where we avoid computing a ns1*ns2 x ns1*ns2 matrix

Need careful choice of endogstatevec:
- Need log scale rather than linear scale since points with a close to zero have large nonlinear changes in utility (meaning the interpolation doesn't work well with large gaps) unlike points where agents have large amounts of assets
"""
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/../')

import numpy as np

# Defaults:{{{1
endogstatevec = np.exp(np.linspace(np.log(0.01), np.log(100), 100))
exogstatevec = [0.01, 0.1]
transmissionarray = np.array([[0.9, 0.1], [0.4, 0.6]])
BETA = 0.95
R = 1.048

def vfi_nobound():
    """
    The boundfunction case works better.

    To get this to work, I needed to set that when C < 0, I return a very negative number which increases in C and which does not depend upon the value of beta* EV (otherwise it can be utility improving during the iterations to lower C when C is already < 0).
    Therefore, in the valuefunction I use betaEV.
    """

    def valuefunction(betaEVfunc, endogstate_now, exogstate, endogstate_future):
        C = endogstate_now * R + exogstate - endogstate_future
        if C > 0:
            return(np.log(C) + betaEVfunc(endogstate_future))
        else:
            # if do hit negative consumption, want to encourage fminbound to raise consumption
            # NEGATIVE UTILITY:
            # so return very negative utility
            # return number need to be quite large since then won't accidentally maximise on C<0
            # but do not set this negative return value too large otherwise fmindbound will think we've converged at the very large negative number
            # -1e10 seems to work well
            # POSITIVE CONSUMPTION:
            # Also if consumption is too low to get positive consumption, need to raise utility by raising consumption
            # This ensures that don't get stuck at the very negative number
            # Need to ensure raise a lot by consumption - otherwise can have situation where betaEV is higher for a lower level of consumption and then get stuck
            # BETTER METHOD:
            # To avoid issues with consumption, just input the utility function in the value function and then return very negative value (rather than utility for single period) when C < 0 and can raise by only small amount C
            return(-1e10 + 1000 * C)

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vfi_1endogstate_func import solvevfi_1endogstate_continuous
    V, pol = solvevfi_1endogstate_continuous(valuefunction, endogstatevec, exogstatevec, transmissionarray, BETA, printinfo = True, s1low = 0, functiontype = 'value-betaEV')

    return(V, pol)


def vfi_bound():
    """
    Use boundfunction to prevent C < 0.
    """

    def rewardfunction(endogstate_now, exogstate, endogstate_future):
        C = endogstate_now * R + exogstate - endogstate_future
        return(np.log(C))
    # can use inputfunction == 'value-betaEV and then input this valuefunction as the inputfunction but easier to just use the rewardfunction
    # def valuefunction(betaEVfunc, endogstate_now, exogstate, endogstate_future):
    #     C = endogstate_now * R + exogstate - endogstate_future
    #     return(np.log(C) + betaEVfunc(endogstate_future))

    def boundfunction(endogstate_now, exogstate):
        s1low = None
        # must save strictly less than all assets today
        s1high = endogstate_now * R + exogstate - 1e-4
        return(s1low, s1high)
        

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vfi_1endogstate_func import solvevfi_1endogstate_continuous
    V, pol = solvevfi_1endogstate_continuous(rewardfunction, endogstatevec, exogstatevec, transmissionarray, BETA, printinfo = True, boundfunction = boundfunction, functiontype = 'reward')

    return(V, pol)


def full(nobound = False):
    print('\nbounded method')
    V, pol = vfi_bound()    
    print(endogstatevec)
    print(V)
    print(pol)

    if nobound is True:
        print('\nno bound method')
        V_nobound, pol_nobound = vfi_nobound()    
        print(V_nobound)
        print(pol_nobound)

        print('\nLargest difference between vfi methods')
        print(np.max(np.abs(V - V_nobound)))
        print(np.max(np.abs(pol - pol_nobound)))

    # print('\nPolicy Probs (necessary for both transmission array methods')
    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vfi_1endogstate_func import getpolprobs_1endogstate_continuous
    polprobs = getpolprobs_1endogstate_continuous(pol, endogstatevec)

    print('\nFull Transmission Array Mean Savings')
    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vfi_1endogstate_func import gentransmissionstararray_1endogstate_polprobs
    transmissionstararray = gentransmissionstararray_1endogstate_polprobs(transmissionarray, polprobs)
    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vfi_1endogstate_func import getstationarydist_1endogstate_full
    fullstatedist_standard, endogstatedist_standard = getstationarydist_1endogstate_full(transmissionstararray, len(endogstatevec))
    meansavings = np.sum(endogstatedist_standard * endogstatevec)
    print(endogstatedist_standard)
    print(meansavings)

    print('\nQuick Transmission Array Mean Savings')
    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vfi_1endogstate_func import getstationarydist_1endogstate_direct
    fullstatedist, endogstatedist = getstationarydist_1endogstate_direct(transmissionarray, polprobs)
    meansavings = np.sum(endogstatedist * endogstatevec)
    print(endogstatedist)
    print(meansavings)

    print('\nDifference Distributions')
    print(np.max(np.abs(endogstatedist_standard - endogstatedist)))


# Run:{{{1
full()
