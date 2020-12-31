#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/../')

import numpy as np

# Defaults:{{{1
BETA = 0.95
R = 1.048

def example_epsteinzin_singleiteration(crra = False):
    """
    Basic idea of example considered here:

    crra is True allows me to compare with CRRA case
    """
    
    rra = 2
    invies = 2

    # two states: Assets today, Income today
    endogstates_now = [1.0, 2.0]
    exogstates_now = [0.0]

    # need a decent number of endogenous future states - otherwise answers can be quite different
    endogstates_future = np.linspace(0.1, 1)

    # exogstates_future = [0.0, 0.2]
    # transmissionarray = np.array([[0.5, 0.5]])

    exogstates_future = [0.0]
    transmissionarray = np.array([[1]])

    # Vprime:
    Vprime = np.empty([len(endogstates_future), len(exogstates_future)])
    for endogstatei in range(len(endogstates_future)):
        a_p = endogstates_future[endogstatei]
        for exogstatei in range(len(exogstates_future)):
            income_p = exogstates_future[exogstatei]

            c_p = a_p + income_p

            if crra is True:
                Vprime[endogstatei, exogstatei] = c_p ** (1 - rra) / (1 - rra)
            else:
                Vprime[endogstatei, exogstatei] = (c_p ** (1 - invies)  ) ** (1/(1 - invies))
            
    if crra is True:
        def inputfunction(s1val, s2val, s1val_new):
            c = s1val + s2val - s1val_new
            return(c**(1-rra)/(1-rra))
    else:
        def inputfunction(BETA, Vfunc, nextperiodprobs, s1val, s2val, s1val_new):
            c = s1val + s2val - s1val_new
            V_overexog = Vfunc(s1val_new)

            sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
            from epsteinzin_func import vf_epsteinzin
            V = vf_epsteinzin(rra, invies, c, BETA, V_overexog, nextperiodprobs)

            return(V)

    def boundfunction(endogstate_now, exogstate):
        s1low = None
        # must save strictly less than all assets today
        s1high = endogstate_now * R + exogstate - 1e-4
        return(s1low, s1high)

    if crra is True:
        functiontype = 'reward'
    else:
        functiontype = 'value-full'

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vf_solveback_1endogstate_func import vf_1endogstate_continuous_oneiteration
    V, pol = vf_1endogstate_continuous_oneiteration(inputfunction, Vprime, endogstates_now, endogstates_future, exogstates_now, exogstates_future, transmissionarray, BETA, functiontype = functiontype, boundfunction = boundfunction)

    print('pol:')
    print(pol)
    print('V:')
    print(V)

# Run:{{{1
print('CRRA:')
example_epsteinzin_singleiteration(crra = True)
print('\n NOT CRRA:')
example_epsteinzin_singleiteration(crra = False)
