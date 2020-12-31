#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/../')

import numpy as np

# Defaults:{{{1
R = 1.048
BETA = 0.95

# Continuous:{{{1
def twoperiod_continuous_example():
    """
    Standard consumption allocation problem with zero return on assets.
    endogstate is a and a'
    exogstate is labor earnings which varies today but not tomorrow
    """
    def u(C):
        if C > 0:
            return(np.log(C))
        else:
            return(-1e90)

    def rewardfunction(endogstate_now, laborincome, endogstate_future):
        C = endogstate_now * R + laborincome - endogstate_future
        return(u(C))

    def boundfunction(endogstate_now, exogstate):
        s1low = None
        # must save strictly less than all assets today
        s1high = endogstate_now * R + exogstate - 1e-4
        return(s1low, s1high)

    
    endogstate_now = [0]
    endogstate_future = np.exp(np.linspace(np.log(1e-4), np.log(10)))
    exogstate_now = [0.1, 1]
    exogstate_future = [0]
    transmissionarray = np.array([[1], [1]])

    def lastperiodutility(endogval, exogval):
        return(u(endogval + exogval))

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vf_solveback_1endogstate_func import Vprime_get
    Vprime = Vprime_get(lastperiodutility, endogstate_future, exogstate_future)

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vf_solveback_1endogstate_func import vf_1endogstate_continuous_oneiteration
    V, pol = vf_1endogstate_continuous_oneiteration(rewardfunction, Vprime, endogstate_now, endogstate_future, exogstate_now, exogstate_future, transmissionarray, BETA, basicchecks = True, boundfunction = boundfunction)

    print(V)
    print(pol)


def threeperiod_continuous_example():
    """
    Standard consumption allocation problem with zero return on assets.
    endogstate is a and a'
    exogstate is labor earnings which varies today but not tomorrow
    """
    def u(C):
        if C > 0:
            return(np.log(C))
        else:
            return(-1e90)

    def rewardfunction(assets, laborincome, assetsprime):
        C = assets + laborincome - assetsprime
        return(u(C))

    def boundfunction(endogstate_now, exogstate):
        s1low = None
        # must save strictly less than all assets today
        s1high = endogstate_now * R + exogstate - 1e-4
        return(s1low, s1high)

    
    endogstate_start = [0]
    endogstate_middleend = np.exp(np.linspace(np.log(1e-4), np.log(10)))
    exogstate_startmiddle = [0.1, 1]
    exogstate_end = [0]
    transmissionarray_end = np.array([[1], [1]])
    transmissionarray_start = np.array([[0.5, 0.5], [0.1, 0.9]])

    inputfunction_list = [rewardfunction, rewardfunction]
    endogstate_list = [endogstate_start, endogstate_middleend, endogstate_middleend]
    exogstate_list = [exogstate_startmiddle, exogstate_startmiddle, exogstate_end]
    boundfunction_list = [boundfunction, boundfunction]
    transmissionarray_list = [transmissionarray_start, transmissionarray_end]
    beta_list = [BETA] * 2

    def lastperiodutility(endogval, exogval):
        return(u(endogval + exogval))

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vf_solveback_1endogstate_func import vf_solveback_continuous
    Vlist, pollist = vf_solveback_continuous(inputfunction_list, lastperiodutility, endogstate_list, exogstate_list, transmissionarray_list, beta_list, basicchecks = True, functiontype_list = 'reward', boundfunction_list = boundfunction_list)

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vf_solveback_1endogstate_func import dist_solveback
    fulldistlist, endogdistlist = dist_solveback([1], [0.5, 0.5], endogstate_list, transmissionarray_list, pollist)

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vf_solveback_1endogstate_func import dist_meanvar
    meandistlist = dist_meanvar(endogdistlist, endogstate_list)

    # print(endogdistlist)
    print(meandistlist)


def manyperiod_continuous_example(T = 3, transmissionstarmethod = True):
    """
    Standard consumption allocation problem with zero return on assets.
    endogstate is a and a'
    exogstate is labor earnings which varies today but not tomorrow
    """
    def u(C):
        if C > 0:
            return(np.log(C))
        else:
            return(-1e90)

    def rewardfunction(assets, laborincome, assetsprime):
        C = assets + laborincome - assetsprime
        return(u(C))

    def boundfunction(endogstate_now, exogstate):
        s1low = None
        # must save strictly less than all assets today
        s1high = endogstate_now * R + exogstate - 1e-4
        return(s1low, s1high)

    
    endogstate_start = [0]
    endogstate_middleend = np.exp(np.linspace(np.log(1e-4), np.log(10)))
    exogstate_startmiddle = [0.1, 1]
    exogstate_end = [0]
    transmissionarray_end = np.array([[1], [1]])
    transmissionarray_middle = np.array([[0.5, 0.5], [0.1, 0.9]])

    inputfunction_list = [rewardfunction] * (T - 1)
    endogstate_list = [endogstate_start] + [endogstate_middleend] * (T - 1)
    exogstate_list = [exogstate_startmiddle] * (T - 1) +  [exogstate_end]
    boundfunction_list = [boundfunction] * (T - 1)
    transmissionarray_list = [transmissionarray_middle] * (T - 2) + [transmissionarray_end]
    beta_list = [BETA] * (T - 1)

    def lastperiodutility(endogval, exogval):
        return(u(endogval + exogval))

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vf_solveback_1endogstate_func import vf_solveback_continuous
    Vlist, pollist = vf_solveback_continuous(inputfunction_list, lastperiodutility, endogstate_list, exogstate_list, transmissionarray_list, beta_list, basicchecks = True, functiontype_list = 'reward', boundfunction_list = boundfunction_list)

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vf_solveback_1endogstate_func import dist_solveback
    fulldistlist, endogdistlist = dist_solveback([1], [0.5, 0.5], endogstate_list, transmissionarray_list, pollist)

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vf_solveback_1endogstate_func import dist_meanvar
    meandistlist = dist_meanvar(endogdistlist, endogstate_list)

    # print(endogdistlist)
    print(meandistlist)
manyperiod_continuous_example(T = 3, transmissionstarmethod = False)
# Discrete:{{{1
def twoperiod_discrete_example():
    """
    Standard consumption allocation problem with zero return on assets.
    endogstate is a and a'
    exogstate is labor earnings which varies today but not tomorrow
    """
    def u(C):
        if C > 0:
            return(np.log(C))
        else:
            return(-1e90)

    endogstate_now = [0]
    endogstate_future = np.exp(np.linspace(np.log(1e-4), np.log(10), 300))
    exogstate_now = [0.1, 1]
    exogstate_future = [0]
    transmissionarray = np.array([[1], [1]])

    ns1 = len(endogstate_now)
    ns1prime = len(endogstate_future)
    ns2 = len(exogstate_now)
    ns2prime = len(exogstate_future)

    rewardarray = np.empty([ns1, ns2, ns1prime])
    for s1 in range(ns1):
        for s2 in range(ns2):
            for s1prime in range(ns1prime):
                C = endogstate_now[s1] * R + exogstate_now[s2] - endogstate_future[s1prime]
                rewardarray[s1, s2, s1prime] = u(C)

    # get second period V
    def lastperiodutility(endogval, exogval):
        return(u(endogval + exogval))
    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vf_solveback_1endogstate_func import Vprime_get
    Vprime = Vprime_get(lastperiodutility, endogstate_future, exogstate_future)

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vf_solveback_1endogstate_func import vf_1endogstate_discrete_oneiteration
    V, pol = vf_1endogstate_discrete_oneiteration(rewardarray, Vprime, transmissionarray, BETA)

    print(V)
    print(pol)
    print([endogstate_future[pol[0, s2]] for s2 in range(ns2)])


def threeperiod_discrete_example(transmissionstarmethod = True):
    def u(C):
        if C > 0:
            return(np.log(C))
        else:
            return(-1e90)

    endogstate_start = [0]
    endogstate_middleend = np.exp(np.linspace(np.log(1e-4), np.log(10), 300))
    exogstate_startmiddle = [0.1, 1]
    exogstate_end = [0]
    transmissionarray_end = np.array([[1], [1]])
    transmissionarray_start = np.array([[0.5, 0.5], [0.1, 0.9]])

    endogstate_list = [endogstate_start, endogstate_middleend, endogstate_middleend]
    exogstate_list = [exogstate_startmiddle, exogstate_startmiddle, exogstate_end]
    T = len(endogstate_list)

    rewardarray_list = []
    for t in range(T - 1):
        ns1 = len(endogstate_list[t])
        ns1prime = len(endogstate_list[t + 1])
        ns2 = len(exogstate_list[t])
        ns2prime = len(exogstate_list[t + 1])

        rewardarray = np.empty([ns1, ns2, ns1prime])
        for s1 in range(ns1):
            for s2 in range(ns2):
                for s1prime in range(ns1prime):
                    C = endogstate_list[t][s1] * R + exogstate_list[t][s2] - endogstate_list[t + 1][s1prime]
                    rewardarray[s1, s2, s1prime] = u(C)

        rewardarray_list.append(rewardarray)

    transmissionarray_list = [transmissionarray_start, transmissionarray_end]
    beta_list = [BETA] * 2

    # get second period V
    def lastperiodutility(endogval, exogval):
        return(u(endogval + exogval))
    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vf_solveback_1endogstate_func import Vprime_get
    Vprime = Vprime_get(lastperiodutility, endogstate_list[-1], exogstate_list[-1])


    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vf_solveback_1endogstate_func import vf_solveback_discrete
    Vlist, pol_list = vf_solveback_discrete(rewardarray_list, Vprime, transmissionarray_list, beta_list)

    # get the distribution
    if transmissionstarmethod is True:
        sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
        from vf_solveback_1endogstate_func import dist_solveback
        fulldistlist, endogdistlist = dist_solveback([1], [0.5, 0.5], endogstate_list, transmissionarray_list, pol_list)

        sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
        from vf_solveback_1endogstate_func import dist_meanvar
        meandistlist = dist_meanvar(endogdistlist, endogstate_list)
    else:
        None

    print(meandistlist)

def manyperiod_discrete_example(T = 3, transmissionstarmethod = True):
    def u(C):
        if C > 0:
            return(np.log(C))
        else:
            return(-1e90)

    endogstate_start = [0]
    endogstate_middleend = np.exp(np.linspace(np.log(1e-4), np.log(10), 300))
    exogstate_startmiddle = [0.1, 1]
    exogstate_end = [0]
    transmissionarray_end = np.array([[1], [1]])
    transmissionarray_middle = np.array([[0.5, 0.5], [0.1, 0.9]])

    endogstate_list = [endogstate_start] + [endogstate_middleend] * (T - 1)
    exogstate_list = [exogstate_startmiddle] * (T - 1) + [exogstate_end]

    rewardarray_list = []
    for t in range(T - 1):
        ns1 = len(endogstate_list[t])
        ns1prime = len(endogstate_list[t + 1])
        ns2 = len(exogstate_list[t])
        ns2prime = len(exogstate_list[t + 1])

        rewardarray = np.empty([ns1, ns2, ns1prime])
        for s1 in range(ns1):
            for s2 in range(ns2):
                for s1prime in range(ns1prime):
                    C = endogstate_list[t][s1] * R + exogstate_list[t][s2] - endogstate_list[t + 1][s1prime]
                    rewardarray[s1, s2, s1prime] = u(C)

        rewardarray_list.append(rewardarray)

    transmissionarray_list = [transmissionarray_middle] * (T - 2) + [transmissionarray_end]
    beta_list = [BETA] * (T - 1)

    # get second period V
    def lastperiodutility(endogval, exogval):
        return(u(endogval + exogval))
    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vf_solveback_1endogstate_func import Vprime_get
    Vprime = Vprime_get(lastperiodutility, endogstate_list[-1], exogstate_list[-1])


    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vf_solveback_1endogstate_func import vf_solveback_discrete
    Vlist, pol_list = vf_solveback_discrete(rewardarray_list, Vprime, transmissionarray_list, beta_list)

    # get the distribution
    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vf_solveback_1endogstate_func import dist_solveback
    fulldistlist, endogdistlist = dist_solveback([1], [0.5, 0.5], endogstate_list, transmissionarray_list, pol_list, transmissionstarmethod = transmissionstarmethod)

    sys.path.append(str(__projectdir__ / Path('submodules/vfi-general/')))
    from vf_solveback_1endogstate_func import dist_meanvar
    meandistlist = dist_meanvar(endogdistlist, endogstate_list)

    print(meandistlist)
