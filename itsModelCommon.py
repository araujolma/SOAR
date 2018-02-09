#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 20:57:12 2018

@author: carlos
"""

import numpy
from atmosphere import rho


def mdlDer(t: float, x: list, alfaProg: callable, betaProg: callable,
           aed: callable, earth: callable)-> list:

    # initialization
    h = x[0]
    v = x[1]
    gamma = x[2]
    M = x[3]

    if numpy.isnan(h):
        print('t: ', t)
        #  print('x: ', x)
        raise Exception('itsme saying: h is not a number')

    # Alpha control calculation
    alfat = alfaProg(t)

    # other calculations
    # btm = betaProg(t)*con['T']/M
    beta, Isp, T = betaProg(t)
    btm = beta*T/M
    sinGamma = numpy.sin(gamma)
    g = earth.g0*(earth.R/(earth.R+h))**2 - (earth.we**2)*(earth.R + h)

    # aerodynamics
    qdinSrefM = 0.5 * rho(h) * (v**2) * aed.s_ref/M
    LM = qdinSrefM * (aed.CL0 + aed.CL1*alfat)
    DM = qdinSrefM * (aed.CD0 + aed.CD2*(alfat**2))

    if v < 1e-6 and v >= 0.0:
        v = 1e-6
    elif v > -1e-6 and v < 0.0:
        v = -1e-6

    # states derivatives
    return [v*sinGamma,  # coefficient
            btm*numpy.cos(alfat) - g*sinGamma - DM,  # coefficient
            btm*numpy.sin(alfat)/v +
            (v/(h+earth.R)-g/v)*numpy.cos(gamma) +
            (LM/v) + 2*earth.we,  # coefficient
            -btm*M/(earth.g0*Isp)]  # coefficient


class modelAttitude():

    def __init__(self, t1: float, t2: float, v1: float, v2: float):
        self.t1 = t1
        self.t2 = t2

        self.v1 = v1
        self.v2 = v2

    def value(self, t: float)-> float:
        if (t >= self.t1) and (t < self.t2):
            ans = ((self.v2 - self.v1)*(1 -
                   numpy.cos(2*numpy.pi*(t - self.t1)/(self.t2 - self.t1)))/2)\
                   + self.v1
            return ans
        else:
            return self.v1

    def multValue(self, t: float):
        N = len(t)
        ans = numpy.full((N, 1), 0.0)
        for jj in range(0, N):
            ans[jj] = self.value(t[jj])

        return ans


class modelTrajectory():

    def __init__(self):

        self.tt = []
        self.xx = []
        self.tp = []
        self.xp = []
        self.tphases = []
        self.mass0 = []
        self.massJet = []

    def append(self, tt, xx):

        self.tt.append(tt)
        self.xx.append(xx)

    def appendStar(self, tt, xx):

        self.tt.append(tt)
        self.xx.append([*xx])

    def appendP(self, tp, xp):

        self.tp.append(tp)
        self.xp.append(xp)

    def numpyArray(self):

        self.tt = numpy.array(self.tt)
        self.xx = numpy.array(self.xx)
        self.tp = numpy.array(self.tp)
        self.xp = numpy.array(self.xp)


class modelAed():

    def __init__(self, con):

        self.s_ref = con['s_ref']
        self.CL0 = con['CL0']
        self.CL1 = con['CL1']
        self.CD0 = con['CD0']
        self.CD2 = con['CD2']


class modelEarth():

    def __init__(self, con):

        self.g0 = con['g0']
        self.R = con['R']
        self.we = con['we']
