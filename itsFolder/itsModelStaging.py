#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:07:46 2018

@author: carlos
"""

import numpy


def stagingCalculate(con, Dv1: float, Dv2: float)-> None:

    efflist = con['efflist']
    Tlist = con['Tlist']

    if con['homogeneous']:

        if con['NStag'] == 0:

            p2 = modelOptimalStagingHomogeneous([efflist[-1]],       Dv2,
                                                [Tlist[-1]], con['Isp'],
                                                con['g0'],  con['Mu'])
            p1 = modelOptimalStagingHomogeneous([efflist[0]], Dv1 + Dv2,
                                                [Tlist[0]], con['Isp'],
                                                con['g0'],  con['Mu'])
            p2.ms = p2.mu + p1.me[0]
            p2.me[0] = 0.0
            p1.NStag0or1(p2)

        elif con['NStag'] == 1:

            p2 = modelOptimalStagingHomogeneous([efflist[-1]],       Dv2,
                                                [Tlist[-1]], con['Isp'],
                                                con['g0'],  con['Mu'])
            p1 = modelOptimalStagingHomogeneous([efflist[0]], Dv1 + Dv2,
                                                [Tlist[0]], con['Isp'],
                                                con['g0'],  con['Mu'])
            p2.me[0] = p1.me[0]
            p1.NStag0or1(p2)

        else:

            p2 = modelOptimalStagingHomogeneous([efflist[-1]], Dv2,
                                                [Tlist[-1]], con['Isp'],
                                                con['g0'],  con['Mu'])
            p1 = modelOptimalStagingHomogeneous(efflist[0:-1], Dv1,
                                                Tlist[0:-1], con['Isp'],
                                                con['g0'], p2.mtot[0])

    else:

        if con['NStag'] > 1:
            Isplist = con['Isplist']

            p2 = modelStagingHeterogeneous([efflist[-1]], [Isplist[-1]],
                                           [Tlist[-1]], Dv2, con['Mu'],
                                           con['g0'], con['tol'])
            # p2.bisec()
            p2.result()
            # p2.show()
            p1 = modelStagingHeterogeneous(efflist[0:-1], Isplist[0:-1],
                                           Tlist[0:-1], Dv1, p2.mtot[0],
                                           con['g0'], con['tol'])
            # p1.bisec()
            p1.result()
            # p1.show()
        else:
            raise Exception('itsme saying: heterogeneous vehicle for'
                            'NStag < 2 is not supported yet!')

    if numpy.isnan(p1.mtot[0]):
        print('p1: ')
        p1.show()
        print('\np2: ')
        p2.show()
        raise Exception('itsme saying: negative masses!')

    if p1.mtot[0]*con['g0'] > Tlist[0]:
        print('T = ', Tlist[0])
        print('P = ', p1.mtot[0]*con['g0'])
        raise Exception('itsme saying: weight greater than thrust!')

    return p1, p2


class modelOptimalStagingHomogeneous():
    # optimalStaging() returns a object with information of
    # optimal staging factor for a homogeneous vehcile
    # The maximal staging reason is defined as reducing the total mass for
    # a defined delta V.
    # Structural eficience and thrust shall variate for diferent stages,  but
    # specific impulse must be the same for all stages
    # Based in Cornelisse (1979)

    def __init__(self, effList: list, dV: float, Tlist: list, Isp: float,
                 g0: float, mu: float):
        self.Tlist = Tlist
        self.e = numpy.array(effList)
        self.T = numpy.array(Tlist)
        self.dV = dV
        self.mu = mu
        self.Isp = Isp
        self.c = Isp*g0
        self.mflux = self.T/self.c

        self._lamb = (numpy.exp(-self.dV/self.c/self.e.size) -
                      self.e)/(1 - self.e)

        phi = (1 - self.e)*(1 - self._lamb)

        # Total sub-rocket mass
        self.__mtotCalculate()
        # Propelant mass on each stage
        self.mp = self.mtot*phi
        # Strutural mass of each stage
        self.me = self.mp*(self.e/(1 - self.e))
        # Duration of each stage burning
        self.tb = self.mp/self.mflux
        # Final burning time of each stage
        self.__tfCalculate()
        # Empty vehicle mass (!= 0.0 only for NStag == 0)
        self.ms = 0.0
        return None

    def __mtotCalculate(self)-> None:

        mtot = self.e*0.0
        N = self.e.size-1
        for ii in range(0,  N+1):
            if ii == 0:
                mtot[N - ii] = self.mu/self._lamb[N - ii]
            else:
                mtot[N - ii] = mtot[N - ii + 1]/self._lamb[N - ii]
        self.mtot = mtot
        return None

    def __tfCalculate(self)-> None:

        tf = self.e*0.0
        N = self.tb.size-1
        for ii in range(0, N+1):
            if ii == 0:
                tf[ii] = self.tb[ii]
            else:
                tf[ii] = self.tb[ii] + tf[ii-1]
        self.tf = tf
        return None

    def NStag0or1(self, p: object)-> None:

        self.dV = self.dV - p.dV
        self.mu = p.mtot[0]
        self.mp[0] = self.mp[0] - p.mp[0]  # Propelant mass on each stage
        self.me[0] = self.me[0]*0.0
        self.tb[0] = self.tb[0] - p.tb[0]  # Duration of each stage burning
        self.tf[0] = self.tf[0] - p.tf[0]
        return None

    def printInfo(self):

        print("\n\rdV =", self.dV)
        print("mu =", self.mu)
        print("me =", self.me)
        print("mp =", self.mp)
        print("mtot =", self.mtot)
        print("mflux =", self.mflux)
        print("tb =", self.tb)
        print("tf =", self.tf)


class modelOptimalStagingHeterogeneous():

    def __init__(self, elist: list, Isplist: list, Tlist: list, vTot: float,
                 Mu: float, g0: float, tol: float):

        self.Mu = Mu
        self.tol = tol
        self.e = numpy.array(elist)
        self.Isplist = Isplist
        self.c = numpy.array(Isplist)*g0
        self.T = numpy.array(Tlist)
        self.vTot = vTot
        self.cMin = numpy.min(self.c)
        self.exp = numpy.exp(1.0)
        self.mflux = self.T/self.c
        self.x = []
        self.v = []
        self.lamb = []
        self.mtot = []
        self.mp = []
        self.me = []
        self.tb = []
        self.tf = []
        self.count = 0

    def functionToBe0(self, x: float) -> float:

        d0 = (1.0 - x/self.c)/self.e
        d1 = d0 ** (self.c/self.vTot)
        error = numpy.prod(d1) - self.exp

        return error

    def bisec(self) -> None:

        count = 0
        x1 = self.cMin/2
        #  print(x1)
        stop = False
        error1 = self.functionToBe0(x1)
        x2 = x1*(1 - self.tol)
        #  print(x2)
        error2 = self.functionToBe0(x2)
        while not stop:

            step = error2*(x2 - x1)/(error2 - error1)

            x1 = x2 + 0.0
            error1 = error2.copy()

            x2 = x2 - step
            error2 = self.functionToBe0(x2)

            count += 1

            if (abs(error2) < self.tol) or (count == 100):
                stop = True

            warnning = 'itsme saying: lagrange multiplier out of interval'
            if (x2 < 0):
                x2 = 0.0
                print(warnning)
            elif (x2 > self.cMin):
                x2 = self.cMin + 0.0
                print(warnning)

        self.x = x2
        self.count = count
        #  print(count)
        return None

    def result(self) -> None:

        self.v = self.c*numpy.log((1 - self.x/self.c)/self.e)
        self.lamb = (numpy.exp(-self.v/self.c) - self.e)/(1 - self.e)

        self.mtot = self.lamb.copy()
        self.me = self.mtot.copy()
        N = len(self.lamb)

        for ii in range(1, N + 1):
            if ii == 1:
                self.mtot[N - ii] = self.Mu/self.lamb[N - ii]
                self.me[N - ii] = self.e[N - ii]*(self.mtot[N - ii] - self.Mu)

            else:
                #  print(N-ii)
                self.mtot[N - ii] = self.mtot[N - ii + 1]/self.lamb[N - ii]
                self.me[N - ii] = self.e[N - ii]*(self.mtot[N - ii] -
                                                  self.mtot[N - ii + 1])

        self.mp = (1 - self.e)*self.me/self.e
        self.tb = self.mp/self.mflux
        self.tf = self.tb.copy()

        for ii in range(1, N):
            self.tf[ii] = self.tf[ii - 1] + self.tb[ii]

        self.Tlist = self.T.tolist()
        self.tf = self.tf.tolist()
        self.tb = self.tb.tolist()
        self.me = self.me.tolist()

        return None

    def show(self) -> None:

        print('inputs:')
        print('e', self.e)
        print('c', self.c)
        print('T', self.T)

        print('\nresults:')
        print('tol', self.tol)
        print('error', self.functionToBe0(self.x))
        print('count', self.count)
        print('vTot', self.vTot)
        print('v', sum(self.v))
        print('mflux', self.mflux)
        print('lamb', self.lamb)
        print('mtot', self.mtot)
        print('mp', self.mp)
        print('me', self.me)
        print('tb', self.tb)
        print('tf', self.tf)

    def printInfo(self)-> None:

        print("\n\rdV =", self.vTot)
        print("mu =", self.Mu)
        print("me =", self.me)
        print("mp =", self.mp)
        print("mtot =", self.mtot)
        print("mflux =", self.mflux)
        print("tb =", self.tb)
        print("tf =", self.tf)

        return None


class modelStagingHeterogeneous():

    def __init__(self, elist: list, Isplist: list, Tlist: list, vTot: float,
                 Mu: float, g0: float, tol: float):

        self.Mu = Mu
        self.tol = tol
        self.e = numpy.array(elist)
        self.Isplist = Isplist
        self.c = numpy.array(Isplist)*g0
        self.T = numpy.array(Tlist)
        self.clne = self.c*numpy.log(self.e)
        self.vTot = vTot
        self.cMin = numpy.min(self.c)
        self.exp = numpy.exp(1.0)
        self.mflux = self.T/self.c
        self.x = []
        self.v = []
        self.lamb = []
        self.mtot = []
        self.mp = []
        self.me = []
        self.tb = []
        self.tf = []
        self.count = 0

        if vTot < 0:
            print('N: ', len(elist))
            print('vTot: ', vTot)
            raise Exception('itsme saying: vTot is smaller than zero!')

    def result(self) -> None:

        N = len(self.e)
        # self.v = self.vTot/N
        # self.v = self.vTot*self.c/numpy.sum(self.c)
        self.v = self.vTot/N + numpy.mean(self.clne) - self.clne
        self.lamb = (numpy.exp(-self.v/self.c) - self.e)/(1 - self.e)

        self.mtot = self.lamb.copy()
        self.me = self.mtot.copy()

        for ii in range(1, N + 1):
            if ii == 1:
                self.mtot[N - ii] = self.Mu/self.lamb[N - ii]
                self.me[N - ii] = self.e[N - ii]*(self.mtot[N - ii] - self.Mu)

            else:
                #  print(N-ii)
                self.mtot[N - ii] = self.mtot[N - ii + 1]/self.lamb[N - ii]
                self.me[N - ii] = self.e[N - ii]*(self.mtot[N - ii] -
                                                  self.mtot[N - ii + 1])

        self.mp = (1 - self.e)*self.me/self.e
        self.tb = self.mp/self.mflux
        self.tf = self.tb.copy()

        for ii in range(1, N):
            self.tf[ii] = self.tf[ii - 1] + self.tb[ii]

        self.Tlist = self.T.tolist()
        self.tf = self.tf.tolist()
        self.tb = self.tb.tolist()
        self.me = self.me.tolist()

        return None

    def show(self) -> None:

        print('inputs:')
        print('e', self.e)
        print('c', self.c)
        print('T', self.T)

        print('\nresults:')
        print('tol', self.tol)
        print('vTot', self.vTot)
        print('v', sum(self.v))
        print('mflux', self.mflux)
        print('lamb', self.lamb)
        print('mtot', self.mtot)
        print('mp', self.mp)
        print('me', self.me)
        print('tb', self.tb)
        print('tf', self.tf)

    def printInfo(self)-> None:

        print("\n\rdV =", self.vTot)
        print("mu =", self.Mu)
        print("me =", self.me)
        print("mp =", self.mp)
        print("mtot =", self.mtot)
        print("mflux =", self.mflux)
        print("tb =", self.tb)
        print("tf =", self.tf)

        return None
