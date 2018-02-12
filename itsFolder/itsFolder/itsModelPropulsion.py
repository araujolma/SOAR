#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:20:20 2018

@author: carlos
"""

import numpy
import matplotlib.pyplot as plt


class modelPropulsion():

    def __init__(self, p1: object, p2: object, tf: float,
                 v1: float, v2: float, softness: float, Isp: float, T: float):

        # improvements for heterogeneous rocket
        self.Isp = Isp
        self.T = T

        self.t1 = p1.tf[-1]
        t2 = tf - p2.tb[-1]
        self.t2 = t2
        self.t3 = tf

        self.v1 = v1
        self.v2 = v2

        f = softness/2

        d1 = self.t1  # width of retangular and 0.5 soft part
        self.c1 = d1*f  # width of the 0.5 soft part
        self.fr1 = d1 - self.c1  # final of the retangular part
        self.fs1 = d1 + self.c1  # final of the retangular part

        d2 = self.t3 - t2  # width of retangular and 0.5 soft part
        self.c2 = d2*f  # width of the 0.5 soft part
        self.r2 = d2 - self.c2  # width of the retangular part
        self.ir2 = t2 + self.c2  # start of the retangular part
        self.is2 = t2 - self.c2  # start of the soft part

        self.dv21 = v2 - v1

        # List of time events and jetsoned masses
        self.tflist = p1.tf[0:-1].tolist() + [self.fr1, self.fs1] + \
            [self.is2, self.ir2, tf]
        self.melist = p1.me[0:-1].tolist() + [0.0, p1.me[-1]] + \
            [0.0, 0.0, p2.me[-1]]

        # List of time events and jetsoned masses for first propulsive part
        self.tflistP1 = p1.tf[0:-1].tolist() + [self.fr1, self.fs1]
        self.melistP1 = p1.me[0:-1].tolist() + [0.0, p1.me[-1]]

        self.fail = False
        if len(p1.tf) > 2:
            if p1.tf[-2] >= self.fr1:
                self.fail = True

        self.tlist1 = p1.tf[0:-1].tolist()+[self.fs1]
        self.Tlist1 = p1.Tlist
        self.Tlist2 = p2.Tlist

        self.Isp1 = p1.Isp
        self.Isp2 = p2.Isp

    def single(self, t: float)-> float:
        if (t <= self.fr1):
            ans = self.v1
        elif (t <= self.fs1):
            cos = numpy.cos(numpy.pi*(t - self.fr1)/(2*self.c1))
            ans = self.dv21*(1 - cos)/2 + self.v1
        elif (t <= self.is2):
            ans = self.v2
        elif (t <= self.ir2):
            cos = numpy.cos(numpy.pi*(t - self.is2)/(2*self.c2))
            ans = -self.dv21*(1 - cos)/2 + self.v2
        elif (t <= self.t3):
            ans = self.v1
        else:
            ans = 0.0

        return ans

    def value(self, t: float)-> float:
        if (t <= self.fr1):
            ans = self.v1
        elif (t <= self.fs1):
            cos = numpy.cos(numpy.pi*(t - self.fr1)/(2*self.c1))
            ans = self.dv21*(1 - cos)/2 + self.v1
        elif (t <= self.is2):
            ans = self.v2
        elif (t <= self.ir2):
            cos = numpy.cos(numpy.pi*(t - self.is2)/(2*self.c2))
            ans = -self.dv21*(1 - cos)/2 + self.v2
        elif (t <= self.t3):
            ans = self.v1
        else:
            ans = 0.0

        return ans

    def mdlDer(self, t: float)-> tuple:

        return self.value(t), self.Isp, self.T

    def multValue(self, t: float):
        N = len(t)
        ans = numpy.full((N, 1), 0.0)
        for jj in range(0, N):
            ans[jj] = self.value(t[jj])

        return ans


class modelPropulsionHetSimple():

    def __init__(self, p1: object, p2: object, tf: float,
                 v1: float, v2: float):

        # Improvements for heterogeneous rocket
        self.fail = False
        # Total list of specific impulse
        self.Isplist = p1.Isplist + [1.0] + p2.Isplist
        # Total list of Thrust
        self.Tlist = p1.Tlist + [0.0] + p2.Tlist
        # Total list of final t
        t2 = tf - p2.tb[-1]
        self.t2 = t2
        self.tflist = p1.tf + [t2, tf]
        self.tf = tf
        # Total list of jettsoned masses
        self.melist = p1.me + [0.0, p2.me[-1]]
        # Total list of Thrust control
        self.vlist = []
        for T in self.Tlist:
            if T > 0.0:
                self.vlist.append(v1)
            else:
                self.vlist.append(v2)
        # number of archs
        self.N = len(self.tflist)
        # List of events for the first propulsive part
        self.tflistP1 = p1.tf
        self.melistP1 = p1.me

    def getIndex(self, t: float)-> int:
        ii = 0
        stop = False
        while not stop:
            if ii == self.N:
                stop = True
            elif t < self.tflist[ii]:
                stop = True
                # ii = ii - 1
            else:
                ii += 1

        return ii

    def single(self, t: float)-> float:
        ii = self.getIndex(t)
        return self.vlist[ii]

    def value(self, t: float)-> float:
        ii = self.getIndex(t)
        if ii == self.N:
            ans = 0.0
        else:
            ans = self.vlist[ii]

        return ans

    def mdlDer(self, t: float)-> tuple:
        ii = self.getIndex(t)
        if ii == self.N:
            ans = 0.0, 1.0, 0.0
        else:
            ans = self.vlist[ii], self.Isplist[ii], self.Tlist[ii]

        return ans

    def multValue(self, t: float):
        N = len(t)
        ans = numpy.full((N, 1), 0.0)
        for jj in range(0, N):
            ans[jj] = self.value(t[jj])
        return ans

    def mdlDerXtime(self, N: int):
        tt = range(0, N)*(self.tf/(N-1))
        v_t = []
        Isp_t = []
        T_t = []

        for t in tt:
            v, Isp, T = self.mdlDer(t)
            v_t.append(v)
            Isp_t.append(Isp)
            T_t.append(T)

        return tt, v_t, Isp_t, T_t

    def show(self):

        print('Isplist', self.Isplist)
        print('Tlist', self.Tlist)
        print('melist', self.melist)
        print('vlist', self.vlist)
        print('tflist', self.tflist)
        raise

    def plot(self)-> None:

        tt, v_t, Isp_t, T_t = self.mdlDerXtime(1000)

        plt.subplot2grid((6, 2), (0, 0), rowspan=2, colspan=2)
        plt.plot(tt, v_t, '-b')
        plt.grid(True)
        plt.ylabel("beta [-]")

        plt.subplot2grid((6, 2), (2, 0), rowspan=2, colspan=2)
        plt.plot(tt, Isp_t, '-b')
        plt.grid(True)
        plt.ylabel("Isp [s]")

        plt.subplot2grid((6, 2), (4, 0), rowspan=2, colspan=2)
        plt.plot(tt, T_t, '-b')
        plt.grid(True)
        plt.ylabel("T [kN]")

        plt.show()
