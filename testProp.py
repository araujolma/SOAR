#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 21:35:00 2018

@author: carlos
"""

# Propulsive model for a heterogeneous rocket. Still on development!

from numpy import full


class modelPropulsion():

    def __init__(self, p1: object, p2: object, tf: float, v1: float,
                 v2: float):

        self.t1 = p1.tf[-1]
        self.t2 = tf - p2.tb[-1]
        self.t3 = tf

        self.v1 = v1
        self.v2 = v2

        self.p1 = p1
        self.p2 = p2

    def __indexFinder(self, t: float, p)-> int:
        ans = 0
        for tf in p.tf:
            if t > tf:
                ans = ans + 1
        return ans

    def __intervalFinder(self, t: float)-> float:
        if t <= self.t1:
            interval = 1
        elif t <= self.t2:
            interval = 2
        elif t <= self.t3:
            interval = 3
        else:
            interval = 4

        return interval

    def beta(self, t: float)-> float:

        interval = self.__intervalFinder(t)

        if interval == 1 and interval == 3:
            beta = self.v1
        else:
            beta = self.v2

        return beta

    def thrustIsp(self, t: float)-> float:

        interval = self.__intervalFinder(t)

        if interval == 1:
            ii = self.__indexFinder(t)
            T = self.p1.T[ii]
            Isp = self.p1.Isp[ii]
        elif interval == 3:
            ii = self.__indexFinder(t - self.t2)
            T = self.p2.T[ii]
            Isp = self.p2.Isp[ii]
        else:
            T = 0.0
            Isp = 1

        return T, Isp

    def value(self, t: float)-> float:

        T, Isp = self.thrustIsp(t)
        return self.beta(t), T, Isp

    def mdlDer(self, t: float)-> tuple:

        return self.beta(t), self.Isp, self.T

    def multValue(self, t: float):
        N = len(t)
        ans = full((N, 1), 0.0)
        for jj in range(0, N):
            ans[jj] = self.value(t[jj])

        return ans
