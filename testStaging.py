#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:35:04 2017

@author: carlos
"""

import numpy


class test():

    def __init__(self, eList, cList, vTot, tol):

        self.tol = tol
        self.e = numpy.array(eList)
        self.c = numpy.array(cList)
        self.vTot = vTot
        self.cMin = numpy.min(self.c)
        self.exp = numpy.exp(1.0)
        self.v = []
        self.lamb = []

    def function(self, x):

        d0 = (1.0 - x/self.c)/self.e
        print(d0)
        d1 = d0 ** (self.c/self.vTot)
        error = numpy.prod(d1) - self.exp

        return error

    def bisec(self):

        count = 0
        x1 = self.cMin/2
        print(x1)
        stop = False
        error1 = self.function(x1)
        x2 = x1*(1 - tol)
        print(x2)
        error2 = self.function(x2)
        while not stop:

            step = error2*(x2 - x1)/(error2 - error1)

            x1 = x2.copy()
            error1 = error2.copy()

            x2 = x2 - step
            print(x2)
            error2 = self.function(x2)

            count += 1

            if (abs(error2) < self.tol) or (count == 100):
                stop = True

        return x2

    def result(self, x):

        self.v = self.c*numpy.log((1 - x/self.c)/self.e)
        self.lamb = (numpy.exp(-self.v/self.c) - self.e)/(1 - self.e)


if __name__ == "__main__":

    eList = [0.1, 0.09, 0.07]
    cList = [3.5, 3.0, 2.8]
    vTot = 7.0
    tol = 1e-9

    stg = test(eList, cList, vTot, tol)

    x = stg.bisec()
    stg.result(x)

    print('\nresults:')
    print(stg.function(x))
    print(stg.v)
    print(stg.lamb)
