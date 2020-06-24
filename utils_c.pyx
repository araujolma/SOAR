# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 18:53:01 2016

@author: levi
"""
import numpy

cimport cython
cimport numpy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def simp(vec, int N, onlyCoef=False):
    """ Simple integration of array according to Simpson's method.
    It can also only yield the coefficients if one wants to do the integration
    by oneself (maybe in an optimized loop)."""


#    coefList = numpy.ones(N)
#    coefList[0] = 17.0/48.0; coefList[N-1] = coefList[0]
#    coefList[1] = 59.0/48.0; coefList[N-2] = coefList[1]
#    coefList[2] = 43.0/48.0; coefList[N-3] = coefList[2]
#    coefList[3] = 49.0/48.0; coefList[N-4] = coefList[3]
#    coefList *= 1.0/(N-1)

    #cdef double[:]

    coefList = numpy.empty(N, dtype=numpy.double)
    cdef Py_ssize_t k
    cdef double oneN = 1.0 / (3.0 * N)
    cdef double twoN = 2.0 * oneN
    cdef double fourN = 2.0 * twoN

    for k in range(1,N-1):
        if k % 2 == 0:
            coefList[k] = twoN
        else:
            coefList[k] = fourN
    #
    coefList[0] = oneN
    coefList[N-1] = oneN

    if onlyCoef:
        return numpy.asarray(coefList)
    else:
        return numpy.asarray(coefList.dot(vec))
