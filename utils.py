# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 18:53:01 2016

@author: levi
"""
import numpy
import datetime

#%% Interpolation methods

def interpV(t,tVec,xVec):
    Nsize = xVec.shape[1]
    ans = numpy.empty(Nsize)
    for k in range(Nsize):
        ans[k] = numpy.interp(t,tVec,xVec[:,k])
    return ans

def interpM(t,tVec,xVec):

    ni = xVec.shape[1]
    nj = xVec.shape[2]
    ans = numpy.empty((ni,nj))
    for i in range(ni):
        for j in range(nj):
            ans[i,j] = numpy.interp(t,tVec,xVec[:,i,j])
    return ans


#%% Calculus and related stuff

def ddt(vec,N):

    dt = 1.0/(N-1)
    dvec = numpy.empty_like(vec)
    dvec[0] = (vec[1]-vec[0])/dt
    dvec[N-1] = (vec[N-1]-vec[N-2])/dt
    for k in range(1,N-1):
        dvec[k] = .5*(vec[k+1]-vec[k-1])/dt
#    for k in range(N-1):
#        dvec[k] = vec[k+1]-vec[k]
#    dvec[N-1] = dvec[N-2]
#    dvec *= (1.0/dt)
    return dvec

def simp(vec, N, onlyCoef=False):
    """ Simple integration of array according to Simpson's method.
    It can also only yield the coefficients if one wants to do the integration
    by oneself (maybe in an optimized loop)."""

#    coefList = numpy.ones(N)
#    coefList[0] = 17.0/48.0; coefList[N-1] = coefList[0]
#    coefList[1] = 59.0/48.0; coefList[N-2] = coefList[1]
#    coefList[2] = 43.0/48.0; coefList[N-3] = coefList[2]
#    coefList[3] = 49.0/48.0; coefList[N-4] = coefList[3]
#    coefList *= 1.0/(N-1)

    coefList = numpy.empty(N)
    oneN = 1.0 / (3.0 * N)

    coefList[0] = oneN
    coefList[1:N:2] = 4.0 * oneN  # set odd coefficientes to 4/3N
    coefList[2:N:2] = 2.0 * oneN  # set even coefficientes to 2/3N
    coefList[N - 1] = oneN

    if onlyCoef:
        return coefList
    else:
        return coefList.dot(vec)

def testAlgn(x,y):
    """Test the alignment of three points on a plane. Returns the determinant
    of the three points, which acutally proportional to the area of the trian-
    gle determined by the points.

    x: array with the x coordinates
    y: array with the y coordinates"""

    A = numpy.ones((3,3))
    A[:,1] = x
    A[:,2] = y
    return numpy.linalg.det(A)

#%% Date, time, etc
def getNowStr():
    """Returns a string with the current time, all non numeric characters
    switched to _'s. """

    thisStr = str(datetime.datetime.now())
    thisStr = thisStr.replace(' ','_')
    thisStr = thisStr.replace('-','_')
    thisStr = thisStr.replace(':','_')
    thisStr = thisStr.replace('.','_')
    return thisStr

if __name__ == "__main__":
    print("In utils.py!")
    print("Testing testAlgn:")

    x = numpy.array([1.0,2.0,3.0])
    y = 5.0*x
    print(testAlgn(x,y))
    x[0]=0.0
    print(testAlgn(x,y))

    N = 501
    x = numpy.linspace(0,1,num=N) ** 3

    Isimp = simp(x,N)
    print("Isimp = "+str(Isimp))

    Itrap = 0.0
    Itrap = .5 * (x[0]+x[-1])
    Itrap += x[1:(N-1)].sum()
    Itrap *= 1.0/(N-1)
    print("Itrap = "+str(Itrap))

