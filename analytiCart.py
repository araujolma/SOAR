#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 08:13:49 2019

@author: levi

This module is for obtaining the analytical solution of probCart.
"""
import numpy
import matplotlib.pyplot as plt
from utils import simp


def try_lambdaR(lr,K=1.,vL=.5,l=1.,a=1.,mustPrint=False,mustPlot=False):
    """Generates a solution with the same "shape" as the optimal solution,
     from a given value of lr (lambda_R), returning the value of the
      residual of the time equation. Hence the solution yielded is not
     necessarily the optimal one, but the closer the residual is to zero,
     the closer the solution is to the optimal one.
     """

    # maximum theoretical speed
    vMaxTheo = numpy.sqrt(a * l)
    if vL > vMaxTheo:
        raise Exception("Limit speed is useless (too high).")

    # this is the speed in which the cart "coasts"
    vs = vL + lr / 2. / K
    if vs > vMaxTheo:
        vs = vMaxTheo

    # total time
    pi = l / vs + vs / a
    # (non-dimensional) time until vL speed is achieved
    tvL = vL/pi/a
    # (non-dimensional) time until vs speed is achieved
    t1s = vs/pi/a

    # speed limit violation
    viol = 100.*(vs/vL-1.)

    # preparing the array for numerical integration
    N = 10001
    tVec = numpy.linspace(0.,1.,num=N)
    v = tVec * 0.
    # this loop assembles the speed profile array
    for k,t in enumerate(tVec):
        if t < t1s:
            v[k] = pi * a * t
        elif t < 1.-t1s:
            v[k] = vs
        else:
            v[k] = vs - pi * a * (t+t1s-1.)

    # residual computation
    int1 = K * simp( (v-vL)**2 * (v>=vL) , N)
    int2 = K * simp( v * (v-vL) * (v>=vL) , N)
    res = 1. + int1 + 2 * int2 - 2.*lr*l/pi

    # cost function value for this solution proposal
    cost = pi * (1. + int1)
    strVio = "Speed violation = {:.2G}%, costFunc = {:.4E}".format(viol,cost)
    if mustPrint:
        strP = "vs = {:.4G}, pi = {:.4G}, tvL = {:.4G}, t1s = {:.4G}".format(
                vs,pi,tvL,t1s)
        print(strP)
        print(strVio)

    if mustPlot:
        plt.plot(tVec,v,label='v')
        plt.plot(tVec,vL+tVec*0.,label='vLim')
        plt.xlabel("Non-dimensional time")
        plt.ylabel("Speed")
        plt.title(strVio)
        plt.grid(True)
        plt.legend()
        plt.show()

    return res

def solve(K=1.,vL=.5,tol=1e-6):
    """Get the analytical solution by finding the proper lambda_R.
        The search is based on a simple bisection method."""


    # Get a lambda_R so that the residual is positive
    lp = 1.; resP = -1.
    while resP <= 0.:
        lp -= 1.
        resP = try_lambdaR(lp, K=K, vL=vL)

    # Get a lambda_R so that the residual is negative
    resN = 1.; ln = 1.  # .5#.387
    while resN >= 0:
        ln += 1.
        resN = try_lambdaR(ln, K=K, vL=vL)

    # now, do the bisection
    res = 1.; lr = 1.
    while abs(res) > tol:
        lr = .5*(lp+ln)
        print("\nTrying lambda_r =",lr)
        res = try_lambdaR(lr,K=K,vL=vL)
        print("res =",res)
        if res < 0:
            ln = lr
        else:
            lp = lr
    #
    print("\nFinal answer: (lambda_r = {:.4E}, residual = {:.4E})".format(
        lr,res))
    try_lambdaR(lr,K=K,vL=vL,mustPrint=True,mustPlot=True)
#


if __name__ == "__main__":
    #print(try_lambdaR(1e-3,mustPrint = True , mustPlot = True))
    #print(try_lambdaR(2.,mustPrint = True , mustPlot = True))
    solve(K=100.,vL=.5)#10000.)


