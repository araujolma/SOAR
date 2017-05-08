#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:09:24 2017

@author: levi
"""
import numpy
import matplotlib.pyplot as plt
from prob_rocket_sgra import declProb, calcGrads, calcPhi
from sgra_simple_rocket_alt import calcP

opt = dict()
opt['initMode'] = 'extSol'#'default'#'extSol'

# declare problem:
sizes,t,x0,u0,pi0,lam,mu,tol,constants,boundary,restrictions = declProb(opt)
phi0 = calcPhi(sizes,x0,u0,pi0,constants,restrictions)

calcP(sizes,x0,u0,pi0,constants,boundary,restrictions,mustPlot=True)
Grads = calcGrads(sizes,x0,u0,pi0,constants,restrictions)

phix = Grads['phix']
N = sizes['N']
n = sizes['n']

phixNum = phix.copy()

E = numpy.zeros((n,N,n))

#for i in range(N):
#    for j in range(n):
#        E[j,i,j] = 1.0

delta = numpy.array([1e-6,1e-7,1e-8,1e-6])
for j in range(n):
    E[j,:,j] = numpy.ones(N)


for j in range(n):
    xp = x0 + delta[j]*E[j,:,:]
    xm = x0 - delta[j]*E[j,:,:]
    phip = calcPhi(sizes,xp,u0,pi0,constants,restrictions)
    phim = calcPhi(sizes,xm,u0,pi0,constants,restrictions)
    phixEj = .5*(phip-phim)/delta[j]
    for i in range(N):
        phixNum[i,:,j] = phixEj[i,:].copy()

erro = (phixNum-phix)

for i in range(n):
    for j in range(n):
        erroMax = numpy.absolute(erro[:,i,j]).max()
        plt.plot(erro[:,i,j])
        plt.grid(True)
        plt.title("i = "+str(i)+", j = "+str(j)+"; erroMax = {:.4E}".format(erroMax))
        plt.show()