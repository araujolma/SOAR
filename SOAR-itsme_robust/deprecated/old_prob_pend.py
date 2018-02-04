#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:23:17 2017

@author: levi
"""

import numpy
import matplotlib.pyplot as plt
#from atmosphere import rho

# ##################
# PROBLEM DOMAIN:
# ##################
def declProb(opt=dict()):
# time discretization
    N = 2000 + 1#20000 + 1 #
    dt = 1.0/(N-1)
    t = numpy.arange(0,1.0+dt,dt)


# matrix sizes
    n = 2
    m = 1
    p = 1
    q = 2  # (Miele 1970)  # 7 (Miele 2003)

# tolerances
    tolP = 1.0e-7
    tolQ = 1.0e-5

# prepare sizes
    sizes = dict()
    sizes['N'] = N
    sizes['n'] = n
    sizes['m'] = m
    sizes['p'] = p
    sizes['q'] = q

#prepare tolerances
    tol = dict()
    tol['P'] = tolP
    tol['Q'] = tolQ

    pi = 1.0*numpy.ones(p)

    x = numpy.zeros((N,n))
#    x[:,0] = .01*numpy.sin(100*2*numpy.pi*t)
    # initial guesses with frequencies less than 100 end up converging!
    x[:,0] = .01*numpy.sin(200*2*numpy.pi*t)
    
    u = numpy.zeros((N,m))


    #x = .5*numpy.pi*numpy.ones((N,n))
    #x[:,0] = .5*numpy.pi*numpy.arange(0.0,1.0+dt,dt)
    #u = numpy.zeros((N,m))
    
    lam = x.copy()
    mu = numpy.zeros(q)
    constants = dict()
    boundary = dict()
    restrictions = dict()
    return sizes,t,x,u,pi,lam,mu,tol,constants,boundary,restrictions


def calcPhi(sizes,x,u,pi,constants,restrictions):
    N = sizes['N']
    n = sizes['n']

    # calculate phi:
    phi = numpy.empty((N,n))

    # dr/dt = pi * v
    # dv/dt = pi * u
    phi[:,0] = pi[0] * x[:,1]
    phi[:,1] = pi[0] * (numpy.tanh(u[:,0])-numpy.sin(x[:,0]))#numpy.sin(u[:,0])

    return phi

def calcPsi(sizes,x,boundary):
    N = sizes['N']

    psi = numpy.array([x[N-1,0]-.5*numpy.pi,x[N-1,1]])

    return psi

def calcF(sizes,x,u,pi,constants,restrictions):
    N = sizes['N']

    f = pi[0]*numpy.ones(N)

    return f

def calcGrads(sizes,x,u,pi,constants,restrictions):
    Grads = dict()

    N = sizes['N']
    n = sizes['n']
    m = sizes['m']
    p = sizes['p']
    #q = sizes['q']

    # Pre-assign functions
    sin = numpy.sin
    cos = numpy.cos
    tanh = numpy.tanh
    array = numpy.array

    Grads['dt'] = 1.0/(N-1)

    phix = numpy.zeros((N,n,n))
    phiu = numpy.zeros((N,n,m))
    phip = numpy.zeros((N,n,p))

    fx = numpy.zeros((N,n))
    fu = numpy.zeros((N,m))
    fp = numpy.ones((N,p))
    
    psix = array([[1.0,0.0],[0.0,1.0]])
    psip = array([[0.0],[0.0]])

    for k in range(N):
        phix[k,:,:] = pi[0]*array([[0.0, 1.0],
                                   [-cos(x[k,0]), 0.0]])

#        phiu[k,:,:] = pi[0]*array([[0.0],
#                                   [cos(u[k])]])
#
#        phip[k,:,:] = array([[x[k,1]],
#                             [sin(u[k])]])

        phiu[k,:,:] = pi[0]*array([[0.0],
                                   [1.0-tanh(u[k])**2]])

        phip[k,:,:] = array([[x[k,1]],
                             [tanh(u[k])-sin(x[k,0])]])
    Grads['phix'] = phix
    Grads['phiu'] = phiu
    Grads['phip'] = phip
    Grads['fx'] = fx
    Grads['fu'] = fu
    Grads['fp'] = fp
#    Grads['gx'] = gx
#    Grads['gp'] = gp
    Grads['psix'] = psix
    Grads['psip'] = psip
    return Grads

def calcI(sizes,x,u,pi,constants,restrictions):
    N = sizes['N']
    f = calcF(sizes,x,u,pi,constants,restrictions)
    I = .5*(f[0]+f[N-1])
    I += f[1:(N-1)].sum()
    I *= 1.0/(N-1)

    return I

def plotSol(sizes,t,x,u,pi,constants,restrictions,opt=dict()):

    plt.subplot2grid((8,4),(0,0),colspan=5)
    plt.plot(t,x[:,0]*180/numpy.pi,)
    plt.grid(True)
    plt.ylabel("theta (deg)")
    if opt.get('mode','sol') == 'sol':
        I = calcI(sizes,x,u,pi,constants,restrictions)
        titlStr = "Current solution: I = {:.4E}".format(I)
        if opt.get('dispP',False):
            P = opt['P']
            titlStr = titlStr + " P = {:.4E} ".format(P)
        if opt.get('dispQ',False):
            Q = opt['Q']
            titlStr = titlStr + " Q = {:.4E} ".format(Q)
    elif opt['mode'] == 'var':
        titlStr = "Proposed variations"
    else:
        titlStr = opt['mode']
    #
    plt.title(titlStr)
    plt.subplot2grid((8,4),(1,0),colspan=5)
    plt.plot(t,x[:,1]*180/numpy.pi,'g')
    plt.grid(True)
    plt.ylabel("omega (deg/s)")
    plt.subplot2grid((8,4),(2,0),colspan=5)
    plt.plot(t,u[:,0],'k')
    plt.grid(True)
    plt.ylabel("u1 [-]")
    plt.subplot2grid((8,4),(3,0),colspan=5)
    plt.plot(t,numpy.tanh(u[:,0]),'r')
    plt.grid(True)
    plt.ylabel("contr [-]")

    plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
    plt.show()
    print("pi =",pi,"\n")
#
