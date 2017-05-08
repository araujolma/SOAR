# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:02:30 2017

@author: araujolma
"""

import numpy
import matplotlib.pyplot as plt
#from atmosphere import rho

# ##################
# PROBLEM DOMAIN:
# ##################
def declProb(opt=dict()):
# time discretization
    N = 1000 + 1#20000 + 1 #
    dt = 1.0/(N-1)
    t = numpy.arange(0,1.0+dt,dt)


# matrix sizes
    n = 3
    m = 1
    p = 1
    q = 1  # (Miele 1970)  # 7 (Miele 2003)

# tolerances
    tolP = 1.0e-8
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

    x = numpy.zeros((N,n))
    #u = numpy.ones((N,m))
    u = numpy.zeros((N,m))

    #x[:,0] = t.copy()
    pi = numpy.ones(p)

    lam = x.copy()
    mu = numpy.zeros(q)
    constants = dict()
    boundary = dict()
    restrictions = dict()
    return sizes,t,x,u,pi,lam,mu,tol,constants,boundary,restrictions


def calcPhi(sizes,x,u,pi,constants,restrictions):
    N = sizes['N']
    n = sizes['n']

    sin = numpy.sin
    cos = numpy.cos

    # calculate phi:
    phi = numpy.empty((N,n))

    # example rocket single stage to orbit with Lift and Drag
    phi[:,0] = pi[0] * x[:,2] * cos(u[:,0])
    phi[:,1] = pi[0] * x[:,2] * sin(u[:,0])
    phi[:,2] = pi[0] * sin(u[:,0])

    return phi

def calcPsi(sizes,x,boundary):
    N = sizes['N']

    psi = numpy.array([x[N-1,0]-1.0])

    return psi

def calcF(sizes,x,u,pi,constants,restrictions):
    N = sizes['N']

    f = numpy.ones(N)

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
    array = numpy.array

    Grads['dt'] = 1.0/(N-1)

    phix = numpy.zeros((N,n,n))
    phiu = numpy.zeros((N,n,m))
    phip = numpy.zeros((N,n,p))

    fx = numpy.zeros((N,n))
    fu = numpy.zeros((N,m))
    fp = numpy.ones((N,p))

    # Gradients from example rocket single stage to orbit with Lift and Drag
    psix = array([[1.0,0.0,0.0]])
    psip = array([[0.0]])

    # atmosphere: numerical gradient

    for k in range(N):
        sinUk = sin(u[k,0])
        cosUk = cos(u[k,0])
        phix[k,:,:] = pi[0]*array([[0.0, 0.0, cosUk],
                                   [0.0, 0.0, sinUk],
                                   [0.0, 0.0, 0.0]])

        phiu[k,:,:] = pi[0]*array([[-x[k,2]*sinUk],
                                   [x[k,2]*cosUk],
                                   [cosUk]])

        phip[k,:,:] = array([[x[k,2]*cosUk],
                             [x[k,2]*sinUk],
                             [sinUk]])

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

    # example rocket single stage to orbit with Lift and Drag
    f = calcF(sizes,x,u,pi,constants,restrictions)
    I = f.sum()

    return I

def plotSol(sizes,t,x,u,pi,constants,restrictions,opt=dict()):

    plt.subplot2grid((8,4),(0,0),colspan=5)
    plt.plot(t,x[:,0],)
    plt.grid(True)
    plt.ylabel("x")
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
    plt.plot(t,x[:,1],'g')
    plt.grid(True)
    plt.ylabel("y")
    plt.subplot2grid((8,4),(2,0),colspan=5)
    plt.plot(t,x[:,2],'r')
    plt.grid(True)
    plt.ylabel("z")
    plt.subplot2grid((8,4),(3,0),colspan=5)
    plt.plot(t,u[:,0],'k')
    plt.grid(True)
    plt.ylabel("u1 [-]")

    plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
    plt.show()
    print("pi =",pi,"\n")
#
