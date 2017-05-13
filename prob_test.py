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

    pi = 4.0*numpy.ones(p)
    
    x = numpy.zeros((N,n))
    u = .5*numpy.pi*numpy.ones((N,m))
    x[500:750,1] = t[0:250]
    x[500:750,0] = .5*t[0:250]*t[0:250]
    for i in range(250):
        x[750+i,1] = t[250-i]   
        u[750+i] = -.5*numpy.pi
        x[750+i,0] = -.5*t[250-i]*t[250-i] + 2*x[749,0]
    u[0:500] = 0
    
    x *= pi 
    
    u[1000] = -.5*numpy.pi
    x[1000,0] = 1.0
    x[1000,1] = 0.0
    
    
    
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

    # calculate phi:
    phi = numpy.empty((N,n))

    # dr/dt = pi * v
    # dv/dt = pi * u
    phi[:,0] = pi[0] * x[:,1]
    phi[:,1] = pi[0] * sin(u[:,0])

    return phi

def calcPsi(sizes,x,boundary):
    N = sizes['N']

    psi = numpy.array([x[N-1,0]-1.0,x[N-1,1]])

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
    array = numpy.array

    Grads['dt'] = 1.0/(N-1)

    phix = numpy.zeros((N,n,n))
    phiu = numpy.zeros((N,n,m))
    phip = numpy.zeros((N,n,p))

    fx = numpy.zeros((N,n))
    fu = numpy.zeros((N,m))
    fp = numpy.ones((N,p))

    # Gradients from example rocket single stage to orbit with Lift and Drag
    psix = array([[1.0,0.0],[0.0,1.0]])
    psip = array([[0.0],[0.0]])

    # atmosphere: numerical gradient

    for k in range(N):
        phix[k,:,:] = pi[0]*array([[0.0, 1.0],
                                   [0.0, 0.0]])

        phiu[k,:,:] = pi[0]*array([[0.0],
                                   [cos(u[k])]])

        phip[k,:,:] = array([[x[k,1]],
                             [sin(u[k])]])

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
    plt.plot(t,u[:,0],'k')
    plt.grid(True)
    plt.ylabel("u1 [-]")

    plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
    plt.show()
    print("pi =",pi,"\n")
#
