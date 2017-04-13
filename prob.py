# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 14:20:56 2016

@author: levi
"""

import numpy

# ##################
# PROBLEM DOMAIN:
# ##################
def declProb():
# time discretization    
    N = 5000 + 1    
    dt = 1.0/(N-1)
    t = numpy.arange(0,1.0+dt,dt)
    
# example 9.1
#    n = 2
#    m = 1
#    p = 0
#    q = 2

# example 9.2
#    n = 2
#    m = 1
#    p = 0
#    q = 2

# example 10.1
#    n = 2
#    m = 1
#    p = 1
#    q = 2

# example 10.2
    n = 3
    m = 1
    p = 1
    q = 1

# prepare sizes
    sizes = dict()
    sizes['N'] = N
    sizes['n'] = n
    sizes['m'] = m
    sizes['p'] = p 
    sizes['q'] = q
    
# initial guess:

# example 9.1
#    x = numpy.zeros((N,n))
#    x[:,0] = t.copy()
#    x[:,1] = 1.0+t.copy()
#    u = numpy.ones((N,m))
#    lam = 0.0*x.copy()
#    mu = numpy.zeros(q)
#    pi = numpy.array([1.0])

# example 9.2
#    x = numpy.zeros((N,n))
#    x[:,1] = 0.3*t
#    u = numpy.zeros((N,m))
#    lam = 0.0*x.copy()
#    mu = numpy.zeros(q)
#    pi = numpy.array([1.0])
      
# example 10.1
#    x = numpy.zeros((N,n))
#    x[:,0] = t.copy()
#    u = numpy.ones((N,m))
#    lam = 0.0*x.copy()
#    mu = numpy.zeros(q)
#    pi = numpy.array([1.0])
    
# example 10.2    
    x = numpy.zeros((N,n))
    x[:,0] = t.copy()
    u = numpy.ones((N,m))
    lam = 0.0*x.copy()
    mu = numpy.zeros(q)
    pi = numpy.array([1.0])
    
    tol = dict()
    tol['P'] = 1.0e-8
    tol['Q'] = 1.0e-5
    return sizes,t,x,u,pi,lam,mu,tol
    
def calcPhi(sizes,x,u,pi):
    N = sizes['N']
    n = sizes['n']

# calculate phi:
    phi = numpy.empty((N,n))
    
# example 9.1
#    phi[:,0] = u[:,0] - x[:,1]**2
#    phi[:,1] = u[:,0] - x[:,0]*x[:,1]
## Optimization test:    
##    for k in range(N):
##        phi[k,0] = u[k] - x[k,1]**2
##        phi[k,1] = u[k] -x[k,0]*x[k,1]    

# example 9.2
#    phi[:,0] = 2*sin(u[:,0]) - 1.0
#    phi[:,1] = x[:,0]
    
# example 10.1
#    phi[:,0] = pi[0] * u[:,0]
#    phi[:,1] = pi[0] * (x[:,0]**2 - u[:,0]**2)
## Optimization test:
##    for k in range(N):
##        phi[k,0] = pi[0] * u[k]
##        phi[k,1] = pi[0] * (x[k,0]**2 - u[k]**2)
    
# example 10.2
    sinu = numpy.sin(u[:,0])
    phi[:,0] = pi[0] * x[:,2] * numpy.cos(u[:,0])
    phi[:,1] = pi[0] * x[:,2] * sinu
    phi[:,2] = pi[0] * sinu

    return phi
    
def calcPsi(sizes,x):
    
    N = sizes['N']

# example 9.1
#    return numpy.array([x[N-1,0]-1.0,x[N-1,1]-2.0])    

# example 9.2
#    return numpy.array([x[N-1,0],x[N-1,1]-1.0]) 
    
# example 10.1
#    return numpy.array([x[N-1,0]-1.0,x[N-1,1]])
    
# example 10.2    
    return numpy.array([x[N-1,0]-1.0])
    
    
def calcF(sizes,x,u,pi):
    N = sizes['N']
    f = numpy.empty(N)
    for k in range(N):
    # example 9.1
#        f[k] = 1.0 + x[k,0]**2 + x[k,1]**2 + u[k]**2

    # example 9.2
#        f[k] = 2*cos(u[k])
        
    # example 10.1 or 10.2 
        f[k] = pi[0]   
    
    return f

def calcGrads(sizes,x,u,pi):
    Grads = dict()
        
    N = sizes['N']
    n = sizes['n']
    m = sizes['m']
    p = sizes['p']
    #q = sizes['q']
    
    Grads['dt'] = 1.0/(N-1)

    phix = numpy.zeros((N,n,n))
    phiu = numpy.zeros((N,n,m))
                    
    if p>0:
        phip = numpy.zeros((N,n,p))
    else:
        phip = numpy.zeros((N,n,1))

    fx = numpy.zeros((N,n))
    fu = numpy.zeros((N,m))
    fp = numpy.zeros((N,p))
    
    for k in range(N):
                
        # Gradients from example 9.1:   
#        psix = numpy.array([[1.0,0.0],[0.0,1.0]])
#        psip = numpy.array([0.0])
#        phix[k,:,:] = numpy.array([[0.0,-2.0*x[k,1]],[-x[k,1],-x[k,0]]])
#        phiu[k,:,:] = numpy.array([[1.0],[1.0]])
#        phip[k,:,:] = numpy.array([0.0])
#        fx[k,:] = numpy.array([2.0*x[k,0],2.0*x[k,1]])
#        fu[k,:] = numpy.array([2.0*u[k]])
#        fp[k,:] = numpy.array([0.0])
                
        # Gradients from example 9.2:
#        psix = numpy.array([[1.0,0.0],[0.0,1.0]])
#        psip = numpy.array([0.0])
#        phix[k,:,:] = numpy.array([[0.0,0.0],[1.0,0.0]])
#        phiu[k,:,:] = numpy.array([[2*cos(u[k])],[0.0]])
#        phip[k,:,:] = numpy.array([0.0])
#        fu[k,:] = numpy.array([-2*sin(u[k])]) 
#        fp[k,:] = numpy.array([0.0])
            
        # Gradients from example 10.1:
#        psix = numpy.array([[1.0,0.0],[0.0,1.0]])
#        psip = numpy.array([[0.0],[0.0]])
#        phix[k,:,:] = numpy.array([[0.0,0.0],[2.0*pi[0]*x[k,0],0.0]])
#        phiu[k,:,:] = numpy.array([[pi[0]],[-2.0*pi[0]*u[k]]])
#        phip[k,:,:] = numpy.array([[u[k,0]],[x[k,0]**2 - u[k,0]**2]])
#        fp[k,0] = 1.0        
        
        # Gradients from example 10.2:
        psix = numpy.array([[1.0,0.0,0.0]])        
        psip = numpy.array([0.0])  
        cosuk = numpy.cos(u[k,0])
        sinuk = numpy.sin(u[k,0])
        zk = x[k,2]
        phix[k,:,:] = pi[0]*numpy.array([[0.0,0.0,cosuk],[0.0,0.0,sinuk],[0.0,0.0,0.0]])
        phiu[k,:,:] = pi[0]*numpy.array([[-zk*sinuk],[zk*cosuk],[cosuk]])
        phip[k,:,:] = numpy.array([[zk*cosuk],[zk*sinuk],[sinuk]])
        fp[k,0] = 1.0    
    
    Grads['phix'] = phix.copy()
    Grads['phiu'] = phiu.copy()
    Grads['phip'] = phip.copy()
    Grads['fx'] = fx.copy()
    Grads['fu'] = fu.copy()
    Grads['fp'] = fp.copy()
#    Grads['gx'] = gx.copy()
#    Grads['gp'] = gp.copy()
    Grads['psix'] = psix.copy()
    Grads['psip'] = psip.copy()        
    
    return Grads
    

def calcI(sizes,x,u,pi):
    # example 9.1
#    N = sizes['N']
#    f = calcF(sizes,x,u,pi)
#    I = f.sum()
    
    # example 9.2
#    N = sizes['N']
#    f = calcF(sizes,x,u,pi)
#    I = f.sum()
   
    # example 10.1 or 10.2
    I = pi[0]
    
    return I