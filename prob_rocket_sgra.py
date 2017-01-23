# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:02:30 2017

@author: munizlgmn
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
    
# example rocket single stage to orbit L=0 D=0
    n = 4
    m = 2
    p = 1 
    q = 3 # (Miele 1970)  # 7 (Miele 2003)     
    grav_e = 9.8     # m/s^2
    Thrust = 500000  # N
    Isp = 450        # s
    r_e = 6371       # km
    GM = 398600.4415 # km^3 s^-2
    s_f = 0.1
    
# prepare sizes
    sizes = dict()
    sizes['N'] = N
    sizes['n'] = n
    sizes['m'] = m
    sizes['p'] = p 
    sizes['q'] = q

# prepare constants
    constants = dict()
    constants['grav_e'] = grav_e
    constants['Thrust'] = Thrust
    constants['Isp'] = Isp
    constants['r_e'] = r_e
    constants['GM'] = GM
    constants['s_f'] = s_f
    
# initial guess:

# example rocket single stage to orbit L=0 D=0
    x = numpy.zeros((N,n))
#    x[:,0] = t.copy()
#    x[:,1] = 1.0+t.copy()
#    x[:,2] = 
#    x[:,3] = 
    u = numpy.ones((N,m))
    lam = 0.0*x.copy()
    mu = numpy.zeros(q)
    pi = numpy.ones((p,1))

    tol = dict()
    tol['P'] = 1.0e-8
    tol['Q'] = 1.0e-5
    return sizes,t,x,u,pi,lam,mu,tol
    
def calcPhi(sizes,x,u,pi):
    N = sizes['N']
    n = sizes['n']
    grav_e = constants['grav_e']
    Thrust = constants['Thrust']
    Isp = constants['Isp']
    r_e = constants['r_e']
    GM = constants['GM']
       
# calculate phi:
    phi = numpy.empty((N,n))

# calculate r and grav:
    r = numpy.empty(N)
    grav = numpy.empty(N)
    r = r_e + x[:,0]
    grav = GM/(r**2)
    
# example rocket single stage to orbit L=0 D=0     
    phi[:,0] = pi[0] * x[:,1] * sin(x[:,2])
    phi[:,1] = (pi[0] * u[:,1] * Thrust * cos(u[:,0]))/(x[:,3]) - grav * sin(x[:,2])
    phi[:,2] = pi[0] * ((u[:,1] * Thrust * sin(u[:,0]))/(x[:,3] * x[:,1]) - cos(x[:,2]) * ((x[:,1]/r) - (grav/x[:,1])))
    phi[:,3] = - (pi[0] * u[:,1] * Thrust)/(grav_e * Isp)

    return phi
    
def calcPsi(sizes,x):    
    N = sizes['N']
  
# example rocket single stage to orbit L=0 D=0
    psi = numpy.array([x[N-1,0]-463.0,x[N-1,1]-7633.0,x[N-1,2]])

    return psi
    
def calcF(sizes,x,u,pi):
    N = sizes['N']
    f = numpy.empty(N)
    grav_e = constants['grav_e']
    Thrust = constants['Thrust']
    Isp = constants['Isp']
    s_f = constants['s_f']
    
    for k in range(N):
# example rocket single stage to orbit L=0 D=0
        f[k] = ((Thrust * pi[0])/(grav_e * (1-s_f) * Isp)) * u[k,2]
   
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
        # Gradients from example rocket single stage to orbit L=0 D=0
        psix = numpy.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0]])        
        psip = numpy.array([[0.0],[0.0],[0.0]])  
#        cosuk = numpy.cos(u[k,0])
#        sinuk = numpy.sin(u[k,0])
#        zk = x[k,2]
#        phix[k,:,:] = pi[0]*numpy.array([[0.0,0.0,cosuk],[0.0,0.0,sinuk],[0.0,0.0,0.0]])
#        phiu[k,:,:] = pi[0]*numpy.array([[-zk*sinuk],[zk*cosuk],[cosuk]])
#        phip[k,:,:] = numpy.array([[zk*cosuk],[zk*sinuk],[sinuk]])
#        fp[k,0] = 1.0    
    
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
# example rocket single stage to orbit L=0 D=0
    N = sizes['N']
    f = calcF(sizes,x,u,pi)
    I = f.sum()
    
    return I