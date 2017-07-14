#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct integration test

Created on Thu Apr 27 14:09:24 2017

@author: levi
"""
import numpy
import matplotlib.pyplot as plt
from prob_rocket_sgra import declProb, calcGrads, calcPhi
from sgra_simple_rocket_alt import calcP, plotSol
from atmosphere import rho

def calcXdot(sizes,x,u,pi,constants,restrictions):
    n = sizes['n']
    grav_e = constants['grav_e']
    Thrust = constants['Thrust']
    Isp = constants['Isp']
    r_e = constants['r_e']
    GM = constants['GM']
    CL0 = constants['CL0']
    CL1 = constants['CL1']
    CD0 = constants['CD0']
    CD2 = constants['CD2']
    s_ref = constants['s_ref']
    alpha_min = restrictions['alpha_min']
    alpha_max = restrictions['alpha_max']
    beta_min = restrictions['beta_min']
    beta_max = restrictions['beta_max']
    sin = numpy.sin
    cos = numpy.cos
    u1 = u[0]
    u2 = u[1]

    # calculate variables alpha and beta
    alpha = (alpha_max + alpha_min)/2 + sin(u1)*(alpha_max - alpha_min)/2
    beta = (beta_max + beta_min)/2 + sin(u2)*(beta_max - beta_min)/2

    # calculate variables CL and CD
    CL = CL0 + CL1*alpha
    CD = CD0 + CD2*(alpha)**2

    # calculate L and D
    
    dens = rho(x[0])
    pDynTimesSref = .5 * dens * (x[1]**2) * s_ref    
    L = CL * pDynTimesSref
    D = CD * pDynTimesSref
    
    # calculate r
    r = r_e + x[0]

    # calculate grav
    grav = GM/r/r

    # calculate phi:
    dx = numpy.empty(n)

    # example rocket single stage to orbit with Lift and Drag
    sinGama = sin(x[2])
    dx[0] = pi[0] * x[1] * sinGama
    dx[1] = pi[0] * ((beta * Thrust * cos(alpha) - D)/x[3] - grav * sinGama)
    dx[2] = pi[0] * ((beta * Thrust * sin(alpha) + L)/(x[3] * x[1]) + cos(x[2]) * ( x[1]/r  -  grav/x[1] ))
    dx[3] = - (pi[0] * beta * Thrust)/(grav_e * Isp)

    return dx

opt = dict()
opt['initMode'] = 'extSol'#'default'#'extSol'

optPlot = dict(); optPlot['mode'] = 'sol'
optPlot['dispP'] = True; optPlot['dispQ'] = False


# declare problem:
sizes,t,x0,u0,pi0,lam,mu,tol,constants,boundary,restrictions = declProb(opt)
P0,Pint0,Ppsi0 = calcP(sizes,x0,u0,pi0,constants,boundary,restrictions)
optPlot['P'] = P0
plotSol(sizes,t,x0,u0,pi0,constants,restrictions,optPlot)
print("P = {:.4E}".format(P0)+", Pint = {:.4E}".format(Pint0)+\
              ", Ppsi = {:.4E}".format(Ppsi0)+"\n")
phi = calcPhi(sizes,x0,u0,pi0,constants,restrictions)


Grads = calcGrads(sizes,x0,u0,pi0,constants,restrictions)
dt = Grads['dt']
#phix = Grads['phix']

N = sizes['N']
n = sizes['n']

x = numpy.empty((N,n))
for k in range(100):
    x[0,:] = x0[0,:].copy()
    for i in range(N-1):
#        derk = calcXdot(sizes,x[i,:],u0[i,:],pi0,constants,restrictions)
#        aux = x[i,:] + dt * derk
#        x[i+1,:] = x[i,:] + .5 * dt * (derk + \
#                     calcXdot(sizes,aux,u0[i+1,:],pi0,constants,restrictions)) 
        k1 = calcXdot(sizes,x[i,:],u0[i,:],pi0,constants,restrictions)
        k2 = calcXdot(sizes,x[i,:]+.5*dt*k1,.5*(u0[i,:]+u0[i+1,:]),pi0,constants,restrictions)
        k3 = calcXdot(sizes,x[i,:]+.5*dt*k2,.5*(u0[i,:]+u0[i+1,:]),pi0,constants,restrictions)
        k4 = calcXdot(sizes,x[i,:]+dt*k3,u0[i+1,:],pi0,constants,restrictions)
        x[i+1,:] = x[i,:] + dt * (k1+k2+k2+k3+k3+k4)/6 
    
    P,Pint,Ppsi = calcP(sizes,x,u0,pi0,constants,boundary,restrictions)
    optPlot['P'] = P
    plotSol(sizes,t,x,u0,pi0,constants,restrictions,optPlot)
    print("P = {:.4E}".format(P)+", Pint = {:.4E}".format(Pint)+\
                  ", Ppsi = {:.4E}".format(Ppsi)+"\n")
    phi = calcPhi(sizes,x,u0,pi0,constants,restrictions)
    input("This was k = "+str(k)+"...")

#for k in range(5):
#    x = x0.copy()
#    for i in range(N-1):
#        derk = phi[i,:]
#        x[i+1] = x[i] + dt * derk
#        phi = calcPhi(sizes,x,u0,pi0,constants,restrictions)
#        x[i+1] = x[i] + .5 * dt * (derk + phi[i+1,:]) 
#    
#    P,Pint,Ppsi = calcP(sizes,x,u0,pi0,constants,boundary,restrictions)
#    optPlot['P'] = P
#    plotSol(sizes,t,x,u0,pi0,constants,restrictions,optPlot)
#    print("P = {:.4E}".format(P)+", Pint = {:.4E}".format(Pint)+\
#                  ", Ppsi = {:.4E}".format(Ppsi)+"\n")
#    phi = calcPhi(sizes,x,u0,pi0,constants,restrictions)
#    input("This was k = "+str(k)+"...")