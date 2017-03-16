# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 21:28:20 2017

@author: levi
"""

import utils
from rockProp import getRockTraj
from prob_rocket_sgra import declProb, calcPhi, calcPsi, calcGrads, calcI
from sgra_simple_rocket import calcP, plotSol

opt = dict()
opt['initMode'] = 'extSol'

# declare problem:
sizes,t,x,u,pi,lam,mu,tol,constants = declProb(opt)


dx = utils.ddt(sizes,x)
phi = calcPhi(sizes,x,u,pi,constants)

#print("Proposed initial guess:")
#P = calcP(sizes,x,u,pi,constants)
#plotSol(sizes,t,x,u,pi,lam,mu,constants)
#print("P =",P)