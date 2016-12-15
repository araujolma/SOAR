# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:33:27 2016

@author: levi
"""

# a file for solving the first example (9.1) in Miele (1970):

# f(x) = 1 + x1**2 + x2**2 + u**2
# dx1 = u - x1**2
# dx2 = u - x1  *x2

import numpy
from scipy.integrate import odeint

def interpV(t,tVec,xVec):
	dims = xVec.shape[1:]
	Nsize = numpy.prod(dims)
	ans = numpy.zeros(Nsize)
	for k in range(Nsize):
		ans[k] = numpy.interp(t,tVec,xVec[:,k])
	
	return numpy.reshape(ans,dims)

def calcLamDot(lam,t,tVec,fxVec,phixVec):
	fxt = interpV(t,tVec,fxVec)
	phixt = interpV(t,tVec,phixVec)
	
	return fxt + phixt.dot(lam)
	

def calcGrads(sizes,x,u):
		print("In calcGrads.")
		Grads = dict()
		
		N = sizes['N']
		n = sizes['n']
		m = sizes['m']
		p = sizes['p']

		phix = numpy.zeros((N,n,n))
		phiu = numpy.zeros((N,n,m))
		
				
		if p>0:
			phip = numpy.zeros((N,n,p))
		else:
			phip = numpy.zeros((N,n,1))

		
		for t in range(N):
			phix[t,:,:] = numpy.array([[0,-2.0*x1[t]],[-x1[t],-x0[t]]])
			phiu[t,:,:] = numpy.array([[1.0],[1.0]])
		
		print("phix =",phix,"\n\nphiu =",phiu)
		Grads['phix'] = phix
		Grads['phiu'] = phiu
		Grads['phip'] = phip
		Grads['gx'] = gx
		Grads['gp'] = gp
		Grads['psix'] = psix
		Grads['psip'] = psip		
		
		return Grads

N = 1000
t = numpy.arange(0,N,1.0)/N

n = 2
x = numpy.zeros((N,n))

m = 1
u = numpy.ones((N,m))

p = 0

sizes = dict()
sizes['N'] = N
sizes['n'] = n
sizes['m'] = m
sizes['p'] = p 

x[:,0] = t.copy()
x[:,1] = 1.0+t.copy()


# FIRST GRADIENT STEP:

Grads = calcGrads(sizes,x,u)

