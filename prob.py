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
	N = 1000 + 1
	
	dt = 1.0/(N-1)
	t = numpy.arange(0,1.0+dt,dt)

	n = 2
	m = 1
	p = 0
	q = 2

	# prepare sizes
	sizes = dict()
	sizes['N'] = N
	sizes['n'] = n
	sizes['m'] = m
	sizes['p'] = p 
	sizes['q'] = q
	
	# initial guess:
	x = numpy.zeros((N,n))
#	x[:,0] = t.copy()	
	
	x[:,0] = t.copy()
	x[:,1] = 1.0+t.copy()
	u = numpy.ones((N,m))
	lam = 0.0*x.copy()
	mu = numpy.zeros(q)
	return sizes,t,x,u,lam,mu
	
def calcPhi(sizes,x,u):
	N = sizes['N']
	n = sizes['n']
	# calculate phi:
	phi = numpy.zeros((N,n))
	for k in range(N):
		phi[k,0] = u[k] - x[k,0]**2#x[k,1]
		phi[k,1] = u[k] -x[k,0]*x[k,1]#u[k]#
	return phi
	
def calcPsi(sizes,x):
	N = sizes['N']
	#return numpy.array([x[N-1,0]-1.0,x[N-1,1]])
	return numpy.array([x[N-1,0]-1.0,x[N-1,1]-2.0])
	
def calcF(sizes,x,u):
	N = sizes['N']
	f = numpy.empty(N)
	
	for k in range(N):
		#f[k] = u[k]**2 #+ x[k,1]**2
		f[k] = 1.0 + x[k,0]**2 + x[k,1]**2 + u[k]**2
	return f

def calcGrads(sizes,x,u):
	Grads = dict()
		
	N = sizes['N']
	n = sizes['n']
	m = sizes['m']
	p = sizes['p']
	#q = sizes['q']

	phix = numpy.zeros((N,n,n))
	phiu = numpy.zeros((N,n,m))
					
	if p>0:
		phip = numpy.zeros((N,n,p))
	else:
		phip = numpy.zeros((N,n,1))

	fx = numpy.zeros((N,n))
	fu = numpy.zeros((N,m))		
		
	psix = numpy.array([[1.0,0.0],[0.0,1.0]])
		
	for k in range(N):
		#phix[k,:,:] = numpy.array([[0.0,1.0],[0.0,0.0]])
		phix[k,:,:] = numpy.array([[0.0,-2.0*x[k,0]],[-x[k,1],-x[k,0]]])
		#phiu[k,:,:] = numpy.array([[0.0],[1.0]])
		phiu[k,:,:] = numpy.array([[1.0],[1.0]])
		#fx[k,:] = numpy.array([0.0,2.0*x[k,1]])
		fx[k,:] = numpy.array([2.0*x[k,0],2.0*x[k,1]])		
		fu[k,:] = numpy.array([2.0*u[k]])
		#numpy.array([2.0*u[k]])
	
	Grads['phix'] = phix.copy()
	Grads['phiu'] = phiu.copy()
	Grads['phip'] = phip.copy()
	Grads['fx'] = fx.copy()
	Grads['fu'] = fu.copy()
#	Grads['gx'] = gx.copy()
#	Grads['gp'] = gp.copy()
	Grads['psix'] = psix.copy()
	#Grads['psip'] = psip.copy()		
	
	return Grads