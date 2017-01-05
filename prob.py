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
	N = 5000 + 1
	
	dt = 1.0/(N-1)
	t = numpy.arange(0,1.0+dt,dt)

	n = 3#2
	m = 1
	p = 1#0#
	q = 1#2

	# prepare sizes
	sizes = dict()
	sizes['N'] = N
	sizes['n'] = n
	sizes['m'] = m
	sizes['p'] = p 
	sizes['q'] = q
	
	# initial guess:
	
	# example of minimum time
	x = numpy.zeros((N,n))
	x[:,0] = t.copy()	
	#x[:,1] = numpy.zeros(N)	
	
#	x[:,0] = t.copy()
#	x[:,1] = 1.0+t.copy()
	

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
#	for k in range(N):
#		phi[k,0] = u[k] - x[k,0]**2#x[k,1]
#		phi[k,1] = u[k] -x[k,0]*x[k,1]#u[k]#	
#	
	# example of minimum time
#	phi[:,0] = pi[0]*x[:,1]
#	phi[:,1] = pi[0]*u[:,0]
	
	# example 10.1
#	phi[:,0] = pi[0] * u[:,0]
#	phi[:,1] = pi[0] * (x[:,0]**2 - u[:,0]**2)
#	for k in range(N):
#		phi[k,0] = pi[0] * u[k]
#		phi[k,1] = pi[0] * (x[k,0]**2 - u[k]**2)
	
	# example 10.2
	sinu = numpy.sin(u[:,0])
	phi[:,0] = pi[0] * x[:,2] * numpy.cos(u[:,0])
	phi[:,1] = pi[0] * x[:,2] * sinu
	phi[:,2] = pi[0] * sinu

	return phi
	
def calcPsi(sizes,x):
	N = sizes['N']
	return numpy.array([x[N-1,0]-1.0])
	#return numpy.array([x[N-1,0]-1.0,x[N-1,1]])
	#return numpy.array([x[N-1,0]-1.0,x[N-1,1]-2.0])
	
def calcF(sizes,x,u,pi):
	N = sizes['N']
	
	# example of minimum time
	f = numpy.empty(N)
	for k in range(N):
		f[k] = pi[0]
		#f[k] = u[k]**2 #+ x[k,1]**2
		#f[k] = 1.0 + x[k,0]**2 + x[k,1]**2 + u[k]**2	
		#f[k] = u[k]**2 + pi[0]
	
	#f = numpy.zeros(N)
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
		
	#psix = numpy.array([[1.0,0.0],[0.0,1.0]])
	#psip = numpy.array([[0.0],[0.0]])
		
	# Gradients from example 10.2:
	psix = numpy.array([[1.0,0.0,0.0]])		
	psip = numpy.array([0.0])		

	
	for k in range(N):
		#phix[k,:,:] = numpy.array([[0.0,1.0],[0.0,0.0]])
		
		#phix[k,:,:] = numpy.array([[0.0,-2.0*x[k,0]],[-x[k,1],-x[k,0]]])
		#phiu[k,:,:] = numpy.array([[0.0],[1.0]])
		#phiu[k,:,:] = numpy.array([[1.0],[1.0]])
		#phip[k,:,:] = numpy.array([[x[k,1]],[u[k]]])		
		
		#fx[k,:] = numpy.array([0.0,2.0*x[k,1]])
		
		#fx[k,:] = numpy.array([2.0*x[k,0],2.0*x[k,1]])		
		#fu[k,:] = numpy.array([2.0*u[k]])
		
		#fp[k,:] = numpy.array([0.0])
		#numpy.array([2.0*u[k]])
		
		
		# example of minimum time
		#phix[k,:,:] = numpy.array([[0.0,pi[0]],[0.0,0.0]])
		#phiu[k,:,:] = numpy.array([[0.0],[pi[0]]])
		#phip[k,:,:] = numpy.array([[x[k,1]],[u[k]]])
		#fu[k,:] = numpy.array([2.0*u[k,0]])
		#fp[k,:] = numpy.array([1.0])
		
		# Gradients from example 10.1:
#		phix[k,:,:] = numpy.array([[0.0,0.0],[2.0*pi[0]*x[k,0],0.0]])
#		phiu[k,:,:] = numpy.array([[pi[0]],[-2.0*pi[0]*u[k]]])
#		phip[k,:,:] = numpy.array([[u[k,0]],[x[k,0]**2 - u[k,0]**2]])
#		fp[k,0] = 1.0		
		
		# Gradients from example 10.2:
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
#	Grads['gx'] = gx.copy()
#	Grads['gp'] = gp.copy()
	Grads['psix'] = psix.copy()
	Grads['psip'] = psip.copy()		
	
	return Grads
	

def calcI(sizes,x,u,pi):
	# example of minimum time
	#N = sizes['N']
	#f = calcF(sizes,x,u,pi)
	#I = f.sum() - .5*(f[0]+f[N-1])
	#I *= 1.0/(N-1)
	#I += pi[0]
	
	# from example 10.1:
	I = pi[0]
	
	return I