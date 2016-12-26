# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:33:27 2016

@author: levi
"""

# a file for solving the first example (9.1) in Miele (1970):

# f(x) = 1 + x1**2 + x2**2 + u**2
# dx1 = u - x1**2
# dx2 = u - x1  *x2

# s.t. x1(0) = 0; x1(1) = 1
#      x2(0) = 1; x2(1) = 2

import numpy
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from numpy.linalg import norm
from utils import interpV, interpM, ddt
from prob import declProb, calcPhi, calcPsi, calcF, calcGrads

#from scipy.integrate import quad

# ##################
# SOLUTION DOMAIN:
# ##################

def calcLamDotGrad(lam,t,tVec,fxInv,phixInv):
	fxt = interpV(t,tVec,fxInv)
	phixt = interpM(t,tVec,phixInv)

	return phixt.dot(lam) - fxt

def calcLamDotRest(lam,t,tVec,phixInv):
	phixt = interpM(t,tVec,phixInv)
	return phixt.dot(lam)

def calcADotGrad(A,t,tVec,phixVec,phiuVec,B):
	#print("In calcADot!")
	phixt = interpM(t,tVec,phixVec)
	phiut = interpM(t,tVec,phiuVec)

	Bt = interpV(t,tVec,B)
	#print('phixt =',phixt)
	#print('phiut =',phiut)
	#print('Bt =',Bt)
	
	return phixt.dot(A) + phiut.dot(Bt)

def calcADotRest(A,t,tVec,phixVec,phiuVec,B,aux):
	#print("In calcADot!")
	phixt = interpM(t,tVec,phixVec)
	phiut = interpM(t,tVec,phiuVec)
	auxt = interpV(t,tVec,aux)
	Bt = interpV(t,tVec,B)
	#print('phixt =',phixt)
	#print('phiut =',phiut)
	#print('Bt =',Bt)
	
	return phixt.dot(A) + phiut.dot(Bt) + auxt

def calcI(sizes,x,u):
	N = sizes['N']
	f = calcF(sizes,x,u)
	I = f.sum() - .5*(f[0]+f[N-1])
	I *= 1.0/(N-1)
	return I

def calcP(sizes,x,u):
	# MISSING: PSI term

	N = sizes['N']
	phi = calcPhi(sizes,x,u)
	dx = ddt(sizes,x)
	#dx = x.copy()
	P = 0.0	
	for t in range(1,N):
	#	dx[t,:] = x[t,:]-x[t-1,:]
		P += norm(dx[t,:]-phi[t,:])**2
	P *= 1.0/N
	return P
	
def calcQ(sizes,x,u,lam,mu):
	# MISSING: PSI term and derivatives with PI
	N = sizes['N']

	# get gradients
	Grads = calcGrads(sizes,x,u)
	phix = Grads['phix']
	phiu = Grads['phiu']	
	fx = Grads['fx']
	fu = Grads['fu']	
	psix = Grads['psix']
	
	dlam = ddt(sizes,lam)
	Q = 0.0	
	for k in range(1,N):
#		dlam[k,:] = lam[k,:]-lam[k-1,:]
		Q += norm(dlam[k,:] - fx[k,:] + phix[k,:,:].transpose().dot( lam[k,:]))**2
		Q += norm(fu[k,:]-phiu[k,:,:].transpose().dot(lam[k,:]))**2
	Q *= 1.0/N
	
	Q += norm(lam[N-1,:]+psix.transpose().dot(mu))
	return Q
		
def grad(sizes,x,u,t,Q0):
	print("In grad.")
	
	print("Q0 =",Q0)
	# get sizes
	N = sizes['N']
	n = sizes['n']	
	m = sizes['m']
	q = sizes['q']
	
	# get gradients
	Grads = calcGrads(sizes,x,u)
	
	phix = Grads['phix']	
	phiu = Grads['phiu']	
	fx = Grads['fx']
	fu = Grads['fu']
	psix = Grads['psix']	

	psixTr = psix.transpose()
	fxInv = fx.copy()
	phixInv = phix.copy()
	phiuTr = numpy.empty((N,m,n))
	for k in range(N):
		fxInv[k,:] = fx[N-k-1,:]
		phixInv[k,:,:] = phix[N-k-1,:,:].transpose()	
		phiuTr[k,:,:] = phiu[k,:,:].transpose()
	
	mu = numpy.zeros(q)
	M = numpy.ones((q+1,q+1))	
	
	arrayA = numpy.empty((q+1,N,n))
	arrayB = numpy.empty((q+1,N,m))
	arrayL = arrayA.copy()
	arrayM = numpy.empty((q+1,q))
	for i in range(q+1):		
		mu = 0.0*mu
		if i<q:
			mu[i] = 1.0
		
		#print("mu =",mu)
		# integrate equation (38) backwards		
		auxLamInit = - psixTr.dot(mu)
		auxLam = odeint(calcLamDotGrad,auxLamInit,t,args=(t,fxInv, phixInv))
		
		B = -fu
		lam = auxLam.copy()
		for k in range(N):
			lam[k,:] = auxLam[N-k-1,:]
			B[k,:] += phiuTr[k,:,:].dot(lam[k,:])
		
#		plt.plot(t,lam)
#		plt.grid(True)
#		plt.xlabel("t")
#		plt.ylabel("lambda")
#		plt.show()
#		# C = 0.0 (unnecessary)
#
#		plt.plot(t,B)
#		plt.grid(True)
#		plt.xlabel("t")
#		plt.ylabel("B")
#		plt.show()		
#		
		A = odeint(calcADotGrad,numpy.zeros(n),t,args=(t,phix,phiu,B))

#		plt.plot(t,A)
#		plt.grid(True)
#		plt.xlabel("t")
#		plt.ylabel("A")
#		plt.show()		
		
		arrayA[i,:,:] = A
		arrayB[i,:,:] = B
		arrayL[i,:,:] = lam
		arrayM[i,:] = mu
		M[1:,i] = psixTr.dot(A[N-1,:])
	#
		
	# Calculations of weights k:

	col = numpy.zeros(q+1)
	col[0] = 1.0
	K = numpy.linalg.solve(M,col)
	print("K =",K)
	
	# summing up linear combinations
	A = 0.0*A	
	B = 0.0*B
	lam = 0.0*lam
	mu = 0.0*mu
	for i in range(q+1):
		A += K[i]*arrayA[i,:,:]
		B += K[i]*arrayB[i,:,:]
		lam += K[i]*arrayL[i,:,:]
		mu += K[i]*arrayM[i,:]
	
	# TODO: calculation of alfa
#	alfa = 1.0e-3

	nx = x.copy()
	nu = u.copy()
	alfa = 1.0
	nx += alfa * A
	nu += alfa * B
	Q = calcQ(sizes,nx,nu,lam,mu)
	print("Q =",Q)
	while Q > Q0:
		alfa *= .5
		nx = x + alfa * A
		nu = u + alfa * B
		Q = calcQ(sizes,nx,nu,lam,mu)
		print("alfa =",alfa,"Q =",Q)
		
	print("alfa =",alfa)	
	
	# incorporation of A, B into the solution
#	x += alfa * A
#	u += alfa * B

	print("Leaving grad.")
	return nx,nu,lam,mu,Q

def rest(sizes,x,u,t):
	print("In rest.")
	
	P0 = calcP(sizes,x,u)
	print("P0 =",P0)	
	
	# get sizes
	N = sizes['N']
	n = sizes['n']	
	m = sizes['m']
	q = sizes['q']

	phi = calcPhi(sizes,x,u)	
	psi = calcPsi(sizes,x)

	aux = phi.copy()
	aux -= ddt(sizes,x)

#	aux = numpy.empty((N,n))	
#	dt = 1.0/N
#	for k in range(1,N):
#		aux[k,:] = phi[k,:] - (x[k,:]-x[k-1,:])/dt	
	
	# get gradients
	Grads = calcGrads(sizes,x,u)
	
	phix = Grads['phix']	
	phiu = Grads['phiu']	
	fx = Grads['fx']
	#fu = Grads['fu']
	psix = Grads['psix']	

	psixTr = psix.transpose()
	fxInv = fx.copy()
	phixInv = phix.copy()
	phiuTr = numpy.empty((N,m,n))

	# column vector for linear system involving k's	
	col = numpy.zeros(q+1)
	col[0] = 1.0
	col[1:] = -psi
	
	for k in range(N):
		fxInv[k,:] = fx[N-k-1,:]
		phixInv[k,:,:] = phix[N-k-1,:,:].transpose()	
		phiuTr[k,:,:] = phiu[k,:,:].transpose()
	
	mu = numpy.zeros(q)
	M = numpy.ones((q+1,q+1))	
	
	arrayA = numpy.empty((q+1,N,n))
	arrayB = numpy.empty((q+1,N,m))
	arrayL = arrayA.copy()
	arrayM = numpy.empty((q+1,q))
	for i in range(q+1):		
		mu = 0.0*mu
		if i<q:
			mu[i] = 1.0
		
		#print("mu =",mu)
		
		# integrate equation (75-2) backwards		
		auxLamInit = - psixTr.dot(mu)
		auxLam = odeint(calcLamDotRest,auxLamInit,t,args=(t,phixInv))
		
		B = numpy.empty((N,m))
		lam = auxLam.copy()
		for k in range(N):
			lam[k,:] = auxLam[N-k-1,:]
			B[k,:] = phiuTr[k,:,:].dot(lam[k,:])
		
#		plt.plot(t,lam)
#		plt.grid(True)
#		plt.xlabel("t")
#		plt.ylabel("lambda")
#		plt.show()
		# C = 0.0 (unnecessary)

#		plt.plot(t,B)
#		plt.grid(True)
#		plt.xlabel("t")
#		plt.ylabel("B")
#		plt.show()		
				
		A = odeint(calcADotRest,numpy.zeros(n),t,args=(t,phix,phiu,B, aux))

#		plt.plot(t,A)
#		plt.grid(True)
#		plt.xlabel("t")
#		plt.ylabel("A")
#		plt.show()		
		
		arrayA[i,:,:] = A
		arrayB[i,:,:] = B
		arrayL[i,:,:] = lam
		arrayM[i,:] = mu
		
		M[1:,i] = psixTr.dot(A[N-1,:])
	#
		
	# Calculations of weights k:

	K = numpy.linalg.solve(M,col)
	print("K =",K)
	
	# summing up linear combinations
	A = 0.0*A	
	B = 0.0*B
	lam = 0.0*lam
	mu = 0.0*mu
	for i in range(q+1):
		A += K[i]*arrayA[i,:,:]
		B += K[i]*arrayB[i,:,:]
		lam += K[i]*arrayL[i,:,:]
		mu += K[i]*arrayM[i,:]
	
	# TODO: calculation of alfa
#	alfaHigh = 1.0
#	alfaLow = 0.0
#	alfa = 1.0e-1#3
	
	nx = x.copy()
	nu = u.copy()
	
	alfa = 1.0
	nx += alfa * A
	nu += alfa * B
	while calcP(sizes,nx,nu) > P0:
		alfa *= .5
		nx = x + alfa * A
		nu = u + alfa * B
		
	print("alfa =",alfa)
	# incorporation of A, B into the solution
#	x += alfa * A
#	u += alfa * B

	print("Leaving rest.")
	return nx,nu,lam,mu

def plotSol(t,x,u,lam,mu):
	P = calcP(sizes,x,u)
	Q = calcQ(sizes,x,u,lam,mu)
	I = calcI(sizes,x,u)
	plt.subplot(2,1,1)
	plt.plot(t,x)
	plt.grid(True)
	plt.xlabel("t")
	plt.ylabel("x")
	plt.title("P = {:.4E}".format(P)+", Q = {:.4E}".format(Q)+", I = {:.4E}".format(I))
	plt.subplot(2,1,2)
	plt.plot(t,u)
	plt.grid(True)
	plt.xlabel("t")
	plt.ylabel("u")
	plt.show()
#


# ##################
# MAIN SEGMENT:
# ##################

# declare problem:
sizes,t,x,u,lam,mu = declProb()
Grads = calcGrads(sizes,x,u)
phiu = Grads['phiu']
plotSol(t,x,u,lam,mu)

#P = calcP(sizes,x,u)
#print("P =",P)

## first restoration step:

while calcP(sizes,x,u) > 1e-6:
	x,u,lam,mu = rest(sizes,x,u,t)
	plotSol(t,x,u,lam,mu)

Q = calcQ(sizes,x,u,lam,mu)
# first gradient step:
while Q > 5e-4:
	while calcP(sizes,x,u) > 1e-6:
		x,u,lam,mu = rest(sizes,x,u,t)
		plotSol(t,x,u,lam,mu)
	x,u,lam,mu,Q = grad(sizes,x,u,t,Q)
	plotSol(t,x,u,lam,mu)
