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
from scipy.integrate import odeint
from numpy.linalg import norm
import matplotlib.pyplot as plt
#from scipy.integrate import quad
	

def interpV(t,tVec,xVec):
#	print("In interpV!")
#	print(numpy.shape(tVec))
#	print(numpy.shape(xVec))	
	#print("t =",t,"tVec =",tVec,"xVec =",xVec)
#	xVec = numpy.squeeze(xVec)
	
	Nsize = xVec.shape[1]

	ans = numpy.empty(Nsize)
	for k in range(Nsize):
		ans[k] = numpy.interp(t,tVec,xVec[:,k])
	return ans

def interpM(t,tVec,xVec):

	ni = xVec.shape[1]
	nj = xVec.shape[2]

	ans = numpy.empty((ni,nj))
	for i in range(ni):
		for j in range(nj):
			ans[i,j] = numpy.interp(t,tVec,xVec[:,i,j])
	return ans

def calcLamDot(lam,t,tVec,fxInv,phixInv):
	fxt = interpV(t,tVec,fxInv)
	phixt = interpM(t,tVec,phixInv)

	return fxt + phixt.dot(lam)
	
def calcADot(A,t,tVec,phixVec,phiuVec,B):
	#print("In calcADot!")
	phixt = interpM(t,tVec,phixVec)
	phiut = interpM(t,tVec,phiuVec)

	Bt = interpV(t,tVec,B)
	#print('phixt =',phixt)
	#print('phiut =',phiut)
	#print('Bt =',Bt)
	
	return phixt.dot(A) + phiut.dot(Bt)

def calcPhi(sizes,x,u):
	# calculate phi:
	phi = numpy.zeros((N,n))
	for k in range(N):
		phi[k,0] = u[k] - x[k,0]**2
		phi[k,1] = u[k] -x[k,0]*x[k,1]
	return phi

def calcF(sizes,x,u):
	N = sizes['N']
	f = numpy.empty(N)
	
	for k in range(N):
		f[k] = 1.0 + x[k,0]**2 + x[k,1]**2 + u[k]**2
	return f

def calcP(sizes,x,u):
	N = sizes['N']
	phi = calcPhi(sizes,x,u)
	dx = x.copy()
	P = 0.0
	for t in range(1,N):
		dx[t,:] = x[t,:]-x[t-1,:]
		P += norm(dx[t,:]-phi[t,:])**2
	P *= 1.0/N
	return P
	
#def calcQ(sizes,)

def calcGrads(sizes,x,u):
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

	fx = numpy.zeros((N,n))
	fu = numpy.zeros((N,m))		
		
	for k in range(N):
		phix[k,:,:] = numpy.array([[0.0,-2.0*x[k,0]],[-x[k,1],-x[k,0]]])
		phiu[k,:,:] = numpy.array([[1.0],[1.0]])
		fx[k,:] = numpy.array([2.0*x[k,0],2.0*x[k,1]])		
	
	Grads['phix'] = phix
	Grads['phiu'] = phiu
	Grads['phip'] = phip
	Grads['fx'] = fx
	Grads['fu'] = fu
#	Grads['gx'] = gx
#	Grads['gp'] = gp
#	Grads['psix'] = psix
#	Grads['psip'] = psip		
	
	return Grads
		
def grad(sizes,x,u,t):
	Grads = calcGrads(sizes,x,u)
	
	phix = Grads['phix']	
	phiu = Grads['phiu']	
	fx = Grads['fx']
	fu = Grads['fu']
	
	q = sizes['q']
	N = sizes['N']
	mu = numpy.zeros(q)
	for i in range(q+1):		
		mu = 0*mu
		if i<q:
			mu[i] = 1.0
		
		# integrate equation (38) backwards
					
		
		#lam[N-1] = -gx - mu*psix
		auxLamInit = numpy.zeros(n)#-gx -mu*psix
		fxInv = fx.copy()
		phixInv = phix.copy()
		for k in range(N):
			fxInv[k,:] = fx[N-k-1,:]
			phixInv[k,:,:] = phix[N-k-1,:,:].transpose()
			
		auxLam = odeint(calcLamDot,auxLamInit,t,args=(t,fxInv,phixInv))
		
		B = -fu
		lam = auxLam.copy()
		for k in range(N):
			lam[k,:] = auxLam[N-k-1,:]
			B[k,:] += phiu[k,:].transpose().dot(lam[k,:])
		
		plt.plot(t,lam)
		plt.grid(True)
		plt.xlabel("t")
		plt.ylabel("lambda")
		plt.show()
		# C = 0.0 (unnecessary)
		
		A = odeint(calcADot,numpy.zeros(n),t,args=(t,phix,phiu,B))
	
		# no k-calculation step, for q = 0...
	
	alfa = 1.0e-3
	x += alfa * A
	u += alfa * B
	
	P = calcP(sizes,x,u)
	print("P =",P)	
	return x,u


N = 100 + 1
dt = 1.0/(N-1)
t = numpy.arange(0,1.0+dt,dt)

n = 2
x = numpy.zeros((N,n))

m = 1
u = numpy.ones((N,m))

p = 0
q = 0

sizes = dict()
sizes['N'] = N
sizes['n'] = n
sizes['m'] = m
sizes['p'] = p 
sizes['q'] = q

x[:,0] = t.copy()
x[:,1] = 1.0+t.copy()

# calculate error P:

P = calcP(sizes,x,u)
print("P =",P)


plt.plot(t,x)
plt.grid(True)
plt.xlabel("t")
plt.ylabel("x")
plt.show()


# calculate error Q:

#Q = calcQ(sizes,)

# FIRST GRADIENT STEP:

x,u = grad(sizes,x,u,t)

f = calcF(sizes,x,u)
plt.plot(t,f)
plt.grid(True)
plt.xlabel("t")
plt.ylabel("f")
plt.show()