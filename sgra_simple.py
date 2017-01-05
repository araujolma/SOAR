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
from prob import declProb, calcPhi, calcPsi, calcF, calcGrads, calcI

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

def calcADotGrad(A,t,tVec,phixVec,phiuVec,phipVec,B,C):
	#print("In calcADot!")
	phixt = interpM(t,tVec,phixVec)
	phiut = interpM(t,tVec,phiuVec)
	phipt = interpM(t,tVec,phipVec)
	Bt = interpV(t,tVec,B)
	
	#print('phixt =',phixt)
	#print('phiut =',phiut)
	#print('Bt =',Bt)
	
	return phixt.dot(A) + phiut.dot(Bt) + phipt.dot(C)

def calcADotRest(A,t,tVec,phixVec,phiuVec,phipVec,B,C,aux):
	#print("In calcADot!")
	phixt = interpM(t,tVec,phixVec)
	phiut = interpM(t,tVec,phiuVec)
	phipt = interpM(t,tVec,phipVec)
	auxt = interpV(t,tVec,aux)
	Bt = interpV(t,tVec,B)
	#print('phixt =',phixt)
	#print('phiut =',phiut)
	#print('Bt =',Bt)
	
	return phixt.dot(A) + phiut.dot(Bt) + phipt.dot(C) + auxt

def calcP(sizes,x,u,pi):

	N = sizes['N']
	phi = calcPhi(sizes,x,u,pi)
	psi = calcPsi(sizes,x)
	dx = ddt(sizes,x)
	#dx = x.copy()
	P = 0.0	
	for t in range(1,N-1):
	#	dx[t,:] = x[t,:]-x[t-1,:]
		P += norm(dx[t,:]-phi[t,:])**2
	P += .5*norm(dx[0,:]-phi[0,:])**2
	P += .5*norm(dx[N-1,:]-phi[N-1,:])**2
	
	P *= 1.0/(N-1)
	
	P += norm(psi)
	return P
	
def calcQ(sizes,x,u,pi,lam,mu):
	# Q expression from (15)

	N = sizes['N']
	p = sizes['p']
	dt = 1.0/(N-1)
	
	# get gradients
	Grads = calcGrads(sizes,x,u,pi)
	phix = Grads['phix']
	phiu = Grads['phiu']
	phip = Grads['phip']	
	fx = Grads['fx']
	fu = Grads['fu']	
	fp = Grads['fp']
	psix = Grads['psix']
	#psip = Grads['psip']
	
	dlam = ddt(sizes,lam)
	Qx = 0.0	
	Qu = 0.0
	auxVecIntQ = numpy.zeros(p)
	for k in range(1,N-1):
#		dlam[k,:] = lam[k,:]-lam[k-1,:]
		Qx += norm(dlam[k,:] - fx[k,:] + phix[k,:,:].transpose().dot( lam[k,:]))**2
		Qu += norm(fu[k,:]-phiu[k,:,:].transpose().dot(lam[k,:]))**2
		auxVecIntQ += fp[k,:] - phip[k,:,:].transpose().dot(lam[k,:])
	#
		
	Qx += .5*(norm(dlam[0,:] - fx[0,:] + phix[0,:,:].transpose().dot( lam[0,:]))**2)
	Qx += .5*(norm(dlam[N-1,:] - fx[N-1,:] + phix[N-1,:,:].transpose().dot( lam[N-1,:]))**2)

	Qu += .5*norm(fu[0,:]-phiu[0,:,:].transpose().dot(lam[0,:]))**2
	Qu += .5*norm(fu[N-1,:]-phiu[N-1,:,:].transpose().dot(lam[N-1,:]))**2
	
	Qx *= dt
	Qu *= dt
	
	auxVecIntQ += .5*(fp[0,:] - phip[0,:,:].transpose().dot(lam[0,:]))
	auxVecIntQ += .5*(fp[N-1,:] - phip[N-1,:,:].transpose().dot(lam[N-1,:]))	
	
	auxVecIntQ *= dt
	Qp = norm(auxVecIntQ)
	Qt = norm(lam[N-1,:] + psix.transpose().dot(mu))
#"Qx =",Qx)

	Q = Qx + Qu + Qp + Qt
	print("Q = {:.4E}".format(Q)+": Qx = {:.4E}".format(Qx)+", Qu = {:.4E}".format(Qu)+", Qp = {:.4E}".format(Qp)+", Qt = {:.4E}".format(Qt))

	return Q

def calcStepGrad(x,u,pi,lam,mu,A,B,C):

#	alfa = 1.0	
#	nx = x + alfa * A
#	nu = u + alfa * B
#	np = pi + alfa * C
#	
#	Q0 = calcQ(sizes,nx,nu,np,lam,mu)
#	print("Q =",Q0)
#
#	Q = Q0
#	alfa = .8
#	nx = x + alfa * A
#	nu = u + alfa * B
#	np = pi + alfa * C
#	nQ = calcQ(sizes,nx,nu,np,lam,mu)
#	cont = 0
#	while (nQ-Q)/Q < -.05 and alfa > 1.0e-11 and cont < 5:
#		cont += 1
#		Q = nQ
#		alfa *= .5
#		nx = x + alfa * A
#		nu = u + alfa * B
#		np = pi + alfa * C
#		nQ = calcQ(sizes,nx,nu,np,lam,mu)
#		print("alfa =",alfa,"Q =",nQ)
#	
#	if Q>Q0:
#		alfa = 1.0

	# "Trissection" method
	alfa = 1.0	
	nx = x + alfa * A
	nu = u + alfa * B
	np = pi + alfa * C
	
	oldQ = calcQ(sizes,nx,nu,np,lam,mu)
	nQ = .9*oldQ
#	print("Q =",Q)
	alfaMin = 0.0
	alfaMax = 1.0
	cont = 0		
	while (nQ-oldQ)/oldQ < -.05 and cont < 5:
		oldQ = nQ
		
		dalfa = (alfaMax-alfaMin)/3.0
		alfaList = numpy.array([alfaMin,alfaMin+dalfa,alfaMax-dalfa,alfaMax])	
		QList = numpy.empty(numpy.shape(alfaList))

		for j in range(4):
			alfa = alfaList[j]
			
			nx = x + alfa * A
			nu = u + alfa * B
			np = pi + alfa * C
	
			Q = calcQ(sizes,nx,nu,np,lam,mu)
			QList[j] = Q
		#
		print("QList:",QList)
		minQ = QList[0]
		indxMinQ = 0
		
		for j in range(1,4):
			if QList[j] < minQ:
				indxMinQ = j
				minQ = QList[j]
		#
		
		alfa = alfaList[indxMinQ]
		nQ = QList[indxMinQ]
		print("nQ =",nQ)
		if indxMinQ == 0:
			alfaMin = alfaList[0]
			alfaMax = alfaList[1]
		elif indxMinQ == 1:
			if QList[0] < QList[2]:
				alfaMin = alfaList[0]
				alfaMax = alfaList[1]
			else:
				alfaMin = alfaList[1]
				alfaMax = alfaList[2]
		elif indxMinQ == 2:
			if QList[1] < QList[3]:
				alfaMin = alfaList[1]
				alfaMax = alfaList[2]
			else:
				alfaMin = alfaList[2]
				alfaMax = alfaList[3]
		elif indxMinQ == 3:
			alfaMin = alfaList[2]
			alfaMax = alfaList[3]
			
		cont+=1
	#
		
	return .5*(alfaMin+alfaMax)
		
def grad(sizes,x,u,pi,t,Q0):
	print("In grad.")
	
	print("Q0 =",Q0)
	# get sizes
	N = sizes['N']
	n = sizes['n']	
	m = sizes['m']
	p = sizes['p']
	q = sizes['q']
	
	# get gradients
	Grads = calcGrads(sizes,x,u,pi)
	
	phix = Grads['phix']	
	phiu = Grads['phiu']	
	phip = Grads['phip']
	fx = Grads['fx']
	fu = Grads['fu']
	fp = Grads['fp']
	psix = Grads['psix']	
	psip = Grads['psip']
	dt = Grads['dt']
	
	# prepare time reversed/transposed versions of the arrays
	psixTr = psix.transpose()
	fxInv = fx.copy()
	phixInv = phix.copy()
	phiuTr = numpy.empty((N,m,n))
	phipTr = numpy.empty((N,p,n))
	for k in range(N):
		fxInv[k,:] = fx[N-k-1,:]
		phixInv[k,:,:] = phix[N-k-1,:,:].transpose()	
		phiuTr[k,:,:] = phiu[k,:,:].transpose()
		phipTr[k,:,:] = phip[k,:,:].transpose()
	psipTr = psip.transpose()
	
	# Prepare array mu and arrays for linear combinations of A,B,C,lam
	mu = numpy.zeros(q)
	M = numpy.ones((q+1,q+1))	
	
	arrayA = numpy.empty((q+1,N,n))
	arrayB = numpy.empty((q+1,N,m))
	arrayC = numpy.empty((q+1,p))
	arrayL = arrayA.copy()
	arrayM = numpy.empty((q+1,q))
	
	for i in range(q+1):		
		mu = 0.0*mu
		if i<q:
			mu[i] = 1.0
		
		#print("mu =",mu)
		# integrate equation (38) backwards for lambda		
		auxLamInit = - psixTr.dot(mu)
		auxLam = odeint(calcLamDotGrad,auxLamInit,t,args=(t,fxInv, phixInv))
		
		# Calculate B
		B = -fu
		lam = auxLam.copy()
		for k in range(N):
			lam[k,:] = auxLam[N-k-1,:]
			B[k,:] += phiuTr[k,:,:].dot(lam[k,:])

		# Calculate C		
		C = numpy.zeros(p)
		for k in range(1,N-1):
			C += fp[k,:] - phipTr[k,:,:].dot(lam[k,:])
		C += .5*(fp[0,:] - phipTr[0,:,:].dot(lam[0,:]))
		C += .5*(fp[N-1,:] - phipTr[N-1,:,:].dot(lam[N-1,:]))
		C *= -dt
		C -= -psipTr.dot(mu) 
		
#		plt.plot(t,B)
#		plt.grid(True)
#		plt.xlabel("t")
#		plt.ylabel("B")
#		plt.show()		
#		
		# integrate diff equation for A
		A = odeint(calcADotGrad,numpy.zeros(n),t, args=(t,phix,phiu,phip,B,C))

#		plt.plot(t,A)
#		plt.grid(True)
#		plt.xlabel("t")
#		plt.ylabel("A")
#		plt.show()		
		
		arrayA[i,:,:] = A
		arrayB[i,:,:] = B
		arrayC[i,:] = C
		arrayL[i,:,:] = lam
		arrayM[i,:] = mu
		M[1:,i] = psix.dot(A[N-1,:]) + psip.dot(C)
	#
		
	# Calculations of weights k:

	col = numpy.zeros(q+1)
	col[0] = 1.0
	K = numpy.linalg.solve(M,col)
	print("K =",K)
	
	# summing up linear combinations
	A = 0.0*A	
	B = 0.0*B
	C = 0.0*C
	lam = 0.0*lam
	mu = 0.0*mu
	for i in range(q+1):
		A += K[i]*arrayA[i,:,:]
		B += K[i]*arrayB[i,:,:]
		C += K[i]*arrayC[i,:]
		lam += K[i]*arrayL[i,:,:]
		mu += K[i]*arrayM[i,:]
	
	# Calculation of alfa
	
	alfa = calcStepGrad(x,u,pi,lam,mu,A,B,C)
		
	nx = x + alfa * A
	nu = u + alfa * B
	np = pi + alfa * C
	Q = calcQ(sizes,nx,nu,np,lam,mu)
		
	print("Leaving grad with alfa =",alfa)	
	
	return nx,nu,np,lam,mu,Q

def calcStepRest(x,u,pi,A,B,C):

	alfa = 1.0	
	nx = x + alfa * A
	nu = u + alfa * B
	np = pi + alfa * C
	
	P0 = calcP(sizes,nx,nu,np)
	print("P =",P0)

	P = P0
	alfa = .8
	nx = x + alfa * A
	nu = u + alfa * B
	np = pi + alfa * C
	nP = calcP(sizes,nx,nu,np)
	cont = 0
	while (nP-P)/P < -.05 and alfa > 1.0e-11 and cont < 5:
		cont += 1
		P = nP
		alfa *= .5
		nx = x + alfa * A
		nu = u + alfa * B
		np = pi + alfa * C
		nP = calcP(sizes,nx,nu,np)
		print("alfa =",alfa,"P =",nP)
		
	return alfa

def rest(sizes,x,u,pi,t):
	print("In rest.")
	
	P0 = calcP(sizes,x,u,pi)
	print("P0 =",P0)	
	
	# get sizes
	N = sizes['N']
	n = sizes['n']	
	m = sizes['m']
	p = sizes['p']
	q = sizes['q']

	# calculate phi and psi
	phi = calcPhi(sizes,x,u,pi)	
	psi = calcPsi(sizes,x)

	# aux: phi - dx/dt
	aux = phi.copy()
	aux -= ddt(sizes,x)

	# get gradients
	Grads = calcGrads(sizes,x,u,pi)
	
	dt = Grads['dt']
	
	phix = Grads['phix']	
	phiu = Grads['phiu']	
	phip = Grads['phip']
	fx = Grads['fx']
	psix = Grads['psix']	
	psip = Grads['psip']
	
	psixTr = psix.transpose()
	fxInv = fx.copy()
	phixInv = phix.copy()
	phiuTr = numpy.empty((N,m,n))
	phipTr = numpy.empty((N,p,n))
	psipTr = psip.transpose()
		
	for k in range(N):
		fxInv[k,:] = fx[N-k-1,:]
		phixInv[k,:,:] = phix[N-k-1,:,:].transpose()	
		phiuTr[k,:,:] = phiu[k,:,:].transpose()
		phipTr[k,:,:] = phip[k,:,:].transpose()
	
	mu = numpy.zeros(q)
	
	# Matrix for linear system involving k's
	M = numpy.ones((q+1,q+1))	

	# column vector for linear system involving k's	[eqs (88-89)]
	col = numpy.zeros(q+1)
	col[0] = 1.0 # eq (88)
	col[1:] = -psi # eq (89)
	
	arrayA = numpy.empty((q+1,N,n))
	arrayB = numpy.empty((q+1,N,m))
	arrayC = numpy.empty((q+1,p))
	arrayL = arrayA.copy()
	arrayM = numpy.empty((q+1,q))
	
	for i in range(q+1):		
		mu = 0.0*mu
		if i<q:
			mu[i] = 1.0
				
		# integrate equation (75-2) backwards		
		auxLamInit = - psixTr.dot(mu)
		auxLam = odeint(calcLamDotRest,auxLamInit,t,args=(t,phixInv))
		
		# equation for Bi (75-3)
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
		
		# equation for Ci (75-4)
		C = numpy.zeros(p)
		for k in range(1,N-1):
			C += phipTr[k,:,:].dot(lam[k,:])
		C += .5*(phipTr[0,:,:].dot(lam[0,:]))
		C += .5*(phipTr[N-1,:,:].dot(lam[N-1,:]))
		C *= dt
		C -= -psipTr.dot(mu)
		
		# integrate equation for A:				
		A = odeint(calcADotRest,numpy.zeros(n),t,args= (t,phix,phiu,phip,B,C,aux))

		# store solution in arrays
		arrayA[i,:,:] = A
		arrayB[i,:,:] = B
		arrayC[i,:] = C
		arrayL[i,:,:] = lam
		arrayM[i,:] = mu
		
		# Matrix for linear system (89)
		M[1:,i] = psix.dot(A[N-1,:])
		M[1:,i] += psip.dot(C)#psip * C
	#
		
	# Calculations of weights k:

	K = numpy.linalg.solve(M,col)
	print("K =",K)
	
	# summing up linear combinations
	A = 0.0*A	
	B = 0.0*B
	C = 0.0*C
	lam = 0.0*lam
	mu = 0.0*mu
	for i in range(q+1):
		A += K[i]*arrayA[i,:,:]
		B += K[i]*arrayB[i,:,:]
		C += K[i]*arrayC[i,:]
		lam += K[i]*arrayL[i,:,:]
		mu += K[i]*arrayM[i,:]
	
#	alfa = 1.0#2.0#
	alfa = calcStepRest(x,u,p,A,B,C)
	nx = x + alfa * A
	nu = u + alfa * B
	np = pi + alfa * C
#	while calcP(sizes,nx,nu,np) > P0:
#		alfa *= .8#.5
#		nx = x + alfa * A
#		nu = u + alfa * B
#		np = pi + alfa * C
#	print("alfa =",alfa)
	
	# incorporation of A, B into the solution
#	x += alfa * A
#	u += alfa * B

	print("Leaving rest with alfa =",alfa)	
	return nx,nu,np,lam,mu

def plotSol(t,x,u,pi,lam,mu):
	P = calcP(sizes,x,u,pi)
	Q = calcQ(sizes,x,u,pi,lam,mu)
	I = calcI(sizes,x,u,pi)
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
	print("pi =",pi)#, ", lambda =",lam,", mu =",mu)
#


# ##################
# MAIN SEGMENT:
# ##################

# declare problem:
sizes,t,x,u,pi,lam,mu,tol = declProb()
Grads = calcGrads(sizes,x,u,pi)
phix = Grads['phix']
phiu = Grads['phiu']
psix = Grads['psix']
psip = Grads['psip']

print("Proposed initial guess:")
plotSol(t,x,u,pi,lam,mu)

tolP = tol['P']
tolQ = tol['Q']

# first restoration step:

while calcP(sizes,x,u,pi) > tolP:
	x,u,pi,lam,mu = rest(sizes,x,u,pi,t)
	#plotSol(t,x,u,pi,lam,mu)

print("\nAfter first rounds of restoration:")
plotSol(t,x,u,pi,lam,mu)

Q = calcQ(sizes,x,u,pi,lam,mu)
# first gradient step:
while Q > tolQ:
	while calcP(sizes,x,u,pi) > tolP:
		x,u,pi,lam,mu = rest(sizes,x,u,pi,t)
		plotSol(t,x,u,pi,lam,mu)
	x,u,pi,lam,mu,Q = grad(sizes,x,u,pi,t,Q)
	plotSol(t,x,u,pi,lam,mu)
