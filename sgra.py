# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 08:07:34 2016

@author: levi
"""

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

class Miss:
	def __init__(self):
		
		self.InitStt = numpy.array([0,1])
		self.FinlStt = numpy.array([1,2])		
		self.NMissCons = 4		
		print("Mission initialized.")
class Sol:
	def __init__(self,thisMiss,s=0):
		# initialize sizes and controls

		# number of states
		n = 2
		
		# number of controls
		m = 1 
		
		# number of parameters		
		p = s 
		
		# number of constraints
		q = thisMiss.NMissCons + 1 #+ n * (s-1) 	
		
		# time discretization
		N = 10  		
		
		self.s = s
		self.n = n
		self.m = m
		self.p = p
		self.q = q
		self.N = N
		
#		# staging information:
#		self.efes = .9*numpy.ones(self.s)
#		self.Isp = 450.0*numpy.ones(self.s)


		# initialization of states, controls, etc
		self.x = numpy.zeros((n,N))
		
#		self.dx = numpy.zeros((N,n))
		self.u = numpy.zeros((m,N))
		self.pi = numpy.ones(p)		
		self.lam = numpy.zeros((n,N))
#		self.dlam = numpy.zeros((N,n))
		self.mu = numpy.ones(q)
		self.timeVec = numpy.arange(0,1,N)
		
		# initialize errors and tolerances:
		self.P = 1.0
		self.Q = 1.0
		
		self.epsP = 1.0e-6
		self.epsQ = 1.0e-6
		
		print("Solution initialized.")	
		
	def grad(self):
		print("In grad.")
		# get gradients of f, g, phi and psi
		Grads = self.calcGrads()
		
		self.intgDiff(Grads)
		
		# fx,fu,fp,phix,phiu,phip,gx,gp,psix,psip =
		
		# Integrate differential system (32), (33), (34-2) for 
		# Ai, Bi, lambdai, Ci, mui

		# Obtain constants ki through eqs (46)-(47)
		
		# Obtain A, B, lambda, C, mu through (44)-(45),
		
		# Compute stepsize alfa
		
		# Compute corrections Dx, Du, Dp (31)
		
		# Obtain xt, ut, pt (18)
		
		if self.calcP() > self.epsP:
			self.rest()
		print("Leaving grad.")
		
	def rest(self):
		print("In rest.")

		while (self.calcP() > self.epsP):
			# compute dxt - phit		
		
			# get gradients of f, g, phi and psi
			phix,phiu,phip,gx,gp,psix,psip = self.calcGrads()
			
			
			# Integrate differential system (75), (76), (77-2) for 
			# Ai, Bi, lambdai, Ci, mui
	
			# Obtain constants ki (88)-(89)
			
			# Obtain At, Bt, lambdat, Ct, mut (86)-(87)
			
			# Compute stepsize alfa
			
			# Compute corrections Dxt, Dut, Dpt with (74)
			
			# Obtain xch, uch, pch (60) 
		
		if self.calcI() < Iant:
			self.grad()
		else:
			self.dummy()
			#If Ineq. (107) is violated, return to the previous 
			# gradient phase and reduce the gradient stepsize 
			# alfa until, after restoration, Ineq. (107) is satisfied.	
			
		print("Leaving rest.")
#	def dx():
#		dx = 
#		return
	def calcGrads(self):
		print("In calcGrads.")
		Grads = dict()
		N = self.N
		n = self.n
		m = self.m
		p = self.p
		x = self.x
		x0 = x[0,:]
		x1 = x[1,:]

		phix = numpy.zeros((N,n,n))
		phiu = numpy.zeros((N,n,m))
		
		gx = numpy.zeros(n)
		gp = []
		psix = numpy.zeros(n)
		psip = numpy.zeros(1)
				
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
	def intgDiff(self,Grads):
		q = self.q
		N = self.N		

		gx = Grads['gx']
		psix = Grads['psix']		
		fx = Grads['fx']
		phix = Grads['phix']		
		
		mu = numpy.zeros(q)
		for i in range(q+1):		
			mu = 0*mu
			if i<q:
				mu[i] = 1.0
			
			# integrate equation (38) backwards
						
			
			#lam[N-1] = -gx - mu*psix
			auxLamInit = -gx -mu*psix
			fxInv = fx.copy()
			phixInv = phix.copy()
			for k in range(N):
				fxInv[k,:,:] = fx[N-k-1,:,:]
				phixInv[k,:,:] = phix[N-k-1,:,:]
				
			auxLam = odeint(calcLamDot,auxLamInit,self.timeVec,args=(fxInv,phixInv))
			
		
	def calcP(self):
		# this method calculates P (associated with current solution)
	
		return self.P
		
	def calcQ(self):
		
		return self.Q		

	def main(self):
		print("In main.")
		
		# compute constraint error:

		
		if currSol.calcP() < currSol.epsP:
			currSol.grad()
		else:
			currSol.rest()
			
#		while (currSol.calcQ > currSol.epsQ) and \
#		(currSol.calcP > currSol.epsP):
#			if currSol.P < currSol.epsP:
#				currSol.grad()
#			else:
#				currSol.rest()
		
			

# initialize mission:
thisMiss = Miss()

print("Teste.")
# initialize solution:
currSol = Sol(thisMiss)


#currSol.calcGrads()
#currSol.main()

## compute constraint error:
#
#if currSol.calcP < currSol.epsP:
#	# step1
#	while (currSol.P > currSol.epsP) and (currSol.Q > currSol.epsQ):
#		currSol.grad()
#else:
#	# step2
#	currSol.rest()


# check Q novamente

# gradient-restoration cycle
# while 

