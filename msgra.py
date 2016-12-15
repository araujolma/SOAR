# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 08:07:34 2016

@author: levi
"""

import numpy
from scipy.integrate import odeint

class Miss:
	def __init__(self):
		self.InitAlt = 0.0
		self.InitVel = 0.0
		self.InitGamma = .5*numpy.pi()
		
		self.FinlAlt = 463.0
		self.FinlVel = 7633.0
		self.FinlGamma = 0.0
		
		self.NMissCons = 6
		

class Sol:
	def __init__(self,thisMiss,s=3):
		# initialize sizes and controls

		# number of states
		n = 4 
		
		# number of controls
		m = 2 
		
		# number of parameters		
		p = s 
		
		# number of constraints
		q = thisMiss.NMissCons + 1 + n * (s-1) 	
		
		# time discretization
		N = 1000  		
		
		self.s = s
		self.n = n
		self.m = m
		self.p = p
		self.q = q
		self.N = N
		
		# staging information:
		self.efes = .9*numpy.ones(self.s)
		self.Isp = 450.0*numpy.ones(self.s)


		# initialization of states, controls, etc
		self.x = numpy.zeros((N,n))
#		self.dx = numpy.zeros((N,n))
		self.u = numpy.zeros((N,m))
		self.pi = numpy.ones(p)		
		self.lam = numpy.zeros((N,n))
#		self.dlam = numpy.zeros((N,n))
		self.mu = numpy.ones(q)
		
		# initialize errors and tolerances:
		self.P = 1.0
		self.Q = 1.0
		
		self.epsP = 1.0e-6
		self.epsQ = 1.0e-6
		
	def grad(self):
		
		# solve LMPBVP for functions A, B, C, lam, mu; rho=1
		
		
		self.calcQ()		
		if self.Q > self.epsQ:
			# the state, control, and parameter are updated via (14)
			# A,B,C = 1.0			
			
			# find alfa via directional search, 
			# J˜ < J; P˜ <= P*
			# alfa = ...

	def rest(self):
		# solve LMPBVP for functions A, B, C, lam, mu; rho=0
		# the state, control, and parameter are updated via (14)
		# A,B,C = 1.0			
	
		# find alfa via bisection process from alfa=1, 
		# J˜ < J; P˜ <= P*
		alfa = 1.0
		Pant = self.P
		while currSol.P > Pant:
			#bisection process
		
		# alfa = ...
		
			
#	def dx():
#		dx = 
#		return
	def calcP(self):
		# this method calculates P (associated with current solution)
	
		return self.P
		
	def calcQ(self):
		return 0.0		

	def main(self):
		# compute constraint error:

		while (currSol.calcQ > currSol.epsQ) and (currSol.calcP > currSol.epsP)
			if currSol.P < currSol.epsP:
				currSol.grad()
			else:
				currSol.rest()
		
			

# initialize mission:
thisMiss = Miss()

# initialize solution:
currSol = Sol(thisMiss,s=1)


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

