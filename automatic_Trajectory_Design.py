# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 11:34:54 2017

@author: carlos
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 09:51:21 2017

@author: carlos

Status:
	Converging for several free parameters values
	However still presents some divergences
	Need more checks for bad conditioned settings

"""

import numpy
import matplotlib.pyplot as plt
from scipy.integrate import ode

# Global variables
GM = 398600.4415       # km^3 s^-2
R = 6371               # km
pi = numpy.pi

def main ():

	tol = 1e-5			# Tolerance factor

	# Free parameters
	h_final = 463.0 	# km
	Mu = 100.0	 	     # Payload mass [kg]		

################
#	Factors:

#	fator_V      # Ajust to find a final V
#	tf           # Ajust to find a final gamma
#	fdv1         # Ajust to find a final h

#	Errors:

#	(v - V_final)/0.01
#	(gamma - gamma_final)/0.01
#	(h - h_final)/10


################

	# Factors intervals
	fsup = numpy.array([1.5,600.0,1.6]) # Superior limit
	finf = numpy.array([0.5,400.0,1.3]) # Inferior limit

	# Initital guess
	factors = (fsup + finf)/2#numpy.array([fator_V,tf,tAoA])

	# Initital display of vehicle trajectory without orbital phase
	displayResults(factors,h_final,Mu,tol)

	# Automatic adjustament
	new_factors = bisection0(fsup,finf,h_final,Mu,tol)
		
	# Results with automatic adjustment 
	displayResults(new_factors,h_final,Mu,tol)

	return None


def bisection0(fsup,finf,h_final,Mu,tol):
	
	def bisection1(fsup,finf,f3,h_final,Mu,tol):

		####################################################################
		# Bissection speed and gamma loop	
	
		print("\n\####################################################\n\r")
		print("\n\rBisection1 initializing")	
	
		# Initializing parameters
		factors1 = (fsup + finf)/2		
		df = (fsup - finf)/10

		# Making the 3 factor variarions null		
		factors1[2] = f3 + 0.0
		df[2] = df[2]*0.0
				
		factors2 = factors1 + df
		errors1, tt, xx = trajectoryDesing(factors1,h_final,Mu,"design",tol)
		continuing = True
		count = 0
		Nmax = 50
		
		# Loop
		while continuing and (count < Nmax):
						
			# Error update
			errors2, tt, xx = trajectoryDesing(factors2,h_final,Mu,"design",tol)		
			
			# Still needs step and factors verifications like bissec0
			der = (factors2 - factors1)/(errors2 - errors1)				
			step = errors2*der
			step[2] = step[2]*0.0
			
			factors3 = factors2 - step		
			errors3, tt, xx = trajectoryDesing(factors3,h_final,Mu,"design",tol)
					
			verify = abs(errors3) < tol
			if verify[0] and verify[1]:
				continuing = False			
				
			else:
				errors1 = errors2
				errors2 = errors3
				factors1 = factors2
				factors2 = factors3
				count += 1

		if count == Nmax:
			print("\n\rBisec1 total iteractions: ", count," (max)")
		else:
			print("\n\rBisec1 total iteractions: ", count)	
		h,v,gama,M = numpy.transpose(xx)
		orbitResults(h,v,gama)
		
		errorh = errors3[2]
		
		return errorh,factors3
	
	##########################################################################
	# Bisection altitude loop
	print("\n\####################################################\n\r")
	print("\n\rBisection0 initializing")
	
	# Step limiter
	df = abs( (fsup[2] - finf[2])/10 )	
	
	# fators initilization
	step = df.copy();	
	f1 = (fsup[2] + finf[2])/2
	f2 = f1 + step
	e1,factors = bisection1(fsup,finf,f1,h_final,Mu,tol)
	count = 0.0	
	stop = False
	
	# Loop
	while not stop:		
		
		# bisection1: Error update from speed and gamma loop
		e2,factors = bisection1(fsup,finf,f2,h_final,Mu,tol)
		
		# Checkings
		
		# Division and step check
		de = (e2 - e1)
		if abs(de) < tol*1e-2:
			
			step = df
			
		else:		
			der = (f2 - f1)/de
			step = e2*der			
			if step > df:
				step = 0.0 + df
			elif step < -df:
				step = 0.0 - df
	
		# Factor definition and check
		f3 = f2 - step
		
		if f3 > fsup[2]:
			f3 = 0.0 + fsup[2]
		elif f3 < finf[2]:
			f3 = 0.0 + finf[2]
		
		# Parameters update
		f1 = f2.copy()
		f2 = f3.copy()
		e1 = e2.copy()
		count += 1

		# Display		
		print("\n\####################################################\n\r")
		print("\n\rBisec0 iteration: ",count)
		print("Errors:   ",e2)
		print("Sup limits: %6.4f, %6.4f, %6.4f" % (   fsup[0],   fsup[1],   fsup[2]))
		print("Factors   : %6.4f, %6.4f, %6.4f" % (factors[0],factors[1],factors[2]))
		print("Inf limits: %6.4f, %6.4f, %6.4f" % (   finf[0],   finf[1],   finf[2]))

		# Loop checks
		if count == 30:
			stop = True
		if abs(e2) < tol:
			stop = True	
			
	return factors
	
def trajectoryDesing(factors,h_final,Mu,typeResult,tol):	

	# example rocket single stage to orbit L=0 D=0
	# initial state condition
	h_initial = 0.0            # km
	V_initial = 1.0e-6         # km/s
	gamma_initial = 90*pi/180  # rad
	# Initial mass definied in bellow	
	
	# final state condition
	V_final = numpy.sqrt(GM/(R+h_final))   # km/s Circular velocity
	gamma_final = 0.0 # rad
		
	Isp = 450              # s
	efes = .95
	g0 = GM/(R**2) #9.8e-3   # [km s^-2] gravity acceleration on earth surface
	AoAmax = 3.0           # graus

	torb = 2*pi*(R + h_final)/V_final

	##########################################################################
     # Trajetory design parameters
	fator_V,tf,fdv1 = factors
	#fdv1 = 1.4 #Ajust to find a final h
	tAoA = 2.0        #Ajust to find a final h

	##########################################################################
	# Initial mass definition and thrust programm
	Dv1 = fdv1*numpy.sqrt(2.0*GM*(1/R - 1/(R+h_final)))
	Dv2 = V_final.copy()

	Dv2 = Dv2*fator_V
	LamMax = 1/(1-efes)
	Lam1 = numpy.exp(Dv1/g0/Isp)
	Lam2 = numpy.exp(Dv2/g0/Isp)
	
	Mp2 = (Lam2-1)*efes*Mu/(1 - Lam2*(1-efes))
	Mp1 = (Lam1-1)*efes*(Mu + (Mp2/efes))/(1 - Lam1*(1-efes))
	Mp = Mp1 + Mp2;
	Me = (1-efes)*Mp/efes
	M0 = Mu + Mp + Me

	T = 40.0e3 # thrust in N
	T *= 1.0e-3 # thrust in kg * km / s^2 [for compatibility purposes...]

	tb1 = Mp1 * g0 * Isp / T
	tb2 = Mp2 * g0 * Isp / T

	# thrust program
	#tabBeta = retPulse(tb1,(tf-tb2),1.0,0.0)
	tVec = numpy.array([tb1,(tf-tb2),tf,tf*1.1])
	vVec = numpy.array([1.0,0.0,1.0,0.0])
	tabBeta = retPulse2(tVec,vVec)

	##########################################################################
	# Attitude program definition
	# Chossing tAoA1 as a fraction of tf results in code bad behavior
	# So a fixed generic number is used
	tAoA1 = 4.4 # [s], maneuver initiates 4.4 seconds from lift off
	tAoA2 = tAoA1 + tAoA

	# Attitude program
	#tabAlpha = retPulse(tAoA1,tAoA2,0.0,-AoAmax*pi/180)
	tVec = numpy.array([tAoA1,tAoA2,tf])
	vVec = numpy.array([0.0,-AoAmax*pi/180,0.0])
	tabAlpha = retPulse2(tVec,vVec)

	##########################################################################
	#Integration

	# initial conditions
	t0 = 0.0
	x0 = numpy.array([h_initial,V_initial,gamma_initial,M0])

     # Integrator setting
     # ode set:
     #         atol: absolute tolerance
     #         rtol: relative tolerance
	ode45 = ode(mdlDer).set_integrator('dopri5',nsteps=1,atol = tol/10,rtol = tol/100)
	ode45.set_initial_value(x0, t0).set_f_params((tabAlpha,tabBeta,T,Isp,g0,R))

	# Phase times, incluiding the initial time in the begining
	
	if (typeResult == "orbital"):
	
		tphases = numpy.array([t0,tAoA1,tAoA2,tb1,(tf-tb2),tf,torb])
		
	else:
		
		tphases = numpy.array([t0,tAoA1,tAoA2,tb1,(tf-tb2),tf])
	
	
	if (typeResult == "design"):
		# Integration using rk45 separated by phases
		# Automatic multiphase integration
		# Light running
		tt,xx,tp,xp = totalIntegration(tphases,ode45,False)
		h,v,gamma,M = xx
		errors = ((v - V_final)/0.01, (gamma - gamma_final)/0.01, (h - h_final)/10)
		errors = numpy.array(errors)
		ans = errors, tt, xx		
		
	elif (typeResult == "plot") or (typeResult == "orbital"):		
		# Integration using rk45 separated by phases
		# Automatic multiphase integration
		# Full running
		print("\n\rDv =",Dv1,"Dv =",Dv2," Lam1 =",Lam1," Lam2 =",Lam2,"LamMax =",LamMax)
		print("\n\rMu =",Mu," Mp =",Mp," Me =",Me,"M0 =",M0,"\n\r")
		tt,xx,tp,xp = totalIntegration(tphases,ode45,True)
		uu = numpy.concatenate([tabAlpha.multValue(tt),tabBeta.multValue(tt)], axis=1)
		up = numpy.concatenate([tabAlpha.multValue(tp),tabBeta.multValue(tp)], axis=1)
		ans = (tt,xx,uu,tp,xp,up)				
		
	return ans

def totalIntegration(tphases,ode45,flagAppend):

	def phaseIntegration(t_initial,t_final,Nref,ode45,tt,xx,tp,xp,flagAppend):
	
		tph = t_final - t_initial
		ode45.first_step = tph/Nref     
		stop1 = False
		while not stop1:
			ode45.integrate(t_final)
			if flagAppend:
				tt.append(ode45.t)
				xx.append(ode45.y)
			if ode45.t >= t_final:
				stop1 = True
		if flagAppend:
			tp.append(ode45.t)
			xp.append(ode45.y)
		else:
			tt = ode45.t
			xx = ode45.y
			
		return tt,xx,tp,xp

	Nref = 5.0 # Number of interval divisions for determine first step 	
	# Output variables
	tt,xx,tp,xp = [],[],[],[]

	for ii in range(1,len(tphases)):
		tt,xx,tp,xp = phaseIntegration(tphases[ii - 1],tphases[ii],Nref,ode45,tt,xx,tp,xp,flagAppend)
		if flagAppend:
			print("Phase integration iteration:",ii)

	tt = numpy.array(tt)
	xx = numpy.array(xx)
	tp = numpy.array(tp)
	xp = numpy.array(xp) 		
		
	return tt,xx,tp,xp

def displayResults(factors,h_final,Mu,tol):

	def plotResults(tt,xx,uu,tp,xp,up,typeFig):
	
		ii = 0
		plt.subplot2grid((6,4),(0,0),rowspan=2,colspan=2)
		plt.hold(True)
		plt.plot(tt,xx[:,ii],'.-b')
		plt.plot(tp,xp[:,ii],'.r')
		plt.hold(False)
		plt.grid(True)
		plt.ylabel("h [km]")
		
		ii = 1
		plt.subplot2grid((6,4),(0,2),rowspan=2,colspan=2)
		plt.hold(True)
		plt.plot(tt,xx[:,ii],'.-b')
		plt.plot(tp,xp[:,ii],'.r')
		plt.hold(False)
		plt.grid(True)
		plt.ylabel("V [km/s]")
		
		ii = 2
		plt.subplot2grid((6,4),(2,0),rowspan=2,colspan=2)
		plt.hold(True)
		plt.plot(tt,xx[:,ii]*180.0/numpy.pi,'.-b')
		plt.plot(tp,xp[:,ii]*180.0/numpy.pi,'.r')
		plt.hold(False)
		plt.grid(True)
		plt.ylabel("gamma [deg]")
		
		ii = 3
		plt.subplot2grid((6,4),(2,2),rowspan=2,colspan=2)
		plt.hold(True)
		plt.plot(tt,xx[:,ii],'.-b')
		plt.plot(tp,xp[:,ii],'.r')
		plt.hold(False)
		plt.grid(True)
		plt.ylabel("m [kg]")
		
		ii = 0
		plt.subplot2grid((6,4),(4,0),rowspan=2,colspan=2)
		plt.hold(True)
		plt.plot(tt,uu[:,ii],'.-b')
		plt.plot(tp,up[:,ii],'.r')
		plt.hold(False)
		plt.grid(True)
		plt.ylabel("alfa [rad]")
		
		ii = 1
		plt.subplot2grid((6,4),(4,2),rowspan=2,colspan=2)
		plt.hold(True)
		plt.plot(tt,uu[:,ii],'.-b')
		plt.plot(tp,up[:,ii],'.r')
		plt.hold(False)
		plt.grid(True)
		plt.xlabel("t")
		plt.ylabel("beta [adim]")
		
		plt.show()				
					
		return None
		
	# Results without orbital phase
	tt0,xx0,uu0,tp0,xp0,up0 = trajectoryDesing(factors,h_final,Mu,"plot",tol)
	h,v,gama,M = numpy.transpose(xx0[-1,:])
	eec = orbitResults(h,v,gama)
	plotResults(tt0,xx0,uu0,tp0,xp0,up0,"rocket traj")
	
	# Results with orbital phase
	if abs(eec-1) > 0.1:	
		# The eccentricity test avoid simultaitons too close of the singularity
		tt0,xx0,uu0,tp0,xp0,up0 = trajectoryDesing(factors,h_final,Mu,"orbital",tol)
		h,v,gama,M = numpy.transpose(xx0[-1,:])
		orbitResults(h,v,gama)
		plotResults(tt0,xx0,uu0,tp0,xp0,up0,"orbital")
		
	return None

def orbitResults(h,v,gama):

	r = R + h
	cosGama = numpy.cos(gama)
	sinGama = numpy.sin(gama)
	momAng = r * v * cosGama
	print("Ang mom:",momAng)
	en = .5 * v * v - GM/r
	print("Energy:",en)
	a = - .5*GM/en
	print("Semi-major axis:",a)
	aux = v * momAng / GM
	e = numpy.sqrt((aux * cosGama - 1)**2 + (aux * sinGama)**2)
	print("Eccentricity:",e)

	print("Final altitude:",h)
	ph = a * (1.0 - e) - R
	print("Perigee altitude:",ph)	
	ah = 2*(a - R) - ph		
	print("Apogee altitude:",ah)
	
	return e

def mdlDer(t,x,arg):
       
	h,v,gama,M = x[0],x[1],x[2],x[3]
	alfaProg,betaProg,T,Isp,g0,R = arg 
	betat = betaProg.value(t)
	alfat = alfaProg.value(t)
    
	btm = betat*T/M
	sinGama = numpy.sin(gama)
	g = g0*(R/(R+h))**2

	return numpy.array([v*sinGama,\
	btm*numpy.cos(alfat) - g*sinGama,\
	btm*numpy.sin(alfat)/v + (v/(h+R)-g/v)*numpy.cos(gama),\
	-btm*M/g0/Isp])	
	
class retPulse():
	
	def __init__(self,t1,t2,v1,v2):
		self.t1 = t1
		self.t2 = t2
		self.v1 = v1
		self.v2 = v2
		
	def value(self,t):
		if (t < self.t1):
			return self.v1
		elif (t < self.t2):
			return self.v2
		else:
			return self.v1
			
	def multValue(self,t):
		N = len(t)
		ans = numpy.full((N,1),self.v1)
		for ii in range(0,N):
			if (t[ii] >= self.t1) and (t[ii] < self.t2):
				ans[ii] = self.v2
		return ans
		
class retPulse2():
	
	def __init__(self,tVec,vVec):
		self.tVec = tVec		
		self.vVec = vVec
		
	def value(self,t):
		ii = 0
		NVec = len(self.tVec)
		stop = False
		while not stop:
			if (t < self.tVec[ii]):
				ans = self.vVec[ii]
				stop = True
			else:
				ii = ii + 1
				if ii == NVec:
					ans = self.vVec[-1]
					stop = True
				
		return ans
							
	def multValue(self,t):
		N = len(t)
		ans = numpy.full((N,1),self.vVec[0])
		for jj in range(0,N):
			ans[jj] = self.value(t[jj])

		return ans			

main()