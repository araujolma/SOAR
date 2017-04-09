# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 15:05:16 2017

@author: Carlos Souza
"""

import numpy
import matplotlib.pyplot as plt
from scipy.integrate import ode
#from numpy.linalg import norm
#from utils import interpV#, interpM, ddt

def main ():
	N = 5000 + 1
	dt = 1.0e-2#1.0/(N-1)
	pi = numpy.pi

	# example rocket single stage to orbit L=0 D=0
	# initial state condition
	h_initial = 0.0            # km
	V_initial = 0.0            # km/s
	gamma_initial = pi/2 # rad
	m_initial = 50000          # kg
	# final state condition
	h_final = 463     # km
	V_final = 7.633   # km/s
	gamma_final = 0.0 # rad
	GM = 398600.4415       # km^3 s^-2
	Isp = 450              # s
	efes = .95
	R = 6371             # km
	g0 = 9.8e-3
	AoAmax = 2.0 # graus

	##########################################################################
	fator_V = 1.06 # Ajust to find a final V
	tf = 480.0 # Ajust to find a final gamma
	tAoA = 0.5 #Ajust to find a final h
	fdv1 = 1.4 #Ajust to find a final h

	Mu = 100.0
	Dv1 = fdv1*numpy.sqrt(2.0*GM*(1/R - 1/(R+h_final)))
	Dv2 = V_final

	##########################################################################
	Dv2 = Dv2*fator_V
	LamMax = 1/(1-efes)
	Lam1 = numpy.exp(Dv1/g0/Isp)
	Lam2 = numpy.exp(Dv2/g0/Isp)
	print("Dv =",Dv1,"Dv =",Dv2," Lam1 =",Lam1," Lam2 =",Lam2,"LamMax =",LamMax)

	Mp2 = (Lam2-1)*efes*Mu/(1 - Lam2*(1-efes))
	Mp1 = (Lam1-1)*efes*(Mu + (Mp2/efes))/(1 - Lam1*(1-efes))
	Mp = Mp1 + Mp2;
	Me = (1-efes)*Mp/efes
	M0 = Mu + Mp + Me
	print("Mu =",Mu," Mp =",Mp," Me =",Me,"M0 =",M0)


	T = 40.0e3 # thrust in N
	T *= 1.0e-3 # thrust in kg * km / s^2 [for compatibility purposes...]

	tb1 = Mp1 * g0 * Isp / T
	tb2 = Mp2 * g0 * Isp / T
	tb = tb1 + tb2

	tabBeta = retPulse(tb1,(tf-tb2),1.0,0.0)


	##########################################################################
	# Chossing tAoA1 as a fraction of tf results in code bad behavior
	# So a fixed generic number is used
	tAoA1 = .01*440
	tAoA2 = tAoA1 + tAoA

	tabAlpha = retPulse(tAoA1,tAoA2,0.0,-AoAmax*pi/180)

	##########################################################################
	#Integration

	# initial conditions
	t0 = 0.0
	x0 = numpy.array([0,1.0e-6,90*pi/180,M0])

     # Integrator setting
     # ode set:
     #         atol: absolute tolerance
     #         rtol: relative tolerance
	ode45 = ode(mdlDer).set_integrator('dopri5',nsteps=1,atol = 1.0e-6,rtol = 1.0e-8)
	ode45.set_initial_value(x0, t0).set_f_params((tabAlpha,tabBeta,T,Isp,g0,R))

	# Phase times, incluiding the initial time in the begining
	tphases = numpy.array([t0,tAoA1,tAoA2,tb1,(tf-tb2),tf])
	
	# Integration using rk45 separated by phases
	# Automatic multiphase integration
	tt,xx,tp,xp = totalIntegration(tphases,ode45)
	
	########################################
	# Alpha and beta results
		
	uu = numpy.concatenate([tabAlpha.multValue(tt),tabBeta.multValue(tt)], axis=1)
	up = numpy.concatenate([tabAlpha.multValue(tp),tabBeta.multValue(tp)], axis=1)
	
	########################################  
     #  Displaing results
	plotResults(tt,xx,uu,tp,xp,up)

	# Orbit calculation
	h,v,gama,M = xx[-1,:]

	orbitResults(h,v,gama,R,GM)

	return None

def totalIntegration(tphases,ode45):

	Nref = 5.0 # Number of interval divisions for determine first step 	
	# Output variables
	tt,xx,tp,xp = [],[],[],[]

	for ii in range(1,len(tphases)):
		tt,xx,tp,xp = phaseIntegration(tphases[ii - 1],tphases[ii],Nref,ode45,tt,xx,tp,xp)	

	tt = numpy.array(tt)
	xx = numpy.array(xx)
	tp = numpy.array(tp)
	xp = numpy.array(xp) 		
		
	return tt,xx,tp,xp
	
def phaseIntegration(t_initial,t_final,Nref,ode45,tt,xx,tp,xp):

	tph = t_final - t_initial
	ode45.first_step = tph/Nref     
	stop1 = False
	while not stop1:
		ode45.integrate(t_final)
		tt.append(ode45.t)
		xx.append(ode45.y)
		if ode45.t >= t_final:
			stop1 = True

	tp.append(ode45.t)
	xp.append(ode45.y)		
		
	return tt,xx,tp,xp

def plotResults(tt,xx,uu,tp,xp,up):

#	plt.figure() 
#	plt.hold(True) 
#	plt.plot(tt,xx[:,0],'.-b')
#	plt.plot(tp,xp[:,0],'.r')
#	plt.hold(False) 
#	plt.xlabel("Time [s]")
#	plt.ylabel("h [km]")
#	plt.show()
# 
#	plt.figure() 
#	plt.hold(True) 
#	plt.plot(tt,xx[:,1],'.-b')
#	plt.plot(tp,xp[:,1],'.r')
#	plt.hold(False) 
#	plt.xlabel("Time [s]")
#	plt.ylabel("V [km/s]")
#	plt.show()
#
#	plt.figure() 
#	plt.hold(True) 
#	plt.plot(tt,xx[:,2],'.-b')
#	plt.plot(tp,xp[:,2],'.r')
#	plt.hold(False) 
#	plt.xlabel("Time [s]")
#	plt.ylabel("gamma [deg]")
#	plt.show()
#
#	plt.figure() 
#	plt.hold(True) 
#	plt.plot(tt,xx[:,3],'.-b')
#	plt.plot(tp,xp[:,3],'.r')
#	plt.hold(False) 
#	plt.xlabel("Time [s]")
#	plt.ylabel("m [kg]")
#	plt.show()
#	
#	plt.figure() 
#	plt.hold(True) 
#	plt.plot(tt,aa,'.-b')
#	plt.plot(tp,ap,'.r')
#	plt.hold(False) 
#	plt.xlabel("Time [s]")
#	plt.ylabel("Alpha [rad]")
#	plt.show()
#	
#	plt.figure() 
#	plt.hold(True) 
#	plt.plot(tt,bb,'.-b')
#	plt.plot(tp,bp,'.r')
#	plt.hold(False) 
#	plt.xlabel("Time [s]")
#	plt.ylabel("Beta [-]")
#	plt.show()
    
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
	
	#plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
	plt.show()				
				
	return None

def orbitResults(h,v,gama,R,GM):

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
	
	return None

def plotRockTraj(t,x,R,tb,tb2):

	pi = numpy.pi
	cos = numpy.cos
	sin = numpy.sin

	N = len(t)
	print("N =",N)
	dt = t[1]-t[0]
	X = numpy.empty(numpy.shape(t))
	Z = numpy.empty(numpy.shape(t))

	sigma = 0.0
	X[0] = 0.0
	Z[0] = 0.0
	for i in range(1,N):
		v = x[i,1]
		gama = x[i,2]
		dsigma = v * cos(gama) / (R+x[i,0])
		sigma += dsigma*dt

		X[i] = X[i-1] + dt * v * cos(gama-sigma)
		Z[i] = Z[i-1] + dt * v * sin(gama-sigma)



	print("sigma =",sigma)
	# get burnout point
	itb = int(tb/dt) - 1
	itb2 = int(tb2/dt) - 1
	h,v,gama,M = x[itb,:]
	print("itb =",itb)
	print("State @burnout time:")
	print("h = {:.4E}".format(h)+", v = {:.4E}".format(v)+\
	", gama = {:.4E}".format(gama)+", m = {:.4E}".format(M))


	plt.plot(X,Z)
	plt.grid(True)
	plt.hold(True)
	# Draw burnout point
	#plt.plot(X[:itb],Z[:itb],'r')
	#plt.plot(X[itb],Z[itb],'or')

#	plt.plot([0.0,0.0],[-1.0,0.0],'k')
#	plt.plot([0.0,sin(sigma)],[-1.0,-1.0+cos(sigma)],'k')
	s = numpy.arange(0,1.01,.01)*sigma
	x = R * cos(.5*pi - s)
	z = R * (sin(.5*pi - s) - 1.0)
	#z = -1 + numpy.sqrt(1-x**2)
	plt.plot(x,z,'k')
	plt.plot(X[:itb],Z[:itb],'r')
	plt.plot(X[itb],Z[itb],'or')
	plt.plot(X[itb2:],Z[itb2:],'g')
	plt.plot(X[itb2],Z[itb2],'og')
	plt.plot(X[1]-1,Z[1],'ok')
	plt.xlabel("X [km]")
	plt.ylabel("Z [km]")

	plt.axis('equal')
	plt.title("Rocket trajectory on Earth")
	plt.show()

	return None
	
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
	
def interpV(t,tVec,xVec):
	
	Nsize = xVec.shape[1]

	ans = numpy.empty(Nsize)
	for k in range(Nsize):
		ans[k] = numpy.interp(t,tVec,xVec[:,k])
	return ans
	
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

main()

