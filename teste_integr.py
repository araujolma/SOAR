# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from utils import interpV#, interpM, ddt

def plotRockTraj(t,x,R):

	cos = numpy.cos
	sin = numpy.sin	

	N = len(t)
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
	plt.plot(X/R,Z/R)
	plt.grid(True)
	plt.hold(True)
	plt.plot([0.0,0.0],[-1.0,0.0],'k')
	plt.plot([0.0,sin(sigma)],[-1.0,-1.0+cos(sigma)],'k')
	x = numpy.arange(0,sin(sigma),.01)
	z = -1 + numpy.sqrt(1-x**2)
	plt.plot(x,z,'k')
	plt.axis('equal')
	plt.show()
	
	return None


def mdlDer(x,t,tVec,alfa,beta,T,Isp,g0,R):
	h,v,gama,M = x[0],x[1],x[2],x[3]
	betat = interpV(t,tVec,beta)
	alfat = interpV(t,tVec,alfa)	
	
	btm = betat*T/M
	sinGama = numpy.sin(gama)
	
	return numpy.array([v*sinGama,\
	btm*numpy.cos(alfat) - g0*sinGama,\
	btm*numpy.sin(alfat)/v + (v/(h+R)-g0/v)*numpy.cos(gama),\
	-btm*M/g0/Isp])



N = 5000 + 1    
dt = 1.0e-3#1.0/(N-1)
pi = numpy.pi
    
# example rocket single stage to orbit L=0 D=0
# initial state condition
h_initial = 0.0            # km  
V_initial = 0.0            # km/s
gamma_initial = numpy.pi/2 # rad
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

Mu = 100.0
Dv = 1.0*numpy.sqrt(V_final**2 + 2.0*GM*(1/R - 1/(R+h_final)))
LamMax = 1/(1-efes)
Lam = numpy.exp(Dv/g0/Isp)
print("Dv =",Dv," Lam =",Lam,"LamMax =",LamMax)

Mp = (Lam-1)*efes*Mu/(1 - Lam*(1-efes))
Me = (1-efes)*Mp/efes
M0 = Mu + Mp + Me	
print("Mu =",Mu," Mp =",Mp," Me =",Me,"M0 =",M0)


T = 75.0e3 # thrust in N
T *= 1.0e-3 # thrust in kg * km / s^2 [for compatibility purposes...]

tb = Mp * g0 * Isp / T
tf = 60.0#1200.0
t = numpy.arange(0,tf+dt,dt)
Nt = numpy.size(t)

beta = numpy.zeros((Nt,1))
tvar = 0.0
i = 0
while tvar <= tb:
	beta[i] = 1.0
	i += 1
	tvar += dt

alfa = numpy.zeros((Nt,1))
tvar = 0.0
for i in range(Nt):
	tvar += dt
	if tvar > .1*tb and tvar < tb:
		alfa[i] = -15*pi/180#-21*pi/180


x0 = numpy.array([0,1.0e-6,90*pi/180,M0])
x = odeint(mdlDer,x0,t,args=(t,alfa,beta,T,Isp,g0,R))

plt.subplot2grid((6,4),(0,0),colspan=4)
plt.plot(t,x[:,0],)
plt.grid(True)
plt.ylabel("h [km]")
plt.subplot2grid((6,4),(1,0),colspan=4)
plt.plot(t,x[:,1],'g')
plt.grid(True)
plt.ylabel("V [km/s]")
plt.subplot2grid((6,4),(2,0),colspan=4)
plt.plot(t,x[:,2]*180.0/pi,'r')
plt.grid(True)
plt.ylabel("gamma [deg]")
plt.subplot2grid((6,4),(3,0),colspan=4)
plt.plot(t,x[:,3],'m')
plt.grid(True)
plt.ylabel("m [kg]")
#plt.subplot2grid((6,4),(4,0),colspan=4)
#plt.plot(t,u[:,0],'k')
#plt.grid(True)
#plt.ylabel("alfa [rad]")
#plt.subplot2grid((6,4),(5,0),colspan=4)
#plt.plot(t,u[:,1],'c')
#plt.grid(True)
#plt.xlabel("t")
#plt.ylabel("beta [adim]")
plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
plt.show()

plotRockTraj(t,x,R)