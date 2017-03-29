# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 21:28:33 2017

@author: Carlos Souza
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy
import matplotlib.pyplot as plt
from scipy.integrate import odeint
#from numpy.linalg import norm
from utils import interpV#, interpM, ddt
#import prob_rocket_sgra

def getRockTraj(printInfo=False):#,constants,boundary,restrictions):
    
    dt = 1e-3#7.0e-4#1.0/(N-1)
    pi = numpy.pi

    # example rocket single stage to orbit L=0 D=0
    
    h_final = 463     # km
    V_final = 7.633   # km/s
    GM = 398600.4415       # km^3 s^-2
    Isp = 450              # s
    s_f = 0.05              
    efes = 1 - s_f
    r_e = 6371             # km
    grav_e = 9.8e-3
    alpha_min = -2*(numpy.pi)/180  # in rads
    alpha_max = 2*(numpy.pi)/180   # in rads
    beta_min = 0
    beta_max = 1
#    h_final = boundary['h_final']
#    V_final = boundary['V_final']
#    grav_e = constants['grav_e']
#    Thrust = constants['Thrust']
#    Isp = constants['Isp']
#    r_e = constants['r_e']
#    GM = constants['GM']
#    s_f = constants['s_f']
#    alpha_min = restrictions['alpha_min']
#    alpha_max = restrictions['alpha_max']
#    beta_min = restrictions['beta_min'] 
#    beta_max = restrictions['beta_max']
    
    ##########################################################################
    fator_V = 1.05  #1.05 # Ajust to find a final V
    tf = 440.0            # Adjust to find a final gamma
    tAoA = 2.0      #2.0  # Adjust to find a final h
    
    Mu = 100.0
    Dv1 = 1.3*numpy.sqrt(2.0*GM*(1/r_e - 1/(r_e+h_final)))
    Dv2 = V_final

    ##########################################################################
    Dv2 = Dv2 * fator_V
    LamMax = 1/(1-efes)
    Lam1 = numpy.exp(Dv1/grav_e/Isp)
    Lam2 = numpy.exp(Dv2/grav_e/Isp)
    if printInfo:
        print("Dv =",Dv1,"Dv =",Dv2," Lam1 =",Lam1," Lam2 =",Lam2,"LamMax =",LamMax)
    
    Mp2 = (Lam2-1)*efes*Mu/(1 - Lam2*(1-efes))
    Mp1 = (Lam1-1)*efes*(Mu + (Mp2/efes))/(1 - Lam1*(1-efes))
    Mp = Mp1 + Mp2;
    Me = (1-efes)*Mp/efes
    M0 = Mu + Mp + Me
    if printInfo:
        print("Mu =",Mu," Mp =",Mp," Me =",Me,"M0 =",M0)


    Thrust = 40.0e3 # thrust in N
    Thrust *= 1.0e-3 # thrust in kg * km / s^2 [for compatibility purposes...]

    tb1 = Mp1 * grav_e * Isp / Thrust
    tb2 = Mp2 * grav_e * Isp / Thrust
    tb = tb1 + tb2

    t = numpy.arange(0,tf+dt,dt)
    Nt = numpy.size(t)
    u1 = numpy.zeros((Nt,1))
    u2 = (-pi/2)*numpy.ones((Nt,1))
    tvar = 0.0
    i = 0
    while tvar <= tf:
        if tvar < tb1:
            u2[i] = pi/2            
        elif tvar > (tf - tb2):
            u2[i] = pi/2            
        i += 1
        tvar += dt
    beta = (beta_max + beta_min)/2 + numpy.sin(u2)*(beta_max - beta_min)/2    
    
    tvar = 0.0
    tAoA1 = .01*tf
    tAoA2 = tAoA1 + tAoA
    for i in range(Nt):
        tvar += dt
        if tvar > tAoA1 and tvar < tAoA2:
            u1[i] = -pi/2
    alpha = (alpha_max + alpha_min)/2 + numpy.sin(u1)*(alpha_max - alpha_min)/2
    ##########################################################################
    plt.plot(t,alpha*180/pi)
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Angle of attack [deg]")
    plt.show()

    plt.plot(t,beta)
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Thrust profile [-]")
    plt.show()
 
    # initial state
    x0 = numpy.array([0,1.0e-6,90*pi/180,M0])
 
    # integrate differential equations for trajectory
    #x = odeint(mdlDer,x0,t,args=(t,alpha,beta,Thrust,Isp,grav_e,r_e))
    x = odeint(mdlDer,x0,t,args=(t,u1,u2,Thrust,Isp,grav_e,r_e,alpha_min,alpha_max,beta_min,beta_max))
    u = numpy.empty((Nt,2))
    for k in range(Nt):
#        u[k,0] = alpha[k]
#        u[k,1] = beta[k]
         u[k,0] = u1[k]
         u[k,1] = u2[k]

    plt.subplot2grid((4,4),(0,0),colspan=4)
    plt.plot(t,x[:,0],)
    plt.grid(True)
    plt.ylabel("h [km]")
    plt.subplot2grid((4,4),(1,0),colspan=4)
    plt.plot(t,x[:,1],'g')
    plt.grid(True)
    plt.ylabel("V [km/s]")
    plt.subplot2grid((4,4),(2,0),colspan=4)
    plt.plot(t,x[:,2]*180.0/pi,'r')
    plt.grid(True)
    plt.ylabel("gamma [deg]")
    plt.subplot2grid((4,4),(3,0),colspan=4)
    plt.plot(t,x[:,3],'m')
    plt.grid(True)
    plt.ylabel("m [kg]")
    plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
    plt.show()

    plotRockTraj(t,x,r_e,tb1,tf-tb2)

    # colocar aqui módulo de calculo de órbita
    h,v,gama,M = x[Nt-1,:]
    r = r_e + h
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
    ph = a * (1.0 - e) - r_e
    print("Perigee altitude:",ph)
    print("Altitude at tf:",x[Nt-1,0])
    print("Speed at tf:",x[Nt-1,1])

    return t/tf,x,u,numpy.array([tf])

def plotRockTraj(t,x,r_e,tb,ts2):

    pi = numpy.pi
    cos = numpy.cos
    sin = numpy.sin

    N = len(t)
    #print("N =",N)
    dt = t[1]-t[0]
    X = numpy.empty(numpy.shape(t))
    Z = numpy.empty(numpy.shape(t))

    sigma = 0.0
    X[0] = 0.0
    Z[0] = 0.0
    for i in range(1,N):
        v = x[i,1]
        gama = x[i,2]
        dsigma = v * cos(gama) / (r_e+x[i,0])
        sigma += dsigma*dt

        X[i] = X[i-1] + dt * v * cos(gama-sigma)
        Z[i] = Z[i-1] + dt * v * sin(gama-sigma)


    #print("sigma =",sigma)
    # get burnout point
    itb = int(tb/dt) - 1
    its2 = int(ts2/dt) - 1
    #h,v,gama,M = x[N-1,:]
    
    #print("State @burnout time:")
    #print("h = {:.4E}".format(h)+", v = {:.4E}".format(v)+\
    #", gama = {:.4E}".format(gama)+", m = {:.4E}".format(M))


    plt.plot(X,Z)
    plt.grid(True)
    plt.hold(True)
    # Draw burnout point
    
    s = numpy.arange(0,1.01,.01)*sigma
    x = r_e * cos(.5*pi - s)
    z = r_e * (sin(.5*pi - s) - 1.0)
    
    plt.plot(x,z,'k')
    plt.plot(X[:itb],Z[:itb],'r')
    plt.plot(X[itb],Z[itb],'or')
    plt.plot(X[its2:],Z[its2:],'g')
    plt.plot(X[its2],Z[its2],'og')
    plt.plot(X[1]-1,Z[1],'ok')
    plt.xlabel("X [km]")
    plt.ylabel("Z [km]")

    plt.axis('equal')
    plt.title("Rocket trajectory on Earth")
    plt.show()

    return None


#def mdlDer(x,t,tVec,alphaProg,betaProg,Thrust,Isp,grav_e,r_e):
def mdlDer(x,t,tVec,u1Prog,u2Prog,Thrust,Isp,grav_e,r_e,alpha_min,alpha_max,beta_min,beta_max):
    h,v,gama,M = x[0],x[1],x[2],x[3]
#    betat = interpV(t,tVec,betaProg)
#    alphat = interpV(t,tVec,alphaProg)
    u1t = interpV(t,tVec,u1Prog)
    u2t = interpV(t,tVec,u2Prog)    
    alphat = (alpha_max + alpha_min)/2 + numpy.sin(u1t)*(alpha_max - alpha_min)/2
    betat = (beta_max + beta_min)/2 + numpy.sin(u2t)*(beta_max - beta_min)/2
    btm = betat*Thrust/M
    sinGama = numpy.sin(gama)
    g = grav_e*(r_e/(r_e+h))**2
    
    return numpy.array([v*sinGama,\
    btm*numpy.cos(alphat) - g*sinGama,\
    btm*numpy.sin(alphat)/v + (v/(h+r_e)-g/v)*numpy.cos(gama),\
    -btm*M/grav_e/Isp])

if __name__ == "__main__":
    getRockTraj(printInfo=True)


