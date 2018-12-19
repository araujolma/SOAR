#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 22:49:46 2018

@author: levi

This is a module for simple integration of the rocket equations,
in order to yield a basic (naive) initial guess for MSGRA.
"""

import numpy
from interf import logger

# TODO: there will be a conflict when this module gets imported in
#  probRock.py, because this module already loads probRock (mainly for
#  using plotSol, I hope calcXDot can be imported separately, so there is
#  no conflict with it.) Anyway, we will probably have to make another
#  version of plotSol to it.
import probRock


# TODO: 'objectify' this module. It would be great for development and
#  testing if the trajectory guess could be generated in here as well,
#  instead of having to run main.py every time.

# TODO: make a module for loading parameters from .its file. Probably just
#  loading some of itsme.py's methods does the trick.


d2r = numpy.pi/180.0
constants = {'r_e': 6371.0,
             'GM': 398600.4415,
             'Thrust': numpy.array([100.0]),
             's_ref': numpy.array([0.7853981633974482e-06]),
             's_f': numpy.array([0.1]),
             'Isp': numpy.array([450.]),
             'CL0': numpy.array([-0.02]),
             'CL1': numpy.array([0.8]),#
             'CD0': numpy.array([0.05]),#0.05,#
             'CD2': numpy.array([0.5]),#0.5,#
             'DampCent': -30.,
             'DampSlop': 10.,
             'Kpf': 0.,
             'PFmode': 'tanh'}
constants['grav_e'] = constants['GM']/(constants['r_e']**2)

h_final = 473.
V_final = numpy.sqrt(constants['GM']/(constants['r_e']+h_final))
missDv = numpy.sqrt((constants['GM']/constants['r_e']) *
                    (2.0-1./(1.+h_final/constants['r_e'])))
boundary = {'h_initial': 20.,#0.,
            'V_initial': .05,#1e-6,
            'gamma_initial': 90. * d2r,#5*numpy.pi,
            'm_initial': 3000,
            'h_final': h_final,
            'V_final': V_final,
            'gamma_final': 0.,
            'mission_dv': missDv}

alfa_max = 3. * numpy.pi / 180.
restrictions = {'alpha_min': -alfa_max,
                'alpha_max': alfa_max,
                'beta_min': 0.,
                'beta_max': 1.,
                'acc_max': 4. * constants['grav_e']}
# 'TargHeig': TargHeig}

# Declare a running solution
sol = probRock.prob()
# Open a logger object, just for printing
sol.log = logger(sol.probName,mode='screen')

sol.constants = constants
# Anything, really, it is just because there should be a mPayl attribute
sol.mPayl = 10.
# Same here:
sol.addArcs = 0
sol.isStagSep = numpy.array([False])

sol.boundary = boundary
sol.restrictions = restrictions
sol.initMode = 'extSol'
# Show the parameters before integration
sol.printPars()
#input("\nIAE?")

m, n, s = 2, 4, 1
# Final, dimensional time
tf = 300.
# Dimensional dt for integration
dtd = 0.005
N = int(tf/dtd)+1#100001#000

# thrust to weight ratio
psi = 2.

h, V = boundary['h_initial'], boundary['V_initial']
gama, M =  boundary['gamma_initial'], boundary['m_initial']
# running x
x_ = numpy.array([h, V, gama, M])
# prepare arrays for integration
x, u = numpy.empty((N,n,s)), numpy.empty((N,m,s))
pi = numpy.empty(s)
# initial condition, first arc
x[0,:,0] = x_

for arc in range(s):
    td = 0.
    for k in range(1,N):
        r = constants['r_e'] + h
        # Non-dimensional thrust
        beta = psi * M * constants['grav_e'] /constants['Thrust']
        #Thr = beta * constants['Thrust']
        #g = constants['GM'] / r / r
        # pitching program
        if td < 10.:
            alfa = (-2.*(td/10.))*d2r
        else:
            alfa = -2 * d2r
        #(-2.*(1.-td/tf)-1.*(td/tf))*d2r#numpy.arcsin((g/V - V/r)*numpy.cos(gama) * (M*V/Thr) )
        u_ = numpy.array([alfa,beta])
        # calculate derivatives
        f = probRock.calcXdot(td, x_, u_, constants, 0)
        # update running solution (Euler)
        x_ += f * dtd
        # update dimensional time
        td += dtd
        # Load states for next iteration
        h, V, gama, M = x_
        # store states and controls
        x[k,:,arc], u[k-1,:,arc] = x_, u_

    # avoiding nonsense values for controls in the last time
    u[-1, :, arc] = u[-2,:,arc]
    # continuity condition for next arc
    if arc < s-1:
        x[0,:,arc+1] = x[-1,:,arc]
    # Store time into pi array
    pi[arc] = td

# print("\n\ntd = {:.1F} s".format(td))
# print("\nStates: Height [km], Speed [km/s], Flight Path Angle [deg], Mass [kg]")
# x[:,2] *= 180./numpy.pi
# print(x)
# u[:,0] *= 180./numpy.pi
# u[:,1] *= constants['Thrust']
# print("\nControls: Angle of attack [deg], Thrust [kN]")
# print(u[1:,:])

# Load constants into sol object
sol.N, sol.m, sol.n, sol.p, sol.s = N, m, n, s, s
sol.dt = 1./(N-1)
sol.t = numpy.linspace(0.,1.,num=N)
# Load the output of the integration process to sol object
sol.x = x
sol.u = sol.calcAdimCtrl(u[:,0,:],u[:,1,:])
sol.pi = pi

# Finally: plot solution and trajectory
sol.plotSol()#mustSaveFig=False)
sol.plotTraj(compare=False)