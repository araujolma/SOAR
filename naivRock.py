#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 22:49:46 2018

@author: levi

This is a module for simple integration of the rocket equations,
in order to yield a basic (naive) initial guess for MSGRA.
"""

import numpy, pprint
from interf import logger
from scipy.signal import place_poles
from itsme import initializeSGRA
from atmosphere import rho
import probRock

d2r = numpy.pi/180.0

def speed_ang_controller(v,gama,M,pars):
    #print("\nIn speed_ang_controller!")
    #print(pars)
    #print(v)

    vOrb, gOrb, rOrb = pars['vOrb'], pars['gOrb'], pars['rOrb']
    lv, lg = pars['lv'], pars['lg']

    dv,dg = v-vOrb, gama

    # Calculate nominal thrust to weight ratio
    psi = - lv * dv / gOrb + dg

    # perform saturations
    if psi < 0.:
        psi = 0.
    Thr = psi * M * gOrb
    if Thr > pars['Thrust']:
        Thr = pars['Thrust']

    beta = Thr / pars['Thrust']
    psi = Thr / M / gOrb

    # calculate nominal angle of attack
    alfa = -(vOrb/gOrb/psi) * (lg * dg + 2. * dv / rOrb)

    # perform saturations
    if alfa > pars['alfa']:
        alfa = pars['alfa']
    elif alfa < -pars['alfa']:
        alfa = -pars['alfa']

    return alfa, beta

def finl_controller(stt,pars):

    h, v, gama, M = stt
    r = pars['r_e']+h
    g = pars['mu']/(r**2)
    psiCosAlfa = pars['acc_v']/g + numpy.sin(gama)
    psiSinAlfa = pars['acc_g']*v/g + (1. - v**2/g/r) * numpy.cos(gama)

    psi = numpy.sqrt(psiSinAlfa**2+psiCosAlfa**2)
    alfa = numpy.arctan2(psiSinAlfa,psiCosAlfa)

    # perform saturations
    if alfa > pars['alfa']:
        alfa = pars['alfa']
    elif alfa < -pars['alfa']:
        alfa = -pars['alfa']

    beta = psi * M * g / pars['Thrust']

    # perform saturations
    if beta < 0.:
        beta = 0.
    elif beta > 1.:
        beta = 1.

    return alfa, beta

def keep_straight_controller(stt,pars):
    h, V, gama, M = stt
    # r = pars['r_e'] + h
    # g = pars['mu'] / (r ** 2)
    beta = .9#0.8
    Thr = beta * pars['Thrust']
    dens = rho(h)
    pDyn = .5 * dens * V * V
    # v dgdt = T*sin(alfa)/M + L/M = 0
    # T * alfa + Fa * (CL0 + alfa * CL1) = 0
    # T/Fa * alfa + CL0 + alfa * CL1 = 0
    alfa = -pars['CL0']/(Thr/(pDyn * pars['S']) + pars['CL1'])
    #L = pDyn * pars['S'] * (pars['CL0'] + alfa * pars['CL1'])
    # perform saturations
    if alfa < -pars['alfa']:
        alfa = -pars['alfa']
    elif alfa > pars['alfa']:
        alfa = pars['alfa']

    #print("alfa = {:.3E} deg, Thr = {:.3E} kN, dens = {:.3E} kg/km³, pDyn = {:.3E} kN/km², L = {:.3E} kN".format(alfa/d2r,Thr,dens,pDyn,L) )

    return alfa, beta

def one_controller(t,stt,pars):
    h, V, gama, M = stt

    r = pars['r_e'] + h
    g = pars['mu'] / (r ** 2)
    dens = rho(h)
    pDynS = .5 * dens * V * V * pars['S']
    CD0, CD2 = pars['CD0'], pars['CD2']
    CL0, CL1 = pars['CL0'], pars['CL1']

    vdot,gdot = pars['vdot'], pars['gdot']
    C1 = vdot/g + numpy.sin(gama)
    C2 = V * gdot/g + (1.-V*V/g/r) * numpy.cos(gama)
    W = M * g

    # The equations:
    # psi * cos(alfa) = vdot/g + sin(gama) + D/W               = C1 + D/W
    # psi * sin(alfa) = v*gdot/g - L/W + (1.-V*V/gr)*cos(gama) = C2 - L/W

    # iterative scheme
    alfa = 0.; Cond = True
    qsmg = pDynS/W
    C1_, C2_ = C1 + qsmg*CD0, C2 + qsmg*CL0
    CL1_, CD2_ = qsmg * CL1, qsmg * CD2
    while Cond:
        Num = C2_ + CL1*alfa
        Den = C1_ + CD2*alfa*alfa
        f = Num / Den - numpy.tan(alfa)
        df = (CL1*Den-Num*2.*CD2*alfa)/(Den**2) - 1./(numpy.cos(alfa))**2
        alfa_new = alfa - f/df
        Cond = abs(alfa_new-alfa) > 1e-6
        alfa = alfa_new
    # while Cond:
    #     DW = pDynS * (CD0 + CD2 * alfa * alfa) / W
    #     LW = pDynS * (CL0 + CL1 * alfa) / W
    #     psi = numpy.sqrt((C1 + DW)**2 + (C2 - LW)**2)
    #     alfa_new = numpy.arctan2(C2-LW,C1+DW)
    #     Cond = abs(alfa_new-alfa) > 1e-6
    #     alfa = alfa_new
    print("alfa = ",alfa_new/d2r)
    print("Error in alpha: ",(alfa_new-alfa)/d2r)
    alfa = alfa_new
    psi = (C1 + qsmg*(CD0 + CD2*alfa*alfa))/numpy.cos(alfa)
    beta = psi * W / pars['Thrust']

    # perform saturations
    if alfa < -pars['alfa']:
        alfa = -pars['alfa']
    elif alfa > pars['alfa']:
        alfa = pars['alfa']


    if beta > 1.:
        beta = 1.

    return alfa, beta


def pole_place_ctrl_gains(pars):
    """Calculate the gains for the pole placement controller """

    # Assemble matrix 'A'
    A = numpy.zeros((3,3))
    A[0,2] = pars['vOrb']
    A[1,2] = -pars['gOrb']
    A[2,0] = -3. * pars['gOrb'] / pars['rOrb']
    A[2,1] = 2./pars['rOrb'] + pars['gOrb']/(pars['vOrb']**2)

    # Assemble matrix 'B'
    B = numpy.zeros((3,2))
    B[1,0] = pars['gOrb']
    B[2,1] = pars['gOrb']

    P = pars['poles']

    fsf = place_poles(A,B,P)

    return fsf.gain_matrix

def pole_place_controller(h,v,gama,M,pars):
    #return alfa, beta
    pass

def calc_initial_rocket_mass(boundary,constants,log):
    """Calculate the initial rocket mass for a given mission and given
    engine parameters."""

    # Isp in km/s
    Isp_kmps = constants['Isp'][0] * constants['grav_e']
    # Structural factor (structural inefficiency)
    s_f = constants['s_f'][0]
    # Maximum deliverable Dv
    max_Dv = - Isp_kmps * numpy.log(s_f)
    # Mission required Dv
    Dv = boundary['mission_dv']
    if Dv > max_Dv:
        msg = 'naivRock: Sorry, unfeasible mission.\n'+ \
              'Mission Dv-Maximum Dv = {:.3E} km/s'.format(Dv-max_Dv)
        raise Exception(msg)
    else:
        msg = "\nMaximum Dv margin: {:.2G}%".format(100.*(max_Dv/Dv-1.))
        log.printL(msg)
        # Carry on with a Dv larger than strictly necessary (giving margin)
        f = 0.99#.5#.933
        print("f = ",f)
        Dv = (1.-f) * Dv + f * max_Dv
    L = numpy.exp(Dv / Isp_kmps)
    print("L = ",L)
    m0 = (1. + (L-1.) / (1. -  L * s_f) ) * constants['mPayl']
    mu_rel = 100. * constants['mPayl'] / m0
    mp_rel = (1. - s_f) * (100. - mu_rel)
    me_rel = 100. - mu_rel - mp_rel
    msg = 'Initial rocket mass: {:.1F} kg,\n'.format(m0) + \
          ' propellant: {:.2G}%, '.format(mp_rel) + \
          ' structure: {:.2G}%, '.format(me_rel) + \
          ' payload: {:.2G}%'.format(mu_rel)
    log.printL(msg)
    return m0

def naivGues(file, extLog=None):

    # Declare a running solution
    sol = probRock.prob()

    if extLog is None:
        # Open a logger object, just for printing
        sol.log = logger(sol.probName)#,mode='screen')
    else:
        # Otherwise, use this provided logger
        sol.log = extLog

    sol.log.printL("\nIn naivGues, preparing parameters to generate an"
                   " initial solution!")

    # Get all parameters from file into the proper places in sol object
    con = initializeSGRA(file).result()
    sol.storePars(con,file)

    m, n, s = sol.m, sol.n, sol.s

    ones = numpy.ones(s)

    #m0 = calc_initial_rocket_mass(boundary, constants, sol.log)
    m0 = 3000.
    sol.boundary['m_initial'] = m0

    # This is only here for bypassing
    alfa_max = 3. * d2r
    sol.restrictions['alpha_min'] = -alfa_max * ones
    sol.restrictions['alpha_max'] =  alfa_max * ones
    boundary, constants = sol.boundary, sol.constants

    # Maximum dimensional time for any arc [s]
    tf = 1000.
    # Dimensional dt for integration [s]
    dtd = 0.1
    Nmax = int(tf/dtd)+1

    h, V = sol.boundary['h_initial'], sol.boundary['V_initial']
    gama, M = sol.boundary['gamma_initial'], sol.boundary['m_initial']
    # running x
    x_ = numpy.array([h, V, gama, M])
    # prepare arrays for integration
    x, u = numpy.zeros((Nmax,n,s)), numpy.zeros((Nmax,m,s))
    pi = numpy.empty(s)
    # initial condition, first arc
    x[0,:,0] = x_

    # Dimensional running time
    td = 0.
    finlElem = numpy.zeros(s,dtype='int')
    sol.log.printL("\nProceeding for naive integrations...")
    tdArc = 0.  # dimensional running time for each arc
    # TODO: replace hardcoded "end of arc" parameters by custom values from
    #  the configuration file.
    for arc in range(s):
        if arc == 0:
            # First arc:
            # - rise vertically
            # - at full thrust
            # - until v = 100 m/s
            msg = "\nEntering 1st arc (vertical rise at full thrust)..."
            sol.log.printL(msg)

            pars = {'Thrust': constants['Thrust'][0],
                    'alfa': sol.restrictions['alpha_max'][0],
                    'CL0': constants['CL0'][0],
                    'CL1': constants['CL1'][0],
                    'S': constants['s_ref'][0]}
            ctrl = lambda t, state: keep_straight_controller(state,pars)
            arcCond = lambda t, state: state[1] < 0.1
        elif arc == 1:
            # Second arc:
            # - maximum turning (half of max ang of attack)
            # - constant specific thrust
            # - until gamma = 5deg
            #msg = "\nEntering 2nd arc (turn at constant alpha)..."
            msg = "\nEntering 2nd arc (1 turn to rule them all)..."
            sol.log.printL(msg)
            pi2 = numpy.pi / 2.
            vf = boundary['V_final']# * 0.8
            tf = (boundary['h_final']-h)*pi2 / \
                 (vf*(1.-1./pi2) + V/pi2)
            vdot = (vf-V)/tf
            gdot = -pi2/tf
            acc_g = vdot/constants['grav_e']
            beta_LB = M*(vdot+constants['grav_e'])/constants['Thrust'][1]
            msg = "\nTime until next arc: {:.3g} s\n".format(tf) + \
                  "Target acceleration: {:.2g} g's\n".format(acc_g) + \
                  "Target turning rate: {:.1g} deg/s".format(-gdot/d2r) + \
                  "\nLower bound for initial beta: {:.2g}".format(beta_LB)
            sol.log.printL(msg)
            if beta_LB >= 1.:
                input("\nThe thrust will certainly saturate in this arc."
                      "\nPress any key to proceed. ")

            pars = {'mu': constants['GM'],
                    'r_e': constants['r_e'],
                    'CD0': constants['CD0'][1],
                    'CD2': constants['CD2'][1],
                    'Thrust': constants['Thrust'][1],
                    'alfa': sol.restrictions['alpha_max'][1],
                    'CL0': constants['CL0'][1],
                    'CL1': constants['CL1'][1],
                    'S': constants['s_ref'][1],
                    't0': td,
                    'vdot': vdot,#'Dv': vf-V,
                    'gdot': gdot}

            ctrl = lambda t, state: one_controller(t,state,pars)
            tf_offset = tf + td
            arcCond = lambda t, state: (t <= tf_offset)
        elif arc == 2:
            msg = "\nEntering 3rd arc (closed-loop orbit insertion)..."
            sol.log.printL(msg)

            # assemble parameter dictionary for controllers
            # noinspection PyDictCreation
            pars = {'rOrb': constants['r_e'] + boundary['h_final'],
                    'vOrb': boundary['V_final'],
                    'Thrust': constants['Thrust'][2],
                    'alfa': sol.restrictions['alpha_max'][2]}
            pars['gOrb'] = constants['GM'] / (pars['rOrb']**2)

            # approach #01: only speed and angle control,
            # linearization by feedback

            #pars['lv'], pars['lg'] = .01, .1
            #ctrl = lambda t, state: speed_ang_controller(state[1],state[2],
            #                                             state[3],pars)

            # approach #02: full state feedback control,
            # pole placement

            #pars['poles'] = numpy.array([-.1001,-.1002,-.1003])/10.
            #K = pole_place_ctrl_gains(pars)
            #sol.log.printL("Gains for controller:\n{}".format(K))
            #pars['K'] = K
            #ctrl = lambda t, state: pole_place_controller(state[0],state[1],
            #                                              state[2],state[3],
            #                                              pars)

            #tolV, tolG = 0.01, 0.1 * d2r
            #arcCond = lambda t, state: abs(state[1]-pars['vOrb']) > tolV or \
            #                            (abs(state[2]) > tolG)

            # approach #03: height, speed and angle control, similar to #01,
            # but with linear reference for speed and angle. Exit condition
            # by time-out, based on pre-calculated time

            # height error
            eh = boundary['h_final'] - x[0,0,2]
            # initial speed and speed error
            v0 = x[0,1,2]; ev = boundary['V_final'] - v0
            # initial flight path angle
            gama0 = x[0,2,2]
            # calculated duration of maneuver
            tf = 6. * eh / gama0 / (3.*v0 + ev)
            # Load parameters to pars dict
            pars['tf'] = tf; pars['eh'] = eh; pars['ev'] = ev
            # "Acceleration" for velocity and flight path angle
            pars['acc_v'] = ev/tf; pars['acc_g'] = -gama0/tf
            # These are for calculating g locally at each height
            pars['mu'] = constants['GM']; pars['r_e'] = constants['r_e']
            acc = pars['acc_v'] / constants['grav_e']
            turn = pars['acc_g'] / d2r
            msg = "\nTime until orbit: {:.3g} s\n".format(tf) + \
                  "Target acceleration: {:.2g} g's\n".format(acc) + \
                  "Target turning rate: {:.1g} deg/s".format(-turn)
            sol.log.printL(msg)
            ctrl = lambda t, state: finl_controller(state,pars)
            tf_offset = tf + td
            arcCond = lambda t, state: (t <= tf_offset)
        else:
            raise(Exception("Undefined controls for arc = {}.".format(arc)))

        # RK4 integration
        k = 1; dtd2, dtd6 = dtd/2., dtd/6.
        while k < Nmax and arcCond(td, x_):
            u_ = ctrl(td, x_)
            # first derivative
            f1 = probRock.calcXdot(td, x_, u_, constants, arc)
            tdpm = td + dtd2
            # x at half step, with f1
            x2 = x_ + dtd2 * f1
            # u at half step, with f1
            u2 = ctrl(tdpm, x2)
            # second derivative
            f2 = probRock.calcXdot(tdpm, x2, u2, constants, arc)
            # x at half step, with f2
            x3 = x_ + dtd2 * f2  # x at half step, with f2
            # u at half step, with f2
            u3 = ctrl(tdpm, x3)
            # third derivative
            f3 = probRock.calcXdot(tdpm, x3, u3, constants, arc)
            td4 = td + dtd
            # x at half step, with f3
            x4 = x_ + dtd * f3  # x at next step, with f3
            # u at half step, with f3
            u4 = ctrl(td4, x4)
            # fourth derivative
            f4 = probRock.calcXdot(td4, x4, u4, constants, arc)
            # update state with all four derivatives f1, f2, f3, f4
            x_ += dtd6 * (f1 + f2 + f2 + f3 + f3 + f4)
            # update dimensional time
            td = td4
            # store states and controls
            x[k,:,arc], u[k-1,:,arc] = x_, u_
            # Increment time index
            k += 1
        # Store the final time index for this arc
        finlElem[arc] = k
        # continuity condition for next arc (if it exists)
        if arc < s - 1:
            x[0, :, arc + 1] = x[k - 1, :, arc]
        # avoiding nonsense values for controls in the last times of the arc
        for j in range(k-1,Nmax):
            # noinspection PyUnboundLocalVariable
            u[j, :, arc] = u_
        # Store time into pi array
        pi[arc] = td-tdArc
        tdArc = td + 0.
        sol.log.printL("  arc complete!")
        st = x[k - 1, :, arc]
        msg = '\nArc duration = {:.2F} s'.format(pi[arc]) + \
              '\nStates at end of arc:\n' + \
              '- height = {:.2F} km'.format(st[0]) + \
              '\n- speed = {:.3F} km/s'.format(st[1]) + \
              '\n- flight path angle = {:.1F} deg'.format(st[2] / d2r) + \
              '\n- mass = {:.1F} kg'.format(st[3])
        sol.log.printL(msg)
    sol.log.printL("\n... naive integrations are complete.")

    # Load constants into sol object
    sol.log.printL("Original N: {}".format(int(max(finlElem))))
    # This line is for redefining N, if so,
    # while still keeping the 100k + 1 "structure"
    sol.log.printL("New N: {}".format(sol.N))
    m, n, s, q, s = sol.m, sol.n, sol.p, sol.q, sol.s
    # Perform interpolations to keep every arc with the same refinement
    xFine, uFine = numpy.empty((sol.N,n,s)), numpy.empty((sol.N,m,s))
    sol.log.printL("\nProceeding to interpolations...")
    for arc in range(s):
        # "Fine" time array
        tFine = sol.t * pi[arc]
        # "Coarse" time array
        tCoar = numpy.linspace(0.,pi[arc],num=finlElem[arc])

        # perform the actual interpolations
        for stt in range(n):
            xFine[:,stt,arc] = numpy.interp(tFine,tCoar,
                                            x[:finlElem[arc],stt,arc])
        for ctr in range(m):
            uFine[:,ctr,arc] = numpy.interp(tFine,tCoar,
                                            u[:finlElem[arc],ctr,arc])
    sol.log.printL("... interpolations complete.")
    # Load interpolated solutions into sol object
    sol.x = xFine
    # Control is stored non-dimensionally
    sol.u = sol.calcAdimCtrl(uFine[:,0,:],uFine[:,1,:])
    sol.pi = pi
    # Current probRock formulation demands a target height array for the
    # first "artificial" arcs
    sol.boundary['TargHeig'] = numpy.array(sol.x[-1,0,:s-1])
    # These arrays get zeros, although that does not make much sense
    sol.lam = numpy.zeros_like(sol.x,dtype='float')
    sol.mu = numpy.zeros(q)
    sol.log.printL("\nNaive solution is ready, returning now!")
    return sol


if __name__ == "__main__":
    # Generate the initial guess
    sol = naivGues('defaults/probRock.its')
    #sol = naivGues('defaults/probRock-naive2.its')

    # Show the parameters obtained
    sol.printPars()

    # Plot solution
    sol.plotSol() # normal
    # non-dimensional time to check better the details in short arcs
    sol.plotSol(piIsTime=False)
    # Plot trajectory
    sol.plotTraj()
    # sol.plotTraj(fullOrbt=True,mustSaveFig=False)
    sol.log.printL("\nFinal error on boundaries:\n"+str(sol.calcPsi()))

    # TESTING the obtained solution with rest
    contRest = 0
    sol.calcP(mustPlotPint=True)
    while sol.P > sol.tol['P']:
        sol.rest()
        contRest += 1
        sol.plotSol()
    #

    sol.showHistP()
    sol.log.printL("\nnaivRock.py execution finished. Bye!\n")
    sol.log.close()
