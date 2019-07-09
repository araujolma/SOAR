#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 11:03:33 2019

This module is meant for obtaining the analytical solution of probLand,
with the acceleration limitation in place.

@author: levi


"""

import numpy
import matplotlib.pyplot as plt
from utils import simp
#from scipy.optimize import optimize
from scipy.optimize import root

pars_def = {'g0Isp': 200.*9.8e-3,      # km/s
            'T': 10.,                  # kN
            'g': 4903.89/(1737.0**2),  # km/s²
            'h0': 10.,                 # km
            'v0': 0.,                  # km/s
            'M0': 100. + 1000./.9}     # kg

def try_pi_noLim(pi, pars = None):
    """Assuming no limitation on the maximum acceleration, this function
    calculates the residual associated with a given value of the total time,
    pi.

    Of course, the "bang-bang" shape of the solution (actually, wait, then
    bang) is built-in into this calculation."""

    if pi <= 0.:
        raise Exception("Time must be positive!")

    if pars is None:
        pars = pars_def

    g0Isp = pars['g0Isp']
    g = pars['g']
    psi0 = pars['T']/pars['M0']/g
    h0 = pars['h0']

    aux = pi * g / g0Isp

    t1 = 1. - (1.-numpy.exp(-aux))/aux/psi0
    print("t1 =",t1)

    res = pi * g0Isp * (1./psi0 + t1 - 1.) + pi*pi*g*(t1-.5) - h0
        #(pi * g0Isp/psi0) * (1. - (1.-numpy.exp(-aux))/aux) + \
        # pi * pi * g * (.5 - (1. - numpy.exp(-aux))/aux) - h0
    #print("res =",res)
    return res, t1

def solve_pi_noLim(tol=1e-6,pars=None):
    """This works on pair with try_pi_noLim.

    Assuming no acceleration limitation, this function performs a bisection
    on pi until the residual is sufficiently close to zero. """

    if pars is None:
        pars = pars_def

    # Get a pi so that the residual is positive
    pip = 1.; resP = -1.
    while resP <= 0.:
        pip *= 2.
        resP,_ = try_pi_noLim(pip, pars=pars)

    # Get a lambda_R so that the residual is negative
    resN = 1.; pin = 1.  # .5#.387
    while resN >= 0:
        pin /= 10.
        resN,_ = try_pi_noLim(pin, pars=pars)

    # now, do the bisection
    res = 1.; pi = 1.
    while abs(res) > tol:
        pi = .5 * (pip + pin)
        print("\nTrying pi =", pi)
        res,t1 = try_pi_noLim(pi, pars=pars)
        print("res =", res)
        if res < 0:
            pin = pi
        else:
            pip = pi
    #
    print("\nFinal answer: (pi = {:.4E}, residual = {:.4E})".format(
        pi, res))
    return pi, t1

def intg_land(pi,t1,lam0,aL,K,pars,mustPlot=False,N=100001):
    """This is the innermost function for the lander problem with acceleration
    limitation. It yields (numerically, of course) a solution to the lander
    problem, assuming the total time pi>0, waiting fraction 0<t1<1, initial
    conditions on costates lam0, limiting acceleration aL and penalty function
    gain K.

    N is the number of elements for discretization of time array, should be
    kept high because the differential equations are solved with the simplest
    (first order) Euler method.

    This function returns the four residuals that must become zero (only) for
    the exact optimal solution, corresponding to the conditions:
    h(1) = 0,
    v(1) = 0,
    lambda_M(1) = 0,
    integral of (H_pi) dt = 0

    where lambda_M is the Mass costate and H = f - lam * phi is the
    Hamiltonian.
    """

    lh, lv0, lm0 = lam0
    g0Isp = pars['g0Isp']
    g = pars['g']

    h0 = pars['h0']
    v0 = pars['v0']
    M0 = pars['M0']

    T = pars['T']
    # initial value for the crossing function xi
    xi0 = (1. + lm0) / g0Isp - lv0 / M0
    msg = "pi = {:.5G}, xi0 = {:.4G}, t1 = {:.4G}, "\
          "lh = {:.4G}, lv0 = {:.4G}".format(pi, xi0, t1, lh, lv0)
    print(msg)

    t = numpy.linspace(0., 1., num=N)
    dtd = t[1] * pi
    # speed costate
    lv = lv0 - pi * lh * t
    # These are the exact solutions assuming free fall
    v = v0 - pi * g * t
    h = h0 + v0 * pi * t - .5 * g * (pi * t) ** 2
    M = M0 + numpy.zeros(N)#0. * t
    lm = lm0 + numpy.zeros(N)#0. * t

    # Perform the integration (for the propulsed part)
    for k,t_ in enumerate(t):
        if t_ > t1:
            M_ = M[k-1]
            # xi equation
            xi_ = (1.+lm[k-1])/g0Isp - lv[k-1]/M_
            xiM2 = xi_ * M_**2/2./K/T
            # fraction of maximum thrust
            b_ = M_ * aL * (1. - xiM2) / T
            # perform obvious saturations
            if b_ > 1.:
                b_ = 1.
            elif b_ < 0.:
                b_ = 0.
            # acceleration provoked by that thrust
            a_ = b_ * T / M_

            # everything is ready, integrate the ODEs
            h[k] = h[k-1] + v[k-1] * dtd
            v[k] = v[k-1] + (a_ - g) * dtd
            M[k] = M_ -  b_ * T * dtd/ g0Isp
            lm[k] = lm[k-1] + dtd * (a_ / M_) * \
                    (lv[k-1] - 2. * K * (a_-aL) * (a_>aL))

    # assemble complete arrays for integration
    xi = (1.+lm)/g0Isp - lv/M
    b = M * aL * (1. - xi * M**2 / 2. / K / T) * (t>=t1) / T
    # saturate again
    for k, b_ in enumerate(b):
        if b_ > 1.:
            b[k] = 1.
        elif b_ < 0.:
            b[k] = 0.
    a = b * T / M
    # this is the term for the integral equation
    intg = (M0-M[-1])/pi + K * simp((a-aL)**2 * (a>=aL),N) + \
            2. * lh * h0 / pi + lv0 * v0 / pi + (T/g0Isp) * simp(lm * b,N)

    # assemble residuals
    resV = numpy.array([h[-1], v[-1], lm[-1], intg])
    res = sum(resV ** 2)

    #if mustPrint:
    print("resV =", resV)
    print("res =", res)

    if mustPlot:
        fig, axs = plt.subplots(3, 2, constrained_layout=True)
        axs[0,0].plot(t*pi,h,label='h')
        #axs[0].set_title('subplot 1')
        axs[0,0].set_ylabel('Height [km]')
        axs[0,0].set_xlabel('Time [s]')
        axs[0, 0].grid()

        msg = 'Analytical solution for K = {:.1G}, aLim = {:.1G}g (Res = {:.3E})'.format(K,aL/g,res)
        fig.suptitle(msg, fontsize=12)

        axs[0,1].plot(t*pi, v, label='v')
        axs[0,1].set_ylabel('Speed [km/s]')
        axs[0,1].set_xlabel('Time [s]')
        axs[0, 1].grid()

        axs[1,0].plot(t*pi, a/aL, label='acc/accLim')
        axs[1,0].set_ylabel('acc/accLim [-]')
        axs[1,0].set_xlabel('Time [s]')
        axs[1, 0].grid()

        axs[1,1].plot(t*pi, b, label='beta')
        axs[1,1].set_ylabel('beta [-]')
        axs[1,1].set_xlabel('Time [s]')
        axs[1, 1].grid()

        axs[2,0].plot(t*pi, M, label='M')
        axs[2,0].set_ylabel('Mass [kg]')
        axs[2,0].set_xlabel('Time [s]')
        axs[2, 0].grid()

        axs[2,1].plot(t*pi, xi, label='xi')
        axs[2,1].set_ylabel('xi [s/km]')
        axs[2,1].set_xlabel('Time [s]')
        axs[2, 1].grid()

        # calculate maximum violation, mean violation, etc
        maxVio = (max(a) / aL - 1.) * 100.
        meanVio = (simp((a - aL) * (a >= aL), N) / (1. - t1) / aL ) * 100.

        plt.figure()
        plt.plot(t*pi,a/aL)
        plt.grid()
        plt.xlabel("Time [s]")
        plt.ylabel("Acc/AccLim")
        plt.title("K = {:.2G}. Violation: max  = {:.2G}%, mean = {:.2G}%".format(K,maxVio,meanVio))

        plt.show()

    return resV

def try_all(x, aL, K, pars=None,mustPrint=False,mustPlot=False):
    """This function does the same thing as try_pi_noLim, but assuming there
    is a limitation to the acceleration. The problem is that with this
    limitation, the shape of the solution is no longer as simple as before.

    In fact, it depends on all initial parameters of the costates, as well as
    total time pi. This dependence, of course, is highly non-linear.

    The approach in this function is to artificially add residual if
    some constraints (e.g., total time>0, height costate<0, etc) are not
    satisfied."""

    print("\nTrying x =",x)
    pi, lh, lv0, lm0 = x

    # High constant for raising the residual in case of violations
    L = 1e10

    # In case of negative time, it is better to just stop now.
    if pi <= 0.:
        return 1e9

    if pars is None:
        pars = pars_def

    g0Isp = pars['g0Isp']
    M0 = pars['M0']
    xi0 = (1.+lm0)/g0Isp - lv0/M0
    # 0 = xi = xi0 + (lh * pi / M0) * t1, hence:
    t1 = -xi0 * M0 / pi / lh

    resAdd = 0.
    if xi0 < 0.:
        resAdd += L * (xi0 **2 + 1.)
    if lh > 0.:
        resAdd += L * (lh**2 + 1.)
    if pi < 0.:
        resAdd += L * (pi**2 + 1.)
    if t1 > 1.:
        resAdd += L * t1 ** 2


    #print("xi0 = {:.4E}, t1 = {:.4E}".format(xi0,t1))

    if resAdd < 1.:
        lam0 = [lh, lv0, lm0]
        resV = intg_land(pi,t1,lam0,aL,K,pars,mustPlot=mustPlot)
    else:
        resV = numpy.zeros(4)
    return resV + resAdd

def try_all2(x, aL, K, pars=None,mustPrint=False,mustPlot=False):
    """This in an alternate version of try_all, with a different set of
     input variables (basically a change of coordinates).

     Since pi > 0, xi0 > 0, 0<t1<1, the input variables are chosen to be
     x0 = log(pi), x1 = arcsin(2*t1-1), x2 = lv0, x3 = log(xi0).

    In this approach, there is no need to artificially add residuals."""

    print("\nTrying x =",x)
    x0, x1, lv0, x3 = x

    pi = numpy.exp(x0)
    xi0 = numpy.exp(x3)
    t1 = .5 * (1. + numpy.sin(x1))

    if pars is None:
        pars = pars_def

    g0Isp = pars['g0Isp']
    M0 = pars['M0']

    lm0 = (xi0 + lv0/M0)*g0Isp - 1.
    #xi0 = (1.+lm0)/g0Isp - lv0/M0
    # 0 = xi = xi0 + (lh * pi / M0) * t1
    #t1 = -xi0 * M0 / pi / lh
    lh = -xi0 * M0 / pi / t1

    lam0 = [lh,lv0,lm0]
    resV = intg_land(pi,t1,lam0,aL,K,pars,mustPlot=mustPlot)

    return resV

def try_all3(x, aL, K, refs=None,pars=None,mustPrint=False,mustPlot=False):
    """This in an alternate version of try_all, with a different set of
     input variables (basically a change of coordinates).

     Since pi > 0, xi0 > 0, 0<t1<1, the input variables are chosen to be
     x0 = log(pi*t1)), x1 = log(pi*(1-t1)), x2 = lv0, x3 = log(xi0).

    In this approach, there is also no need to artificially add residuals."""

    print("\nTrying x =",x)
    x0, x1, x2, x3 = x
    if refs is None:
        print("Going with default refs...")
        refs = numpy.ones(4)

    pi1 = numpy.exp(x0) * refs[0]
    pi2 = numpy.exp(x1) * refs[1]
    pi = pi1 + pi2
    t1 = pi1 / pi

    lv0 = x2 * refs[2]
    xi0 = numpy.exp(x3) * refs[3]
    if pars is None:
        pars = pars_def

    g0Isp = pars['g0Isp']
    M0 = pars['M0']

    lm0 = (xi0 + lv0/M0)*g0Isp - 1.
    #xi0 = (1.+lm0)/g0Isp - lv0/M0
    # 0 = xi = xi0 + (lh * pi / M0) * t1
    lh = -xi0 * M0 / pi / t1

    lam0 = [lh,lv0,lm0]
    resV = intg_land(pi,t1,lam0,aL,K,pars,mustPlot=mustPlot)

    return resV


def getInitGuesLargK(aL,pars=None, mustPlot=False, numPlot=1000):
    """This function performs the calculations for the durations and
    initial values for costates in a solution that consists of two phases:
    no propulsion (free-fall) then constant-deceleration until full stop
    at height=0.

    This is never the "true" variational solution, but actually a limit when the penalty function
    gain K tends to infinity.
    """

    if pars is None:
        pars = pars_def

    g, g0Isp = pars['g'], pars['g0Isp']
    h0, v0, M0 = pars['h0'], pars['v0'], pars['M0']
    # quadratic equation for times (engine off: pi1, engine on: pi2)
    fat = aL/g - 1.
    A = -.5 * g * (fat + fat**2.)
    B = 0.
    C = h0 + .5 * v0**2/g

    pi2 = (-B - numpy.sqrt(B*B - 4.*A*C))/2./A
    pi1 = v0/g + pi2 * fat

    # total time and initial fraction
    pi = pi1 + pi2
    t1 = pi1 / pi

    # The initial values for the costates must be obtained by a linear system
    Mat = numpy.zeros((3,3))
    Col = numpy.zeros(3)

    coef = pi * aL / g0Isp

    exp_1t1 = numpy.exp(coef*(1.-t1))
    #texp_1t1_0 = exp_1t1 - t1
    #exp_1t1_0 = exp_1t1 - 1.

    # lm(1) = 0 condition
    #Mat[0, 0] = -(pi**2 * aL / M0) * (texp_1t1_0/coef - exp_1t1_0/coef/coef)
    Mat[0, 0] = -(pi ** 2 * aL / M0) * \
                (  ( (coef - 1.) * exp_1t1 -coef * t1 + 1.)/coef/coef  )
    Mat[0, 1] =  (g0Isp / M0) * (exp_1t1 - 1.)
    Mat[0, 2] = 1.

    # integral equation
    Col[1] = M0 * (1. - 1./exp_1t1) / pi
    Mat[1, 0] = -2. * h0 / pi + pi * aL * (1.-t1*t1) / 2.
    Mat[1, 1] = -v0 / pi - aL * (1. - t1)
    Mat[1, 2] = -M0 / pi

    # xi(t1) = 0
    Col[2] =  1. / g0Isp
    Mat[2, 0] = -pi1 / M0 # pi1, not pi
    Mat[2, 1] = 1. / M0
    Mat[2, 2] = -1. / g0Isp

    lams = numpy.linalg.solve(Mat,Col)
    lh0, lv0, lm0 = lams

    # obtain xi0, just for printing...
    xi0 = (1. + lm0) / g0Isp - lv0 / M0
    msg = "Finished initial guess.\npi = {:.5G}, t1 = {:.4G}".format(pi,t1)
    msg += "\nlh = {:.4G}, lv0 = {:.4G}, ".format(lh0,lv0)
    msg += "lm0 = {:.4G}, xi0 = {:.4G}".format(lm0,xi0)
    print(msg)

    if mustPlot:
        t = numpy.linspace(0.,1.,num=numPlot)
        td = t * pi

        lh = numpy.zeros_like(t) + lh0
        lv = lv0 - pi * lh0 * t
        #

        t1d = pi * t1
        vt1 = v0 - t1d * g
        ht1 = h0 + v0 * t1d - .5 * g * t1d**2
        exp = numpy.exp(coef*(t-t1))

        v = (v0 - g * td)*(t<t1) + (vt1 + (aL-g) * (td-t1d)) * (t>=t1)
        h = (h0 + v0 * td - .5 * g * td**2) * (t<t1) + \
            (ht1 + vt1 * (td - t1d) + .5 * (aL - g) * (td - t1d) ** 2) * (t>=t1)
        M = (M0+0.*t) * (t<t1) + (M0/exp)*(t>=t1)

        # texp_tt1 = (t-t1) * exp
        # lm = (lm0+0.*t) * (t<t1) + \
        #      (lm0 + (lv0 * g0Isp/M0) * (exp-1.) +
        #        -(pi*pi*aL*lh/M0)*( (t1/coef-1./coef/coef) * exp_tt1 +
        #                              texp_tt1/coef ) ) * (t>=t1)
        lm = (lm0 + 0. * t) * (t < t1) + \
             (lm0 + (lv0 * g0Isp / M0) * (exp-1.) +
              -(pi * pi * aL * lh / M0) *
                 ((coef * t - 1.) * exp - coef * t1 + 1.) / coef / coef
             ) * (t >= t1)

        fig, axs = plt.subplots(3, 2, constrained_layout=True)
        axs[0,0].plot(t*pi,h,label='h')
        #axs[0].set_title('subplot 1')
        axs[0,0].set_ylabel('Height [km]')
        axs[0,0].set_xlabel('Time [s]')#'Non-dim time [-]')
        axs[0, 0].grid()

        msg = 'Analytical solution for K -> \\infty, aLim = {:.1G}g '.format(aL/g)
        fig.suptitle(msg, fontsize=12)

        axs[1,0].plot(t*pi, v, label='v')
        axs[1,0].set_ylabel('Speed [km/s]')
        axs[1,0].set_xlabel('Time [s]')#'Non-dim time [-]')
        axs[1,0].grid()

        axs[2,0].plot(t*pi, M, label='M')
        axs[2,0].set_ylabel('Mass [kg]')
        axs[2,0].set_xlabel('Time [s]')#'Non-dim time [-]')
        axs[2, 0].grid()

        axs[0, 1].plot(t * pi, lh, label='h')
        # axs[0].set_title('subplot 1')
        axs[0, 1].set_ylabel('Height co. [kg/km]')
        axs[0, 1].set_xlabel('Time [s]')  # 'Non-dim time [-]')
        axs[0, 1].grid()

        axs[1, 1].plot(t * pi, lv, label='v')
        axs[1, 1].set_ylabel('Speed co. [kg s/km]')
        axs[1, 1].set_xlabel('Time [s]')  # 'Non-dim time [-]')
        axs[1, 1].grid()

        axs[2, 1].plot(t * pi, lm, label='M')
        axs[2, 1].set_ylabel('Mass co. [-]')
        axs[2, 1].set_xlabel('Time [s]')  # 'Non-dim time [-]')
        axs[2, 1].grid()

        # xi = (1.+lm)/g0Isp - lv/M
        # plt.figure()
        # plt.plot(t*pi,xi)
        # plt.grid()
        # plt.xlabel("Time [s]")
        # plt.ylabel("xi [s/m]")

        plt.show()

    return pi, t1, lams

if __name__ == "__main__":

    aL = 4. * pars_def['g']
    K = 1e8#3e4#7.5e3

    # try_pi_noLim(0.)
    # try_pi_noLim(1.)
    # try_pi_noLim(10.)
    # try_pi_noLim(100.)
    # try_pi_noLim(1000.)
    #pi,t1 = solve_pi_noLim()
    #print("pi = {:.5G}, t1 = {:5G}".format(pi,t1))


    #try_all((1.228e2,-6.,0.,0.),aL,K,pars=pars,mustPrint=True,mustPlot=True)
#    try_all((1.23e2, -6., 0., 0.), aL, K, pars=pars, mustPrint=True, mustPlot=True)
#    try_all((1.228e2, -7., 0., 0.), aL, K, pars=pars , mustPrint=True, mustPlot=True)
#    try_all((1.228e2, -6., 1., 0.), aL, K, pars=pars , mustPrint=True, mustPlot=True)
#    try_all((1.228e2, -6., 0., 1.), aL, K, pars=pars , mustPrint=True, mustPlot=True)

    # basic 1
    #x0 = numpy.array([1.228e2,-6.,0.,0.])
    #sol = root(try_all,x0,(aL,K))
    #print(sol)
    #try_all(sol.x, aL, K, mustPrint=True, mustPlot=True)

    # basic 2
    #x0 = numpy.array([numpy.log(1.228e2),numpy.arcsin(2.*0.8 -1),0.,numpy.log(0.5)])
    #sol = root(try_all2, x0, (aL, K))
    #print(sol)
    #try_all2(sol.x, aL, K, mustPrint=True, mustPlot=True)

    # THIS ONE WORKS (for these pars, anyway...)
    #sol = root(try_all3, (numpy.log(1.228e2*0.8), numpy.log(1.228e2*0.2), 0., numpy.log(0.5)), (aL, K))
    #print(sol)
    #try_all3(sol.x, aL, K, mustPrint=True, mustPlot=True)

    # This, unfortunately, does not (K>1e4), not even for these pars
    #refs = [pi*t1, pi*(1.-t1), 1., 0.5]
    #sol = root(try_all3, numpy.zeros(4), (aL, K, refs))  # ,method='lm')

    # This is the solution for K=1e4, rebased as refs for
    #refs = [pi * t1, pi * (1. - t1), 1., 0.5] #previous refs
    #x0 = numpy.array([-0.00895419,  0.04372009,  0.95353502, -0.08928519])

    #pi1 = numpy.exp(x0[0]) * refs[0]
    #pi2 = numpy.exp(x0[1]) * refs[1]
    #lv0 = 0.#x0[2] * refs[2]
    #xi0 = numpy.exp(x0[3]) * refs[3]

    #refs = numpy.array([pi1, pi2, lv0, xi0]) #rebasing refs
    #sol = root(try_all3, numpy.zeros(4), (aL, K, refs))  # ,method='lm')
    #print(sol)
    #try_all3(sol.x, aL, K, refs=refs, mustPrint=True, mustPlot=True)

    #sol = root(try_all3, (numpy.log(pi*t1), numpy.log(pi*(1.-t1)), 0., numpy.log(0.5)), (aL, K))
    #print(sol)
    #try_all3(sol.x, aL, K, refs=refs, mustPrint=True, mustPlot=True)

    # new crazy experiments
    # refs = [pi * t1, pi * (1. - t1), 1., 0.5] #previous refs
    # x0 = numpy.zeros(4)
    # e1 = numpy.array([1, 0, 0, 0])
    # e2 = numpy.array([0, 1, 0, 0])
    # e3 = numpy.array([0, 0, 1, 0])
    # e4 = numpy.array([0, 0, 0, 1])
    # h = 1e-5
    # resBase = try_all3(x0, aL, K, refs=refs, mustPrint=True, mustPlot=True)
    # res1 = try_all3(x0 + h * e1, aL, K, refs=refs, mustPrint=True, mustPlot=True)
    # res2 = try_all3(x0 + h * e2, aL, K, refs=refs, mustPrint=True, mustPlot=True)
    # res3 = try_all3(x0 + h * e3, aL, K, refs=refs, mustPrint=True, mustPlot=True)
    # res4 = try_all3(x0 + h * e4, aL, K, refs=refs, mustPrint=True, mustPlot=True)


    # Newest attempt: use the guess for large K

    pi, t1, lams = getInitGuesLargK(aL, mustPlot=True)
    xi0 = (1. + lams[2]) / pars_def['g0Isp'] - lams[1] / pars_def['M0']
    initGues = numpy.array((numpy.log(pi*t1), numpy.log(pi*(1.-t1)), lams[1], numpy.log(xi0)))
    sol = root(try_all3, initGues, (aL, K,numpy.ones(4)))
    print(sol)
    try_all3(sol.x, aL, K, mustPrint=True, mustPlot=True)
