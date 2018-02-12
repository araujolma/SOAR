#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 20:57:12 2018

@author: carlos
"""

import numpy
from atmosphere import rho
import matplotlib.pyplot as plt
from scipy.integrate import ode
from itsFolder.itsModelStaging import stagingCalculate
from itsFolder.itsModelPropulsion import (modelPropulsion,
                                          modelPropulsionHetSimple)


def mdlDer(t: float, x: list, alfaProg: callable, betaProg: callable,
           aed: callable, earth: callable)-> list:

    # initialization
    h = x[0]
    v = x[1]
    gamma = x[2]
    M = x[3]

    if numpy.isnan(h):
        print('t: ', t)
        #  print('x: ', x)
        raise Exception('itsme saying: h is not a number')

    # Alpha control calculation
    alfat = alfaProg(t)

    # other calculations
    # btm = betaProg(t)*con['T']/M
    beta, Isp, T = betaProg(t)
    btm = beta*T/M
    sinGamma = numpy.sin(gamma)
    g = earth.g0*(earth.R/(earth.R+h))**2 - (earth.we**2)*(earth.R + h)

    # aerodynamics
    qdinSrefM = 0.5 * rho(h) * (v**2) * aed.s_ref/M
    LM = qdinSrefM * (aed.CL0 + aed.CL1*alfat)
    DM = qdinSrefM * (aed.CD0 + aed.CD2*(alfat**2))

    if v < 1e-8:
        v = 1e-8

    # states derivatives
    return [v*sinGamma,  # coefficient
            btm*numpy.cos(alfat) - g*sinGamma - DM,  # coefficient
            btm*numpy.sin(alfat)/v +
            (v/(h+earth.R)-g/v)*numpy.cos(gamma) +
            (LM/v) + 2*earth.we,  # coefficient
            -btm*M/(earth.g0*Isp)]  # coefficient


class model():

    def __init__(self, factors: list, con: dict):

        self.factors = factors
        self.con = con
        self.traj = modelTrajectory()
        self.flagAppend = False
        self.simulCounter = 0

        #######################################################################
        # Delta V estimates
        fdv2, ftf, fdv1 = self.factors
        Dv1 = fdv1*con['Dv1ref']
        tf = ftf*con['tref']
        Dv2 = con['V_final'] - fdv2*con['vxref']
        self.Dv2 = Dv2
        # print('V_final: ', con['V_final'])

        #######################################################################
        # Staging calculation
        self.p1, self.p2 = stagingCalculate(self.con, Dv1, Dv2)

        #######################################################################
        # Thrust program
        if con['homogeneous']:
            tabBeta = modelPropulsion(self.p1, self.p2, tf, 1.0, 0.0,
                                      con['softness'], con['Isp'], con['T'])
        else:
            tabBeta = modelPropulsionHetSimple(self.p1, self.p2, tf, 1.0, 0.0)
        if tabBeta.fail:
            raise Exception('itsme saying: Softness too high!')
        self.tabBeta = tabBeta
        # self.tabBeta.show()

        #######################################################################
        # Attitude program definition
        self.tAoA2 = con['tAoA1'] + con['tAoA']
        tabAlpha = modelAttitude(con['tAoA1'], self.tAoA2, 0,
                                 -con['AoAmax']*con['d2r'])
        self.tabAlpha = tabAlpha

        self.aed = modelAed(con)
        self.earth = modelEarth(con)

    def __integrate(self, ode45: object, t0: float, x0)-> None:

        self.simulCounter += 1
        self.traj.tphases = self.tphases
        self.traj.massJet = self.mjetsoned

        # Output variables
        self.traj.append(t0, x0)
        self.traj.appendP(t0, x0)

        N = len(self.tphases)
        initialValue = ode45.set_initial_value
        integrate = ode45.integrate
        appendP = self.traj.appendP
        for ii in range(1, N):

            # Time interval configuration
            t_initial = self.tphases[ii - 1] + self.con['contraction']
            t_final = self.tphases[ii] - self.con['contraction']

            # Stage separation mass reduction
            y_initial = self.traj.xp[-1]
            y_initial[3] = y_initial[3] - self.mjetsoned[ii-1]

            self.traj.mass0.append(y_initial[3].copy())

            # integration
            initialValue(y_initial, t_initial)
            ode45.first_step = (t_final - t_initial)*0.001
            integrate(t_final)

            if ii != N-1:
                appendP(ode45.t+self.con['contraction'], ode45.y)

            # Phase itegration display
            if self.flagAppend:
                if ii == 1:
                    print("Phase integration iteration: 1", end=',  '),
                elif ii == (len(self.tphases) - 1):
                    print('')
                else:
                    print(ii, end=',  ')

        # Final itegration procedures
        # Final stage separation mass reduction
        y_final = ode45.y.copy()
        y_final[3] = y_final[3] - self.mjetsoned[-1]
        self.traj.appendP(ode45.t+self.con['contraction'], y_final)

        # Final point appending
        self.traj.append(self.traj.tp[-1], self.traj.xp[-1])
        self.traj.numpyArray()
        return None

    def __calcAedTab(self, tt, xx, uu)-> tuple:

        con = self.con
        LL = tt.copy()
        DD = tt.copy()
        CCL = tt.copy()
        CCD = tt.copy()
        QQ = tt.copy()
        for ii in range(0,  len(tt)):

            h = xx[ii, 0]
            v = xx[ii, 1]
            alfat = uu[ii, 0]
            # Aerodynamics
            CL = con['CL0'] + con['CL1']*alfat
            CD = con['CD0'] + con['CD2']*(alfat**2)

            qdin = 0.5 * rho(h) * (v**2)
            L = qdin * con['s_ref'] * CL
            D = qdin * con['s_ref'] * CD

            LL[ii] = L
            DD[ii] = D
            CCL[ii] = CL
            CCD[ii] = CD
            QQ[ii] = qdin

        return LL,  DD,  CCL,  CCD,  QQ

    def __cntrCalculate(self, tabAlpha, tabBeta):

        self.uu = numpy.concatenate([tabAlpha.multValue(self.traj.tt),
                                     tabBeta.multValue(self.traj.tt)],  axis=1)
        self.up = numpy.concatenate([tabAlpha.multValue(self.traj.tp),
                                     tabBeta.multValue(self.traj.tp)],  axis=1)

    def simulate(self, typeResult: str)-> None:
        #######################################################################
        con = self.con
        self.simulCounter += 1

        #######################################################################
        # Integration
        # Initial conditions
        t0 = 0.0
        x0 = [con['h_initial'], con['V_initial'],
              con['gamma_initial'], self.p1.mtot[0]]

        #######################################################################
        # Phase times and jetsonned masses
        self.__phasesSetting(t0, con, typeResult)
        # tphases = [t0, con['tAoA1'], self.tAoA2] + self.tabBeta.tflistP1
        # mjetsoned = [0.0, 0.0, 0.0] + self.tabBeta.melistP1

# =============================================================================
#         arg = numpy.argsort(tphases0)
#         tphases = []
#         mjetsoned = []
#         for ii in arg:
#             tphases.append(tphases0[ii])
#             mjetsoned.append(mjetsoned0[ii])
#
# =============================================================================

        #######################################################################
        # Integrator setting
        # ode set:
        #         atol: absolute tolerance
        #         rtol: relative tolerance
        ode45 = ode(mdlDer).set_integrator('dopri5', atol=con['tol']/1,
                                           rtol=con['tol']/10)
        ode45.set_initial_value(x0,  t0)
        ode45.set_f_params(self.tabAlpha.value, self.tabBeta.mdlDer,
                           self.aed, self.earth)

        # Integration using rk45 separated by phases
        if (typeResult == "design"):
            # Fast running
            self.__integrate(ode45, t0, x0)
        else:
            # Slow running
            # Check phases time monotonic increse
            for ii in range(1, len(self.tphases)):
                if self.tphases[ii - 1] >= self.tphases[ii]:
                    print('tphases = ', self.tphases)
                    raise Exception('itsme saying: tphases does ' +
                                    'not increase monotonically!')

            # Integration
            ode45.set_solout(self.traj.appendStar)
            self.__integrate(ode45, t0, x0)
            self.__cntrCalculate(self.tabAlpha, self.tabBeta)

            # Check solution time monotonic increse
            if self.con['contraction'] > 0.0:
                for ii in range(1, len(self.traj.tt)):
                    if self.traj.tt[ii - 1] >= self.traj.tt[ii]:
                        print('ii = ', ii, ' tt[ii-1] = ', self.traj.tt[ii-1],
                              ' tt[ii] = ', self.traj.tt[ii])
                        raise Exception('itsme saying: tt does not' +
                                        ' increase monotonically!')

        self.__errorCalculate()

        return None

    def displayInfo(self)-> None:

        con = self.con
        p1 = self.p1
        p2 = self.p2
        if con['NStag'] == 0:
            print("\n\rSpacecraft properties (NStag == 0):")
            print("Empty spacecraft mass = ", p2.ms)

        elif con['NStag'] == 1:
            print("\n\rSSTO properties (NStag == 1):")

        elif con['NStag'] == 2:
            print("\n\rTSTO vehicle (NStag == 2):")

        else:
            print("\n\rStaged vehicle (NStag == %i):" % con['NStag'])

        print("\n\rVehicle properties on ascending phases:")
        p1.printInfo()
        print("\n\rVehicle properties on orbital phases:")
        p2.printInfo()

        print('\n\rtphases: ', self.tphases)
        print('\n\rmjetsoned: ', self.mjetsoned)
        print('\n\r')

    def plotResults(self):

        (tt, xx, uu, tp, xp, up) = (self.traj.tt, self.traj.xx, self.uu,
                                    self.traj.tp, self.traj.xp, self.up)

        ii = 0
        plt.subplot2grid((6, 4), (0, 0), rowspan=2, colspan=2)
        plt.plot(tt, xx[:, ii], '.-b', tp, xp[:, ii], '.r')
        plt.grid(True)
        plt.ylabel("h [km]")

        ii = 1
        plt.subplot2grid((6, 4), (0, 2), rowspan=2, colspan=2)
        plt.plot(tt, xx[:, ii], '.-b', tp, xp[:, ii], '.r')

        plt.grid(True)
        plt.ylabel("V [km/s]")

        ii = 2
        plt.subplot2grid((6, 4), (2, 0), rowspan=2, colspan=2)
        plt.plot(tt, xx[:, ii]*180.0/numpy.pi, '.-b',
                 tp, xp[:, ii]*180.0/numpy.pi, '.r')
        plt.grid(True)
        plt.ylabel("gamma [deg]")

        ii = 3
        plt.subplot2grid((6, 4), (2, 2), rowspan=2, colspan=2)
        plt.plot(tt, xx[:, ii], '.-b', tp, xp[:, ii], '.r')
        plt.grid(True)
        plt.ylabel("m [kg]")

        ii = 0
        plt.subplot2grid((6, 4), (4, 0), rowspan=2, colspan=2)
        plt.plot(tt, uu[:, ii]*180/numpy.pi, '.-b',
                 tp, up[:, ii]*180/numpy.pi, '.r')
        plt.grid(True)
        plt.ylabel("alfa [deg]")

        ii = 1
        plt.subplot2grid((6, 4), (4, 2), rowspan=2, colspan=2)
        plt.plot(tt, uu[:, ii], '.-b',
                 tp, up[:, ii], '.r')
        plt.grid(True)
        plt.xlabel("t")
        plt.ylabel("beta [adim]")

        plt.show()

        return None

    def plotResultsAed(self):

        tt = self.traj.tt
        tp = self.traj.tp
        # Aed plots
        LL,  DD,  CCL,  CCD,  QQ = self.__calcAedTab(tt,
                                                     self.traj.xx, self.uu)
        Lp,  Dp,  CLp,  CDp,  Qp = self.__calcAedTab(tp,
                                                     self.traj.xp, self.up)

        plt.subplot2grid((6, 2), (0, 0), rowspan=2, colspan=2)
        plt.plot(tt, LL, '.-b', tp, Lp, '.r', tt, DD, '.-g', tp, Dp, '.r')
        plt.grid(True)
        plt.ylabel("L and D [kN]")

        plt.subplot2grid((6, 2), (2, 0), rowspan=2, colspan=2)
        plt.plot(tt, CCL, '.-b', tp, CLp, '.r', tt, CCD, '.-g', tp, CDp, '.r')
        plt.grid(True)
        plt.ylabel("CL and CD [-]")

        plt.subplot2grid((6, 2), (4, 0), rowspan=2, colspan=2)
        plt.plot(tt, QQ, '.-b', tp, Qp, '.r')
        plt.grid(True)
        plt.ylabel("qdin [kPa]")

        plt.show()

        return None

    def orbitResults(self):

        GM = self.con['GM']
        R = self.con['R']

        h, v, gama, M = numpy.transpose(self.traj.xx[-1, :])
        r = R + h

        cosGama = numpy.cos(gama)
        sinGama = numpy.sin(gama)
        vt = v*cosGama + self.con['we']*r
        vr = v*sinGama
        v = numpy.sqrt(vt**2 + vr**2)
        cosGama = vt/v
        sinGama = vr/v

        momAng = r * v * cosGama
        print("Ang mom:", momAng)
        en = .5 * v * v - GM/r
        print("Energy:", en)
        a = - .5*GM/en
        print("Semi-major axis:", a)
        aux = v * momAng / GM
        e = numpy.sqrt((aux * cosGama - 1)**2 + (aux * sinGama)**2)
        print("Eccentricity:", e)

        print("Final altitude:", h)
        ph = a * (1.0 - e) - R
        print("Perigee altitude:", ph)
        ah = 2*(a - R) - ph
        print("Apogee altitude:", ah)
        print('\n\r')

        self.e = e


class modelAttitude2():

    def __init__(self, t1: float, t2: float, v1: float, v2: float):
        self.t1 = t1
        self.t2 = t2

        self.v1 = v1
        self.v2 = v2

    def value(self, t: float)-> float:
        if (t >= self.t1) and (t <= self.t2):
            ans = self.v2
        else:
            ans = self.v1

        return ans

    def multValue(self, t: float):
        N = len(t)
        ans = numpy.full((N, 1), 0.0)
        for jj in range(0, N):
            ans[jj] = self.value(t[jj])

        return ans


class modelAttitude():

    def __init__(self, t1: float, t2: float, v1: float, v2: float):
        self.t1 = t1
        self.t2 = t2

        self.v1 = v1
        self.v2 = v2

    def value(self, t: float)-> float:
        if (t >= self.t1) and (t < self.t2):
            ans = ((self.v2 - self.v1)*(1 -
                   numpy.cos(2*numpy.pi*(t - self.t1)/(self.t2 - self.t1)))/2)\
                   + self.v1
            return ans
        else:
            return self.v1

    def multValue(self, t: float):
        N = len(t)
        ans = numpy.full((N, 1), 0.0)
        for jj in range(0, N):
            ans[jj] = self.value(t[jj])

        return ans


class modelTrajectory():

    def __init__(self):

        self.tt = []
        self.xx = []
        self.tp = []
        self.xp = []
        self.tphases = []
        self.mass0 = []
        self.massJet = []

    def append(self, tt, xx):

        self.tt.append(tt)
        self.xx.append(xx)

    def appendStar(self, tt, xx):

        self.tt.append(tt)
        self.xx.append([*xx])

    def appendP(self, tp, xp):

        self.tp.append(tp)
        self.xp.append(xp)

    def numpyArray(self):

        self.tt = numpy.array(self.tt)
        self.xx = numpy.array(self.xx)
        self.tp = numpy.array(self.tp)
        self.xp = numpy.array(self.xp)


class modelAed():

    def __init__(self, con):

        self.s_ref = con['s_ref']
        self.CL0 = con['CL0']
        self.CL1 = con['CL1']
        self.CD0 = con['CD0']
        self.CD2 = con['CD2']


class modelEarth():

    def __init__(self, con):

        self.g0 = con['g0']
        self.R = con['R']
        self.we = con['we']
