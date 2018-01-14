# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:14:40 2017

@author: Carlos

Initial Trajectory Setup ModulE

Version: Objetification

"""

import numpy
import configparser
import matplotlib.pyplot as plt
from scipy.integrate import ode
from atmosphere import rho
from time import clock


def its(*arg):

    # arguments analisys
    if len(arg) == 0:
        fname = 'default.its'
    elif len(arg) == 1:
        fname = arg[0]
    else:
        raise Exception('itsme saying: too many arguments on its')

    print("itsme: Inital Trajectory Setup Module")
    print("Opening case: ", fname)

    problem1 = problem(fname)

    solution1 = problem1.solveForInitialGuess()

    solution1.displayResults()

    solution2 = problem1.solveForFineTune()

    solution2.basic.plotResultsAed()

    solution2.displayResults()

    if not solution2.converged():
            print('itsme saying: solution has not converged :(')

    return solution2


def sgra(fname: str):

    # arguments analisys

    print("itsme: Inital Trajectory Setup Module")
    print("Opening case: ", fname)

    solution = problem(fname).solveForFineTune()

    solution.basic.displayInfo()
    solution.basic.orbitResults()

    print('Initial states:', solution.basic.traj.xx[0])
    print('Final   states:', solution.basic.traj.xx[-1])

    if not solution.converged():
            print('itsme saying: solution has not converged :(')

    return solution.sgra()


def mdlDer(t: float, x: list, alfaProg: callable, betaProg: callable,
           aed: callable, earth: callable)-> list:

    # initialization
    h = x[0]
    v = x[1]
    gamma = x[2]
    M = x[3]

    if numpy.isnan(h):
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

    # states derivatives
    return [v*sinGamma,  # coefficient
            btm*numpy.cos(alfat) - g*sinGamma - DM,  # coefficient
            btm*numpy.sin(alfat)/v +
            (v/(h+earth.R)-g/v)*numpy.cos(gamma) +
            (LM/v) + 2*earth.we,  # coefficient
            -btm*M/(earth.g0*Isp)]  # coefficient


def itsTester():
    # Some aplications of itsme functions and objects
    its()
    # test list
    testList = ['itsme_test_cases/caseEarthRotating.its',
                'itsme_test_cases/caseMoon.its',
                'itsme_test_cases/caseMu050h200NStag2.its',
                'itsme_test_cases/caseMu050h600NStag2.its',
                'itsme_test_cases/caseMu100h463NStag0.its',
                'itsme_test_cases/caseMu100h463NStag1.its',
                'itsme_test_cases/caseMu100h463NStag2.its',
                'itsme_test_cases/caseMu100h463NStag3.its',
                'itsme_test_cases/caseMu100h463NStag4.its',
                'itsme_test_cases/caseMu100h463NStag5.its',
                'itsme_test_cases/caseMu100h1500NStag3.its',
                'itsme_test_cases/caseMu150h500NStag4.its']

    for case in testList:
        if not its(case).converged():
            raise Exception('itsme saying: solution did not converge')

    problem('itsme_test_cases/' +
            'caseEarthRotating.its').solveForFineTune()
    problem('itsme_test_cases/' +
            'caseMoon.its').solveForFineTune()
    problem('itsme_test_cases/' +
            'caseMu150h500NStag4.its').solveForFineTune()

    sgra('default.its')


class problem():

    def __init__(self, fileAdress: str):

        # TODO: solve the codification problem on configuration files
        configuration = problemConfiguration(fileAdress)
        configuration.environment()
        configuration.initialState()
        configuration.finalState()
        configuration.vehicle()
        configuration.trajectory()
        configuration.solver()

        self.con = configuration.con
        self.fsup = self.con['fsup']
        self.finf = self.con['finf']

    def solveForInitialGuess(self):
        #######################################################################
        # First guess trajectory
        model1 = model(self.con['guess'], self.con)
        model1.simulate("design")
        self.tabAlpha = model1.tabAlpha
        self.tabBeta = model1.tabBeta
        self.errors = model1.errors
        self.factors = self.con['guess']

        return solution(self.con['guess'], self.con)

    def solveForFineTune(self)-> object:
        # Bisection altitude loop
        self.iteractionAltitude = problemIteractions('Altitude')
        self.iteractionSpeedAndAng = problemIteractions('All')

        print("\n#################################" +
              "######################################")
        errors, factors = self.__bisecAltitude()

        # Final test
        model1 = model(factors, self.con)
        model1.simulate("design")

        print("ITS the end (lol)")

        self.displayErrorsFactors(errors, factors)
        print('\nTotal number of trajectory simulations: ',
              (self.iteractionAltitude.count +
               self.iteractionSpeedAndAng.count))

        solution1 = solution(factors, self.con)
        solution1.iteractionAltitude = self.iteractionAltitude
        solution1.iteractionSpeedAndAng = self.iteractionSpeedAndAng

        if not solution1.converged():
            print('itsme saying: solution has not converged :(')

        return solution1

    def __bisecAltitude(self)-> tuple:
        #######################################################################
        # Bisection altitude loop
        index = 2
        tol = self.con['tol']

        # Initilization
        stop = False
        df = abs((self.fsup[index] - self.finf[index])/self.con['Ndiv'])

        factors = (self.fsup + self.finf)/2
        errors, factors = self.__bisecSpeedAndAng(factors)
        self.iteractionSpeedAndAng.update(errors, factors)

        e1 = errors[index]
        f1 = factors[index]
        self.iteractionAltitude.update([e1], [f1])

        f2 = f1 - df

        # Loop
        update = self.iteractionAltitude.update
        while not stop and (self.iteractionAltitude.count <= self.con['Nmax']):

            # bisecSpeedAndAng: Error update from speed and gamma loop
            factors[index] = f2
            errors, factors = self.__bisecSpeedAndAng(factors)
            e2 = errors[index]
            update([e2], [f2])

            # Loop checks
            if (abs(e2) < tol):
                stop = True
                title = "bisecAltitude final iteration: "

            else:
                # Calculation of the new factor
                f2 = self.iteractionAltitude.newFactor(0, df, self.con)
                title = "bisecAltitude iteration: "

            # Display information
            print(title, self.iteractionAltitude.count - 1)

            self.displayErrorsFactors(errors, factors)

        return errors, factors

    def __bisecSpeedAndAng(self, factors1: list)-> tuple:
        #######################################################################
        # Bissection speed and gamma loop
        # Initializing parameters
        # Loop initialization
        stop = False
        self.iteractionSpeedAndAng.reset()

        # Fators initilization
        df = abs((self.fsup - self.finf)/self.con['Ndiv'])
        # Making the 3 factor variarions null
        df[2] = 0.0

        model1 = model(factors1, self.con)
        model1.simulate("design")
        errors1 = model1.errors
        self.iteractionSpeedAndAng.update(errors1, factors1)

        factors2 = factors1 - df

        # Loop
        update = self.iteractionSpeedAndAng.update

        while ((not stop) and
               (self.iteractionSpeedAndAng.countLocal <= self.con['Nmax'])):
            # Error update
            model1 = model(factors2, self.con)
            model1.simulate("design")
            errors2 = model1.errors

            update(errors2, factors2)

            converged = abs(errors2) < self.con['tol']
            if converged[0] and converged[1]:

                stop = True
                # Display information
                if self.iteractionSpeedAndAng.countLocal == self.con['Nmax']:
                    print("bisecSpeedAndAng total iterations: ",
                          self.iteractionSpeedAndAng.countLocal,
                          " (max)")
                else:
                    print("bisecSpeedAndAng total iterations: ",
                          self.iteractionSpeedAndAng.countLocal)

                self.displayErrorsFactors(errors2, factors2)

            elif self.iteractionSpeedAndAng.stationary():

                stop = True
                # Display information
                print('stationary process')
                if self.iteractionSpeedAndAng.countLocal == self.con['Nmax']:
                    print("bisecSpeedAndAng total iterations: ",
                          self.iteractionSpeedAndAng.countLocal,
                          " (max)")
                else:
                    print("bisecSpeedAndAng total iterations: ",
                          self.iteractionSpeedAndAng.countLocal)

                self.displayErrorsFactors(errors2, factors2)

            else:
                # real iteractions
                f0 = self.iteractionSpeedAndAng.newFactor(0, df[0], self.con)
                f1 = self.iteractionSpeedAndAng.newFactor(1, df[1], self.con)
                factors2 = [f0, f1, factors2[2]]

        return errors2, factors2

    def displayErrorsFactors(self, errors: list, factors: list)-> None:

        num = "8.6e"
        print(("Errors    : %"+num+",  %"+num+",  %"+num)
              % (errors[0], errors[1], errors[2]))
        print(("Sup limits: %"+num+",  %"+num+",  %"+num)
              % (self.fsup[0], self.fsup[1], self.fsup[2]))
        print(("Factors   : %"+num+",  %"+num+",  %"+num)
              % (factors[0], factors[1], factors[2]))
        print(("Inf limits: %"+num+",  %"+num+",  %"+num)
              % (self.finf[0], self.finf[1], self.finf[2]))
        print("\n#################################" +
              "######################################")


class problemConfiguration():

    def __init__(self, fileAdress: str):
        # TODO: solve the codification problem on configuration files
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.config.read(fileAdress)
        self.con = dict()

    def environment(self):
        # Enviromental constants
        section = 'enviroment'
        items = self.config.items(section)
        for para in items:
            self.con[para[0]] = self.config.getfloat(section, para[0])
        # [km/s2] gravity acceleration on earth surface
        self.con['g0'] = self.con['GM']/(self.con['R']**2)

        # General constants
        self.con['pi'] = numpy.pi
        self.con['d2r'] = numpy.pi/180.0

    def initialState(self):
        # Initial state constants
        self.con['h_initial'] = self.config.getfloat('initial', 'h')
        self.con['V_initial'] = self.config.getfloat('initial', 'V')
        self.con['gamma_initial'] = \
            self.config.getfloat('initial', 'gamma')*self.con['d2r']

    def finalState(self):
        # Final state constants
        self.con['h_final'] = self.config.getfloat('final', 'h')
        # Circular velocity
        vc = numpy.sqrt(self.con['GM']/(self.con['R']+self.con['h_final']))
        # Rotating referencial velocity effect
        ve = self.con['we']*(self.con['R']+self.con['h_final'])
        self.con['V_final'] = vc - ve  # km/s Final velocity
        self.con['gamma_final'] = \
            self.config.getfloat('final', 'gamma')*self.con['d2r']

    def vehicle(self):
        # Vehicle parameters
        section = 'vehicle'

        if not self.config.has_option(section, 'homogeneous'):
            self.con['homogeneous'] = True
        else:
            self.con['homogeneous'] = \
                self.config.getboolean(section, 'homogeneous')

        # This flag show indicates if the vehicle shall be considered as having
        # the same values of structural mass and thrust for all stages
        if self.con['homogeneous']:
            self.__getVehicleHomogeneous(self.config)
        else:
            self.__getVehicleHeterogeneous(self.config)

    def trajectory(self):
        # Trajectory parameters
        section = 'trajectory'
        items = self.config.items(section)
        for para in items:
            self.con[para[0]] = self.config.getfloat(section, para[0])
        # Time of one orbit using the final velocity
        self.con['torb'] = 2*self.con['pi']*(self.con['R'] +
                                             self.con['h_final']
                                             )/self.con['V_final']

    def solver(self):
        # Solver parameters
        section = 'solver'
        self.con['tol'] = self.config.getfloat(section, 'tol')
        self.con['contraction'] = (1.0e4)*numpy.finfo(float).eps

        # Superior and inferior limits
        auxstr = self.config.get(section, 'guess')
        auxstr = auxstr.split(', ')
        auxnum = []
        for n in auxstr:
            auxnum = auxnum+[float(n)]
        guess = numpy.array(auxnum)

        auxstr = self.config.get(section, 'limit')
        auxstr = auxstr.split(', ')
        auxnum = []
        for n in auxstr:
            auxnum = auxnum+[float(n)]
        limit = numpy.array(auxnum)

        self.con['guess'] = guess
        self.con['fsup'] = guess + limit
        self.con['finf'] = guess - limit

        if self.config.has_option(section, 'Nmax'):
            self.con['Nmax'] = self.config.getint(section, 'Nmax')
        else:
            self.con['Nmax'] = 100

        if self.config.has_option(section, 'Ndiv'):
            self.con['Ndiv'] = self.config.getint(section, 'Ndiv')
        else:
            self.con['Ndiv'] = 10

        # Reference values
        iniEst = problemInitialEstimate(self.con)
        self.con['Dv1ref'] = iniEst.dv
        self.con['tref'] = iniEst.t
        self.con['vxref'] = iniEst.vx

    def __getVehicleHomogeneous(self, config):

        section = 'vehicle'
        items = config.items(section)

        for para in items:
            self.con[para[0]] = config.getfloat(section, para[0])

        # Number of stages
        self.con['NStag'] = config.getint('vehicle', 'NStag')
        self.con['Isp1'] = self.con['Isp']
        self.con['Isp2'] = self.con['Isp']

        # This flag show indicates if the vehicle shall be considered as having
        # the same
        # values of structural mass and thrust for all stages
        efflist = []
        Tlist = []
        if self.con['NStag'] > 1:
            for jj in range(0, self.con['NStag']):
                efflist = efflist+[self.con['efes']]
                Tlist = Tlist+[self.con['T']]
        else:
            # This cases are similar to NStag == 2,  the differences are:
            # for NStag == 0 no mass is jetsoned
            # for NStag == 1 all structural mass is jetsoned at the end of all
            # burning
            for jj in range(0, 2):
                efflist = efflist+[self.con['efes']]
                Tlist = Tlist+[self.con['T']]

        self.con['efflist'] = efflist
        self.con['Tlist'] = Tlist

    def __getVehicleHeterogeneous(self, config):

        section = 'vehicle'
        items = config.items(section)

        for para in items:
            if (para[0] != 'efes') and (para[0] != 'T'):
                self.con[para[0]] = config.getfloat(section, para[0])

        self.con['NStag'] = config.getint(section, 'NStag')  # Number of stages

        print(self.con['Isp1'])

        auxstr = config.get(section, 'efes')
        auxstr = auxstr.split(', ')
        auxnum = []
        for n in auxstr:
            auxnum = auxnum+[float(n)]
        self.con['efflist'] = auxnum

        auxstr = config.get(section, 'T')
        auxstr = auxstr.split(', ')
        auxnum = []
        for n in auxstr:
            auxnum = auxnum+[float(n)]
        self.con['Tlist'] = auxnum


class problemIteractions():

    def __init__(self, name: str):

        self.name = name
        self.errorsList = []
        self.factorsList = []
        self.count = 0
        self.countLocal = 0

    def reset(self):

        self.countLocal = 0

    def update(self, errors: list, factors: list)-> None:

        self.count += 1
        self.countLocal += 1
        self.errorsList.append(errors)
        self.factorsList.append(factors)
        return None

    def newFactor(self, index: int, df: list, con: dict)-> float:

        # Checkings
        # Division and step check
        de = self.errorsList[-1][index] - self.errorsList[-2][index]
        if abs(de) == 0:  # <= con['tol']*1e-2:
            step = 0.0

        else:
            der = (self.factorsList[-1][index] -
                   self.factorsList[-2][index])/de
            step = self.errorsList[-1][index]*der
            if step > df:
                step = 0.0 + df
            elif step < -df:
                step = 0.0 - df

        # Factor definition and check
        f3 = self.factorsList[-1][index] - step

        if f3 > con['fsup'][index]:
            f3 = 0.0 + con['fsup'][index]
        elif f3 < con['finf'][index]:
            f3 = 0.0 + con['finf'][index]

        return f3

    def stationary(self)-> bool:

        stat = False
        if self.countLocal > 10:
            e = numpy.array([numpy.array(self.errorsList[-3][0:2]),
                             numpy.array(self.errorsList[-2][0:2]),
                             numpy.array(self.errorsList[-1][0:2])])

            ff = numpy.std(e, 0)/abs(numpy.mean(e, 0))

            for f in ff:
                stat = stat or (f < 1e-12)

        return stat

    def plotLogErrors(self, legendList: list):

        if self.count != 0:
            err = numpy.array(self.errorsList)
            for e in err:
                e = numpy.array(e)
            err = numpy.array(err)

            if numpy.ndim(err) == 2:
                for ii in range(0, len(self.errorsList[-1])):
                    plt.semilogy(range(0, len(err[:, ii])),
                                 abs(err[:, ii]), label=legendList[ii])

            else:
                plt.semilogy(range(0, len(err)),
                             abs(err), label=legendList[0])
            plt.grid(True)
            plt.ylabel("erros []")
            plt.xlabel("iteration number []")
            plt.title(self.name)
#            if self.name == 'All errors':
            plt.legend()
            plt.show()
            return None

    def plotFactors(self, legendList: list):

        if self.count != 0:
            err = numpy.array(self.factorsList)
            for e in err:
                e = numpy.array(e)
            err = numpy.array(err)

            if numpy.ndim(err) == 2:
                for ii in range(0, len(self.factorsList[-1])):
                    plt.plot(range(0, self.count),
                             abs(err[:, ii]), label=legendList[ii])

            else:
                plt.plot(range(0, self.count),
                         abs(err), label=legendList[0])
            plt.grid(True)
            plt.ylabel("factors []")
            plt.xlabel("iteration number []")
            plt.title(self.name)
#            if self.name == 'All errors':
            plt.legend()
            plt.show()
            return None


class problemInitialEstimate():

    def __init__(self, con: dict):

        efflist = con['efflist']
        Tlist = con['Tlist']

        lamb = numpy.exp(0.5*con['V_final']/(con['Isp2']*con['g0']))
        self.Mu = con['Mu']*lamb
        self.hf = con['h_final']
        self.M0max = Tlist[0]/con['g0']
        self.c = con['Isp1']*con['g0']
        self.mflux = numpy.mean(Tlist[0:-1])/self.c
        self.GM = con['GM']
        self.R = con['R']
        self.e = numpy.exp(numpy.mean(numpy.log(efflist[0:-1])))
        self.g0 = con['g0'] - con['R']*(con['we'] ** 2)
        self.t1max = self.M0max/self.mflux
        self.fail = False

        self.__newtonRaphson()
        self.__dvt()

        self.vx = 0.5*self.V1*(con['R'] + self.h1)/(con['R']+self.hf)

    def __newtonRaphson(self):

        N = 100
        t1max = self.t1max

        dt = t1max/N
        t1 = 0.1*t1max

        cont = 0
        erro = 1.0

        while (abs(erro) > 1e-6) and not self.fail:

            erro = self.__hEstimate(t1) - self.hf
            dedt = (self.__hEstimate(t1+dt) - self.__hEstimate(t1-dt))/(2*dt)
            t1 = t1 - erro/dedt
            cont += 1
            if cont == N:
                raise Exception('itsme saying: initialEstimate failed')

        self.cont = cont
        self.t1 = t1
        return None

    def __hEstimate(self, t1: float)-> float:

        mflux = self.mflux
        Mu = self.Mu
        g0 = self.g0
        c = self.c

        Mp = mflux*t1
        Me = Mp*self.e/(1 - self.e)
        M0 = Mu + Me + Mp
        x = (Mu + Me)/M0
        V1 = c*numpy.log(1/x) - g0*t1
        h1 = (c*M0/mflux) * ((x*numpy.log(x) - x) + 1) - g0*(t1**2)/2
        h = h1 + self.GM/(self.GM/(self.R + h1) - (V1**2)/2) - self.R

        return h

    def __dvt(self)-> None:

        t1 = self.t1
        mflux = self.mflux
        Mu = self.Mu
        g0 = self.g0
        c = self.c

        M0 = Mu + mflux*t1
        V1 = c*numpy.log(M0/Mu) - g0*t1
        dv = V1 + g0*t1
        x = Mu/M0
        h1 = (c*M0/mflux)*((x*numpy.log(x) - x) + 1) - g0*(t1**2)/2

        h = self.__hEstimate(self.t1)

        rr = (self.R + h1)/(self.R + h)
        if rr > 1:
            raise Exception('itsme saying: initialEstimate failed (h1 > h)')
        theta = numpy.arccos(numpy.sqrt(rr))
        ve = numpy.sqrt(2*self.GM/(self.R + h))

        t = t1 + ((self.R + h)/ve) * (numpy.sin(2*theta)/2 + theta)

        self.h1 = h1
        self.h = h
        self.t = t
        self.dv = dv
        self.V1 = V1

        return None


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
        # print('V_final: ', con['V_final'])

        #######################################################################
        # Staging calculation
        self.__stagingCalculate(Dv1, Dv2)

        #######################################################################
        # Thrust program
        tabBeta = modelPropulsion(self.p1, self.p2, tf, 1.0, 0.0,
                                  con['softness'], con['Isp'], con['T'])
        if tabBeta.fail:
            raise Exception('itsme saying: Softness too high!')
        self.tabBeta = tabBeta

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

    def __errorCalculate(self):

        h, v, gamma, M = self.traj.xp[-1]
        errors = ((v - self.con['V_final'])/0.01,
                  (gamma - self.con['gamma_final'])/0.1,
                  (h - self.con['h_final'])/10)
        self.errors = numpy.array(errors)

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

    def __stagingCalculate(self, Dv1: float, Dv2: float)-> None:

        con = self.con
        efflist = con['efflist']
        Tlist = con['Tlist']

        if con['homogeneous']:

            if con['NStag'] == 0:

                p2 = modelOptimalStagingHomogeneous([efflist[-1]],       Dv2,
                                                    [Tlist[-1]], con['Isp'],
                                                    con['g0'],  con['Mu'])
                p1 = modelOptimalStagingHomogeneous([efflist[0]], Dv1 + Dv2,
                                                    [Tlist[0]], con['Isp'],
                                                    con['g0'],  con['Mu'])
                p2.ms = p2.mu + p1.me[0]
                p2.me[0] = 0.0
                p1.NStag0or1(p2)

            elif con['NStag'] == 1:

                p2 = modelOptimalStagingHomogeneous([efflist[-1]],       Dv2,
                                                    [Tlist[-1]], con['Isp'],
                                                    con['g0'],  con['Mu'])
                p1 = modelOptimalStagingHomogeneous([efflist[0]], Dv1 + Dv2,
                                                    [Tlist[0]], con['Isp'],
                                                    con['g0'],  con['Mu'])
                p2.me[0] = p1.me[0]
                p1.NStag0or1(p2)

            else:

                p2 = modelOptimalStagingHomogeneous([efflist[-1]], Dv2,
                                                    [Tlist[-1]], con['Isp'],
                                                    con['g0'],  con['Mu'])
                p1 = modelOptimalStagingHomogeneous(efflist[0:-1], Dv1,
                                                    Tlist[0:-1], con['Isp'],
                                                    con['g0'], p2.mtot[0])

        else:
            raise Exception('itsme saying: heterogeneous vehicle is' +
                            ' not supported yet!')

        if p1.mtot[0]*con['g0'] > Tlist[0]:
            raise Exception('itsme saying: weight greater than thrust!')

        self.p1 = p1
        self.p2 = p2

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
        tphases = [t0, con['tAoA1'], self.tAoA2] + self.tabBeta.tflist
        mjetsoned = [0.0, 0.0, 0.0] + self.tabBeta.melist
        if (typeResult == "orbital"):
            tphases = tphases + [con['torb']]
            mjetsoned = mjetsoned + [0.0]

        self.tphases = tphases
        self.mjetsoned = mjetsoned

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
            for ii in range(1, len(tphases)):
                if tphases[ii - 1] >= tphases[ii]:
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


class modelOptimalStaging():
    # optimalStaging() returns a object with information of optimal staging
    # factor
    # The maximal staging reason is defined as reducing the total mass for a
    # defined delta V.
    # Structural eficience and thrust shall variate for diferent stages,  but
    # specific impulse must be the same for all stages
    # Based in Cornelisse (1979)

    def __init__(self, effList: list, dV, Tlist, con, Isp, g0, mu):
        self.Tlist = Tlist
        self.e = numpy.array(effList)
        self.T = numpy.array(self.Tlist)
        self.dV = dV
        self.mu = mu
        self.Isp = Isp
        self.c = Isp*g0
        self.mflux = self.T/self.c

        if con['homogeneous']:
            self._mE = self.e[0]
            self._m1E = 1 - self.e[0]
        else:
            self._mE = self.__calcGeoMean(self.e)
            self._m1E = self.__calcGeoMean(1 - self.e)

        self.fail = False
        self.__lambCalculate()
        self.phi = (1 - self.e)*(1 - self.lamb)

        self.__mtotCalculate()  # Total sub-rocket mass
        self.mp = self.mtot*self.phi  # Propelant mass on each stage
        self.me = self.mp*(self.e/(1 - self.e))  # Strutural mass of each stage
        # Duration of each stage burning
        self.tb = self.mp/self.mflux
        self.__tfCalculate()  # Final burning time of each stage

        self.ms = 0.0  # Empty vehicle mass (!= 0.0 only for NStag == 0)

    def __calcGeoMean(self, a)-> float:

        m = 1.0
        for v in a:
            m = m*v
        m = m ** (1/a.size)
        return m

    def __lambCalculate(self):

        LtN = (numpy.exp(-self.dV/(self.c*self.e.size)) - self._mE)/self._m1E
        self.LtN = LtN

        if LtN <= 0:
            raise Exception('itsme saying: optimalStaing failed')

        if self._mE == 0.0:
            raise Exception('itsme saying: There is a null' +
                            ' structual efficience')

        self.lamb = (self.e/(1 - self.e))*(LtN*self._m1E/self._mE)

    def __mtotCalculate(self):

        mtot = self.e*0.0
        N = self.e.size-1
        for ii in range(0, N+1):
            if ii == 0:
                mtot[N - ii] = self.mu/self.lamb[N - ii]
            else:
                mtot[N - ii] = mtot[N - ii + 1]/self.lamb[N - ii]
        self.mtot = mtot

    def __tfCalculate(self):

        tf = self.e*0.0
        N = self.tb.size-1
        for ii in range(0, N+1):
            if ii == 0:
                tf[ii] = self.tb[ii]
            else:
                tf[ii] = self.tb[ii] + tf[ii-1]
        self.tf = tf

    def printInfo(self):

        print("dV =", self.dV)
        print("mu =", self.mu)
        print("mp =", self.mp)
        print("me =", self.me)
        print("mtot =", self.mtot)
        print("mflux =", self.mflux)
        print("tb =", self.tb)
        print("tf =", self.tf, "\n\r")


class modelOptimalStagingHomogeneous():
    # optimalStaging() returns a object with information of
    # optimal staging factor for a homogeneous vehcile
    # The maximal staging reason is defined as reducing the total mass for
    # a defined delta V.
    # Structural eficience and thrust shall variate for diferent stages,  but
    # specific impulse must be the same for all stages
    # Based in Cornelisse (1979)

    def __init__(self, effList: list, dV: float, Tlist: list, Isp: float,
                 g0: float, mu: float):
        self.Tlist = Tlist
        self.e = numpy.array(effList)
        self.T = numpy.array(Tlist)
        self.dV = dV
        self.mu = mu
        self.Isp = Isp
        self.c = Isp*g0
        self.mflux = self.T/self.c

        self._lamb = (numpy.exp(-self.dV/self.c/self.e.size) -
                      self.e)/(1 - self.e)

        phi = (1 - self.e)*(1 - self._lamb)

        # Total sub-rocket mass
        self.__mtotCalculate()
        # Propelant mass on each stage
        self.mp = self.mtot*phi
        # Strutural mass of each stage
        self.me = self.mp*(self.e/(1 - self.e))
        # Duration of each stage burning
        self.tb = self.mp/self.mflux
        # Final burning time of each stage
        self.__tfCalculate()
        # Empty vehicle mass (!= 0.0 only for NStag == 0)
        self.ms = 0.0
        return None

    def __mtotCalculate(self)-> None:

        mtot = self.e*0.0
        N = self.e.size-1
        for ii in range(0,  N+1):
            if ii == 0:
                mtot[N - ii] = self.mu/self._lamb[N - ii]
            else:
                mtot[N - ii] = mtot[N - ii + 1]/self._lamb[N - ii]
        self.mtot = mtot
        return None

    def __tfCalculate(self)-> None:

        tf = self.e*0.0
        N = self.tb.size-1
        for ii in range(0, N+1):
            if ii == 0:
                tf[ii] = self.tb[ii]
            else:
                tf[ii] = self.tb[ii] + tf[ii-1]
        self.tf = tf
        return None

    def NStag0or1(self, p: object)-> None:

        self.dV = self.dV - p.dV
        self.mu = p.mtot[0]
        self.mp[0] = self.mp[0] - p.mp[0]  # Propelant mass on each stage
        self.me[0] = self.me[0]*0.0
        self.tb[0] = self.tb[0] - p.tb[0]  # Duration of each stage burning
        self.tf[0] = self.tf[0] - p.tf[0]
        return None

    def printInfo(self):

        print("\n\rdV =", self.dV)
        print("mu =", self.mu)
        print("me =", self.me)
        print("mp =", self.mp)
        print("mtot =", self.mtot)
        print("mflux =", self.mflux)
        print("tb =", self.tb)
        print("tf =", self.tf)


class modelPropulsion():

    def __init__(self, p1: object, p2: object, tf: float,
                 v1: float, v2: float, softness: float, Isp: float, T: float):
        
        # improvements for heterogeneous rocket
        self.Isp = Isp
        self.T = T
        
        self.t1 = p1.tf[-1]
        t2 = tf - p2.tb[-1]
        self.t3 = tf

        self.v1 = v1
        self.v2 = v2

        f = softness/2

        d1 = self.t1  # width of retangular and 0.5 soft part
        self.c1 = d1*f  # width of the 0.5 soft part
        self.fr1 = d1 - self.c1  # final of the retangular part
        self.fs1 = d1 + self.c1  # final of the retangular part

        d2 = self.t3 - t2  # width of retangular and 0.5 soft part
        self.c2 = d2*f  # width of the 0.5 soft part
        self.r2 = d2 - self.c2  # width of the retangular part
        self.ir2 = t2 + self.c2  # start of the retangular part
        self.is2 = t2 - self.c2  # start of the soft part

        self.dv21 = v2 - v1

        # List of time events and jetsoned masses
        self.tflist = p1.tf[0:-1].tolist() + [self.fr1, self.fs1] + \
            [self.is2, self.ir2, tf]
        self.melist = p1.me[0:-1].tolist() + [0.0, p1.me[-1]] + \
            [0.0, 0.0, p2.me[-1]]

        self.fail = False
        if len(p1.tf) > 2:
            if p1.tf[-2] >= self.fr1:
                self.fail = True

        self.tlist1 = p1.tf[0:-1].tolist()+[self.fs1]
        self.Tlist1 = p1.Tlist
        self.Tlist2 = p2.Tlist

        self.Isp1 = p1.Isp
        self.Isp2 = p2.Isp

    def single(self, t: float)-> float:
        if (t <= self.fr1):
            ans = self.v1
        elif (t <= self.fs1):
            cos = numpy.cos(numpy.pi*(t - self.fr1)/(2*self.c1))
            ans = self.dv21*(1 - cos)/2 + self.v1
        elif (t <= self.is2):
            ans = self.v2
        elif (t <= self.ir2):
            cos = numpy.cos(numpy.pi*(t - self.is2)/(2*self.c2))
            ans = -self.dv21*(1 - cos)/2 + self.v2
        elif (t <= self.t3):
            ans = self.v1
        else:
            ans = 0.0

        return ans
    
    def value(self, t: float)-> float:
        if (t <= self.fr1):
            ans = self.v1
        elif (t <= self.fs1):
            cos = numpy.cos(numpy.pi*(t - self.fr1)/(2*self.c1))
            ans = self.dv21*(1 - cos)/2 + self.v1
        elif (t <= self.is2):
            ans = self.v2
        elif (t <= self.ir2):
            cos = numpy.cos(numpy.pi*(t - self.is2)/(2*self.c2))
            ans = -self.dv21*(1 - cos)/2 + self.v2
        elif (t <= self.t3):
            ans = self.v1
        else:
            ans = 0.0

        return ans
    
    def mdlDer(self, t: float)-> tuple:
        
        return self.value(t), self.Isp, self.T

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


class solution():

    def __init__(self, factors: list, con: dict):

        self.factors = factors
        self.con = con

        model1 = model(factors, con)
        model1.simulate("plot")
        self.basic = model1

        model2 = model(factors, con)
        model2.simulate("orbital")
        self.orbital = model2

        self.iteractionAltitude = problemIteractions('')
        self.iteractionSpeedAndAng = problemIteractions('')

    def displayResults(self):

        # Results without orbital phase
        self.basic.displayInfo()
        self.basic.orbitResults()
        self.basic.plotResults()
        self.iteractionAltitude.plotLogErrors(['h'])
        self.iteractionSpeedAndAng.plotLogErrors(['V', 'gamma', 'h'])
        self.iteractionSpeedAndAng.plotFactors(['V', 'gamma', 'h'])

        # Results with orbital phase
        if abs(self.basic.e - 1) > 0.1:
            # The eccentricity test avoids simulations too close
            # of the singularity
            self.orbital.orbitResults()
            self.orbital.plotResults()

        print('Initial states:', self.basic.traj.xx[0])
        print('Final   states:', self.basic.traj.xx[-1])

        return None

    def converged(self):

        convergence = True
        for error in self.basic.errors:
            if abs(error) >= self.basic.con['tol']:
                convergence = False

        return convergence

    def sgra(self):

        ans = self.basic.traj.tt, self.basic.traj.xx, self.basic.uu,\
              self.basic.tabAlpha, self.basic.tabBeta, self.con,\
              self.basic.traj.tphases, self.basic.traj.mass0,\
              self.basic.traj.massJet

        return ans


if __name__ == "__main__":

    start = clock()
    its()
    print('Execution time: ', clock() - start, ' s')
    # input("Press any key to finish...")
