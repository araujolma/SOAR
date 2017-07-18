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


def sgra(fname):

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


def mdlDer(t, x, arg):

    # initialization
    h, v, gama, M = x[0], x[1], x[2], x[3]
    alfaProg, betaProg, con = arg

    if numpy.isnan(h):
        raise Exception('itsme saying: h is not a number')

    # controls calculation
    betat = betaProg.value(t)
    alfat = alfaProg.value(t)

    # other calculations
    btm = betat*con['T']/M
    sinGama = numpy.sin(gama)
    g = con['g0']*(con['R']/(con['R']+h))**2 - (con['we']**2)*(con['R'] + h)

    # aerodynamics
    qdinSrefM = 0.5 * rho(h) * (v**2) * con['s_ref']/M
    LM = qdinSrefM * (con['CL0'] + con['CL1']*alfat)
    DM = qdinSrefM * (con['CD0'] + con['CD2']*(alfat**2))

    # states derivatives
    return numpy.array([v*sinGama,  # coefficient
                        btm*numpy.cos(alfat) - g*sinGama - DM,  # coefficient
                        btm*numpy.sin(alfat)/v +
                        (v/(h+con['R'])-g/v)*numpy.cos(gama) +
                        (LM/v) + 2*con['we'],  # coefficient
                        -btm*M/(con['g0']*con['Isp'])])  # coefficient


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

    def __init__(self, fileAdress):

        # TODO: solve the codification problem on configuration files
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(fileAdress)
        self.con = dict()

        #######################################################################
        # Enviromental constants
        section = 'enviroment'
        items = config.items(section)
        for para in items:
            self.con[para[0]] = config.getfloat(section, para[0])
        # [km/s2] gravity acceleration on earth surface
        self.con['g0'] = self.con['GM']/(self.con['R']**2)

        #######################################################################
        # General constants
        self.con['pi'] = numpy.pi
        self.con['d2r'] = numpy.pi/180.0

        #######################################################################
        # Initial state constants
        self.con['h_initial'] = config.getfloat('initial', 'h')
        self.con['V_initial'] = config.getfloat('initial', 'V')
        self.con['gamma_initial'] = config.getfloat('initial',
                                                    'gamma')*self.con['d2r']

        #######################################################################
        # Final state constants
        self.con['h_final'] = config.getfloat('final', 'h')
        # Circular velocity
        vc = numpy.sqrt(self.con['GM']/(self.con['R']+self.con['h_final']))
        # Rotating referencial velocity effect
        ve = self.con['we']*(self.con['R']+self.con['h_final'])
        self.con['V_final'] = vc - ve  # km/s Final velocity
        self.con['gamma_final'] = config.getfloat('final',
                                                  'gamma')*self.con['d2r']

        #######################################################################
        # Vehicle parameters
        section = 'vehicle'
        items = config.items(section)

        if not config.has_option(section, 'homogeneous'):
            self.con['homogeneous'] = True
        else:
            self.con['homogeneous'] = config.getboolean(section, 'homogeneous')

        # This flag show indicates if the vehicle shall be considered as having
        # the same values of structural mass and thrust for all stages
        if self.con['homogeneous']:
            self.__getVehicleHomogeneous(config)
        else:
            self.__getVehicleHeterogeneous(config)

        #######################################################################
        # Trajectory parameters
        section = 'trajectory'
        items = config.items(section)
        for para in items:
            self.con[para[0]] = config.getfloat(section, para[0])
        # Time of one orbit using the final velocity
        self.con['torb'] = 2*self.con['pi']*(self.con['R'] +
                                             self.con['h_final']
                                             )/self.con['V_final']

        #######################################################################
        # Solver parameters
        section = 'solver'
        self.con['tol'] = config.getfloat(section, 'tol')
        self.con['contraction'] = 0.0  # (1.0e4)*numpy.finfo(float).eps

        # Superior and inferior limits
        auxstr = config.get(section, 'guess')
        auxstr = auxstr.split(', ')
        auxnum = []
        for n in auxstr:
            auxnum = auxnum+[float(n)]
        guess = numpy.array(auxnum)

        auxstr = config.get(section, 'limit')
        auxstr = auxstr.split(', ')
        auxnum = []
        for n in auxstr:
            auxnum = auxnum+[float(n)]
        limit = numpy.array(auxnum)

        self.con['guess'] = guess
        self.con['fsup'] = guess + limit
        self.con['finf'] = guess - limit

        if config.has_option(section, 'Nmax'):
            self.con['Nmax'] = config.getint(section, 'Nmax')
        else:
            self.con['Nmax'] = 100

        if config.has_option(section, 'Ndiv'):
            self.con['Ndiv'] = config.getint(section, 'Ndiv')
        else:
            self.con['Ndiv'] = 10

        #######################################################################
        # Reference values
        iniEst = problemInitialEstimate(self.con)
        self.con['Dv1ref'] = iniEst.dv
        self.con['tref'] = iniEst.t
        self.con['vxref'] = iniEst.vx

        return None

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

    def solveForFineTune(self):
        #######################################################################
        # Bisection altitude loop
        tol = self.con['tol']
        fsup = self.con['fsup']
        finf = self.con['finf']
        self.iteractionAltitude = problemIteractions('Altitude error')
        self.iteractionSpeedAndAng = problemIteractions('All errors')
        sepStr = "\n#################################" +\
                 "######################################"

        # Parameters initialization
        # Loop initialization
        stop = False
        count = 0

        # Fators initilization
        df = abs((fsup[2] - finf[2])/self.con['Ndiv'])
        factors = (fsup + finf)/2
        step = df.copy()
        f1 = (fsup[2] + finf[2])/2
        e1, factors = self.__bisecSpeedAndAng(factors, f1)
        f2 = f1 + step

        # Loop
        while (not stop) and (count <= self.con['Nmax']):

            # bisecSpeedAndAng: Error update from speed and gamma loop
            e2, factors = self.__bisecSpeedAndAng(factors, f2)

            # Loop checks
            if (abs(e2) < tol):
                stop = True
                # Display final information
                num = "8.6e"
                print(sepStr)
                print("bisecAltitude final iteration: ", count)
                print(("Error     : %"+num) % e2)
                print(("Sup limits: %"+num+",  %"+num+",  %"+num) %
                      (fsup[0], fsup[1], fsup[2]))
                print(("Factors   : %"+num+",  %"+num+",  %"+num) %
                      (factors[0], factors[1], f2))
                print(("Inf limits: %"+num+",  %"+num+",  %"+num) %
                      (finf[0], finf[1], finf[2]))

            else:
                # Calculation of the new factor
                # Checkings
                # Division and step check
                self.iteractionAltitude.update(e2)
                de = e2 - e1
                # TODO: a new step check procedure is necessary
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

                # Display information
                num = "8.6e"
                print(sepStr)
                print("bisecAltitude iteration: ", count)
                print(("Error     : %"+num) % e2)
                print(("Sup limits: %"+num+",  %"+num+",  %"+num) %
                      (fsup[0], fsup[1], fsup[2]))

                print(("Factors   : %"+num+",  %"+num+",  %"+num) %
                      (factors[0], factors[1], f2))

                print(("Inf limits: %"+num+",  %"+num+",  %"+num) %
                      (finf[0], finf[1], finf[2]))
            # if end

        self.iteractionAltitude.update(e2)
        traj = model(factors, self.con)
        traj.simulate("design")

        num = "8.6e"
        print(sepStr)
        print("ITS the end (lol)")
        print(("Error     : %"+num+",  %"+num+",  %"+num) %
              (traj.errors[0], traj.errors[1], traj.errors[2]))
        print(("Sup limits: %"+num+",  %"+num+",  %"+num) %
              (fsup[0], fsup[1], fsup[2]))
        print(("Factors   : %"+num+",  %"+num+",  %"+num) %
              (factors[0], factors[1], factors[2]))
        print(("Inf limits: %"+num+",  %"+num+",  %"+num) %
              (finf[0], finf[1], finf[2]))
        print('\nTotal number of trajectory simulations: ',
              (self.iteractionAltitude.count +
               self.iteractionSpeedAndAng.count))

        solution1 = solution(factors, self.con)
        solution1.iteractionAltitude = self.iteractionAltitude
        solution1.iteractionSpeedAndAng = self.iteractionSpeedAndAng

        if not solution1.converged():
            print('itsme saying: solution has not converged :(')

        return solution1

    def __bisecSpeedAndAng(self, factors1, f3):
        #######################################################################
        # Bissection speed and gamma loop
        fsup = self.con['fsup']
        finf = self.con['finf']
        con = self.con
        self.iteractionSpeedAndAng.reset()

        # Initializing parameters
        # Loop initialization
        stop = False
        count = 0
        Nmax = self.con['Nmax']

        # Fators initilization
        df = abs((fsup - finf)/20)
        # Making the 3 factor variarions null
        factors1[2] = f3 + 0.0
        df[2] = 0.0
        factors2 = factors1 + df
        model1 = model(factors1, con)
        model1.simulate("design")
        errors1 = model1.errors
        step = df + 0.0
        factors3 = factors2 + 0.0

        # Loop
        while (not stop) and (count <= self.con['Nmax']):
            # Error update
            model1 = model(factors2, con)
            model1.simulate("design")
            errors2 = model1.errors

            converged = abs(errors2) < con['tol']
            if converged[0] and converged[1]:

                stop = True
                # Display information
                print("\n###################################" +
                      "####################################")
                if count == Nmax:
                    print("bisecSpeedAndAng total iterations: ",  count,
                          " (max)")
                else:
                    print("bisecSpeedAndAng total iterations: ",  count)
                num = "8.6e"
                print(("Errors    : %"+num+",  %"+num)
                      % (errors2[0], errors2[1]))

                print(("Sup limits: %"+num+",  %"+num+",  %"+num)
                      % (fsup[0], fsup[1], fsup[2]))

                print(("Factors   : %"+num+",  %"+num+",  %"+num)
                      % (factors2[0], factors2[1], factors2[2]))

                print(("Inf limits: %"+num+",  %"+num+",  %"+num)
                      % (finf[0], finf[1], finf[2]))

            elif self.iteractionSpeedAndAng.stationary():

                stop = True
                # Display information
                print("\n###################################" +
                      "####################################")
                print('stationary process')
                if count == Nmax:
                    print("bisecSpeedAndAng total iterations: ",  count,
                          " (max)")
                else:
                    print("bisecSpeedAndAng total iterations: ",  count)
                num = "8.6e"
                print(("Errors    : %"+num+",  %"+num)
                      % (errors2[0], errors2[1]))

                print(("Sup limits: %"+num+",  %"+num+",  %"+num)
                      % (fsup[0], fsup[1], fsup[2]))

                print(("Factors   : %"+num+",  %"+num+",  %"+num)
                      % (factors2[0], factors2[1], factors2[2]))

                print(("Inf limits: %"+num+",  %"+num+",  %"+num)
                      % (finf[0], finf[1], finf[2]))

            else:
                self.iteractionSpeedAndAng.update(errors2)
                self.iteractionSpeedAndAng.stationary()
                de = errors2 - errors1
                for ii in range(0, 2):
                    # Division and step check
                    # TODO: a new step check procedure is necessary

                    if de[ii] == 0:
                        step[ii] = 0.0
                    else:
                        step[ii] = errors2[ii]*(factors2[ii] -
                                                factors1[ii])/de[ii]

                    if step[ii] > df[ii]:
                        step[ii] = df[ii] + 0.0
                    elif step[ii] < -df[ii]:
                        step[ii] = -df[ii] + 0.0
                    # if end

                    # factor check
                    factors3[ii] = factors2[ii] - step[ii]

                    if factors3[ii] > fsup[ii]:
                        factors3[ii] = fsup[ii] + 0.0
                    elif factors3[ii] < finf[ii]:
                        factors3[ii] = finf[ii] + 0.0
                    # if end

                # for end
                errors1 = errors2 + 0.0
                factors1 = factors2 + 0.0
                factors2 = factors3 + 0.0
                count += 1
                # print('bisecSpeedAndAng errors', errors2)
            # if end
        # while end
        # Define output
        # print('bisecSpeedAndAng count', count)
        errorh = errors2[2]
        return errorh, factors2


class problemIteractions():

    def __init__(self, name):

        self.name = name
        self.errorsList = []
        self.count = 0
        self.countLocal = 0

    def reset(self):

        self.countLocal = 0

    def update(self, errors):

        self.count += 1
        self.countLocal += 1
        self.errorsList.append(errors)

    def displayErrors(self):

        if self.count != 0:
            err = numpy.array(self.errorsList)
            for e in err:
                e = numpy.array(e)
            err = numpy.array(err)
            plt.plot(range(0, self.count), err)
            plt.grid(True)
            plt.ylabel("erros []")
            plt.xlabel("iteration number []")
            plt.title(self.name)
            plt.show()

    def stationary(self):

        stat = False
        if self.countLocal > 10:
            e = numpy.array([numpy.array(self.errorsList[-3][0:2]),
                             numpy.array(self.errorsList[-2][0:2]),
                             numpy.array(self.errorsList[-1][0:2])])

            ff = numpy.std(e, 0)/abs(numpy.mean(e, 0))

            for f in ff:
                stat = stat or (f < 1e-12)

        return stat

    def displayLogErrors(self):

        if self.count != 0:
            err = numpy.array(self.errorsList)
            for e in err:
                e = numpy.array(e)
            err = numpy.array(err)
            plt.semilogy(range(0, self.count), abs(err))
            plt.grid(True)
            plt.ylabel("erros []")
            plt.xlabel("iteration number []")
            plt.title(self.name)
            if self.name == 'All':
                plt.legend(['V', 'gamma', 'h'])
            plt.show()


class problemInitialEstimate():

    def __init__(self, con):

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
            if cont == 100:
                raise Exception('itsme saying: initialEstimate failed')

        self.cont = cont
        self.t1 = t1
        return None

    def __hEstimate(self, t1):

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

    def __dvt(self):

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

    def __init__(self, factors, con):

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
                                  con['softness'])
        if tabBeta.fail:
            raise Exception('itsme saying: Softness too high!')
        self.tabBeta = tabBeta

        #######################################################################
        # Attitude program definition
        self.tAoA2 = con['tAoA1'] + con['tAoA']
        tabAlpha = modelAttitude(con['tAoA1'], self.tAoA2, 0,
                                 -con['AoAmax']*con['d2r'])
        self.tabAlpha = tabAlpha

    def __integrate(self, ode45, t0, x0):

        self.simulCounter += 1

        # Output variables
        self.traj.append(t0, x0)
        self.traj.appendP(t0, x0)

        N = len(self.tphases)
        for ii in range(1, N):

            # Time interval configuration
            t_initial = self.tphases[ii - 1] + self.con['contraction']
            t_final = self.tphases[ii] - self.con['contraction']

            # Stage separation mass reduction
            y_initial = self.traj.xp[-1]
            y_initial[3] = y_initial[3] - self.mjetsoned[ii-1]

            # integration
            ode45.set_initial_value(y_initial, t_initial)
            ode45.first_step = (t_final - t_initial)*0.001
            ode45.integrate(t_final)

            if ii != N-1:
                self.traj.appendP(ode45.t+self.con['contraction'], ode45.y)

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

    def __errorCalculate(self):

        h, v, gamma, M = self.traj.xp[-1]
        errors = ((v - self.con['V_final'])/0.01,
                  (gamma - self.con['gamma_final'])/0.1,
                  (h - self.con['h_final'])/10)
        self.errors = numpy.array(errors)

        return None

    def __calcAedTab(self, tt, xx, uu):

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

    def __stagingCalculate(self, Dv1, Dv2):

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

    def displayInfo(self):

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

    def simulate(self, typeResult):
        #######################################################################
        con = self.con
        self.simulCounter += 1

        #######################################################################
        # Integration
        # Initial conditions
        t0 = 0.0
        x0 = numpy.array([con['h_initial'], con['V_initial'],
                          con['gamma_initial'], self.p1.mtot[0]])

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
        ode45.set_initial_value(x0,  t0).set_f_params(
                               (self.tabAlpha, self.tabBeta, con))

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
        plt.hold(False)
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
        LL,  DD,  CCL,  CCD,  QQ = self.__calcAedTab(self.traj.tt,
                                                     self.traj.xx, self.uu)
        Lp,  Dp,  CLp,  CDp,  Qp = self.__calcAedTab(self.traj.tp,
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

    def __init__(self, effList, dV, Tlist, con, Isp, g0, mu):
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

    def __calcGeoMean(self, a):

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

    def __init__(self, effList, dV, Tlist, Isp, g0, mu):
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

    def __mtotCalculate(self):

        mtot = self.e*0.0
        N = self.e.size-1
        for ii in range(0,  N+1):
            if ii == 0:
                mtot[N - ii] = self.mu/self._lamb[N - ii]
            else:
                mtot[N - ii] = mtot[N - ii + 1]/self._lamb[N - ii]
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

    def NStag0or1(self, p):

        self.dV = self.dV - p.dV
        self.mu = p.mtot[0]
        self.mp[0] = self.mp[0] - p.mp[0]  # Propelant mass on each stage
        self.me[0] = self.me[0]*0.0
        self.tb[0] = self.tb[0] - p.tb[0]  # Duration of each stage burning
        self.tf[0] = self.tf[0] - p.tf[0]

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

    def __init__(self, p1, p2, tf, v1, v2, softness):
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

    def value(self, t):
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

    def multValue(self, t):
        N = len(t)
        ans = numpy.full((N, 1), 0.0)
        for jj in range(0, N):
            ans[jj] = self.value(t[jj])

        return ans

    def thrust(self, t):

        T = 0.0
        tlist = self.tlist1
        Tlist = self.Tlist1
        for ii in range(0, len(tlist)):
            if t <= tlist[ii]:
                T = Tlist[ii]

        if t > self.is2:
            T = self.Tlist2[0]

        return T

    def Isp(self, t):

        if t <= self.fs1:
            Isp = self.Isp1
        else:
            Isp = self.Isp2

        return Isp


class modelAttitude():

    def __init__(self, t1, t2, v1, v2):
        self.t1 = t1
        self.t2 = t2

        self.v1 = v1
        self.v2 = v2

    def value(self, t):
        if (t >= self.t1) and (t < self.t2):
            ans = ((self.v2 - self.v1)*(1 -
                   numpy.cos(2*numpy.pi*(t - self.t1)/(self.t2 - self.t1)))/2)\
                   + self.v1
            return ans
        else:
            return self.v1

    def multValue(self, t):
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


class solution():

    def __init__(self, factors, con):

        self.factors = factors

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
        self.iteractionAltitude.displayLogErrors()
        self.iteractionSpeedAndAng.displayLogErrors()

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
              self.basic.tabAlpha, self.basic.tabBeta
        return ans


if __name__ == "__main__":

    its()
    # input("Press any key to finish...")
