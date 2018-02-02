# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:14:40 2017

@author: Carlos

Initial Trajectory Setup ModulE

Version: Heterogeneous

"""

import numpy
import configparser
import matplotlib.pyplot as plt
from time import clock
from itsModel import model
from itsModelConfiguration import modelConfiguration
from itsModelInitialEstimate import modelInitialEstimate


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
                'itsme_test_cases/genericHet.its',
                'itsme_test_cases/falcon9.its',
                'itsme_test_cases/electron.its',
                'itsme_test_cases/longMarch4B.its',
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
        configuration.trajectory()
        configuration.solver()

        self.con = configuration.con

        modelConf = modelConfiguration(self.con)
        modelConf.vehicle()

        self.con = modelConf.con

        # Reference values
        iniEst = modelInitialEstimate(self.con)
        self.con['Dv1ref'] = iniEst.dv
        self.con['tref'] = iniEst.t
        self.con['vxref'] = iniEst.vx

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
        self.con['itsFile'] = fileAdress

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

        if self.config.has_option(section, 'fracVel'):
            self.con['fracVel'] = self.config.getfloat(section, 'fracVel')
        else:
            self.con['fracVel'] = 0.7


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
