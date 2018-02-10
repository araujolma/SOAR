# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:14:40 2017

@author: Carlos

Initial Trajectory Setup ModulE

Version: Heterogeneous

"""

import numpy
import os
from time import clock
from itsModel import model
from itsModelConfiguration import modelConfiguration
from itsModelInitialEstimate import modelInitialEstimate
from itsmeSimple import itsInitial
from itsmeCommon import problemConfiguration, problemIteractions, solution


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

    con = initialize(fname)

    problem1 = problem(con)

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

    con = initialize(fname)
    solution = problem(con).solveForFineTune()

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
    folder = 'itsme_test_cases'
    testList = os.listdir('./' + folder)

    print(testList)

    for case in testList:
        if not its(folder + '/' + case).converged():
            print(case)
            raise Exception('itsme saying: solution did not converge')

    con = initialize(folder + '/caseEarthRotating.its')
    problem(con).solveForFineTune()
    con = initialize(folder + '/caseMoon.its')
    problem(con).solveForFineTune()
    con = initialize(folder + '/caseMu150h500NStag4.its')
    problem(con).solveForFineTune()

    sgra('default.its')


def initialize(fileAdress):

    # TODO: solve the codification problem on configuration files
    configuration = problemConfiguration(fileAdress)
    configuration.environment()
    configuration.initialState()
    configuration.finalState()
    configuration.trajectory()
    configuration.solver()

    con = configuration.con

    modelConf = modelConfiguration(con)
    modelConf.vehicle()

    con = modelConf.con

    iniEst = modelInitialEstimate(con)
    con['Dv1ref'] = iniEst.dv
    con['tref'] = iniEst.t
    con['vxref'] = iniEst.vx

    if con['NStag'] > 1:  # not con['homogeneous']:

        iniEst = modelInitialEstimate(con)
        con['Dv1ref'] = iniEst.dv
        con['tref'] = iniEst.t
        con['vxref'] = iniEst.vx

        iniEst = itsInitial(con)
        # input("Press Enter to continue...")

        con['Dv1ref'] = iniEst[0]
        con['tref'] = iniEst[1]
        con['vxref'] = iniEst[2]

        guess = numpy.array([1, 1, 1])
        limit = guess*0.5

        con['guess'] = guess
        con['fsup'] = guess + limit
        con['finf'] = guess - limit

    return con


class problem():

    def __init__(self, con: dict):

        self.con = con
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
        print("itsme fine turn")
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


if __name__ == "__main__":

    start = clock()
    its()
    print('Execution time: ', clock() - start, ' s')
    # input("Press any key to finish...")
