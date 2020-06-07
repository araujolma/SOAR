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
import sys

try:
    from time import clock
except ImportError:
    # changed for python 3.8
    from time import perf_counter as clock

from itsFolder.itsModelComplex import model
from itsFolder.itsModelConfiguration import modelConfiguration
from itsFolder.itsProblem import problem, problemConfiguration
from itsFolder.itsModelInitialEstimate import initialEstimate
from itsFolder.itsmeSimple import secondaryEstimate

sys.path.append('/itsFolder')


def zeroFun(x, problem1, Mu0):

    problem1.con['Mu'] = x
    solution2 = problem1.solveForFineTune()
    error = solution2.basic.traj.xx[-1][-1] - Mu0

    return error


def its(*arg):

    # arguments analisys
    if len(arg) == 0:
        fname = 'default.its'
    elif len(arg) == 1:
        fname = arg[0]
    else:
        raise Exception('itsme saying: too many arguments on its')

    print("itsme: Initial Trajectory Setup Module")
    print("Opening case: ", fname)

    con = initialize(fname).resultShow()

    problem1 = problem(con, model)

    solution1 = problem1.solveForInitialGuess()

    solution1.displayResults()

    solution2 = problem1.solveForFineTune()
    solution2.displayResults()

    if not solution2.converged():
            print('itsme saying: solution has not converged :(')
            print('Case: ', fname)

    return solution2


def sgra(fname: str):

    # arguments analisys

    print("itsme: Initial Trajectory Setup Module")
    print("Opening case: ", fname)

    con = initializeSGRA(fname).result()
    solution = problem(con, model).solveForFineTune()

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
        try:
            its(folder + '/' + case)

        except:
            print('Fail case: ', case)

    testList2 = ['/caseEarthRotating.its',
                 '/caseMoon.its',
                 '/caseMu150h500NStag4.its']

    for case in testList2:
        try:
            con = initialize(folder + case).result()
            problem(con, model).solveForFineTune()

        except:
            print('Fail case: ', case)

    #sgra('teste.its')


class initialize():

    def __init__(self, fileAdress):

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

        self.con = con

    def result(self):

        con = self.con
        # using a quite simple estimate to start
        iniEst = initialEstimate(con)
        con = iniEst.result()

        tol = con['tol']

        if con['NStag'] > 1:  # not con['homogeneous']:

            con['tol'] = tol*10

            # using a better model on the estimate
            iniEst = secondaryEstimate(con)
            # iniEst.displayResults()
            con = iniEst.result()

            # the secondary estimate is so good that the guess bellow has been
            # effective
            guess = numpy.array([1, 1, 1])
            limit = numpy.array([1, 1, 1])*0.9

            con['guess'] = guess
            con['fsup'] = guess + limit
            con['finf'] = guess - limit

            con['tol'] = tol

        return con

    def resultShow(self):

        con = self.con
        # using a quite simple estimate to start
        iniEst = initialEstimate(con)
        con = iniEst.result()

        tol = con['tol']

        if con['NStag'] > 1:  # not con['homogeneous']:

            con['tol'] = tol*10

            # using a better model on the estimate
            iniEst = secondaryEstimate(con)
            iniEst.displayResults()
            con = iniEst.result()

            # the secondary estimate is so good that the guess bellow has been
            # effective
            guess = numpy.array([1, 1, 1])
            limit = numpy.array([1, 1, 1])*0.9

            con['guess'] = guess
            con['fsup'] = guess + limit
            con['finf'] = guess - limit

            con['tol'] = tol

        return con


class problemConfigurationSGRA(problemConfiguration):

    def trajmods(self):
        """Trajectory modification parameters.
        This should not interfere with the itsme functioning at all."""

        section = 'trajmods'
        self.con['DampCent'] = self.config.getfloat(section, 'DampCent')
        self.con['DampSlop'] = self.config.getfloat(section, 'DampSlop')
        targHeigStr = self.config.get(section, 'TargHeig')
        targHeigStr = targHeigStr.split(', ')
        targHeig = list()
        for nStr in targHeigStr:
            targHeig.append(float(nStr))
        self.con['TargHeig'] = numpy.array(targHeig)

    def accel(self):
        """Acceleration limitation parameters.
        This should not interfere with the itsme functioning at all."""

        section = 'accel'
        self.con['acc_max'] = self.config.getfloat(section, 'acc_max')
        self.con['PFmode'] = self.config.get(section, 'PFmode')
        self.con['acc_max_relTol'] = self.config.getfloat(section,
                                                          'acc_max_relTol')
        self.con['PFtol'] = self.config.getfloat(section, 'PFtol')

    def sgra(self):
        """ "Internal" settings for SGRA.
        This should not interfere with the itsme functioning at all."""

        # Floating point parameters
        for key in ['tolP', 'tolQ', 'GSS_PLimCte', 'GSS_stopStepLimTol',\
                    'GSS_stopObjDerTol', 'GSS_findLimStepTol']:
            self.con[key] = self.config.getfloat('sgra',key)

        # Integer parameters
        for key in ['N', 'GSS_stopNEvalLim']:
            self.con[key] = self.config.getint('sgra',key)

        # Array-like parameters
        PiLowStr = self.config.get('sgra', 'pi_min').split(', ')
        PiLow = list()
        for nStr in PiLowStr:
            if nStr.lower() == 'none':
                PiLow.append(None)
            else:
                PiLow.append(float(nStr))
        self.con['pi_min'] = PiLow#numpy.array(PiLow)

        PiHighStr = self.config.get('sgra', 'pi_max').split(', ')
        PiHigh = list()
        for nStr in PiHighStr:
            if nStr.lower() == 'none':
                PiHigh.append(None)
            else:
                PiHigh.append(float(nStr))
        self.con['pi_max'] = PiHigh#numpy.array(PiHigh)


class initializeSGRA(initialize):

    def __init__(self, fileAdress):

        # TODO: solve the codification problem on configuration files
        configuration = problemConfigurationSGRA(fileAdress)
        configuration.environment()
        configuration.initialState()
        configuration.finalState()
        configuration.trajectory()
        configuration.trajmods()
        configuration.accel()
        configuration.sgra()
        configuration.solver()

        con = configuration.con

        modelConf = modelConfiguration(con)
        modelConf.vehicle()

        con = modelConf.con

        self.con = con


if __name__ == "__main__":

    start = clock()
    its()
    print('Execution time: ', clock() - start, ' s')
    # input("Press any key to finish...")
