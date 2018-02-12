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
from itsModel import model


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
            self.con['Ndiv'] = 50

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

        if not con['homogeneous']:
            model1.tabBeta.plot()

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
