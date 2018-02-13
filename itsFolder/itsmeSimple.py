# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:14:40 2017

@author: Carlos

Initial Trajectory Setup ModulE

Submodule itsmeSimple: a version of itsme which a simpler and easier to
converge model. This submodule generates initial values for itsme, improving
its convergence.

"""

from itsFolder.itsProblem import problem as problemBasic
from itsFolder.itsModelSimple import model as modelSimpleClass


class secondaryEstimate():

    def __init__(self, con):

        self.con = con

        print("itsmeM: Modified Inital Trajectory Setup Module")
        print("Opening case: ", con['itsFile'])

        problem1 = problem(con, modelSimpleClass)

        self.solution2 = problem1.solveForFineTune()

        self.iniEst = [self.solution2.Dv1_final, self.solution2.tf_final,
                       self.solution2.vx_final]

    def displayResults(self):

        self.solution2.displayResults()

    def result(self):

        self.con['Dv1ref'] = self.iniEst[0]
        self.con['tref'] = self.iniEst[1]
        self.con['vxref'] = self.iniEst[2]

        return self.con


class problem(problemBasic):

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
        print("itsme secondary estimate")
