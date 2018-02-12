# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:14:40 2017

@author: Carlos

Initial Trajectory Setup ModulE

Version: Heterogeneous

Submodule itsmeSimple: a version of itsme which a simpler and easier to
converge model. This submodule generates initial values for itsme, improving
its convergence.

"""

from itsFolder.itsmeCommon import problem as problemBasic
from itsFolder.itsmeCommon import solutionClass as solutionBasic
from itsFolder.itsModelSimple import model as modelSimpleClass


def itsInitial(con):

    print("itsmeM: Modified Inital Trajectory Setup Module")
    print("Opening case: ", con['itsFile'])

    problem1 = problem(con)

    solution2 = problem1.solveForFineTune()

    solution2.displayResults()

    return solution2.Dv1_final, solution2.tf_final, solution2.vx_final


class problem(problemBasic):

    def displayErrorsFactors(self, errors: list, factors: list)-> None:

        num = "8.6e"
        print("itsme simplified")
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

    def model(self, factors):

        model1 = modelSimpleClass(factors, self.con)
        return model1

    def solution(self, factors):

        solution1 = solutionSimpleClass(factors, self.con)
        return solution1


class solutionSimpleClass(solutionBasic):

    def model(self, factors):

        model1 = modelSimpleClass(factors, self.con)
        return model1
