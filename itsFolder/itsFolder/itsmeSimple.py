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

from itsFolder.itsProblem import problem
from itsFolder.itsModelSimple import model as modelSimpleClass


def secondaryEstimate(con):

    print("itsmeM: Modified Inital Trajectory Setup Module")
    print("Opening case: ", con['itsFile'])

    problem1 = problem(con, modelSimpleClass)

    solution2 = problem1.solveForFineTune()

    solution2.displayResults()

    return solution2.Dv1_final, solution2.tf_final, solution2.vx_final
