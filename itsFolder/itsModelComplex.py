# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:14:40 2017

@author: Carlos

Initial Trajectory Setup ModulE

Version: Objetification

"""

import numpy
import sys
from itsFolder.itsModelCommon import model as modelBase

sys.path.append('/..')


class model(modelBase):

    def __errorCalculate(self):

        h, v, gamma, M = self.traj.xp[-1]
        errors = ((v - self.con['V_final'])/0.01,
                  (gamma - self.con['gamma_final'])/0.1,
                  (h - self.con['h_final'])/10)
        self.errors = numpy.array(errors)

        return None

    def __phasesSetting(self, t0, con, typeResult)-> tuple:

        self.tphases = [t0, con['tAoA1'], self.tAoA2] + self.tabBeta.tflist
        self.mjetsoned = [0.0, 0.0, 0.0] + self.tabBeta.melist

        if (typeResult == "orbital"):
            self.tphases = self.tphases + [con['torb']]
            self.mjetsoned = self.mjetsoned + [0.0]
