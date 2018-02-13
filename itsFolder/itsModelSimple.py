# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:14:40 2017

@author: Carlos

Initial Trajectory Setup ModulE

Submodule itsModelSimple: a simplied version of itsModule. This submodule is
used in itsmeSimple.

"""

import numpy
import sys
from itsFolder.itsModelCommon import model as modelBase

sys.path.append('/..')


class model(modelBase):

    def __integrate(self, ode45: object, t0: float, x0)-> None:

        modelBase.__integrate(self, ode45, t0, x0)
        self.__orbitData()

    def __errorCalculate(self):

        errors = ((self.traj.av + self.Dv2 - self.con['V_final'])/0.01,
                  (self.tabBeta.t2/self.traj.at - 1),
                  (self.traj.ah - self.con['h_final'])/10)
        self.errors = numpy.array(errors)

        return None

    def __phasesSetting(self, t0, con, typeResult)-> tuple:

        self.tphases = [t0, con['tAoA1'], self.tAoA2] + self.tabBeta.tflistP1
        self.mjetsoned = [0.0, 0.0, 0.0] + self.tabBeta.melistP1

    def __orbitData(self)-> None:

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

        en = 0.5 * v * v - GM/r

        a = - 0.5*GM/en

        aux = v * momAng / GM
        e = numpy.sqrt((aux * cosGama - 1)**2 + (aux * sinGama)**2)

        ah = a * (1.0 + e) - R

        ve = self.con['we']*(R + ah)

        av = momAng/(R + ah) - ve

        self.traj.av = av
        self.traj.ah = ah

        # Reference time calculated with excentric anomaly equations
        E1 = numpy.arccos((1 - (R + h)/a)/e)
        E2 = numpy.pi - 2*self.con['gamma_final']

        M1 = E1 - e*numpy.sin(E1)
        M2 = E2 - e*numpy.sin(E2)

        n = numpy.sqrt(GM/(a**3))

        self.traj.at = (M2 - M1)/n

        return None
