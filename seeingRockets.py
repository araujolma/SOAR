#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seeingRockets.py: a code for rocket visualization

Created on Tue Mar  6 19:50:50 2018

@author: carlos

"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy


class rocket():

    def __init__(self, centralBody, boosterBody, Nbooster, d):

        self.cb = centralBody.copy()
        self.bb = boosterBody.copy()
        self.Nb = Nbooster
        self.d = d
        self.Lmax = 0.0

        self.calculate()

    def copy(self):

        return rocket(self.cb, self.bb, self.Nb, self.d)

    def calculate(self):

        dt = 2*numpy.pi/self.Nb
        self.bList = []
        for ii in range(0, self.Nb):
            aux = self.bb.copy()
            aux.translate([self.d, 0.0, 0.0])
            aux.rotateZ(dt*ii)
            self.bList.append(aux)
            if self.Lmax < aux.z1:
                self.Lmax = aux.z1

        self.bList.append(self.cb)
        if self.Lmax < self.cb.z1:
            self.Lmax = self.cb.z1

        	

    def plot(self, ax):

        for body in self.bList:
            body.plot(ax)

        return ax


class body():

    def __init__(self, stgList):

        self.stgList = []
        for stg in stgList:
            self.stgList.append(stg.copy())

        self.r0 = 0.0
        self.r1 = 0.0
        self.z0 = 0.0
        self.z1 = 0.0

        self.calculate()

    def copy(self):

        return body(self.stgList)

    def calculate(self):

        self.r0 = self.stgList[0].r0
        self.z0 = self.stgList[0].z0

        for stg in self.stgList:
            stg.translate([0.0, 0.0, self.z1])
            self.z1 = self.z1 + stg.z1

        self.r1 = self.stgList[-1].r1

    def translate(self, delta):

        for stg in self.stgList:
            stg.translate(delta)

    def rotateZ(self, theta):

        for stg in self.stgList:
            stg.rotateZ(theta)

    def plot(self, ax):

        for stg in self.stgList:
            stg.plot(ax)

        return ax


class stage():

    def __init__(self, modList):

        self.modList = []
        for mod in modList:
            self.modList.append(mod.copy())

        self.r0 = 0.0
        self.r1 = 0.0
        self.z0 = 0.0
        self.z1 = 0.0

        self.calculate()

    def copy(self):

        return stage(self.modList)

    def calculate(self):
        self.r0 = self.modList[0].r0
        self.z0 = self.modList[0].z0

        for mod in self.modList:
            mod.translate([0.0, 0.0, self.z1])
            self.z1 = self.z1 + mod.z1

        self.r1 = self.modList[-1].r1

    def translate(self, delta):

        for mod in self.modList:
            mod.translate(delta)

    def rotateZ(self, theta):

        for mod in self.modList:
            mod.rotateZ(theta)

    def plot(self, ax):

        for mod in self.modList:
            mod.plot(ax)

        return ax


class module():

    def __init__(self, z, r, Nt, color):

        self.color = color
        self.z = z
        self.r = r
        self.Nt = Nt
        self.Nr = len(self.r)

        self.calculate()

    def copy(self):

        return self.__init__(self.z, self.r, self.Nt, self.color)

    def calculate(self):

        self.r0 = self.r[0]
        self.r1 = self.r[-1]
        self.z0 = self.z[0]
        self.z1 = self.z[-1]

        rr = []
        zz = []
        theta = numpy.linspace(0, 2*numpy.pi, self.Nt)
        ones = numpy.linspace(1, 1, self.Nt)

        for i in range(0, self.Nr-1):

            rr.append(numpy.linspace(self.r[i], self.r[i+1], 2))
            zz.append(numpy.linspace(self.z[i], self.z[i+1], 2))

        x = numpy.outer(numpy.cos(theta), rr)
        y = numpy.outer(numpy.sin(theta), rr)
        z = numpy.outer(ones, zz)

        self.surf = surfClass(x, y, z, self.color)

    def plot(self, ax):

        ax = self.surf.plot(ax)
        return ax

    def plotPerfil(self):

        plt.plot(self.z, self.r)
        plt.show()

    def translate(self, delta):

        self.surf.translate(delta)

    def rotateZ(self, theta):

        self.surf.rotateZ(theta)


class surfClass():

    def __init__(self, x, y, z, color):

        self.x = x
        self.y = y
        self.z = z
        self.color = color

    def plot(self, ax):

        ax.plot_surface(self.x, self.y, self.z, color=self.color)

        return ax

    def translate(self, delta):

        self.x = self.x + delta[0]
        self.y = self.y + delta[1]
        self.z = self.z + delta[2]

    def rotateZ(self, theta):

        x0 = self.x.copy()
        y0 = self.y.copy()
        self.x = x0*numpy.cos(theta) + y0*numpy.sin(theta)
        self.y = y0*numpy.cos(theta) - x0*numpy.sin(theta)

######################
# Specialized modules


class cyl(module):

    def __init__(self, L, R, Nt, color):

        self.L = L
        self.R = R
        self.Nt = Nt
        self.color = color
        self.z = [0, L, L]
        self.r = [R, R, 0]

        self.Nr = len(self.r)
        self.calculate()

    def copy(self):

        return cyl(self.L, self.R, self.Nt, self.color)


class cone(module):

    def __init__(self, L, R1, R2, Nt, color):

        self.L = L
        self.R1 = R1
        self.R2 = R2
        self.Nt = Nt
        self.color = color
        self.z = [0, L, L]
        self.r = [R1, R2, 0]

        self.Nr = len(self.r)
        self.calculate()

    def copy(self):

        return cone(self.L, self.R1, self.R2, self.Nt, self.color)


class coife(module):

    def __init__(self, L, R, rho, Nt, color):

        self.L = L
        self.R = R
        self.rho = rho
        self.Nt = Nt
        self.color = color

        self.perfil()

        self.z = self.zc
        self.r = self.rc
        self.Nr = len(self.r)

        self.calculate()

    def copy(self):

        return coife(self.L, self.R, self.rho, self.Nt, self.color)

    def perfil(self):

        N = 20
        ang_ref = numpy.arctan2(self.R - self.rho, self.L - self.rho)
        theta = numpy.array(range(0, N))/(N - 1)
        theta = (1 - theta)*numpy.pi/2
        self.rc = []
        self.zc = []
        for ang in theta:

            if ang < ang_ref:

                self.rc.append(self.rho*numpy.cos(ang) + self.R - self.rho)
                self.zc.append(self.rho*numpy.sin(ang))

            else:

                self.rc.append(self.rho*numpy.cos(ang))
                self.zc.append(self.rho*numpy.sin(ang) + self.L - self.rho)


if __name__ == "__main__":

    # Example of use:

    # colors based on a grey scale.
    c1 = numpy.array([1,1,1])

    cyl1 = cyl( 1, 1, 50, c1)
    cyl2 = cyl(10, 1, 50, c1*0.8)
    cyl3 = cyl( 4, 1, 50, c1*0.6)
    cyl4 = cyl( 6, 1, 50, c1*0.4)
    coife1 = coife(3, 1, 0.3, 50, c1*0.5)

    stg1 = stage([cyl3, cyl2, cyl1])
    stg2 = stage([cyl4, cyl2, cyl1])
    stg3 = stage([cyl1, coife1])

    body1 = body([stg1, stg2, stg3])
    body2 = body([stg2, stg3])

    rocket1 = rocket(body1, body2, 4, 2.5)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2 = rocket1.plot(ax2)
    ax2.set_xlim(-rocket1.Lmax/2, rocket1.Lmax/2)
    ax2.set_ylim(-rocket1.Lmax/2, rocket1.Lmax/2)
    ax2.set_zlim(0, rocket1.Lmax)

    plt.show()
