#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seeingRockets.py: a code for rocket visualization
Created on Tue Mar  6 19:50:50 2018
@author: carlos
Requirement:
    mayavi
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy
import sys
from mayavi import mlab
from tvtk.api import tvtk
from tvtk.common import configure_input_data

sys.path.append('/mayavi_master')


#############################################################################
# Functions


def plotWord(word, h, b, r, ang, color, mlab):

    dh = b*1.2
    wordList = []
    for letter in word:
        ret = retangle(b, b, color, letter)
        wordList.insert(0, ret)

    for ret in wordList:

        ret.rotateYXZ(numpy.pi/2, -numpy.pi/2, ang)
        ret.translate([-r*numpy.cos(ang), r*numpy.sin(ang), h])
        h += dh
        mlab = ret.plotLetter(mlab)

    return mlab

#############################################################################
# Basic classes


class surfClass():

    def __init__(self, x, y, z, color):

        self.x = x
        self.y = y
        self.z = z
        self.color = color

        r = numpy.sqrt(x**2 + y**2)

        self.z0 = self.z.min()
        self.z1 = self.z.max()
        self.r0 = r.min()
        self.r1 = r.max()

    def copy(self):

        return surfClass(self.x, self.y, self.z, self.color)

    def isSurf(self):

        return True

    def translate(self, delta):

        self.x = self.x + delta[0]
        self.y = self.y + delta[1]
        self.z = self.z + delta[2]

    def rotateZ(self, theta):

        x0 = self.x.copy()
        y0 = self.y.copy()
        self.x = x0*numpy.cos(theta) + y0*numpy.sin(theta)
        self.y = y0*numpy.cos(theta) - x0*numpy.sin(theta)

    def rotateY(self, theta):

        x0 = self.z.copy()
        y0 = self.x.copy()
        self.z = x0*numpy.cos(theta) + y0*numpy.sin(theta)
        self.x = y0*numpy.cos(theta) - x0*numpy.sin(theta)

    def rotateX(self, theta):

        x0 = self.y.copy()
        y0 = self.z.copy()
        self.y = x0*numpy.cos(theta) + y0*numpy.sin(theta)
        self.z = y0*numpy.cos(theta) - x0*numpy.sin(theta)

    def rotateYXZ(self, angY, angX, angZ):

        self.rotateY(angY)
        self.rotateX(angX)
        self.rotateZ(angZ)

    def scale(self, alpha):

        self.x = self.x*alpha
        self.y = self.y*alpha
        self.z = self.z*alpha

    def inclinate(self, a):

        self.x = self.x - a*self.z

    def opacity(self, op):

        self.color = [self.color[0], self.color[1], self.color[2], op]

    def plot(self, ax):

        aux = (self.color[0], self.color[1], self.color[2])

        if len(self.color) is 3:

            ax.mesh(self.x, self.y, self.z, color=aux)

        else:

            ax.mesh(self.x, self.y, self.z, color=aux, opacity=self.color[3])

        return ax

    def matPlot(self, ax):

        ax.plot_surface(self.x, self.y, self.z, antialiased=False,
                        color=[self.color[0], self.color[1], self.color[2]],
                        linewidth=0)

        return ax


class group():

    def __init__(self, objList):

        self.objList = []
        self.z0 = 0.0
        self.z1 = 0.0
        self.r0 = 0.0
        self.r1 = 0.0

        for obj in objList:

            self.objList.append(obj.copy())
            self.z0 = numpy.min([self.z0, obj.z0])
            self.z1 = numpy.max([self.z1, obj.z1])
            self.r0 = numpy.min([self.r0, obj.r0])
            self.r1 = numpy.max([self.r1, obj.r1])

    def copy(self):

        return group(self.objList)

    def isSurf(self):

        return False

    def fixCopy(self):

        surfList = []
        surfList = self._getSurf(surfList)
        return group(surfList)

    def _getSurf(self, surfList):

        for obj in self.objList:
            if obj.isSurf():
                surfList.append(obj.copy())
            else:
                surfList = obj._getSurf(surfList)

        return surfList

    def around(self, group2, N, d):

        dt = 2*numpy.pi/N
        for ii in range(0, N):
            aux = group2.fixCopy()
            aux.translate([d, 0.0, 0.0])
            aux.rotateZ(dt*ii)
            self.objList.append(aux)

    def aroundAngles(self, group2, angles, d):

        for ang in angles:
            aux = group2.copy()
            aux.translate([d, 0.0, 0.0])
            aux.rotateZ(ang)
            self.objList.append(aux)

    def translate(self, delta):

        for obj in self.objList:
            obj.translate(delta)

    def rotateZ(self, theta):

        for obj in self.objList:
            obj.rotateZ(theta)

    def rotateY(self, theta):

        for obj in self.objList:
            obj.rotateY(theta)

    def rotateX(self, theta):

        for obj in self.objList:
            obj.rotateX(theta)

    def rotateYXZ(self, angY, angX, angZ):

        for obj in self.objList:
            obj.rotateYXZ(angY)

    def scale(self, alpha):

        for obj in self.objList:
            obj.scale(alpha)

    def opacity(self, op):

        for obj in self.objList:
            obj.opacity(op)

    def inclinate(self, a):

        for obj in self.objList:
            obj.inclinate(a)

    def plot(self, ax):

        for obj in self.objList:
            obj.plot(ax)

        return ax

    def matPlot(self, ax):

        for obj in self.objList:
            obj.matPlot(ax)

        return ax

#############################################################################
# Ineherited classes


class lineClass(surfClass):

    def plot(self, ax):

        aux = (self.color[0], self.color[1], self.color[2])
        ax.plot3d(self.x, self.y, self.z, color=aux)

        return ax

    def matPlot(self, ax):

        ax.plot(self.x, self.y, self.z,
                color=[self.color[0], self.color[1], self.color[2]])

        return ax

    def copy(self):

        return lineClass(self.x, self.y, self.z, self.color)


class stack(group):

    def __init__(self, objList):

        self.objList = []
        for obj in objList:
            self.objList.append(obj.copy())

        self.r0 = 0.0
        self.r1 = 0.0
        self.z0 = 0.0
        self.z1 = 0.0

        self.calculate()

    def copy(self):

        return stack(self.objList)

    def calculate(self):

        self.r0 = self.objList[0].r0
        self.z0 = self.objList[0].z0

        for obj in self.objList:
            obj.translate([0.0, 0.0, self.z1])
            self.z1 = self.z1 + obj.z1

        self.r1 = self.objList[-1].r1


class module(group):

    def __init__(self, z, r, Nt, color):

        self.color = color
        self.z = z
        self.r = r
        self.Nt = Nt
        self.Nr = len(self.r)

        self.objList = []
        self.calculate()

    def copy(self):

        return module(self.z, self.r, self.Nt, self.color)

    def opacity(self, op):

        self.color = [self.color[0], self.color[1], self.color[2], op]

        for obj in self.objList:
            obj.opacity(op)

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

        self.objList = [surfClass(x, y, z, self.color)]

    def plotPerfil(self):

        plt.plot(self.z, self.r)
        plt.show()


class interTruss(group):

    def __init__(self, L, R, Nt, color):

        self.color = color
        self.L = L
        self.R = R
        self.Nt = Nt

        self.objList = []
        self.calculate()

        self.z0 = 0.0
        self.z1 = L
        self.r0 = R
        self.r1 = R

    def calculate(self):

        self.r0 = 0.0
        self.r1 = self.R
        self.z0 = 0.0
        self.z1 = self.L

        theta = numpy.linspace(0, 2*numpy.pi, self.Nt)

        x00 = numpy.cos(theta)*self.R
        y00 = numpy.sin(theta)*self.R
        z00 = theta*0.0 + self.z0

        alpha = numpy.pi/self.Nt

        x11 = numpy.cos(theta + alpha)*self.R
        y11 = numpy.sin(theta + alpha)*self.R
        z11 = theta*0.0 + self.z1

        x = []
        y = []
        z = []

        for ii in range(0, len(x00)):

            x.append(x00[ii])
            x.append(x11[ii])
            y.append(y00[ii])
            y.append(y11[ii])
            z.append(z00[ii])
            z.append(z11[ii])

        ii = 0
        x.append(x00[ii])
        x.append(x11[ii])
        y.append(y00[ii])
        y.append(y11[ii])
        z.append(z00[ii])
        z.append(z11[ii])

        self.objList = [lineClass(numpy.array(x), numpy.array(y),
                                  numpy.array(z), self.color)]

    def copy(self):

        return self.fixCopy()


class coneTruss(group):

    def __init__(self, L, R0, R1, Nt, color):

        self.color = color
        self.L = L
        self.R0 = R0
        self.R1 = R1
        self.Nt = Nt

        self.objList = []

        self.z0 = 0.0
        self.z1 = L
        self.r0 = R0
        self.r1 = R1

        self.calculate()

    def calculate(self):

        theta = numpy.linspace(0, 2*numpy.pi, self.Nt)

        x00 = numpy.cos(theta)*self.R0
        y00 = numpy.sin(theta)*self.R0
        z00 = theta*0.0 + self.z0

        alpha = numpy.pi/self.Nt

        x11 = numpy.cos(theta + alpha)*self.R1
        y11 = numpy.sin(theta + alpha)*self.R1
        z11 = theta*0.0 + self.z1

        x = []
        y = []
        z = []

        for ii in range(0, len(x00)):

            x.append(x00[ii])
            x.append(x11[ii])
            y.append(y00[ii])
            y.append(y11[ii])
            z.append(z00[ii])
            z.append(z11[ii])

        ii = 0
        x.append(x00[ii])
        x.append(x11[ii])
        y.append(y00[ii])
        y.append(y11[ii])
        z.append(z00[ii])
        z.append(z11[ii])

        self.objList = [lineClass(numpy.array(x), numpy.array(y),
                                  numpy.array(z), self.color)]

    def copy(self):

        return self.fixCopy()


class cyl(module):

    def __init__(self, L, R, Nt, color):

        self.L = L
        self.R = R
        self.Nt = Nt
        self.color = color
        self.z = [0, L]
        self.r = [R, R]

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
        self.z = [0, L]
        self.r = [R1, R2]

        self.Nr = len(self.r)
        self.calculate()

    def copy(self):

        return cone(self.L, self.R1, self.R2, self.Nt, self.color)


class curveCone(module):

    def __init__(self, L, R1, R2, RR, Nt, color):

        self.L = L
        self.R1 = R1
        self.R2 = R2
        self.RR = RR
        self.Nt = Nt
        self.color = color

        self.arcCalculate()

        self.Nr = len(self.r)
        self.calculate()

    def copy(self):

        return curveCone(self.L, self.R1, self.R2, self.RR,
                         self.Nt, self.color)

    def arcCalculate(self):

        if self.RR < 0.0:
            signal = -1.0
        else:
            signal = 1.0

        RR = signal*self.RR

        p1 = numpy.array([self.R1, 0])
        p2 = numpy.array([self.R2, self.L])

        pm = (p1 + p2)/2
        v1 = (p2 - p1)/2

        d = numpy.sqrt(v1[0]**2 + v1[1]**2)

        if RR < d:
            self.RR = signal*d

        v1 = v1/d
        v2 = numpy.array([v1[1], -v1[0]])

        x = numpy.sqrt(RR**2 - d**2)
        thetaMax = numpy.arcsin(d/RR)

        theta = numpy.linspace(-thetaMax, thetaMax, 20)

        self.r = []
        self.z = []

        for ang in theta:
            p = pm + v1*numpy.sin(ang)*RR + signal*v2*(RR*numpy.cos(ang) - x)
            self.r.append(p[0])
            self.z.append(p[1])

        return None


class toroCone(module):

    def __init__(self, L, R1, R2, RR, Nt, color):

        self.L = L
        self.R1 = R1
        self.R2 = R2
        self.RR = RR
        self.Nt = Nt
        self.color = color

        self.arcCalculate()

        self.Nr = len(self.r)
        self.calculate()

    def copy(self):

        return toroCone(self.L, self.R1, self.R2, self.RR,
                        self.Nt, self.color)

    def arcCalculate(self):

        if self.RR < 0.0:
            signal = -1.0
        else:
            signal = 1.0

        RR = signal*self.RR

        p1 = numpy.array([self.R1, 0])
        p2 = numpy.array([self.R2, self.L])

        pm = (p1 + p2)/2
        v1 = (p2 - p1)/2

        d = numpy.sqrt(v1[0]**2 + v1[1]**2)

        if RR < d:
            self.RR = signal*d

        v1 = v1/d
        v2 = numpy.array([v1[1], -v1[0]])

        x = numpy.sqrt(RR**2 - d**2)
        thetaMax = numpy.pi - numpy.arcsin(d/RR)

        theta = numpy.linspace(-thetaMax, thetaMax, 50)

        self.r = []
        self.z = []

        for ang in theta:
            p = pm + v1*numpy.sin(ang)*RR + signal*v2*(RR*numpy.cos(ang) + x)
            self.r.append(p[0])
            self.z.append(p[1])

        return None


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


class brackets(group):

    def __init__(self, d, r, z0, L, y, color):

        cyl1 = cyl(d, r, 50, color)
        cyl1.rotateY(-numpy.pi/2)

        cyl1.translate([0.0, -y, z0])
        cyl1 = cyl1.fixCopy()
        cyl2 = cyl1.copy()
        cyl2.translate([0.0, 0.0, L])
        cyl2 = cyl2.fixCopy()
        cyl3 = cyl2.copy()
        cyl3.translate([0.0, 2*y, 0.0])
        cyl3 = cyl3.fixCopy()
        cyl4 = cyl3.copy()
        cyl4.translate([0.0, 0, -L])
        cyl4 = cyl4.fixCopy()

        self.objList = [cyl1, cyl2, cyl3, cyl4]

        self.z0 = 0.0
        self.z1 = 0.0
        self.r0 = 0.0
        self.r1 = 0.0

        for obj in self.objList:

            self.z0 = numpy.min([self.z0, obj.z0])
            self.z1 = numpy.max([self.z1, obj.z1])
            self.r0 = numpy.min([self.r0, obj.r0])
            self.r1 = numpy.max([self.r1, obj.r1])

    def opacity(self, op):

        self.color = [self.color[0], self.color[1], self.color[2], op]

        for obj in self.objList:
            obj.opacity(op)


class nozzle(stack):

    def __init__(self, L, R, Nt, color):

        self.L = L
        self.R = R
        self.Nt = Nt
        self.color = color

        c2 = numpy.array([1, 1, 1])*0.1
        cyl1 = cyl(L, R, Nt, color)
        cone1 = cone(L, R*0.75, R*0.25, Nt, c2)
        cone2 = cone(0, R*0.25, 0.0, Nt, c2)
        cone2.translate([0.0, 0.0, L])
        cone3 = cone(0, R*0.95, R*0.55, Nt, color)
        cone3.translate([0.0, 0.0, L/2])

        self.objList = [cyl1, cone1, cone2, cone3]

        self.r0 = R
        self.r1 = R
        self.z0 = 0.0
        self.z1 = L

    def copy(self):

        return self.fixCopy()

    def opacity(self, op):

        self.color = [self.color[0], self.color[1], self.color[2], op]

        for obj in self.objList:
            obj.opacity(op)


class thin(group):
    # TODO: correct this name

    def __init__(self, rootChord, tipChord, span, swipeAngle, color):

        self.rc = rootChord
        self.tc = tipChord
        self.s = span
        self.sa = swipeAngle
        self.color = color

        z0 = span*numpy.tan(swipeAngle)
        self.z = [[rootChord, tipChord - z0], [0.0, 0.0 - z0]]
        self.x = [[0.0, span], [0.0, span]]
        self.y = [[0.0, 0.0], [0.0, 0.0]]

        self.x = numpy.array(self.x)
        self.y = numpy.array(self.y)
        self.z = numpy.array(self.z)

        obj = surfClass(self.x, self.y, self.z, self.color)

        self.objList = [obj]

        self.z0 = -z0
        self.z1 = rootChord
        self.r0 = 0.0
        self.r1 = span

    def copy(self):

        return thin(self.rc, self.tc, self.s, self.sa, self.color)

    def opacity(self, op):

        self.color = [self.color[0], self.color[1], self.color[2], op]

        for obj in self.objList:
            obj.opacity(op)


class retangle(group):

    def __init__(self, a, b, color, letter):

        self.a = a
        self.b = b
        self.color = color
        self.orientation = (0.0, 0.0, 0.0)
        self.letter = letter

        x = [0.0, a, a, 0.0, 0.0]
        y = [0.0, 0.0, b, b, 0.0]
        z = [0.0, 0.0, 0.0, 0.0, 0.0]

        x = numpy.array(x) - a/2
        y = numpy.array(y)
        z = numpy.array(z)

        r = numpy.sqrt(x**2 + y**2)

        self.objList = [lineClass(x, y, z, self.color)]

        self.z0 = z.min()
        self.z1 = z.max()
        self.r0 = r.min()
        self.r1 = r.max()

    def copy(self):

        return retangle(self.a, self.b, self.color)

    def rotateYXZ(self, angY, angX, angZ):

        f = -180/numpy.pi

        self.rotateY(angY)
        self.rotateX(angX)
        self.rotateZ(angZ)
        self.orientation = (angX*f, angY*f, angZ*f)

    def plotLetter(self, mlab):

        line = self.objList[0]

        v1 = [line.x[1] - line.x[0], line.y[1] - line.y[0],
              line.z[1] - line.z[0]]
        v2 = [line.x[2] - line.x[1], line.y[2] - line.y[1],
              line.z[2] - line.z[1]]

        v1 = numpy.array(v1)
        v2 = numpy.array(v2)

        v1 = v1/numpy.linalg.norm(v1)
        v2 = v2/numpy.linalg.norm(v2)

        vtext = tvtk.VectorText()
        vtext.text = self.letter
        text_mapper = tvtk.PolyDataMapper()
        configure_input_data(text_mapper, vtext.get_output())
        vtext.update()

        p2 = tvtk.Property(color=(self.color[0], self.color[1], self.color[2]))
        text_actor = tvtk.Follower(mapper=text_mapper, property=p2)

        text_actor.position = (0, 0, 0)

        auxx = text_actor._get_x_range()
        auxy = text_actor._get_y_range()

        yc = (auxy[1] - auxy[0])/2

        alpha = self.b/(2*yc)

        xm = (auxx[1] + auxx[0])*alpha/2
        y0 = auxy[0]

        vCorr = (self.a/2 - xm)*v1 - y0*v2

        text_actor.orientation = self.orientation
        text_actor.scale = numpy.array([alpha, alpha, 1])

        auxx = text_actor._get_x_range()
        auxy = text_actor._get_y_range()

        text_actor.position = (line.x[0] + vCorr[0],
                               line.y[0] + vCorr[1], line.z[0] + vCorr[2])

        fig = mlab.gcf()
        fig.scene.add_actor(text_actor)

        return mlab


class nozzleInclinated(stack):

    def __init__(self, L, R, Nt, phi, color):

        self.L = L
        self.R = R
        self.Nt = Nt
        self.color = color
        self.phi = phi

        f = 3/4

        if len(color) is 4:

            c2 = numpy.array([1, 1, 1, 1])*0.1
            c2[3] = color[3]

        else:

            c2 = numpy.array([1, 1, 1])*0.1

        c2 = numpy.array([1, 1, 1])*0.1
        cyl1 = cyl(L*f, R, Nt, color)
        cyl1.translate([0.0, 0.0, L*(1-f)])

        Lt = 0.65*L
        Rt1 = 0.6*R
        Rt2 = 0.3*R

        cone1 = cone(Lt, Rt1, Rt2, Nt, c2)
        cone2 = cone(0, Rt2, 0.0, Nt, c2)
        cone2.translate([0.0, 0.0, Lt])
        tub = stack([cone1, cone2])
        tub.translate([0.0, 0.0, -Lt])
        tub.rotateY(-phi)
        tub.translate([0.0, 0.0, Lt])

        cone3 = cone(L/2, R*0.95, R*0.4, Nt, color)
        cone3.translate([0.0, 0.0, L/2])
        cone4 = cone(0, R*0.4, 0, Nt, color)
        cone4.translate([0.0, 0.0, L])

        self.objList = [cyl1, tub, cone3, cone4]

        self.r0 = R
        self.r1 = R
        self.z0 = 0.0
        self.z1 = L

    def copy(self):

        return self.fixCopy()

    def opacity(self, op):

        self.color = [self.color[0], self.color[1], self.color[2], op]

        for obj in self.objList:
            obj.opacity(op)


if __name__ == "__main__":

    # Example of use:

    # colors based on a grey scale.
    cmap = cm.get_cmap('Greys')
    red = cm.get_cmap('Reds')

    c1 = numpy.array([1.0, 1.0, 1.0])

    # stages desing
    # stg1
    cone1 = cone(1.5, 1.5, 1.0, 50, c1)
    disc1 = cone(0.0, 1.4, 0.0, 50, c1)
    cyl3 = cyl(10, 1.5, 50, c1*0.8)
    nozzle1 = nozzle(2, 1.5, 50, c1*0.5)
    stg1 = stack([nozzle1, cyl3, disc1, cone1])

    # stg2
    cyl1 = cyl(1, 1, 50, c1)
    cyl2 = cyl(6, 1, 50, c1*0.8)
    nozzle2 = nozzle(1.5, 1, 50, c1*0.5)
    stg2 = stack([nozzle2, cyl2, cyl1])

    # stg3
    cyl4 = cyl(2.5, 1, 50, c1*0.8)
    stg3 = stack([nozzle2, cyl4, cyl1])
    stg3.opacity(0.3)

    # coife booster
    cyl5 = cyl(0.5, 1, 50, c1*0.7)
    coife1 = coife(3, 1, 0.3, 50, c1*0.7)
    stgCoifeB = stack([cyl5, coife1])

    # coife Central Body
    cone2 = cone(1.5, 1.0, 1.3, 50, c1*0.7)
    cyl6 = cyl(3, 1.3, 50, c1*0.7)
    coife1 = coife(3, 1.3, 0.3, 50, cmap(0.3))
    stgCoifeC = stack([cone2, cyl6, coife1])

    # Central body design
    rocket1 = stack([stg1, stg2, stg3, stgCoifeC])

    # Booster design
    booster1 = stack([stg2, stgCoifeB])
    brackets1 = brackets(1, 0.1, 2, 5, 0.5, c1*0.2)
    booster1.around(brackets1, 1, 0.5)
    booster1.scale(0.7)
    booster1.rotateZ(numpy.pi)
    thin1 = thin(1, 0.5, 0.5, 0.1*numpy.pi, c1*0.8)
    booster1.around(thin1, 1, 0.7)

    # Final rocket
    rocket1.around(booster1, 4, 2.4)
    # t2 = t1.fixCopy()
    # t1.translate([0, 0, -2])

    # Display
    fig = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
    mlab.clf()
    mlab = rocket1.plot(mlab)

    # text

    b = 0.6
    ang = numpy.pi/4

    plotWord('1STG', 3, b, 1.55, ang, c1*0.1, mlab)

    plotWord('2STG', 16, b, 1.05, ang, c1*0.1, mlab)

    plotWord('3STG', 24, 0.35, 1.05, ang, c1*0.1, mlab)

    # TODO: solve the bug in saving figures
    # mlab.savefig(filename='test.png')
    mlab.show()
