#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seeingRockets.py: a code for rocket visualization

Created on Tue Mar  6 19:50:50 2018

@author: carlos

Requirement:
    mayavi

"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy
import sys
from mayavi import mlab
from seeingRockets import *

sys.path.append('./mayavi_master')
sys.path.append('./')


def rd107():

    red = cm.get_cmap('Reds')
    red = red(0.9)
    grey = cm.get_cmap('Greys')
    grey = grey(0.5)

    d1 = 0.25
    d2 = d1*0.85
    L = 0.2

    cone1 = cone(L, d1, d2, 50, red)
    disk1 = cone(0, d2*1.1, 0.0, 50, red)
    internal = stack([cone1, disk1])
    internal = internal.fixCopy()
    disk2 = cone(0, d1*1.05, d1, 50, red)
    cone2 = cone(1.5*L, d1*1.1, d2*1.1, 50, grey)
    external = stack([disk2, cone2])
    external = external.fixCopy()

    tub = group([external, internal])
    tub.translate([0.0, 0.0, -L*1.5])

    engine = group([])
    engine.around(tub, 4, 2*d1)
    engine.rotateZ(numpy.pi/4)

    return engine.fixCopy()

if __name__ == "__main__":

    # Example of use:

    greenMap = cm.get_cmap('Greys')
    greyMap = cm.get_cmap('Greys')
    orangeMap = cm.get_cmap('Oranges')
    blueMap = cm.get_cmap('Blues')

    gg = 0.6

    engineL = 1.5

    # booster
    Dm = 0.5
    L1 = 7
    eng = rd107()
    eng.translate([-0.15, 0.0, 0.0])
    disk1 = cone(0, 1, 0, 50, greyMap(0.2))
    disk1 = stack([eng, disk1])
    cyl1 = cyl(engineL, 1, 50, orangeMap(0.5))
    cone1 = cone(L1, 1, Dm, 50, greenMap(gg))
    angle = numpy.arctan((1-Dm)/L1)
    cone1 = cone1.fixCopy()
    L2 = 2
    cone2 = cone(L2, Dm, 0, 50, greenMap(gg))
    cone2.inclinate(0.3*Dm/L2)
    cone2 = cone2.fixCopy()
    booster1 = stack([disk1, cyl1, cone1, cone2])
    thin1 = thin(2/3, 0.0, 1, 0.0, orangeMap(0.5))
    booster1.around(thin1, 1, 1.0)
    booster1.rotateY(angle)

    # central body

    # stage1
    D2 = 0.8
    D3 = 0.9
    disk1 = cone(0, D2, 0, 50, greyMap(0.2))
    eng = rd107()
    disk1 = stack([eng, disk1])
    cyl22 = cyl(engineL, D2, 50, orangeMap(0.5))
    cyl2 = cyl(L1, D2, 50, greenMap(gg))
    cone3 = cone(L2, D2, 1, 50, greenMap(gg))
    cyl3 = cone(3, 1, D3, 50, greenMap(gg))
    cone22 = cone(0.4, D3, 0.0, 50, greenMap(gg))
    truss1 = interTruss(0.7, D3-0.025, 10, greenMap(gg))
    inter1 = group([truss1, cone22])

    stg1 = stack([disk1, cyl22, cyl2, cone3, cyl3, inter1])

    # stage2
    disk3 = cone(0, D3, 0, 50, greyMap(0.2))
    cyl4 = cyl(engineL, D3, 50, orangeMap(0.5))
    cyl5 = cyl(4, D3, 50, greenMap(gg))
    cone55 = cone(1, D3, 1, 50, greenMap(gg))
    stg2 = stack([disk3, cyl4, cyl5, cone55])

    # coife1
    cyl6 = cyl(4.5, 1, 50, greyMap(0.0))
    cone4 = cone(1.5, 1, 0.2, 50, greyMap(0.0))
    cyl7 = cyl(2, 0.2, 50, greyMap(0.0))
    cone5 = cone(0.2, 0.2, 0.15, 50, greyMap(0.0))
    cyl8 = cyl(1.5, 0.15, 50, greyMap(0.0))
    cone6 = cone(0.15, 0.15, 0.0, 50, greyMap(0.0))
    coife1 =stack([cyl6, cone4, cyl7, cone5, cyl8, cone6])

    # rocket
    rocket1 = stack([stg1, stg2, coife1])
    rocket1.around(booster1, 4, 1.93)



    bcolor = blueMap(0.3)
    bcolor = (bcolor[0], bcolor[1], bcolor[2])
    mlab.figure(1, bgcolor=bcolor, fgcolor=(0, 0, 0), size=(400, 300))
    mlab.clf()
    mlab = rocket1.plot(mlab)
    mlab.show()

#==============================================================================
#     #  Case using matplotlib (image quality low)
#     fig2 = plt.figure()
#     ax2 = fig2.add_subplot(111, projection='3d')
#     ax2 = rocket1.matPlot(ax2)
#     Lmax = rocket1.z1
#     ax2.set_xlim(-Lmax/2, Lmax/2)
#     ax2.set_ylim(-Lmax/2, Lmax/2)
#     ax2.set_zlim(0, Lmax)
#     plt.show()
#==============================================================================
