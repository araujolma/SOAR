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
import numpy
import sys
from mayavi import mlab
from seeingRockets import *

sys.path.append('./mayavi_master')
sys.path.append('./')

if __name__ == "__main__":

    # Example of use:

    # colors based on a grey scale.
    c1 = numpy.array([1.0, 1.0, 1.0])

    cyl1 = cyl(1, 1, 50, c1)
    cyl2 = cyl(10, 1, 50, c1*0.8)
    cyl5 = cyl(0.5, 1, 50, c1*0.7)
    coife1 = coife(3, 1, 0.3, 50, c1*0.7)
    nozzle2 = nozzle(1.5, 1, 50, c1*0.5)

    stg2 = stack([nozzle2, cyl2, cyl1])
    stgCoife = stack([cyl5, coife1])

    frac = 0.45
    rocket1 = stack([stg2, stgCoife])
    rocket2 = rocket1.copy()

    for ii in range(0, 4):

        rocket3 = rocket2.copy()
        rocket3.scale(0.3)

        rocket2 = rocket1.copy()
        rocket2.around(rocket3, 3, 1.34)
        rocket2 = rocket2.fixCopy()


    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
    mlab.clf()
    mlab = rocket2.plot(mlab)
    #mlab.savefig(filename='test.png')
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
