#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 18:05:26 2017

@author: levi
"""

import numpy
import matplotlib.pyplot as plt

x = numpy.arange(-3,3,.01)

y = numpy.tanh(x)

plt.plot(x,y)
plt.grid(True)
plt.show()

# sinh(t) = .5*(exp(t)-exp(-t))
# cosh(t) = .5*(exp(t)+exp(-t))
# sinh'(t) = .5(exp(t)+exp(-t)) = cosh(t)
# cosh'(t) = .5(exp(t)-exp(-t)) = sinh(t)
# tanh'(t) = (sinh'(t)cosh(t) - sinh(t)cosh'(t) )/(cosh2(t))
#          = (cosh2(t) - sinh2(t))/(cosh2(t))
#          = 1-tanh2(t)
y = 1-numpy.tanh(x)**2

plt.plot(x,y)
plt.grid(True)
plt.show()
