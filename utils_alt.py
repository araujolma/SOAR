# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 18:53:01 2016

@author: levi
"""
import numpy 

def ddt(sizes,vec):
    N = sizes['N']
    dt = 1.0/(N-1)
    dvec = numpy.empty_like(vec)
    
    for k in range(N-1):
        dvec[k] = (vec[k+1]-vec[k])/dt
    dvec[N-1] = dvec[N-2]
    
    return dvec
