#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 12:02:44 2017

@author: levi
"""

#from interf import ITman
#debug = ITman()
#
#solInit = debug.loadSol('probRock_solInit.pkl')
#solInitRest = debug.loadSol('probRock_solInitRest.pkl')
#sol = debug.loadSol('probRock_currSolDeb5.pkl')
##sol.plotSol(mustSaveFig=False,subPlotAdjs=(.06,.1,.98,.91,.20,.63))
#sol.plotSol(mustSaveFig=False, intv=[0.0,30.0],subPlotAdjs=(.06,.1,.98,.91,.20,.63))
#
#sol.plotSol(opt={'mode':'lambda'},mustSaveFig=False, intv=[0.0,30.0],subPlotAdjs=(.06,.1,.98,.91,.20,.63))
##sol.plotSol(opt={'mode':'lambda'},mustSaveFig=False, intv=[100.0,200.0],subPlotAdjs=(.06,.1,.98,.91,.20,.63))


def fterte(a=2.0,b=3.0,c=5.0):
    return a*b*c


terteDict = {'a': 7.0,
             'b': 11.0, 
             'c':13.0}
print(fterte(**terteDict))