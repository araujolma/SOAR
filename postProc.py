#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 17:47:23 2019

@author: levi

A module for post processing.
"""

import interf
import probRock as prob
import matplotlib.pyplot as plt

def loadSol(fileName):
    """ Load and return a solution for inspection."""

    ITman = interf.ITman(probName=prob.prob.probName,isInteractive=True)
    sol = ITman.loadSol(path=fileName)

    return sol

def IvsQ(fileList):
    """Plot a graph for I history vs. Q history."""

    # Produce a list if it is not the case
    if not isinstance(fileList,list):
        fileList = [fileList]

    for file in fileList:
        sol = loadSol(file)
        histI = sol.histI[0:(sol.NIterGrad + 1)]
        histQ = sol.histQ[0:(sol.NIterGrad + 1)]
        plt.semilogx(histQ,histI,label=file)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Hist Q')
    plt.ylabel('Hist I')
    plt.title('I vs. Q')
    plt.show()