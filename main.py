#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:19:46 2017

@author: levi
"""

import datetime, time
#import matplotlib.pyplot as plt

from interf import ITman
#import probRock as prob
#import prob101 as prob
import probBrac as prob
#import probCart as prob
#import probCartMod as prob
#import probSmpl as prob
#import os            
#%%
            
# ##################
# MAIN SEGMENT:
# ##################
if __name__ == "__main__":
    print('------------------------------------------------------------------')
    print('\nRunning main.py!')
    print(datetime.datetime.now())
    print('\n')
    
    sol = prob.prob()
    #GradStat = GradStat()
    ITman = ITman(probName=sol.probName)
    ITman.greet()
    
    start_time = time.time()
    sol,solInit = ITman.setInitSol(sol)
    
    sol = ITman.frstRestRnds(sol)

    sol = ITman.gradRestCycl(sol,solInit)
    
    sol = ITman.restRnds(sol)
    
    print("\n\n")
    print("##################################################################")
    print("                   OPTIMIZATION FINISHED!                         ")
    print("##################################################################")
    ITman.saveSol(sol,ITman.probName+'_currSol.pkl')
    sol.showHistP()

    sol.showHistQ()
    sol.showHistI()
    sol.showHistGradStep()
    
    print("\n\n")
    print("##################################################################")
    print("                   THIS IS THE FINAL SOLUTION:                    ")
    print("##################################################################")
    
    sol.plotSol()

    print("\n################################################################")
    print("=== First Guess + MSGRA execution: %s seconds ===\n" % \
          (time.time() - start_time))
    
    for k in range(3):
        print("\a")
        time.sleep(.1)