#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:19:46 2017

@author: levi
"""

import datetime, time
#import matplotlib.pyplot as plt

from interf import ITman
import probRock as prob
#import prob101 as prob
#import probBrac as prob
#import probCart as prob
#import probCartMod as prob
#import probSmpl as prob
#import os            
#%%
            
# ##################
# MAIN SEGMENT:
# ##################
if __name__ == "__main__":
    print("-"*66)
    print('\nRunning main.py!')
    print(datetime.datetime.now())
    print('\n')

    sol = prob.prob()
    ITman = ITman(probName=sol.probName,confFile='teste.its')
    sol.log = ITman.log

    try:
        ITman.greet()
        
        start_time = time.time()
        sol,solInit = ITman.setInitSol(sol)

        sol = ITman.frstRestRnds(sol)

        sol = ITman.gradRestCycl(sol,solInit)

        sol = ITman.restRnds(sol)

        line = "#"*66
        ITman.log.printL("\n\n")
        ITman.log.printL(line)
        ITman.log.printL("                      OPTIMIZATION FINISHED!                      ")
        ITman.log.printL(line)
        ITman.saveSol(sol,ITman.probName+'_currSol.pkl')
        sol.showHistP()
    
        sol.showHistQ()
        sol.showHistI()
        sol.showHistGradStep()

        ITman.log.printL("\n\n")
        ITman.log.printL(line)
        ITman.log.printL("                   THIS IS THE FINAL SOLUTION:                    ")
        ITman.log.printL(line)

        sol.plotSol()
    
        ITman.log.printL("\n"+line)
        ITman.log.printL("=== First Guess + MSGRA execution: %s seconds ===\n" % \
              (time.time() - start_time))

        for k in range(3):
            print("\a")
            time.sleep(.1)

    except KeyboardInterrupt:
        ITman.log.printL("\n\n\nUser has stopped the program.")
        raise
    except:
        ITman.log.printL("\n\n\nI'm sorry, something bad happened.")
        raise
    finally:
        ITman.log.printL("Terminating now.")
        ITman.log.close()
