#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:19:46 2017

@author: levi
"""

import datetime, time, sys
from interf import ITman

#import probRock as prob
#confFile = 'teste.its'
#import prob101 as prob
#confFile = 'default_prob101.its'
#import probBrac as prob
#confFile = 'default_probBrac.its'
#import probCart as prob
#confFile = 'default_probCart.its'
#import probCartMod as prob
#import probSmpl as prob
#import os
#%%

# ##################
# MAIN SEGMENT:
# ##################
if __name__ == "__main__":
    print("-"*66)
    print('\nRunning main.py with arguments:')
    print(sys.argv)
    print(datetime.datetime.now())

    overrideWithProb = None#'cart'

    if overrideWithProb is None and (len(sys.argv) == 1):
        import probRock as prob
        confFile = 'teste.its'
    else:
        if overrideWithProb is None:
            probName = sys.argv[1].lower()
        else:
            probName = overrideWithProb

        if 'cart' in probName:
            import probCart as prob
            confFile = 'defaults/probCart.its'
        elif 'brac' in probName:
            import probBrac as prob
            confFile = 'defaults/probBrac.its'
        elif '101' in probName:
            import prob101 as prob
            confFile = 'defaults/prob101.its'
        else:
            import probRock as prob
            confFile = 'defaults/probRock.its'

        if len(sys.argv) > 2:
            confFile = sys.argv[2]


    sol = prob.prob()

    ITman = ITman(probName=sol.probName,confFile=confFile)
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
        ITman.saveSol(sol,ITman.log.folderName+'/currSol.pkl')
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
