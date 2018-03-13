#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:19:46 2017

@author: levi
"""

import datetime, time, sys, os
from interf import ITman


#%%

# ##################
# MAIN SEGMENT:
# ##################
if __name__ == "__main__":
    print("-"*66)
    print('\nRunning main.py with arguments:')
    print(sys.argv)
    print(datetime.datetime.now())

    overrideWithProb = '9_1'#None#'brac'#'cart'#'9_1'#'9_2'#'10_1'#'10_2'

    if overrideWithProb is None and (len(sys.argv) == 1):
        # Default of the default, rocket problem
        import probRock as prob
        confFile = 'defaults/probRock.its'
    else:
        if len(sys.argv) > 1:
            probName = sys.argv[1].lower()
        else:
            probName = overrideWithProb

        if 'cart' in probName:
            import probCart as prob
            confFile = 'defaults'+ os.sep + 'probCart.its'
        elif 'brac' in probName:
            import probBrac as prob
            confFile = 'defaults'+ os.sep + 'probBrac.its'
        elif '9_1' in probName:
            import prob9_1 as prob
            confFile = 'defaults'+ os.sep + 'prob9_1.its'
        elif '9_2' in probName:
            import prob9_2 as prob
            confFile = 'defaults'+ os.sep + 'prob9_2.its'
        elif '10_1' in probName:
            import prob10_1 as prob
            confFile = 'defaults'+ os.sep + 'prob10_1.its'
        elif '10_2' in probName:
            import prob10_2 as prob
            confFile = 'defaults'+ os.sep + 'prob10_2.its'
        elif 'lqr' in probName:
            import probLQR as prob
            confFile = 'defaults' + os.sep + 'probLQR.its'

        else:
            print("\nSorry, I did not understand the problem instance:" + \
                  '\n   "' + probName + '"\n'\
                  "Let's carry on with the rocket problem, shall we?")
            import probRock as prob
            confFile = 'defaults'+ os.sep + 'probRock.its'

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
