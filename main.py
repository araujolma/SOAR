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

    # This is the default problem, for users who want to just run "main.py"
    defaultProb = 'zer'#None#'zer'#'brac'#'cart'#'9_1'#'9_2'#'10_1'#'10_2'

    if defaultProb is None and (len(sys.argv) == 1):
        # Default of the default, rocket problem
        import probRock as prob
        confFile = 'defaults/probRock.its'
    else:
        # if the user runs the program from the command line,
        # check for the problem choice as an optional argument.
        if len(sys.argv) > 1:
            probName = sys.argv[1].lower()
        else:
            probName = defaultProb

        if 'cart' in probName:
            import probCart as prob
            confFile = 'defaults'+ os.sep + 'probCart.its'
        elif 'brac' in probName:
            import probBrac as prob
            confFile = 'defaults'+ os.sep + 'probBrac.its'
        elif 'lqr' in probName:
            import probLQR as prob
            confFile = 'defaults'+ os.sep + 'probLQR.its'
        elif 'zer' in probName:
            import probZer as prob
            confFile = 'defaults'+ os.sep + 'probZer.its'
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
        else:
            if 'rock' not in probName:
                print("\nSorry, I didn't understand the problem instance:" + \
                      '   "' + probName + '"...\n' + \
                      "I will carry on with probRock, ok?\n")
            import probRock as prob
            confFile = 'defaults'+ os.sep + 'probRock.its'

        if len(sys.argv) > 2:
            confFile = sys.argv[2]


    sol = prob.prob()

    ITman = ITman(probName=sol.probName,confFile=confFile)
    sol.log = ITman.log

    try:
        # Greet the user, confirm basic settings
        ITman.greet()
        #input("main: Done greeting. I will set init sol now.\n")
        start_time = time.time()
        # Set initial solution
        sol,solInit = ITman.setInitSol(sol)
        # Perform the first restoration rounds
        #input("main: Done setting init sol. I will restore sol now.\n")
        sol = ITman.frstRestRnds(sol)
        # Proceed to the gradient-restoration cycles
        #input("main: Done with first restorations. I will go to GR cycle now.\n")
        sol = ITman.gradRestCycl(sol,solInit)
        # Final restorations
        # TODO: NECESSÁRIO?
        #input("main: Done with GR cycle. I will go to final restorations now.\n")
        sol,_ = ITman.restRnds(sol)

        # Display final messages, show solution and convergence reports
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

        # TODO: [debug] retirar isso
        ITman.log.printL("\nHistQ: "+str(sol.histQ[:(sol.NIterGrad+1)]))
        ITman.log.printL("\nHistI: "+str(sol.histI[:(sol.NIterGrad+1)]))

        for k in range(3):
            print("\a")
            time.sleep(.1)

    # Manage the exceptions. Basically, the logger must be safely shut down.
    except KeyboardInterrupt:
        ITman.log.printL("\n\n\nUser has stopped the program.")
        raise
    except:
        ITman.log.printL("\n\n\nI'm sorry, something bad happened.")
        raise
    finally:
        ITman.log.printL("Terminating now.")
        ITman.log.close()
