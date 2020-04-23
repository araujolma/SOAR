#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:19:46 2017

@author: levi
"""

import datetime, time, sys, os

line = "#" * 66

def main(args,isManu=True,destFold=''):

    from interf import ITman

    # This is the default problem, for users who want to just run "main.py"
    defaultProb = None#'zer'#'brac'#'cart'#'9_1'#'9_2'#'10_1'#'10_2'

    if defaultProb is None and (len(args) == 1):
        # Default of the default, rocket problem
        import probRock as prob
        confFile = 'defaults' + os.sep + 'probRock - saturn1B.its'
    else:
        # if the user runs the program from the command line,
        # check for the problem choice as an optional argument.
        if len(args) > 1:
            probName = args[1].lower()
        else:
            probName = defaultProb.lower()

        if 'brac' in probName:
            import probBrac as prob
            confFile = 'defaults'+ os.sep + 'probBrac.its'
        elif 'cart' in probName:
            import probCart as prob
            confFile = 'defaults'+ os.sep + 'probCart.its'
        elif 'land' in probName:
            import probLand as prob
            confFile = 'defaults' + os.sep + 'probLand.its'
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
                print("\nSorry, I didn't understand the problem " + \
                      "instance:" + '   "' + probName + '"...\n' + \
                      "I will carry on with probRock, ok?\n")
            import probRock as prob
            confFile = 'defaults'+ os.sep + 'probRock.its'

        if len(args) > 2:
            confFile = args[2]


    sol = prob.prob()

    # Declare iterations manager
    ITman = ITman(probName=sol.probName,confFile=confFile,isManu=isManu,
                  destFold=destFold)
    # the logger object for the solution is the same for the iterations manager
    sol.log = ITman.log

    try:
        ## RUN STATUS: initializing
        ITman.log.repoStat('init')

        # Greet the user, confirm basic settings
        ITman.greet()
        start_time = time.time()
        # Set initial solution
        sol,solInit = ITman.setInitSol(sol)
        # Perform the first restoration rounds
        sol = ITman.frstRestRnds(sol)
        ## RUN STATUS: in progress
        ITman.log.repoStat('inProg')
        # Proceed to the gradient-restoration cycles
        sol = ITman.gradRestCycl(sol,solInit)

        # Display final messages, show solution and convergence reports
        msg = "\n\n\n" + line + '\n' + (' '*22) + \
              "OPTIMIZATION FINISHED!" + (' '*22) + '\n' + line
        ITman.log.printL(msg)
        ## RUN STATUS: sucess!
        ITman.log.repoStat('ok')
        # Save solution to disk
        ITman.saveSol(sol,ITman.log.folderName+'/finalSol.pkl')
        # Show all convergence histories
        sol.showHistP(); sol.showHistQ(); sol.showHistI(); sol.showHistQvsI()
        sol.showHistGradStep(); sol.showHistGRrate(); sol.showHistObjEval()

        msg = "\n\n\n" + line + '\n' + (' '*19) + \
              "THIS IS THE FINAL SOLUTION:" + (' '*19) + '\n' + line
        ITman.log.printL(msg)

        # Show solution, compare it with initial guess, show trajectory
        sol.plotSol(); sol.compWith(solInit); sol.plotTraj()

        red = 100. * (sol.histI[1] - sol.I) / sol.histI[1]
        msg = '\nFinal value of I: {:.7E},\n"initial" value of I: {:.7E} ' \
              '(after restoration).\nReduction: {:.3G}%'.format(sol.I, sol.histI[1], red)
        ITman.log.printL(msg)

        msg = '\nFinal values: P = {:.3E}, Q = {:.3E}.\n'.format(sol.P, sol.Q)
        ITman.log.printL(msg)

        ITman.log.printL("\n"+line)
        msg = "=== First Guess + MSGRA execution: %s seconds ===\n" % \
              (time.time() - start_time)
        ITman.log.printL(msg)

        # This does not add that much information, but...
        ITman.log.printL("\nHistQ: "+str(sol.histQ[:(sol.NIterGrad+1)]))
        ITman.log.printL("\nHistI: "+str(sol.histI[:(sol.NIterGrad+1)]))

        # Make a sound!
        ITman.bell()

        # Check Hamiltonian conditions (just to be sure)
        sol.checkHamMin(mustPlot=True)

    # Manage the exceptions.
    # Basically, the logger must be safely shut down.
    except KeyboardInterrupt:
        ITman.log.printL("\n\n\nmain: User has stopped the program.")
        ## RUN STATUS: stopped by user
        ITman.log.repoStat('usrStop')#,iterN=sol.NIterGrad)
        raise
    except:
        ITman.log.printL("\n\n\nmain: I'm sorry, something bad happened.")
        # TODO: the exception should be printed onto the log file!!

        ## RUN STATUS: error
        ITman.log.repoStat('err')#,iterN=sol.NIterGrad)
        raise
    finally:
        ITman.log.printL("main: Terminating now.\n")
        ITman.log.close()

if __name__ == "__main__":
    print(line)
    print('\nRunning main.py with arguments:')
    print(sys.argv)
    print(datetime.datetime.now())

    main(sys.argv)

