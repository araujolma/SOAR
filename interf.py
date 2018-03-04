#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:35:29 2017

@author: levi
"""

import dill, datetime, pprint, os, shutil
from utils import getNowStr#, logPrint

class logger():
    """ Class for the handler of log messages."""

    def __init__(self,probName,mode='both'):
        # Mode ('both', 'file' or 'screen') sets the target output
        self.mode = mode
        # Results folder for this run
        self.folderName = probName + '_' + getNowStr()
        # Create the folder
        os.makedirs(self.folderName)

        try:
            self.fhand = open(self.folderName + os.sep + 'log.txt','w+')
        except:
            print("Sorry, could not open/create the file!")
            exit()

    def printL(self,msg,mode=''):
        if mode in '':
            mode = self.mode

        if mode in 'both':
            print(msg)
            self.fhand.write('\n'+msg)
        elif mode in 'file':
            self.fhand.write('\n'+msg)
        elif mode in 'screen':
            print(msg)

    def pprint(self,obj):
        if self.mode in 'both':
            pprint.pprint(obj)
            pprint.pprint(obj,self.fhand)
        elif self.mode in 'file':
            pprint.pprint(obj,self.fhand)
        elif self.mode in 'screen':
            pprint.pprint(obj)

    def close(self):
        self.fhand.close()


class ITman():
    """Class for the ITerations MANager.

    Only one object from this class is intended to be active during a program
    run. It manages the iterations in the solution, commands solution saving,
    plotting, debug modes, etc.

    """

    bscImpStr = "\n >> "
    dashStr = '\n-------------------------------------------------------------'

    def __init__(self,confFile='',probName='prob'):
        self.probName = probName
        self.defOpt = 'newSol'#'loadSol'#
        self.initOpt = 'extSol'
        self.confFile = confFile
        self.loadSolDir = 'defaults' + os.sep + probName+'_solInitRest.pkl'
        self.loadAltSolDir = ''
        #'solInitRest.pkl'#'solInit.pkl'#'currSol.pkl'
        self.GRplotSolRate = 1
        self.GRsaveSolRate = 5
        self.GRpausRate = 1000#1000#10
        self.GradHistShowRate = 5
        self.RestPlotSolRate = 5
        self.RestHistShowRate = 5
        self.parallelOpt = {'gradLMPBVP': True,
                            'restLMPBVP': True}
        self.overrideParallel()
        self.log = logger(probName)
        # Create directory for logs and stuff

    def overrideParallel(self):
        """Override the parallel configurations in order to prevent entering
        into parallel mode in Windows systems. (issue #63, btw)"""

        if os.sep != '/':
            msg = "\n\n" + '-'*88 + \
                  "Overriding the parallel settings to False!" + \
                  '-'*88 + "\n\n"
            self.log.printL(msg)
            self.parallelOpt = {'gradLMPBVP':False,#True,
                                'restLMPBVP':False}#True}
            # windows systems!
            self.parallelOpt

    def prntDashStr(self):
        self.log.printL(self.dashStr)

    def prom(self):
        inp = input(self.bscImpStr)
        self.log.printL(self.bscImpStr+inp,mode='file')
        return inp

    def printPars(self):
        self.log.printL("\nThese are the attributes for the" + \
                        " Iterations manager:\n")
        self.log.pprint(self.__dict__)

    def greet(self):
        """This is the first command to be run at the beggining of the
        MSGRA.

        The idea is to let the user choose whether he/she wants to load a
        previously prepared solution, or generate a new one."""


        # First greetings to user
        self.prntDashStr()
        self.log.printL("\nWelcome to SGRA!\n")
        # Inform problem
        self.log.printL("Loading settings for problem: "+self.probName)
        # Inform results folder for this run
        self.log.printL('Saving results and log in '+self.log.folderName+'.')
        self.prntDashStr()
        # Show parameters for ITman
        self.printPars()
        self.log.printL("\n(You can always change these here in '" + \
                            __name__ + ".py').")

        # Default option: generate new solution from scratch
        if self.defOpt == 'newSol':
            msg = "\nDefault starting option (defOpt) is to " + \
                  "generate new initial guess.\n" + \
                  "Hit 'enter' to do it, or any other key to " + \
                  "load a previously started solution."
            self.log.printL(msg)
            inp = self.prom()
            if inp == '':
                # Proceed with solution generating.
                # Find out solution generating mode (default, external, naive)
                # TODO: not all these options apply to every problem. Fix this
                self.isNewSol = True
                msg = "\nOk, default mode (initOpt) is '" + self.initOpt + "'."
                msg += "\nHit 'enter' to proceed with it, " + \
                       "or 'd' for 'default',\nor 'n' for 'naive'. " + \
                       "See '" + self.probName + ".py' for details. "
                self.log.printL(msg)
                inp = self.prom().lower()
                if inp=='d':
                    self.initOpt='default'
                    self.log.printL("\nProceeding with 'default' mode.\n")
                elif inp=='n':
                    self.initOpt='naive'
                    self.log.printL("\nProceeding with 'naive' mode.\n")
                else:
                    self.initOpt='extSol'
                    self.log.printL("\nProceeding with 'extSol' mode.\n" + \
                                    "External file for configurations: " + \
                                    self.confFile + "\n")

                return
            else:
                # execution only gets here if the default init is to generate
                # new init guess, but user wants to load solution

                self.isNewSol = False
                self.log.printL("\nGreat, let's load a solution then!")
                # This loop makes sure the user types something that looks like
                # a valid solution.

                # TODO: it should not be so hard to actually check the
                # existence of the file...
                keepAsk = True
                while keepAsk:
                    msg = "\nThe default path to loading " + \
                          "a solution (loadSolDir) is: " + self.loadSolDir + \
                          "\nHit 'enter' to load it, or type the " + \
                          "path to the alternative solution to be loaded."
                    self.log.printL(msg)

                    inp = self.prom()
                    if inp == '':
                        keepAsk = False
                    else:
                        # at least an extension check...
                        if inp.lower().endswith('.pkl'):
                            self.loadSolDir = inp
                            keepAsk = False
                        else:
                            self.log.printL('\nSorry, this is not a valid ' + \
                                            "solution path. Let's try again.")

                #
                self.log.printL("\nOk, proceeding with " + self.loadSolDir + \
                                "...")
                # Now, ask for a comparing solution.
                self.log.printL("\nOne more question: do you have an " + \
                                "alternative path to load \na solution " + \
                                "that serves as a comparing baseline?")
                keepAsk = True
                while keepAsk:
                    msg = "\nHit 'enter' to use the same path " + "(" + \
                          self.loadSolDir + "),\nor type the path to the" + \
                          " alternative solution to be loaded."
                    self.log.printL(msg)

                    inp = self.prom()
                    if inp == '':
                        keepAsk = False
                        self.loadAltSolDir = self.loadSolDir
                    else:
                        if inp.lower().endswith('.pkl'):
                            self.loadSolDir = inp
                            keepAsk = False
                        else:
                            self.log.printL('\nSorry, this is not a valid ' + \
                                            "solution path. Let's try again.")
                #
                self.log.printL("\nOk, proceeding with\n" +
                                self.loadAltSolDir + "\nas a comparing base!")
                #

        elif self.defOpt == 'loadSol':
            msg = "\nDefault starting option (defOpt) is to load solution." + \
                  "\nThe default path to do it (loadSolDir) is: " + \
                  self.loadSolDir + " .\nHit 'enter' to do it, hit " + \
                  "'I' to generate new initial guess,\n" + \
                  "or type the path to alternative solution to be loaded."
            self.log.printL(msg)

            inp = self.prom()
            if inp == '':
                self.isNewSol = False
            elif inp == 'i' or inp == 'I':
                self.isNewSol = True
                self.log.printL("\nOk, generating new initial guess...\n")
            else:
                self.isNewSol = False
                self.loadSolDir = inp
            return

    def checkPars(self,sol):
        keepLoop = True
        while keepLoop:
            self.prntDashStr()
            sol.printPars()
            sol.plotSol()
            self.prntDashStr()
            print("\a")
            msg = "\nAre these parameters OK?\n" + \
                  "Press 'enter' to continue, or update the configuration " + \
                  "file:\n    (" + self.confFile + ")\n" + \
                  "and then press any other key to reload the " + \
                  "parameters from that file.\n" + \
                  "Please notice that this only reloads parameters, not " + \
                  "necessarily\nregenerating the initial guess."
            self.log.printL(msg)

            inp = self.prom()
            if inp != '':
                self.log.printL("\nFine, reloading parameters...")
                sol.loadParsFromFile(file=self.confFile)
            else:
                self.log.printL("\nGreat! Moving on then...\n")
                keepLoop = False


    def loadSol(self,path=''):
        if path == '':
            path = self.loadSolDir

        self.log.printL("\nReading solution from '"+path+"'.")
        with open(path,'rb') as inpt:
            sol = dill.load(inpt)
        sol.log = self.log
        return sol

    def saveSol(self,sol,path=''):
        if path == '':
            path = self.probName + '_sol_' + getNowStr() + '.pkl'

        sol.log = None
        self.log.printL("\nWriting solution to '"+path+"'.")
        with open(path,'wb') as outp:
            dill.dump(sol,outp,-1)

        sol.log = self.log

    def setInitSol(self,sol):
        msg = "Setting initial solution.\n" + \
              "Please wait, you will be asked to confirm it later.\n\n"
        self.log.printL(msg)
        if self.isNewSol:
            # declare problem:
            solInit = sol.initGues({'initMode':self.initOpt,\
                                    'confFile':self.confFile})
            self.log.printL("Saving a copy of the configuration file " + \
                            "in this run's folder.")
            self.confFile = shutil.copy2(self.confFile, self.log.folderName +
                                         os.sep)
            self.saveSol(sol,self.log.folderName + os.sep + 'solInit.pkl')
        else:
            # load previously prepared solution
            self.log.printL('Loading "current" solution...')
            sol = self.loadSol()
            self.log.printL("Saving a copy of the default configuration " + \
                            "file in this run's folder.\n(Just for " + \
                            "alterring later, if necessary).")
            self.confFile = shutil.copy2(self.confFile, self.log.folderName + \
                                         os.sep)
            self.log.printL('Loading "initial" solution ' + \
                            '(for comparing purposes only)...')
            solInit = self.loadSol(path=self.loadAltSolDir)

        # Plot obtained solution, check parameters
        self.prntDashStr()
        self.log.printL("\nProposed initial guess:\n")
        sol.plotSol()
        self.checkPars(sol)

        # Calculate P values, store them
        P,Pint,Ppsi = sol.calcP()
        sol.histP[sol.NIterRest] = P
        sol.histPint[sol.NIterRest] = Pint
        sol.histPpsi[sol.NIterRest] = Ppsi

        # Calculate Q values, just for show. They don't even mean anything.
        Q,Qx,Qu,Qp,Qt = sol.calcQ()

        # Plot trajectory
        sol.plotTraj()

        # Setting debugging options (rest and grad)
        sol.dbugOptRest.setAll(opt={'pausRest':False,
                           'pausCalcP':False,
                           'plotP_int':False,
                           'plotP_intZoom':False,
                           'plotIntP_int':False,
                           'plotSolMaxP':False,
                           'plotRsidMaxP':False,
                           'plotErr':False,
                           'plotCorr':False,
                           'plotCorrFin':False})
        flag = False#True#
        sol.dbugOptGrad.setAll(opt={'pausGrad':flag,
                           'pausCalcQ':flag,
                           'prntCalcStepGrad':True,
                           'plotCalcStepGrad': flag,#True,
                           'pausCalcStepGrad':flag,#True,
                           'plotQx':flag,
                           'plotQu':flag,
                           'plotLam':flag,
                           'plotQxZoom':flag,
                           'plotQuZoom':flag,
                           'plotQuComp':flag,
                           'plotQuCompZoom':flag,
                           'plotSolQxMax':flag,
                           'plotSolQuMax':flag,
                           'plotCorr':flag,
                           'plotCorrFin':flag,
                           'plotF':flag,#True,
                           'plotFint':flag,
                           'plotI':flag})
#        sol.log = self.log
#        solInit.log = self.log

        return sol,solInit

    def showHistPCond(self,sol):
        if sol.NIterRest % self.RestHistShowRate == 0:
            return True
        else:
            return False

    def plotSolRestCond(self,sol):
        if sol.NIterRest % self.RestPlotSolRate == 0:
            return True
        else:
            return False

    def restRnds(self,sol):
        contRest = 0
        origDbugOptRest = sol.dbugOptRest.copy()

        while sol.P > sol.tol['P']:
            sol.rest(parallelOpt=self.parallelOpt)
            contRest += 1

        sol.showHistP()

        self.log.printL("\nEnd of restoration rounds (" + str(contRest) + \
                        "). Solution so far:")
        sol.plotSol()
        sol.dbugOptRest.setAll(opt=origDbugOptRest)
        return sol

    def frstRestRnds(self,sol):
        self.prntDashStr()
        self.log.printL("\nBeginning first restoration rounds...\n")
        sol.P,_,_ = sol.calcP()
        sol = self.restRnds(sol)

        self.saveSol(sol,self.log.folderName + os.sep + 'solInitRest.pkl')

        return sol

    def showHistQCond(self,sol):
        if sol.NIterGrad % self.GradHistShowRate == 0:
            return True
        else:
            return False

    def showHistICond(self,sol):
        if sol.NIterGrad % self.GradHistShowRate == 0:
            return True
        else:
            return False

    def showHistGradStepCond(self,sol):
        if sol.NIterGrad % self.GradHistShowRate == 0:
            return True
        else:
            return False

    def showHistGRrateCond(self,sol):
        return True

    def plotSolGradCond(self,sol):
        if sol.NIterGrad % self.GRplotSolRate == 0:
            return True
        else:
            return False

    def saveSolCond(self,sol):
        #return False

        if sol.NIterGrad % self.GRsaveSolRate==0:
            return True
        else:
            return False

    def gradRestPausCond(self,sol):
        if sol.NIterGrad % self.GRpausRate==0:
            return True
        else:
            return False

    def gradRestCycl(self,sol,altSol=None):

        self.prntDashStr()
        self.log.printL("\nBeginning gradient-restoration rounds...")

        do_GR_cycle = True
        last_grad = 0
        next_grad = 0
        while do_GR_cycle:
            #input("\nComeçando novo ciclo!")
            sol.P,_,_ = sol.calcP()
            sol = self.restRnds(sol)
            I, _, _ = sol.calcI()
            isParallel = self.parallelOpt.get('gradLMPBVP',False)
            A,B,C,lam,mu = sol.LMPBVP(rho=1.0,isParallel=isParallel)
            sol.Q,_,_,_,_ = sol.calcQ()

            if sol.Q <= sol.tol['Q']:
                self.log.printL("Terminate program. Solution is sol_r.")
                do_GR_cycle = False

            else:
                #input("\nVamos tentar dar um passo de grad pra frente!")
                next_grad += 1
                self.log.printL("\nNext grad counter = " + str(next_grad))
                self.log.printL("\nLast grad counter = " + str(last_grad))

                keep_walking_grad = True
                alfa_g_0 = 1.0

                while keep_walking_grad:
                    #input("\nProcurando passo a partir de "+str(alfa_g_0))
                    alfa_g_old,sol_new = sol.grad(alfa_g_0,A,B,C,lam,mu)
                    sol_new = self.restRnds(sol_new)
                    I_new, _, _ = sol_new.calcI()

                    if I_new < I:
                        I = I_new # PARA QUE SERVE ESTE COMANDO?
                        sol = sol_new
                        keep_walking_grad = False
                        next_grad += 1
                        sol.updtHistQ(alfa_g_old,mustPlotQs=True)
                        self.log.printL("\nNext grad counter = " + \
                                        str(next_grad))
                        self.log.printL("\nLast grad counter = " + \
                                        str(last_grad))
                        #input("\nDeu certo, passo dado!")
                    else:
                        last_grad += 1
                        self.log.printL("\nNext grad counter = " + \
                                        str(next_grad))
                        self.log.printL("\nLast grad counter = " + \
                                        str(last_grad))
                        alfa_g_0 = alfa_g_old
                        #input("\nNão deu certo... vamos tentar de novo!")
                    #
                #
            #

            if self.showHistQCond(sol):
                sol.showHistQ()

            if self.showHistICond(sol):
                sol.showHistI()


            if self.showHistGradStepCond(sol):
                sol.showHistGradStep()

            if self.showHistGRrateCond(sol):
                sol.showHistGRrate()

            if self.saveSolCond(sol):
                self.prntDashStr()
                self.saveSol(sol,self.log.folderName + os.sep + 'currSol.pkl')

            if self.plotSolGradCond(sol):
                self.prntDashStr()
                self.log.printL("\nSolution so far:")
                sol.plotSol()
                sol.plotF()
                sol.plotTraj()
                if altSol is not None:
                    sol.compWith(altSol,'Initial guess')

            if self.gradRestPausCond(sol):
                print("\a")
                self.prntDashStr()
                self.log.printL(datetime.datetime.now())
                msg = "\nAfter " + str(sol.NIterGrad) + \
                      " gradient iterations,\n" + \
                      "Grad-Rest cycle pause condition has been reached.\n" + \
                      "Press any key to continue, or ctrl+C to stop.\n" + \
                      "Load last saved solution to go back to GR cycle."
                self.log.printL(msg)
                self.prom()
            #
        #

        return sol
