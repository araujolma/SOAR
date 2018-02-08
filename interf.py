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
#        print("\nHi, I'm a new logger object!")
        self.mode = mode
        self.folderName = probName + '_' + getNowStr()
#        print("Creating a new folder:",self.folderName)
        os.makedirs(self.folderName)

        try:
            self.fhand = open(self.folderName + '/log.txt','w+')
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
        self.loadSolDir = probName+'_solInitRest.pkl'
        #'solInitRest.pkl'#'solInit.pkl'#'currSol.pkl'
        self.GRplotSolRate = 1
        self.GRsaveSolRate = 5
        self.GRpausRate = 1000#1000#10
        self.GradHistShowRate = 5
        self.RestPlotSolRate = 5
        self.RestHistShowRate = 5
        self.parallelOpt = {'gradLMPBVP':True,
                         'restLMPBVP':True}

        self.log = logger(probName)
        # Create directory for logs and stuff


    def prntDashStr(self):
        self.log.printL(self.dashStr)

    def prom(self):
        inp = input(self.bscImpStr)
        self.log.printL(self.bscImpStr+inp,mode='file')
        return inp

    def printPars(self):
        self.log.printL("\nThese are the attributes for the Iterations manager:\n")
        self.log.pprint(self.__dict__)

    def greet(self):
        """This is the first command to be run at the beggining of the
        MSGRA.

        The idea is to let the user choose whether he/she wants to load a
        previously prepared solution, or generate a new one."""


        self.prntDashStr()
        self.log.printL("\nWelcome to SGRA!\n")
        self.log.printL("Loading settings for problem: "+self.probName)
        self.log.printL('Saving results and log in '+self.log.folderName+'.')
        self.prntDashStr()
        self.printPars()
        self.log.printL("\n(You can always change these here in '"+__name__+".py').")

        if self.defOpt == 'newSol':
            self.log.printL("\nDefault starting option (defOpt) is to generate new initial guess.")
            self.log.printL("Hit 'enter' to do it, or any other key to load a "+\
                  "previous solution.")
            inp = self.prom()
            if inp == '':
                self.isNewSol = True
                self.log.printL("\nOk, default mode (initOpt) is '"+self.initOpt+"'.")
                self.log.printL("Hit 'enter' to proceed with it, or 'd' for 'default',")
                self.log.printL("or 'n' for 'naive'. See '" + self.probName + \
                      ".py' for details. ")
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
                self.isNewSol = False
                # execution only gets here if the default init is to generate
                # new init guess, but user wants to load solution
                self.log.printL("\nOk, default path to loading solution (loadSolDir) is: '"+\
                      self.loadSolDir+"'.")
                self.log.printL("Hit 'enter' to load it, or type the path to "+\
                      "alternative solution to be loaded.")
                inp = self.prom()
                if inp != '':
                    self.loadSolDir = inp

        elif self.defOpt == 'loadSol':
            self.log.printL("\nDefault starting option (defOpt) is to load solution.")
            self.log.printL("The default path to loading solution (loadSolDir) is: "+self.loadSolDir)
            self.log.printL("Hit 'enter' to do it, 'I' to generate new initial guess,")
            self.log.printL("or type the path to alternative solution to be loaded.")

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
        self.log.printL(self.dashStr)
        sol.printPars()
        sol.plotSol()
        self.prntDashStr()
        print("\a")
        self.log.printL("\nAre these parameters OK?")
        self.log.printL("Press any key to continue, or ctrl+C to stop.")
        self.prom()

    def loadSol(self,path=''):
        if path == '':
            path = self.loadSolDir

        self.log.printL("\nLoading solution from '"+path+"'.")
        with open(path,'rb') as inpt:
            sol = dill.load(inpt)
        return sol

    def saveSol(self,sol,path=''):
        if path == '':
           path = self.probName + '_sol_' + getNowStr() + '.pkl'

        self.log.printL("\nWriting solution to '"+path+"'.")
        with open(path,'wb') as outp:
            dill.dump(sol,outp,-1)

    def setInitSol(self,sol):
        self.log.printL("Setting initial solution.")
        self.log.printL("Please wait, you will be asked to confirm it later.\n\n")
        if self.isNewSol:
            # declare problem:
            solInit = sol.initGues({'initMode':self.initOpt,\
                                    'confFile':self.confFile})
            self.log.printL("Saving a copy of the configuration file in this run's folder.")
            shutil.copy2(self.confFile,self.log.folderName+"/")
            self.saveSol(sol,self.log.folderName+'/solInit.pkl')
        else:
            # load previously prepared solution
            self.log.printL('Loading "current" solution...')
            sol = self.loadSol()
            self.log.printL('Loading "initial" solution (for comparing purposes only)...')
            solInit = self.loadSol(self.probName+'_solInit.pkl')#sol.copy()

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
        self.log.printL("End of restoration rounds. Solution so far:")
        sol.plotSol()
        sol.dbugOptRest.setAll(opt=origDbugOptRest)
        return sol

    def frstRestRnds(self,sol):
        self.prntDashStr()
        self.log.printL("\nBeginning first restoration rounds...\n")
        sol.P,_,_ = sol.calcP()
        sol = self.restRnds(sol)

        self.saveSol(sol,self.log.folderName + '/solInitRest.pkl')

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
        self.log.printL("\nBeginning gradient rounds...")
        sol.Q,_,_,_,_ = sol.calcQ()
        while sol.Q > sol.tol['Q']:
            sol = self.restRnds(sol)

            sol.grad(parallelOpt=self.parallelOpt)

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
                self.saveSol(sol,self.log.folderName + '/currSol.pkl')

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
                self.log.printL("\nAfter "+str(sol.NIterGrad)+" gradient iterations,")
                self.log.printL("Grad-Rest cycle pause condition has been reached.")
                self.log.printL("Press any key to continue, or ctrl+C to stop.")
                self.log.printL("Load last saved solution to go back to GR cycle.")
                self.prom()
        #

        return sol
