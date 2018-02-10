#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:35:29 2017

@author: levi
"""

import dill, datetime, pprint

class ITman():
    """Class for the ITerations MANager.

    Only one object from this class is intended to be active during a program
    run. It manages the iterations in the solution, commands solution saving, 
    plotting, debug modes, etc.
    
    """

    bscImpStr = "\n >> "
    dashStr = '\n-------------------------------------------------------------'
    
    def __init__(self,probName='currSol'):
        self.probName = probName
        self.defOpt = 'newSol'#'loadSol'#
        self.initOpt = 'extSol'
        self.caseName = 'default2st'
#        self.isNewSol = False
        self.loadSolDir = probName+'_solInitRest.pkl'
        #'solInitRest.pkl'#'solInit.pkl'#'currSol.pkl'
        self.mustPlotGrad = True
        self.mustPlotRest = False
        self.mustPlotSol = True
        self.GRplotSolRate = 1
        self.GRsaveSolRate = 5
        self.GRpausRate = 1000#1000#10
        self.GradHistShowRate = 5
        self.RestPlotSolRate = 5
        self.RestHistShowRate = 5
        self.parallelOpt = {'gradLMPBVP':True,
                         'restLMPBVP':True}
    
    def prntDashStr(self):
        print(self.dashStr)
        
    def prom(self):
        inp = input(self.bscImpStr)
        return inp
    
    def printPars(self):
        print("\nThese are the attributes for the Iterations manager:\n")
        pprint.pprint(self.__dict__)
    
    def greet(self):
        """This is the first command to be run at the beggining of the 
        MSGRA. 
        
        The idea is to let the user choose whether he/she wants to load a 
        previously prepared solution, or generate a new one."""
        
        self.prntDashStr()      
        print("\nWelcome to SGRA!")
        print("Loading settings for problem: "+self.probName)
        self.prntDashStr()      
        self.printPars()     
        print("(You can always change these here in '"+__name__+".py').")
        
        if self.defOpt == 'newSol':
            print("\nDefault starting option (defOpt) is to generate new initial guess.")
            print("Hit 'enter' to do it, or any other key to load a "+\
                  "previous solution.")
            inp = self.prom()
            if inp == '':
                self.isNewSol = True
                print("\nOk, default mode (initOpt) is '"+self.initOpt+"'.")
                print("Hit 'enter' to proceed with it, or 'd' for 'default',")
                print("or 'n' for 'naive'. See '" + self.probName + \
                      ".py' for details. ")
                inp = self.prom().lower()
                if inp=='d':
                    self.initOpt='default'
                    print("\nProceeding with 'default' mode.\n")
                elif inp=='n':
                    self.initOpt='naive'
                    print("\nProceeding with 'naive' mode.\n")                    
                else:
                    self.initOpt='extSol'
                    print("\nProceeding with 'extSol' mode.\n")
                    
                return
            else:
                self.isNewSol = False
                # execution only gets here if the default init is to generate 
                # new init guess, but user wants to load solution
                print("\nOk, default path to loading solution (loadSolDir) is: '"+\
                      self.loadSolDir+"'.")
                print("Hit 'enter' to load it, or type the path to "+\
                      "alternative solution to be loaded.")
                inp = input(self.bscImpStr)
                if inp != '':
                    self.loadSolDir = inp
                
        elif self.defOpt == 'loadSol':
            print("\nDefault starting option (defOpt) is to load solution.")
            print("The default path to loading solution (loadSolDir) is: "+self.loadSolDir)
            print("Hit 'enter' to do it, 'I' to generate new initial guess,")
            print("or type the path to alternative solution to be loaded.")
            inp = input(self.bscImpStr)
            if inp == '':
                self.isNewSol = False
            elif inp == 'i' or inp == 'I':
                self.isNewSol = True
                print("\nOk, generating new initial guess...\n")
            else:
                self.isNewSol = False
                self.loadSolDir = inp  
            return
        
    def checkPars(self,sol):
        print(self.dashStr)
        sol.printPars()
        sol.plotSol()
        self.prntDashStr()
        print("\a")
        print("\nAre these parameters OK?")
        print("Press any key to continue, or ctrl+C to stop.")
        input(self.bscImpStr)
        
    def loadSol(self,path=''):
        if path == '':
            path = self.loadSolDir

        print("\nLoading solution from '"+path+"'.")        
        with open(path,'rb') as inpt:
            sol = dill.load(inpt)
        return sol

    def saveSol(self,sol,path=''):
        if path == '':
           path = self.probName + '_sol_' + \
           str(datetime.datetime.now()).replace(' ','_') + '.pkl'

        print("\nWriting solution to '"+path+"'.")        
        with open(path,'wb') as outp:
            dill.dump(sol,outp,-1)
   
    def setInitSol(self,sol):
        print("Setting initial solution.")
        print("Please wait, you will be asked to confirm it later.\n\n")
        if self.isNewSol:
            # declare problem:
            solInit = sol.initGues({'initMode':self.initOpt})
            self.saveSol(sol,self.probName+'_solInit.pkl')
        else:
            # load previously prepared solution
            print('Loading "current" solution...')
            sol = self.loadSol()
            print('Loading "initial" solution (for comparing purposes only)...')
            solInit = self.loadSol(self.probName+'_solInit.pkl')#sol.copy()
        
        ####
        #sol.plotSol(mustSaveFig=False,intv=[100.0,300.0])
        #sol.calcP(mustPlotPint=True)
        #input("In interf.py: Checa lá o ..._Pint.pdf!")
        ####
        
        # Plot obtained solution, check parameters
        self.prntDashStr()
        print("\nProposed initial guess:\n")
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
        print("End of restoration rounds. Solution so far:")
        sol.plotSol()
        sol.dbugOptRest.setAll(opt=origDbugOptRest)
        return sol
    
    def frstRestRnds(self,sol):
        self.prntDashStr()
        print("\nBeginning first restoration rounds...\n")
        sol.P,_,_ = sol.calcP()
        sol = self.restRnds(sol)
        
        self.saveSol(sol,self.probName+'_solInitRest.pkl')
    
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
        print("\nBeginning gradient rounds...")
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
                self.saveSol(sol,self.probName+'_currSol.pkl')
            
            if self.plotSolGradCond(sol):
                self.prntDashStr()
                print("\nSolution so far:")
                sol.plotSol()
                sol.plotF()
                sol.plotTraj()
                if altSol is not None:
                    sol.compWith(altSol,'Initial guess')
            
            if self.gradRestPausCond(sol):
                print("\a")
                self.prntDashStr()
                print(datetime.datetime.now())
                print("\nAfter "+str(sol.NIterGrad)+" gradient iterations,")
                print("Grad-Rest cycle pause condition has been reached.")
                print("Press any key to continue, or ctrl+C to stop.")                
                print("Load last saved solution to go back to GR cycle.")
                self.prom()
        #
        
        return sol   
