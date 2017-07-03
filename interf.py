#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:35:29 2017

@author: levi
"""

import dill,datetime

class ITman():
    # iterations manager
    bscImpStr = "\n >> "
    
    def __init__(self):
        self.defOpt = 'loadSol'#'newSol'
        self.initOpt = 'extSol'
        self.isNewSol = False
        self.loadSolDir = "solInitRestored.pkl"
        self.mustPlotGrad = True
        self.mustPlotRest = False
        self.mustPlotSol = True
        self.NSilent = 0
        self.mustAsk = True
        
    def greet(self):
        print("\n------------------------------------------------------------")
        print("\nWelcome to SGRA!\n")

       
        if self.defOpt == 'newSol':
            print("Default starting option is to generate new initial guess.")
            print("Hit 'enter' to do it, or any other key to load a previous solution.")
            inp = input(self.bscImpStr)
            if inp == '':
                self.isNewSol = True
                print("\nOk, default mode is '"+self.initOpt+"'.")
                print("Hit 'enter' to proceed with it, or 'd' for 'default',")
                print("or 'n' for 'naive'. See 'rockSol.py' for details. ")
                inp = input(self.bscImpStr).lower()
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
                print("\nOk, default path to loading solution is: '"+\
                      self.loadSolDir+"'.")
                print("Hit 'enter' to load it, or type the path to alternative solution to be loaded.")
                inp = input(self.bscImpStr)
                if inp != '':
                    self.loadSolDir = inp
                
        elif self.defOpt == 'loadSol':
            print("Default starting option is to load solution.")
            print("The default path to loading solution is: "+self.loadSolDir)
            print("Hit 'enter' to do it, 'I' to generate new initial guess,")
            print("or type the path to alternative solution to be loaded.")
            inp = input(self.bscImpStr)
            if inp == '':
                self.isNewSol = False
            elif inp == 'i' or inp == 'I':
                self.isNewSol = True
            else:
                self.isNewSol = False
                self.loadSolDir = inp  
            return
        
    def checkPars(self,sol):
        
        print("--------------------------------------------------------------")
        sol.printPars()
        print("--------------------------------------------------------------")
        print("\nAre these parameters OK?")
        print("Press any key to continue, or ctrl+C to stop.")
        input(self.bscImpStr)
        
    def loadSol(self):
        path = self.loadSolDir# 'sol_init.pkl'#'sol_'+str(datetime.datetime.now())+'.pkl'

        print("\nLoading solution from '"+path+"'.")        
        with open(path,'rb') as inpt:
            sol = dill.load(inpt)
        return sol

    def saveSol(self,sol,path=None):
        if path is None:
           path = 'sol_'+str(datetime.datetime.now())+'.pkl'

        print("\nWriting solution to '"+path+"'.")        
        with open(path,'wb') as outp:
            dill.dump(sol,outp,-1)
            
class GradStat:
    def __init__(self):
        self.mustPlotGrad = True
        self.mustPlotRest = False
        self.mustPlotSol = True
        self.NSilent = 0
        self.mustAsk = True
    
    def printMsg(self):
        self.printStatus()
        print("\nEnter commands for changing debug mode, or just hit 'enter'.")

    def parseStr(self,thisStr):
        print("input =",thisStr)
        if thisStr[0]=='g':
            if thisStr[1:].isnumeric():
                n = int(thisStr[1:])
                if n>0:
                    self.NSilent=n
                    print("Waiting",n,"runs before new prompt.")
            elif thisStr[1:]=='p':
                self.mustPlotGrad = True
            elif thisStr[1:]=='n':
                self.mustPlotGrad = False
        elif thisStr[0]=='r':
            if thisStr[1:]=='p':
                self.mustPlotRest = True
            elif thisStr[1:]=='n':
                self.mustPlotRest = False
        elif thisStr[0]=='s':
            if thisStr[1:]=='p':
                self.mustPlotSol = True
            elif thisStr[1:]=='n':
                self.mustPlotSol = False
        else:
            print("Ignoring unrecognized command '"+thisStr+"'...")
        
    def endOfLoop(self):
        print("\a")
        if self.NSilent>0:
            print("\n",self.NSilent,"more runs remaining before new prompt.")
            self.NSilent -= 1
        else:
            if self.mustAsk:
                self.printMsg()
                inp = input(">> ")
                inp.lower()
                if not(inp=='\n'):
                    inpList = inp.split()    
                    for k in range(len(inpList)):
                        self.parseStr(inpList[k])
                    self.printStatus()

                print("\nOk. Back to main loop.")
    def printStatus(self):
        print("\nStatus:")
        print("mustPlotGrad:",self.mustPlotGrad)
        print("mustPlotRest:",self.mustPlotRest)
        print("mustPlotSol:",self.mustPlotSol)
        print("NSilent:",self.NSilent)
        print("mustAsk:",self.mustAsk)
        
    def test(self):
        while True:
            print("In test mode. This will run forever.\n")
            self.endOfLoop()