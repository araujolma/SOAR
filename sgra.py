#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:07:35 2017

@author: levi
"""

import rest_sgra, grad_sgra, numpy, copy, pprint
import matplotlib.pyplot as plt

class sgra():
    def __init__(self):
        # these numbers should not make any sense; 
        # they should change with the problem
        N,n,m,p,q,s = 50000,4,2,1,3,2

        self.N = N
        self.n = n
        self.m = m
        self.p = p
        self.q = q
        self.s = s

        self.x = numpy.zeros((N,n))
        self.u = numpy.zeros((N,m))
        self.pi = numpy.zeros(p)
        self.lam = numpy.zeros((N,n))
        self.mu = numpy.zeros(q)
        
        self.boundary = {}
        self.constants = {}
        self.restrictions = {}
        
        self.P = 1.0
        self.Q = 1.0
        self.I = 1.0
        
        # Basic maximum number of iterations for grad/rest. 
        # May be overriden in the problem definition
        MaxIterRest = 10000
        self.MaxIterRest = MaxIterRest
        self.NIterRest = 0
        self.histStepRest = numpy.zeros(MaxIterRest)
        self.histP = numpy.zeros(MaxIterRest)
        self.histPint = numpy.zeros(MaxIterRest)
        self.histPpsi = numpy.zeros(MaxIterRest)
        
        MaxIterGrad = 1000
        self.MaxIterGrad = MaxIterGrad
        self.NIterGrad = 0
        self.histStepGrad = numpy.zeros(MaxIterGrad)
        self.histQ = numpy.zeros(MaxIterGrad)
        self.histQx = numpy.zeros(MaxIterGrad)
        self.histQu = numpy.zeros(MaxIterGrad)
        self.histQp = numpy.zeros(MaxIterGrad)
        self.histQt = numpy.zeros(MaxIterGrad)

        self.histI = numpy.zeros(MaxIterGrad)
        
        # Debugging options
        tf = False
        self.dbugOptRest = {'pausRest':tf,
                            'pausCalcP':tf,
                            'plotP_int':tf,
                            'plotP_intZoom':tf,
                            'plotIntP_int':tf,
                            'plotSolMaxP':tf,
                            'plotRsidMaxP':tf,
                            'plotCorr':tf}
        tf = False
        self.dbugOptGrad = {'pausGrad':tf,
                            'pausCalcQ':tf,
                            'prntCalcStepGrad': tf,
                            'plotCalcStepGrad': tf,
                            'pausCalcStepGrad':tf,
                            'plotQx':tf,
                            'plotQu':tf,
                            'plotQxZoom':tf,
                            'plotQuZoom':tf,
                            'plotSolQxMax':tf,
                            'plotSolQuMax':tf,
                            'plotCorr':tf}
   
    def setAllDbugOptRest(self,tf):
        for key in self.dbugOptRest.keys():
            self.dbugOptRest[key] = tf
    
    def setAllDbugOptGrad(self,tf):
        for key in self.dbugOptGrad.keys():
            self.dbugOptRest[key] = tf
            
    def copy(self):
        return copy.deepcopy(self)
    
    def aplyCorr(self,alfa,corr):
        
        self.x  += alfa * corr['x']
        self.u  += alfa * corr['u']
        self.pi += alfa * corr['pi']
        
    def initGues(self):
        # implemented by child classes
        pass

    def printPars(self):
        dPars = self.__dict__
#        keyList = dPars.keys()
        print("These are the attributes for the current solution:\n")
        pprint.pprint(dPars)
#%% Just for avoiding compatibilization issues with other problems
    
    def plotTraj(self):
        print("plotTraj: unimplemented method.")
        pass
    
    def compWith(self,*args,**kwargs):
        print("compWith: unimplemented method.")
        pass
    
    def plotSol(self,*args,**kwargs):
        print("plotSol: unimplemented method.")
        pass
#%% RESTORATION-WISE METHODS
    
    def rest(self,*args,**kwargs):
        rest_sgra.rest(self,*args,**kwargs)

    def calcStepRest(self,*args,**kwargs):
        return rest_sgra.calcStepRest(self,*args,**kwargs)        
        
    def calcP(self,*args,**kwargs):
        return rest_sgra.calcP(self,*args,**kwargs)   
    
    def updtHistP(self,alfa):
        
        NIterRest = self.NIterRest+1

        P,Pint,Ppsi = self.calcP()
        self.P = P
        self.histP[NIterRest] = P
        self.histPint[NIterRest] = Pint
        self.histPpsi[NIterRest] = Ppsi
        self.histStepRest[NIterRest] = alfa
        self.NIterRest = NIterRest
        
    def showHistP(self):
        IterRest = numpy.arange(0,self.NIterRest+1,1)

        if self.histP[IterRest].any() > 0:
            plt.semilogy(IterRest,self.histP[IterRest],'b',label='P')

        if self.histPint[IterRest].any() > 0:
            plt.semilogy(IterRest,self.histPint[IterRest],'k',label='P_int')

        if self.histPpsi[IterRest].any() > 0:
            plt.semilogy(IterRest,self.histPpsi[IterRest],'r',label='P_psi')
        
        plt.plot(IterRest,self.tol['P']+0.0*IterRest,'-.b',label='tolP')
        print("\nConvergence report on P:")
        plt.grid(True)
        plt.xlabel("Iterations")
        plt.ylabel("P values")
        plt.legend()
        plt.show()

#%% GRADIENT-WISE METHODS

    def grad(self,*args,**kwargs):
        grad_sgra.grad(self,*args,**kwargs)
        
    def calcStepGrad(self,*args,**kwargs):
        return grad_sgra.calcStepGrad(self,*args,**kwargs)

    def calcQ(self,*args,**kwargs):
        return grad_sgra.calcQ(self,*args,**kwargs)
    
    def updtHistQ(self,alfa):
    
        
        NIterGrad = self.NIterGrad+1
        
        Q,Qx,Qu,Qp,Qt = self.calcQ()
        self.Q = Q
        self.histQ[NIterGrad] = Q
        self.histQx[NIterGrad] = Qx
        self.histQu[NIterGrad] = Qu
        self.histQp[NIterGrad] = Qp
        self.histQt[NIterGrad] = Qt
        self.histStepGrad[NIterGrad] = alfa
        
        I = self.calcI()
        self.histI[NIterGrad] = I        
        self.I = I
        
        self.NIterGrad = NIterGrad
        
    def showHistQ(self):
        IterGrad = numpy.arange(1,self.NIterGrad+1,1)

        if self.histQ[IterGrad].any() > 0:
            plt.semilogy(IterGrad,self.histQ[IterGrad],'b',label='Q')

        if self.histQx[IterGrad].any() > 0:
            plt.semilogy(IterGrad,self.histQx[IterGrad],'k',label='Qx')

        if self.histQu[IterGrad].any() > 0:
            plt.semilogy(IterGrad,self.histQu[IterGrad],'r',label='Qu')

        if self.histQp[IterGrad].any() > 0:
            plt.semilogy(IterGrad,self.histQp[IterGrad],'g',label='Qp')

        if self.histQt[IterGrad].any() > 0:
            plt.semilogy(IterGrad,self.histQt[IterGrad],'y',label='Qt')

        plt.title("Convergence report on Q")
        plt.grid(True)
        plt.xlabel("Iterations")
        plt.ylabel("Q values")
        plt.legend()
        plt.show()
        
    def showHistI(self):
        IterGrad = numpy.arange(1,self.NIterGrad+1,1)
        
        plt.title("Convergence report on I")
        plt.plot(IterGrad,self.histI[IterGrad])
        plt.grid(True)
        plt.xlabel("Iterations")
        plt.ylabel("I values")
        plt.show()