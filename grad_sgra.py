#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:39:08 2017

@author: levi
"""

import numpy
from utils import ddt, testAlgn
import matplotlib.pyplot as plt

class stepMngr():
    """ Class for the step manager """
    def __init__(self,k=1e3):
        self.cont = -1
        self.histStep = list()
        self.histI = list()
        self.histP = list()
        self.histQ = list()
        self.histObj = list()
        self.k = k

    def calcObj(self,P,Q,I):
        return (I + (self.k)*P)
    
    def getLast(self):
        P = self.histP[self.cont]
        Q = self.histQ[self.cont]
        I = self.histI[self.cont]
        Obj = self.histObj[self.cont]

        return P,Q,I,Obj
    
    def tryStep(self,sol,corr,alfa,mustPrnt=False):
        self.cont += 1
        
        if mustPrnt:
            print("\nTrying alfa = {:.4E}".format(alfa))
            
        newSol = sol.copy()
        newSol.aplyCorr(alfa,corr)
        
        P,_,_ = newSol.calcP()
        Q,_,_,_,_ = newSol.calcQ()
        I = newSol.calcI()
        
        Obj = self.calcObj(P,Q,I)

        if mustPrnt:
            print("\nResults:\nalfa = {:.4E}".format(alfa)+\
                  " P = {:.4E}".format(P)+" Q = {:.4E}".format(Q)+\
                  " I = {:.4E}".format(I)+" Obj = {:.7E}".format(Obj))

        self.histStep.append(alfa)
        self.histP.append(P)
        self.histQ.append(Q)
        self.histI.append(I)
        self.histObj.append(Obj)
        
        return self.getLast()

def calcQ(self):
    # Q expression from (15)
    #print("\nIn calcQ.\n")
    N,n,m,p,s = self.N,self.n,self.m,self.p,self.s
    dt = 1.0/(N-1)

#    x = self.x
#    u = self.u
    lam = self.lam
    mu = self.mu
    
    
    # get gradients
    Grads = self.calcGrads()
    phix = Grads['phix']
    phiu = Grads['phiu']
    phip = Grads['phip']
    fx = Grads['fx']
    fu = Grads['fu']
    fp = Grads['fp']
    psiy = Grads['psiy']
    psip = Grads['psip']
    dlam = ddt(lam,N)

    Qx = 0.0
    Qu = 0.0
    Qp = 0.0
    Qt = 0.0
    Q = 0.0
    auxVecIntQp = numpy.zeros((p,s))
    
    errQx = numpy.empty((N,n,s)); normErrQx = numpy.empty((N,s))
    errQu = numpy.empty((N,m,s)); normErrQu = numpy.empty((N,s))
    errQp = numpy.empty((N,p,s)); #normErrQp = numpy.empty(N)
    
    z = numpy.empty(2*n*s)
    for arc in range(s):
        z[2*arc*n : (2*arc+1)*n] = -lam[0,:,arc]
        z[(2*arc+1)*n : (2*arc+2)*n] = lam[N-1,:,arc]
        
        for k in range(N):
            errQx[k,:,arc] = dlam[k,:,arc] - fx[k,:,arc] + \
                             phix[k,:,:,arc].transpose().dot(lam[k,:,arc])
            errQu[k,:,arc] = fu[k,:,arc] +  \
                            - phiu[k,:,:,arc].transpose().dot(lam[k,:,arc])
            errQp[k,:,arc] = fp[k,:,arc] + \
                            - phip[k,:,:,arc].transpose().dot(lam[k,:,arc])
        
            normErrQx[k,arc] = errQx[k,:,arc].transpose().dot(errQx[k,:,arc])
            normErrQu[k,arc] = errQu[k,:,arc].transpose().dot(errQu[k,:,arc])
        
            Qx += normErrQx[k,arc]
            Qu += normErrQu[k,arc]
            auxVecIntQp[:,arc] += errQp[k,:,arc]
        #
        Qx -= .5*(normErrQx[0,arc]+normErrQx[N-1,arc])
        Qu -= .5*(normErrQu[0,arc]+normErrQu[N-1,arc])
        Qx *= dt
        Qu *= dt

        auxVecIntQp[:,arc] -= .5*(errQp[0,:,arc]+errQp[N-1,:,arc])
    
    auxVecIntQp *= dt
    
    resVecIntQp = numpy.zeros(p)
    for arc in range(s):
        resVecIntQp += auxVecIntQp[:,arc]
    Qp = resVecIntQp.transpose().dot(resVecIntQp)
    
    resVecQp = psip.transpose().dot(mu)
    Qp += resVecQp.transpose().dot(resVecQp)
    
    errQt = z + psiy.transpose().dot(mu)
    Qt = errQt.transpose().dot(errQt)

    Q = Qx + Qu + Qp + Qt
    print("Q = {:.4E}".format(Q)+": Qx = {:.4E}".format(Qx)+\
          ", Qu = {:.4E}".format(Qu)+", Qp = {:.7E}".format(Qp)+\
          ", Qt = {:.4E}".format(Qt))

    self.Q = Q
    somePlot = False
    for key in self.dbugOptGrad.keys():
        if ('plotQ' in key) or ('PlotQ' in key):
            if self.dbugOptGrad[key]:
                somePlot = True
                break
    if somePlot:
        print("\nDebug plots for this calcQ run:\n")
        self.plotSol()

        indMaxQu = numpy.argmax(normErrQu, axis=0)

        for arc in range(s):
            print("\nArc =",arc,"\n")
            ind1 = numpy.array([indMaxQu[arc]-20,0]).max()
            ind2 = numpy.array([indMaxQu[arc]+20,N]).min()
            
            if self.dbugOptGrad['plotQu']:
                plt.plot(self.t,normErrQu[:,arc])
                plt.grid(True)
                plt.title("Integrand of Qu")
                plt.show()                    
                                
            #for zoomed version:
            if self.dbugOptGrad['plotQuZoom']:
                plt.plot(self.t[ind1:ind2],normErrQu[ind1:ind2,arc],'o')
                plt.grid(True)
                plt.title("Integrand of Qu (zoom)")
                plt.show()
                
#            if self.dbugOptGrad['plotCtrl']:
#                if self.m==2:
#                    alfa,beta = self.calcDimCtrl()
#                    plt.plot(self.t,alfa[:,arc]*180.0/numpy.pi)
#                    plt.title("Ang. of attack")
#                    plt.show()
#                    
#                    plt.plot(self.t,beta[:,arc]*180.0/numpy.pi)
#                    plt.title("Thrust profile")
#                    plt.show()
            if self.dbugOptGrad['plotQuComp']:
                plt.plot(self.t,errQu[:,0,arc])
                plt.grid(True)
                plt.title("Qu: component 1")
                plt.show()
                
                if m>1:
                    plt.plot(self.t,errQu[:,1,arc])
                    plt.grid(True)
                    plt.title("Qu: component 2")
                    plt.show()

            if self.dbugOptGrad['plotQuCompZoom']:
                plt.plot(self.t[ind1:ind2],errQu[ind1:ind2,0,arc])
                plt.grid(True)
                plt.title("Qu: component 1 (zoom)")
                plt.show()
                
                if m>1:
                    plt.plot(self.t[ind1:ind2],errQu[ind1:ind2,1,arc])
                    plt.grid(True)
                    plt.title("Qu: component 2 (zoom)")
                    plt.show()

            if self.dbugOptGrad['plotLam']:
                plt.plot(self.t,lam[:,0,arc])
                plt.grid(True)
                plt.title("Lambda_h")
                plt.show()
                
                if n>1:
                    plt.plot(self.t,lam[:,1,arc])
                    plt.grid(True)
                    plt.title("Lambda_v")
                    plt.show()

                if n>2:
                    plt.plot(self.t,lam[:,2,arc])
                    plt.grid(True)
                    plt.title("Lambda_gama")
                    plt.show()

                if n>3:
                    plt.plot(self.t,lam[:,3,arc])
                    plt.grid(True)
                    plt.title("Lambda_m")
                    plt.show()

    # TODO: break these plots into more conditions

#    if numpy.array(self.dbugOptGrad.values()).any:
#        print("\nDebug plots for this calcQ run:")
#        
#        if self.dbugOptGrad['plotQx']:
#            plt.plot(self.t,normErrQx)
#            plt.grid(True)
#            plt.title("Integrand of Qx")
#            plt.show()
#        
#        if self.dbugOptGrad['plotQu']:                
#            plt.plot(self.t,normErrQu)
#            plt.grid(True)
#            plt.title("Integrand of Qu")
#            plt.show()
#
#        # for zoomed version:
#        indMaxQx = normErrQx.argmax()
#        ind1 = numpy.array([indMaxQx-20,0]).max()
#        ind2 = numpy.array([indMaxQx+20,N-1]).min()
#        
#        if self.dbugOptGrad['plotQxZoom']:
#            plt.plot(self.t[ind1:ind2],normErrQx[ind1:ind2],'o')
#            plt.grid(True)
#            plt.title("Integrand of Qx (zoom)")
#            plt.show()
#        
#        if self.dbugOptGrad['plotSolQxMax']:
#            print("\nSolution on the region of MaxQx:")
#            self.plotSol(intv=numpy.arange(ind1,ind2,1,dtype='int'))
#            
#        # for zoomed version:
#        indMaxQu = normErrQu.argmax()
#        ind1 = numpy.array([indMaxQu-20,0]).max()
#        ind2 = numpy.array([indMaxQu+20,N-1]).min()
#        
#        if self.dbugOptGrad['plotQuZoom']:
#            plt.plot(self.t[ind1:ind2],normErrQu[ind1:ind2],'o')
#            plt.grid(True)
#            plt.title("Integrand of Qu (zoom)")
#            plt.show()
#        
#        if self.dbugOptGrad['plotSolQuMax']:
#            print("\nSolution on the region of MaxQu:")
#            self.plotSol(intv=numpy.arange(ind1,ind2,1,dtype='int'))
#        
#        
##        if n==4 and m==2:
##            
##            plt.plot(tPlot[ind1:ind2],errQx[ind1:ind2,0])
##            plt.grid(True)
##            plt.ylabel("Qx_h")
##            plt.show()        
##    
##            plt.plot(tPlot[ind1:ind2],errQx[ind1:ind2,1],'g')
##            plt.grid(True)
##            plt.ylabel("Qx_V")
##            plt.show()
##    
##            plt.plot(tPlot[ind1:ind2],errQx[ind1:ind2,2],'r')
##            plt.grid(True)
##            plt.ylabel("Qx_gamma")
##            plt.show()
##    
##            plt.plot(tPlot[ind1:ind2],errQx[ind1:ind2,3],'m')
##            plt.grid(True)
##            plt.ylabel("Qx_m")
##            plt.show()
##            
##            print("\nStates, controls, lambda on the region of maxQx:")
##
##            plt.plot(tPlot[ind1:ind2],x[ind1:ind2,0])
##            plt.grid(True)
##            plt.ylabel("h [km]")
##            plt.show()        
##    
##            plt.plot(tPlot[ind1:ind2],x[ind1:ind2,1],'g')
##            plt.grid(True)
##            plt.ylabel("V [km/s]")
##            plt.show()
##    
##            plt.plot(tPlot[ind1:ind2],x[ind1:ind2,2]*180/numpy.pi,'r')
##            plt.grid(True)
##            plt.ylabel("gamma [deg]")
##            plt.show()
##    
##            plt.plot(tPlot[ind1:ind2],x[ind1:ind2,3],'m')
##            plt.grid(True)
##            plt.ylabel("m [kg]")
##            plt.show()
##    
##            plt.plot(tPlot[ind1:ind2],u[ind1:ind2,0],'k')
##            plt.grid(True)
##            plt.ylabel("u1 [-]")
##            plt.show()
##    
##            plt.plot(tPlot[ind1:ind2],u[ind1:ind2,1],'c')
##            plt.grid(True)
##            plt.xlabel("t")
##            plt.ylabel("u2 [-]")
##            plt.show()
##            
##            print("Lambda:")
##            
##            plt.plot(tPlot[ind1:ind2],lam[ind1:ind2,0])
##            plt.grid(True)
##            plt.ylabel("lam_h")
##            plt.show()        
##    
##            plt.plot(tPlot[ind1:ind2],lam[ind1:ind2,1],'g')
##            plt.grid(True)
##            plt.ylabel("lam_V")
##            plt.show()
##    
##            plt.plot(tPlot[ind1:ind2],lam[ind1:ind2,2],'r')
##            plt.grid(True)
##            plt.ylabel("lam_gamma")
##            plt.show()
##    
##            plt.plot(tPlot[ind1:ind2],lam[ind1:ind2,3],'m')
##            plt.grid(True)
##            plt.ylabel("lam_m")
##            plt.show()
##            
###            print("dLambda/dt:")
###            
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,0])
###            plt.grid(True)
###            plt.ylabel("dlam_h")
###            plt.show()        
###    
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,1])
###            plt.grid(True)
###            plt.ylabel("dlam_V")
###            plt.show()
###    
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,2],'r')
###            plt.grid(True)
###            plt.ylabel("dlam_gamma")
###            plt.show()
###    
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,3],'m')
###            plt.grid(True)
###            plt.ylabel("dlam_m")
###            plt.show()
###            
###            print("-phix*lambda:")
###            
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,0]-errQx[ind1:ind2,0])
###            plt.grid(True)
###            plt.ylabel("-phix*lambda_h")
###            plt.show()        
###    
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,1]-errQx[ind1:ind2,1],'g')
###            plt.grid(True)
###            plt.ylabel("-phix*lambda_V")
###            plt.show()
###    
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,2]-errQx[ind1:ind2,2],'r')
###            plt.grid(True)
###            plt.ylabel("-phix*lambda_gamma")
###            plt.show()
###    
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,3]-errQx[ind1:ind2,3],'m')
###            plt.grid(True)
###            plt.ylabel("-phix*lambda_m")
###            plt.show()
##
##            print("\nBlue: dLambda/dt; Black: -phix*lam")
##            
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,0])
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,0]-errQx[ind1:ind2,0],'k')
##            plt.grid(True)
##            plt.ylabel("z_h")
##            plt.show()        
##    
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,1])
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,1]-errQx[ind1:ind2,1],'k')
##            plt.grid(True)
##            plt.ylabel("z_V")
##            plt.show()
##    
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,2])
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,2]-errQx[ind1:ind2,2],'k')
##            plt.grid(True)
##            plt.ylabel("z_gamma")
##            plt.show()
##    
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,3])
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,3]-errQx[ind1:ind2,3],'k')
##            plt.grid(True)
##            plt.ylabel("z_m")
##            plt.show()

    if self.dbugOptGrad['pausCalcQ']:
        input("calcQ in debug mode. Press any key to continue...")
    return Q,Qx,Qu,Qp,Qt

def calcStepGrad(self,corr):
    
    print("\nIn calcStepGrad.\n")
    
    prntCond = self.dbugOptGrad['prntCalcStepGrad']
    # Get initial status (Q0, no correction applied)

    print("\nalfa :",0.0)
    Q0,_,_,_,_ = self.calcQ()
    P0,_,_ = self.calcP()
    I0 = self.calcI()
    stepMan = stepMngr(k = 1e-7*I0/P0)#(k = 1e-9*I0/P0)
    # TODO: ideias
    # usar tolP ao inves de P0
    # usar P-tolP ao inves de P
    Obj0 = stepMan.calcObj(P0,Q0,I0)
    print("I0 = {:.4E}".format(I0)+" Obj0 = {:.4E}".format(Obj0))

#    if prntCond:
#        print("P0 = {:.4E}".format(P0))
#        print("I0 = {:.4E}\n".format(I0))
    
    # Get status associated with integral correction (alfa=1.0)
    if prntCond:
        print("\n> Trying alfa = 1.0 first, fingers crossed...")
    alfa = 1.0
    P1,Q1,I1,Obj1 = stepMan.tryStep(self,corr,alfa,prntCond)

#    if prntCond:
#        print("P1 = {:.4E}".format(P1))
#        print("I1 = {:.4E}\n".format(I1))
        
    # Search for a better starting point for alfa
    Obj = Obj1; keepLook = False; dAlfa = 1.0
    if Obj>1.1*Obj0:#Obj>Obj0:#I/I0 >= 10.0:#Q>Q0:#
        if prntCond:
            print("\n> Whoa! Going back to safe region of alphas...\n")
        keepLook = True
        dAlfa = 0.1
        cond = lambda nObj,Obj: nObj>Obj0 #nQ/Q0>1.1
    # Or increase alfa, if the conditions seem Ok for it
    elif Obj<Obj0:#se:#if Q<Q0:
        if prntCond:
            print("\n> This seems boring. Going forward!\n")
        keepLook = True
        dAlfa = 10.0
        cond = lambda nObj,Obj: nObj<Obj

    nObj = Obj.copy()
    while keepLook:
        Obj = nObj.copy()
        alfa *= dAlfa
        nP,nQ,nI,nObj = stepMan.tryStep(self,corr,alfa,prntCond)            
        keepLook = cond(nObj,Obj)
    #
    if dAlfa > 1.0:
        alfa /= dAlfa
    elif dAlfa < 1.0:
        Obj = nObj.copy()

    
    # Now Obj is not so much bigger than Obj0. Start "bilateral analysis"
    if prntCond:
        print("\n> Starting bilateral analysis...\n")
    alfa0 = alfa
    alfa = 1.2*alfa0
    PM,QM,IM,ObjM = stepMan.tryStep(self,corr,alfa,prntCond)
    
    alfa = .8*alfa0 
    Pm,Qm,Im,Objm = stepMan.tryStep(self,corr,alfa,prntCond)
    
    # Start refined search
    print("\nObj =",Obj)
    if Objm < Obj: 
        if self.dbugOptGrad['prntCalcStepGrad']:
            print("\n> Beginning search for decreasing alfa...")
        # if the tendency is to still decrease alfa, do it...
        nObj = Obj.copy(); keepSearch = True
        while keepSearch and alfa > 1.0e-15:
            Obj = nObj.copy()
            alfa *= .8
            nP,nQ,nI,nObj = stepMan.tryStep(self,corr,alfa,prntCond)

            # TODO: use testAlgn to test alignment of points, 
            # and use it to improve calcStepGrad. E.G.:
            
#            isAlgn = (abs(testAlgn(histAlfa[(cont-4):(cont-1)],\
#                           histQ[(cont-4):(cont-1)])) < 1e-3)
#
#            if isAlgn:
#                m = (histQ[cont-2]-histQ[cont-3]) / \
#                    (histAlfa[cont-2]-histAlfa[cont-3])
#                b = histQ[cont-2] - m * histAlfa[cont-2]
#                print("m =",m,"b =",b)
                        
            if nObj < Obj0:
                keepSearch = ((nObj-Obj)/Obj < -.1)#-.001)#nQ<Q#
        alfa /= 0.8
    else:

        # no overdrive!
#        if prntCond:
#                print("\n> Apparently alfa =",alfa0,"is the best.")
#        alfa = alfa0 # BRASIL
        
        if Obj <= ObjM: 
            if prntCond:
                print("\n> Apparently alfa =",alfa0,"is the best.")
            alfa = alfa0 # BRASIL
        else:
            if prntCond:
                print("\n> Beginning search for increasing alfa...")
            # There still seems to be a negative gradient here. Objncrease alfa!
            nObj = ObjM.copy()
            alfa = 1.2*alfa0; keepSearch = True#(nPint>Pint1M)
            while keepSearch:
                Obj = nObj.copy()
                alfa *= 1.2
                nP,nQ,nI,nObj = stepMan.tryStep(self,corr,alfa,prntCond)
                
                keepSearch = nObj<Obj
                #if nPint < Pint0:
            alfa /= 1.2
     
       
    # after all this analysis, plot the history of the tried alfas, and 
    # corresponding Q's        
    if self.dbugOptGrad['plotCalcStepGrad']:
        
        histAlfa = stepMan.histStep
        histP = stepMan.histP
        histQ = stepMan.histQ
        histI = stepMan.histI
        histObj = stepMan.histObj
        
        # Ax1: convergence history of Q
        fig, ax1 = plt.subplots()
        ax1.loglog(histAlfa, histQ, 'ob')
        linhAlfa = numpy.array([min(histAlfa),max(histAlfa)])
        linQ0 = Q0 + 0.0*numpy.empty_like(linhAlfa)
        ax1.loglog(linhAlfa,linQ0,'--b')
        ax1.set_xlabel('alpha')
        ax1.set_ylabel('Q', color='b')
        ax1.tick_params('y', colors='b')
        ax1.grid(True)
        
        # Ax2: convergence history of P
        ax2 = ax1.twinx()
        ax2.loglog(histAlfa,histP,'or')
        linP0 = P0 + 0.0*numpy.empty_like(linhAlfa)
        ax2.loglog(linhAlfa,linP0,'--r')    
        ax2.set_ylabel('P', color='r')
        ax2.tick_params('y', colors='r')
        ax2.grid(True)

        # Get index for applied alfa
        for k in range(len(histAlfa)):
            if abs(histAlfa[k]-alfa)<1e-14:
                break
            
        # Get final values of Q and P, plot them in squares
        Q = histQ[k]; P = histP[k]
        ax1.loglog(alfa,Q,'sb'); ax2.loglog(alfa,P,'sr')
        
        plt.title("Q and P versus Grad Step for this grad run")
        plt.show()
     
        # Plot history of I
        plt.loglog(histAlfa,histI,'o')
        linI = I0 + 0.0*numpy.empty_like(linhAlfa)
        plt.loglog(linhAlfa,linI,'--')
        plt.plot(alfa,histI[k],'s')
        plt.ylabel("I")
        plt.xlabel("alpha")
        plt.title("I versus grad step for this grad run")
        plt.grid(True)
        plt.show()
        
        # Plot history of Obj
        plt.loglog(histAlfa,histObj,'o')
        linObj = Obj0 + 0.0*numpy.empty_like(linhAlfa)
        plt.loglog(linhAlfa,linObj,'--')
        plt.plot(alfa,histObj[k],'s')
        plt.ylabel("Obj")
        plt.xlabel("alpha")
        plt.title("Obj versus grad step for this grad run")
        plt.grid(True)
        plt.show()

        
    if prntCond:           
        print("\n> Chosen alfa = {:.4E}".format(alfa)+", Q = {:.4E}".format(Q))
        print("> Number of objective evaluations:",stepMan.cont)
        
    if self.dbugOptGrad['pausCalcStepGrad']:
        input("\n> Run of calcStepGrad terminated. Press any key to continue.")
      
    return alfa


def grad(self):
    
    print("\nIn grad, Q0 = {:.4E}.".format(self.Q))

    # Calculate corrections
    A,B,C,lam,mu = self.LMPBVP(rho=1.0)
 
    # Store corrections in solution
    self.lam = lam
    self.mu = mu
    corr = {'x':A,'u':B,'pi':C}
 
    # Calculation of alfa
    alfa = self.calcStepGrad(corr)

    # Apply correction and update Q history
    self.aplyCorr(alfa,corr)
    self.updtHistQ(alfa)
    
    # update P just to ensure proper restoration afterwards
    P,_,_ = self.calcP()
    self.P = P
    print("Leaving grad with alfa =",alfa)
    print("Delta pi = ",alfa*C)
    
    if self.dbugOptGrad['pausGrad']:
        input('Grad in debug mode. Press any key to continue...')
