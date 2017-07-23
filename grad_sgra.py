#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:39:08 2017

@author: levi
"""

import numpy
from utils import ddt, testAlgn
import matplotlib.pyplot as plt

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

    return Q,Qx,Qu,Qp,Qt

def calcStepGrad(self,corr):
    
    print("\nIn calcStepGrad.\n")
    cont = 0; prntCond = self.dbugOptGrad['prntCalcStepGrad']
    # Get initial status (Q0, no correction applied)
    Q0,_,_,_,_ = self.calcQ(); cont += 1
    P0,_,_ = self.calcP()
    if prntCond:
        print("P0 = {:.4E}".format(P0))
        I0 = self.calcI()
        print("I0 = {:.4E}\n".format(I0))
    
    # Get status associated with integral correction (alfa=1.0)
    newSol = self.copy()
    newSol.aplyCorr(1.0,corr)
    Q1,_,_,_,_ = newSol.calcQ(); cont += 1
    histQ = [Q1]; histAlfa = [1.0]
    P1,_,_ = newSol.calcP(); histP = [P1]
    if prntCond:
        print("\n> Trying alfa = 1.0 first, fingers crossed...")
        print("P1 = {:.4E}".format(P1))
        I1 = newSol.calcI()
        print("I1 = {:.4E}\n".format(I1))
        
    # Search for a better starting point for alfa, one that does not make Q
    # more than 10 times bigger.
    Q = Q1; alfa = 1.0; keepLook = False; dAlfa = 1.0 #;  contGran = 0
    if Q/Q0 >= 10.0:#Q>Q0:#
        if prntCond:
            print("\n> Whoa! Going back to safe region of alphas...\n")
        keepLook = True
        dAlfa = 0.1
        cond = lambda nQ,Q: nQ>Q0 #nQ/Q0>1.1
    # Or increase alfa, if the conditions seem Ok for it
    elif Q<Q0:#se:#if Q<Q0:
        if prntCond:
            print("\n> This seems boring. Going forward!\n")
        keepLook = True
        dAlfa = 10.0
        cond = lambda nQ,Q: nQ<Q

    nQ = Q
    while keepLook:
        Q = nQ
        alfa *= dAlfa
        newSol = self.copy()
        newSol.aplyCorr(alfa,corr)
        nQ,_,_,_,_ = newSol.calcQ(); cont += 1
        histQ.append(nQ); histAlfa.append(alfa)
        nP,_,_ = newSol.calcP(); histP.append(nP)
        if prntCond:
            print("alfa =",alfa,", P = {:.4E}".format(nP),\
                  ", Q = {:.6E}".format(nQ),\
                  " (Q0 = {:.6E})\n".format(Q0))
            
        keepLook = cond(nQ,Q)
    #
    if dAlfa > 1.0:
        alfa /= dAlfa
    
    # Now Q is not so much bigger than Q0. Start "bilateral analysis"
    if prntCond:
        print("\n> Starting bilateral analysis...\n")
    alfa0 = alfa
    alfa = 1.2*alfa0 
    newSol = self.copy()
    newSol.aplyCorr(alfa,corr)
    QM,_,_,_,_ = newSol.calcQ(); cont += 1
    histQ.append(QM); histAlfa.append(alfa)
    PM,_,_ = newSol.calcP(); histP.append(PM)
    if prntCond:
        print("alfa =",alfa,", P = {:.4E}".format(PM),\
              ", Q = {:.6E}".format(QM),\
              " (Q0 = {:.6E})\n".format(Q0))
    
    
    alfa = .8*alfa0 
    newSol = self.copy()
    newSol.aplyCorr(alfa,corr)
    Qm,_,_,_,_ = newSol.calcQ(); cont += 1
    histQ.append(Qm); histAlfa.append(alfa)
    Pm,_,_ = newSol.calcP(); histP.append(Pm)
    if prntCond:
        print("alfa =",alfa,", P = {:.4E}".format(Pm),\
              ", Q = {:.6E}".format(Qm),\
              " (Q0 = {:.6E})\n".format(Q0))
    
    # Start refined search
    
    if Qm < Q: 
        if self.dbugOptGrad['prntCalcStepGrad']:
            print("\n> Beginning search for decreasing alfa...")
        # if the tendency is to still decrease alfa, do it...
        nQ = Q; keepSearch = True#(nQ<Q0)
        while keepSearch and alfa > 1.0e-15:
            Q = nQ
            alfa *= .8
            newSol = self.copy()
            newSol.aplyCorr(alfa,corr)
            nQ,_,_,_,_ = newSol.calcQ(); cont += 1
            histQ.append(nQ); histAlfa.append(alfa)
            nP,_,_ = newSol.calcP(); histP.append(nP)
            if prntCond:
                print("alfa = {:.6E}".format(alfa),", Q = {:.6E}".format(nQ),\
                      ", Q0 = {:.6E}".format(Q0),", dQ = {:6E}".format(nQ-Q0))

            # TODO: use testAlgn to test alignment of points, 
            # and use it to improve calcStepGrad.
            isAlgn = (abs(testAlgn(histAlfa[(cont-4):(cont-1)],\
                           histQ[(cont-4):(cont-1)])) < 1e-3)

            if isAlgn:
                m = (histQ[cont-2]-histQ[cont-3]) / \
                    (histAlfa[cont-2]-histAlfa[cont-3])
                b = histQ[cont-2] - m * histAlfa[cont-2]
                print("m =",m,"b =",b)
                        
            if nQ < Q0:
                keepSearch = ((nQ-Q)/Q < -.001)#nQ<Q#
        alfa /= 0.8
    else:
        
        # no overdrive!
        if prntCond:
                print("\n> Apparently alfa =",alfa0,"is the best.")
        alfa = alfa0 # BRASIL
        
#        if Q <= QM: 
#            if prntCond:
#                print("\n> Apparently alfa =",alfa0,"is the best.")
#            alfa = alfa0 # BRASIL
#        else:
#            if prntCond:
#                print("\n> Beginning search for increasing alfa...")
#            # There still seems to be a negative gradient here. Increase alfa!
#            nQ = QM
#            alfa = 1.2*alfa0; keepSearch = True#(nPint>Pint1M)
#            while keepSearch:
#                Q = nQ
#                alfa *= 1.2
#                newSol = self.copy()
#                newSol.aplyCorr(alfa,corr)
#                nQ,_,_,_,_ = newSol.calcQ(); cont += 1
#                histQ.append(nQ); histAlfa.append(alfa)
#                
#                P,_,_ = newSol.calcP()
#                
#                if prntCond:
#                    print("alfa = {:.4E}".format(alfa),", Q = {:.4E}".format(nQ),\
#                          " (Q0 = {:.4E})".format(Q0),\
#                          "P = {:.4E}".format(P)+"\n")
#                keepSearch = nQ<Q
#                #if nPint < Pint0:
#            alfa /= 1.2
     
       
    # after all this analysis, plot the history of the tried alfas, and 
    # corresponding Q's        
    if self.dbugOptGrad['plotCalcStepGrad']:
        fig, ax1 = plt.subplots()
        
        # Ax1: convergence history of Q
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
     
    if prntCond:           
        print("\n> Chosen alfa = {:.4E}".format(alfa)+", Q = {:.4E}".format(Q))
        print("> Number of calcQ evaluations:",cont)
        
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
