#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:39:08 2017

@author: levi
"""

import numpy
from utils import ddt
import matplotlib.pyplot as plt

def calcQ(self,dbugOpt={}):
    # Q expression from (15)
    #print("\nIn calcQ.\n")
    N = self.N
    n = self.n
    m = self.m
    p = self.p
    dt = 1.0/(N-1)

    x = self.x
    u = self.u
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
    psix = Grads['psix']
    psip = Grads['psip']
    dlam = ddt(lam,N)

    Qx = 0.0
    Qu = 0.0
    Qp = 0.0
    Qt = 0.0
    Q = 0.0
    auxVecIntQp = numpy.zeros(p)
    
    errQx = numpy.empty((N,n)); normErrQx = numpy.empty(N)
    errQu = numpy.empty((N,m)); normErrQu = numpy.empty(N)
    errQp = numpy.empty((N,p)); #normErrQp = numpy.empty(N)
    
    for k in range(N):
        errQx[k,:] = dlam[k,:] - fx[k,:] + phix[k,:,:].transpose().dot(lam[k,:])
        errQu[k,:] = fu[k,:]-phiu[k,:,:].transpose().dot(lam[k,:])
        errQp[k,:] = fp[k,:] - phip[k,:,:].transpose().dot(lam[k,:])
        
        normErrQx[k] = errQx[k,:].transpose().dot(errQx[k,:])
        normErrQu[k] = errQu[k,:].transpose().dot(errQu[k,:])
        
        Qx += normErrQx[k]
        Qu += normErrQu[k]
        auxVecIntQp += errQp[k,:]
    #
    Qx -= .5*(normErrQx[0]+normErrQx[N-1])
    Qu -= .5*(normErrQu[0]+normErrQu[N-1])
    Qx *= dt
    Qu *= dt

    auxVecIntQp -= .5*(errQp[0,:]+errQp[N-1,:])
    auxVecIntQp *= dt
    
    auxVecIntQp += psip.transpose().dot(mu)

    Qp = auxVecIntQp.transpose().dot(auxVecIntQp)
    
    errQt = lam[N-1,:] + psix.transpose().dot(mu)
    Qt = errQt.transpose().dot(errQt)

    Q = Qx + Qu + Qp + Qt
    print("Q = {:.4E}".format(Q)+": Qx = {:.4E}".format(Qx)+\
          ", Qu = {:.4E}".format(Qu)+", Qp = {:.7E}".format(Qp)+\
          ", Qt = {:.4E}".format(Qt))

    # TODO: break these plots into more conditions

    if dbugOpt.get('plot',False):
        tPlot = numpy.arange(0,1.0+dt,dt)
        
        plt.plot(tPlot,normErrQx)
        plt.grid(True)
        plt.title("Integrand of Qx")
        plt.show()
        
        plt.plot(tPlot,normErrQu)
        plt.grid(True)
        plt.title("Integrand of Qu")
        plt.show()

        # for zoomed version:
        indMaxQx = normErrQx.argmax()
        ind1 = numpy.array([indMaxQx-20,0]).max()
        ind2 = numpy.array([indMaxQx+20,N-1]).min()
        plt.plot(tPlot[ind1:ind2],normErrQx[ind1:ind2],'o')
        plt.grid(True)
        plt.title("Integrand of Qx (zoom)")
        plt.show()
        
        if n==4 and m==2:
            
            plt.plot(tPlot[ind1:ind2],errQx[ind1:ind2,0])
            plt.grid(True)
            plt.ylabel("Qx_h")
            plt.show()        
    
            plt.plot(tPlot[ind1:ind2],errQx[ind1:ind2,1],'g')
            plt.grid(True)
            plt.ylabel("Qx_V")
            plt.show()
    
            plt.plot(tPlot[ind1:ind2],errQx[ind1:ind2,2],'r')
            plt.grid(True)
            plt.ylabel("Qx_gamma")
            plt.show()
    
            plt.plot(tPlot[ind1:ind2],errQx[ind1:ind2,3],'m')
            plt.grid(True)
            plt.ylabel("Qx_m")
            plt.show()
            
            print("\nStates, controls, lambda on the region of maxQx:")

            plt.plot(tPlot[ind1:ind2],x[ind1:ind2,0])
            plt.grid(True)
            plt.ylabel("h [km]")
            plt.show()        
    
            plt.plot(tPlot[ind1:ind2],x[ind1:ind2,1],'g')
            plt.grid(True)
            plt.ylabel("V [km/s]")
            plt.show()
    
            plt.plot(tPlot[ind1:ind2],x[ind1:ind2,2]*180/numpy.pi,'r')
            plt.grid(True)
            plt.ylabel("gamma [deg]")
            plt.show()
    
            plt.plot(tPlot[ind1:ind2],x[ind1:ind2,3],'m')
            plt.grid(True)
            plt.ylabel("m [kg]")
            plt.show()
    
            plt.plot(tPlot[ind1:ind2],u[ind1:ind2,0],'k')
            plt.grid(True)
            plt.ylabel("u1 [-]")
            plt.show()
    
            plt.plot(tPlot[ind1:ind2],u[ind1:ind2,1],'c')
            plt.grid(True)
            plt.xlabel("t")
            plt.ylabel("u2 [-]")
            plt.show()
            
            print("Lambda:")
            
            plt.plot(tPlot[ind1:ind2],lam[ind1:ind2,0])
            plt.grid(True)
            plt.ylabel("lam_h")
            plt.show()        
    
            plt.plot(tPlot[ind1:ind2],lam[ind1:ind2,1],'g')
            plt.grid(True)
            plt.ylabel("lam_V")
            plt.show()
    
            plt.plot(tPlot[ind1:ind2],lam[ind1:ind2,2],'r')
            plt.grid(True)
            plt.ylabel("lam_gamma")
            plt.show()
    
            plt.plot(tPlot[ind1:ind2],lam[ind1:ind2,3],'m')
            plt.grid(True)
            plt.ylabel("lam_m")
            plt.show()
            
#            print("dLambda/dt:")
#            
#            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,0])
#            plt.grid(True)
#            plt.ylabel("dlam_h")
#            plt.show()        
#    
#            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,1])
#            plt.grid(True)
#            plt.ylabel("dlam_V")
#            plt.show()
#    
#            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,2],'r')
#            plt.grid(True)
#            plt.ylabel("dlam_gamma")
#            plt.show()
#    
#            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,3],'m')
#            plt.grid(True)
#            plt.ylabel("dlam_m")
#            plt.show()
#            
#            print("-phix*lambda:")
#            
#            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,0]-errQx[ind1:ind2,0])
#            plt.grid(True)
#            plt.ylabel("-phix*lambda_h")
#            plt.show()        
#    
#            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,1]-errQx[ind1:ind2,1],'g')
#            plt.grid(True)
#            plt.ylabel("-phix*lambda_V")
#            plt.show()
#    
#            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,2]-errQx[ind1:ind2,2],'r')
#            plt.grid(True)
#            plt.ylabel("-phix*lambda_gamma")
#            plt.show()
#    
#            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,3]-errQx[ind1:ind2,3],'m')
#            plt.grid(True)
#            plt.ylabel("-phix*lambda_m")
#            plt.show()

            print("\nBlue: dLambda/dt; Black: -phix*lam")
            
            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,0])
            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,0]-errQx[ind1:ind2,0],'k')
            plt.grid(True)
            plt.ylabel("z_h")
            plt.show()        
    
            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,1])
            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,1]-errQx[ind1:ind2,1],'k')
            plt.grid(True)
            plt.ylabel("z_V")
            plt.show()
    
            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,2])
            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,2]-errQx[ind1:ind2,2],'k')
            plt.grid(True)
            plt.ylabel("z_gamma")
            plt.show()
    
            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,3])
            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,3]-errQx[ind1:ind2,3],'k')
            plt.grid(True)
            plt.ylabel("z_m")
            plt.show()
    return Q,Qx,Qu,Qp,Qt

def calcStepGrad(self,corr,dbugOpt={}):

    print("\nIn calcStepGrad.\n")
    
    Q0,_,_,_,_ = self.calcQ(dbugOpt)
    P0,_,_ = self.calcP(dbugOpt)
    print("P0 = {:.4E}".format(P0))
    I0 = self.calcI()
    print("I0 = {:.4E}\n".format(I0))
    
    newSol = self.copy()
    newSol.aplyCorr(.8,corr,dbugOpt)
    Q1m,_,_,_,_ = newSol.calcQ(dbugOpt)
    P1m,_,_ = newSol.calcP(dbugOpt)
    print("P1m = {:.4E}".format(P1m))
    I1m = newSol.calcI()
    print("I1m = {:.4E}\n".format(I1m))

    newSol = self.copy()
    newSol.aplyCorr(1.0,corr,dbugOpt)
    Q1,_,_,_,_ = newSol.calcQ(dbugOpt)
    P1,_,_ = newSol.calcP(dbugOpt)    
    print("P1 = {:.4E}".format(P1))
    I1 = newSol.calcI()
    print("I1 = {:.4E}\n".format(I1))

    newSol = self.copy()
    newSol.aplyCorr(1.2,corr,dbugOpt)
    Q1M,_,_,_,_ = newSol.calcQ(dbugOpt)
    P1M,_,_ = newSol.calcP(dbugOpt)
    print("P1M = {:.4E}".format(P1M))
    I1M = newSol.calcI()
    print("I1M = {:.4E}\n".format(I1M))
    
    histQ = [Q1M,Q1,Q1m]
    histAlfa = [1.2,1.0,0.8]
    
    if Q1 >= Q1m or Q1 >= Q0:
        # alfa = 1.0 is too much. Reduce alfa.
        
        nQ = Q1; alfa=.8
        cont = 0; keepSearch = (nQ>Q0)
        while keepSearch and alfa > 1.0e-15:
            cont += 1
            Q = nQ
            alfa *= .8
            newSol = self.copy()
            newSol.aplyCorr(alfa,corr,dbugOpt)
            nQ,_,_,_,_ = newSol.calcQ(dbugOpt)
            print("alfa =",alfa,", Q = {:.6E}".format(nQ),\
                  " (Q0 = {:.6E})\n".format(Q0))
            histQ.append(nQ)
            histAlfa.append(alfa)
            print(Q0-nQ)
            if nQ < Q0:
                print("fact = ",(nQ-Q)/Q,"\n")
                keepSearch = ((nQ-Q)/Q < -.001)#nQ<Q#
       
    else:
        
#        return 1.0
        
        if Q1 <= Q1M:
            # alfa = 1.0 is likely to be best value. 
            # Better not to waste time and return 1.0 
            alfa = 1.0
        else:
            # There is still a descending gradient here. Increase alfa!
            nQ = Q1M
            alfa=1.2; cont = 0; keepSearch = True#(nPint>Pint1M)
            while keepSearch:
                cont += 1
                Q = nQ
                alfa *= 1.2
                newSol = self.copy()
                newSol.aplyCorr(alfa,corr,dbugOpt)
                nQ,_,_,_,_ = newSol.calcQ(dbugOpt)
                print("alfa =",alfa,", Q = {:.4E}".format(nQ),\
                      " (Q0 = {:.4E})".format(Q0),"\n")
                histQ.append(nQ)
                histAlfa.append(alfa)
                keepSearch = nQ<Q
                #if nPint < Pint0:
            alfa /= 1.2
    
    plt.loglog(histAlfa,histQ,'o')
    plt.loglog(histAlfa[0:3],histQ[0:3],'ok')
    linhAlfa = numpy.array([min(histAlfa),max(histAlfa)])
    linQ0 = Q0 + 0.0*numpy.empty_like(linhAlfa)
    plt.loglog(linhAlfa,linQ0,'--')
    plt.grid(True)
    plt.xlabel("alfa")
    plt.ylabel("Q")
    plt.title("Q versus Grad Step for current Grad run")
    plt.show()
    input("What now?")
    
    return alfa


def grad(self,dbugOpt={}):
    
    print("In grad, Q0 = {:.4E}.".format(self.Q))
#    print("Q0 =",Q0,"\n")
    # get sizes
    N,n,m,p,q = self.N,self.n,self.m,self.p,self.q
    dt = 1.0/(N-1)
    
    x = self.x
    
    # get gradients
    Grads = self.calcGrads()

    phix = Grads['phix']
    phiu = Grads['phiu']
    phip = Grads['phip']
    fx = Grads['fx']
    fu = Grads['fu']
    fp = Grads['fp']
    psix = Grads['psix']
    psip = Grads['psip']

    # prepare time reversed/transposed versions of the arrays
    psixTr = psix.transpose()
    fxInv = fx.copy()
    phixInv = phix.copy()
    phiuTr = numpy.empty((N,m,n))
    phipTr = numpy.empty((N,p,n))
    for k in range(N):
        fxInv[k,:] = fx[N-k-1,:]
        phixInv[k,:,:] = phix[N-k-1,:,:].transpose()
        phiuTr[k,:,:] = phiu[k,:,:].transpose()
        phipTr[k,:,:] = phip[k,:,:].transpose()
    psipTr = psip.transpose()

    # Prepare array mu and arrays for linear combinations of A,B,C,lam
    mu = numpy.zeros(q)
    auxLam = 0*x
    lam = 0*x
    M = numpy.ones((q+1,q+1))

    arrayA = numpy.empty((q+1,N,n))
    arrayB = numpy.empty((q+1,N,m))
    arrayC = numpy.empty((q+1,p))
    arrayL = arrayA.copy()
    arrayM = numpy.empty((q+1,q))
    
    #initGuesArray = 1e-7*numpy.random.randn(n-1,q)
    #print("initGuesArray =",initGuesArray)

    #optPlot = dict()
    for i in range(q+1):
        
        print("\n> Integrating solution "+str(i+1)+" of "+str(q+1)+"...\n")
        
        mu = 0.0*mu
        if i<q:
            mu[i] = 1.0#e-7
            #mu[i] = ((-1)**i)*1.0e-8#10#1.0#e-5#
            #mu[ind1[i]] = 1.0e-8
            #mu[ind2[i]] = -1.0e-8
            #mu = initGuesArray[:,i]

        # integrate equation (38) backwards for lambda
        auxLam[0,:] = - psixTr.dot(mu)

        #auxLam[0,:] = initGuesArray[:,i]

        print(" auxLamInit =",auxLam[0,:])
        # Euler implicit
        I = numpy.eye(n)
        for k in range(N-1):
            auxLam[k+1,:] = numpy.linalg.solve(I-dt*phixInv[k+1,:],\
                  auxLam[k,:]-fxInv[k,:]*dt)
        #
        
        #auxLam = numpy.empty((N,n))
        #auxLam[0,:] = auxLamInit
        #for k in range(N-1):
        #    auxLam[k+1,:] = auxLam[k,:] + dt*(phixInv[k,:,:].dot(auxLam[k-1,:]))
        

        # Calculate B
        B = -fu.copy()
        for k in range(N):
            lam[k,:] = auxLam[N-k-1,:]
            B[k,:] += phiuTr[k,:,:].dot(lam[k,:])


        ##################################################################
        # TESTING LAMBDA DIFFERENTIAL EQUATION
        if i<q: #otherwise, there's nothing to test here...
            dlam = ddt(lam,N)
            erroLam = numpy.empty((N,n))
            normErroLam = numpy.empty(N)
            for k in range(N):
                erroLam[k,:] = dlam[k,:]+phix[k,:,:].transpose().dot(lam[k,:])-fx[k,:]
                normErroLam[k] = erroLam[k,:].transpose().dot(erroLam[k,:])                
                
            if dbugOpt.get('plotLamErr',False):
                print("\nLambda Error:")
                print("Cannot plot anymore. :( ")
                # TODO: include lambda error plotting mode
                
                #optPlot['mode'] = 'states:LambdaError'
                #plotSol(sizes,t,erroLam,numpy.zeros((N,m)),numpy.zeros(p),\
                #    constants,restrictions,optPlot)

            maxNormErroLam = normErroLam.max()
            print("maxNormErroLam =",maxNormErroLam)
            if dbugOpt.get('plotLam',False) and (maxNormErroLam > 0):
                plt.semilogy(normErroLam)
                plt.grid()
                plt.title("ErroLam")
                plt.show()            
        
        ##################################################################            
                
        # Calculate C
        C = numpy.zeros(p)
        for k in range(1,N-1):
            C += fp[k,:] - phipTr[k,:,:].dot(lam[k,:])
        C += .5*(fp[0,:] - phipTr[0,:,:].dot(lam[0,:]))
        C += .5*(fp[N-1,:] - phipTr[N-1,:,:].dot(lam[N-1,:]))
        C *= -dt #yes, the minus sign is on purpose!
        C -= -psipTr.dot(mu)

        if dbugOpt.get('plotLam',False):
            print("Cannot plot lambda anymore... for now!")
            #optPlot['mode'] = 'states:Lambda'
            #plotSol(sizes,t,lam,B,C,constants,restrictions,optPlot)

        # integrate diff equation for A
        A = numpy.zeros((N,n))
        
        for k in range(N-1):
            derk = phix[k,:,:].dot(A[k,:]) + phiu[k,:,:].dot(B[k,:]) + \
            phip[k,:,:].dot(C)
            aux = A[k,:] + dt*derk
            A[k+1,:] = A[k,:] + .5*dt*(derk + \
                               phix[k+1,:,:].dot(aux) + \
                               phiu[k+1,:,:].dot(B[k+1,:]) + \
                               phip[k+1,:,:].dot(C))

#        for k in range(N-1):
#            A[k+1,:] = numpy.linalg.solve(I-dt*phix[k+1,:,:],\
#                    A[k,:] + dt*(phiu[k,:,:].dot(B[k,:]) + phip[k,:,:].dot(C)))

        #A = numpy.zeros((N,n))
        #for k in range(N-1):
        #    A[k+1,:] = A[k,:] + dt*(phix[k,:,:].dot(A[k,:]) +  \
        #                            phiu[k,:,:].dot(B[k,:]) + \
        #                            phip[k,:].dot(C[k,:]))
        

        dA = ddt(A,N)        
        erroA = numpy.empty((N,n))
        normErroA = numpy.empty(N)  
        for k in range(N):
            erroA[k,:] = dA[k,:]-phix[k,:,:].dot(A[k,:]) -phiu[k,:,:].dot(B[k,:]) -phip[k,:,:].dot(C)
            normErroA[k] = erroA[k,:].dot(erroA[k,:])
        
        if dbugOpt.get('plotAErr',False):
            print("\nA Error:")
            print("Cannot plot anymore. :( ")
            #optPlot['mode'] = 'states:AError'
            #plotSol(sizes,t,erroA,B,C,\
            #        constants,restrictions,optPlot)            
        
        maxNormErroA = normErroA.max()
        print("maxNormErroA =",maxNormErroA)
        if dbugOpt.get('plotAErr',False) and (maxNormErroA > 0):
            plt.semilogy(normErroA)
            plt.grid()
            plt.title("ErroA")
            plt.show()
        
        arrayA[i,:,:] = A
        arrayB[i,:,:] = B
        arrayC[i,:] = C
        arrayL[i,:,:] = lam
        arrayM[i,:] = mu
        
        #if mustPlot:
        #    optPlot['mode'] = 'var'
        #    plotSol(sizes,t,A,B,C,constants,restrictions,optPlot)

        M[1:,i] = psix.dot(A[N-1,:]) + psip.dot(C)
    #

    # Calculations of weights k:

    col = numpy.zeros(q+1)
    col[0] = 1.0
    
    print("M =",M)
    print("det(M) =",numpy.linalg.det(M))
    print("col =",col)
    K = numpy.linalg.solve(M,col)
    print("K =",K)
    print("Residual =",M.dot(K)-col)
    # summing up linear combinations
    A = 0.0*A
    B = 0.0*B
    C = 0.0*C
    lam = 0.0*lam
    mu = 0.0*mu
    for i in range(q+1):
        A += K[i]*arrayA[i,:,:]
        B += K[i]*arrayB[i,:,:]
        C += K[i]*arrayC[i,:]
        lam += K[i]*arrayL[i,:,:]
        mu += K[i]*arrayM[i,:]
    
    
    ##########################################
    
    dlam = ddt(lam,N)
    dA = ddt(A,N)
    erroLam = numpy.empty((N,n))
    erroA = numpy.empty((N,n))
    normErroLam = numpy.empty(N)
    normErroA = numpy.empty(N)
    for k in range(N):
        erroLam[k,:] = dlam[k,:]+phix[k,:,:].transpose().dot(lam[k,:])-fx[k,:]
        normErroLam[k] = erroLam[k,:].transpose().dot(erroLam[k,:])                
        erroA[k,:] = dA[k,:]-phix[k,:,:].dot(A[k,:]) -phiu[k,:,:].dot(B[k,:]) -phip[k,:,:].dot(C)
        normErroA[k] = erroA[k,:].dot(erroA[k,:])
    
    if dbugOpt.get('plotAErrFin',False):
        print("\nFINAL A Error:")
        print("Cannot plot anymore. :( ")
        #optPlot['mode'] = 'states:AError'
        #plotSol(sizes,t,erroA,B,C,\
        #        constants,restrictions,optPlot)    
    maxNormErroA = normErroA.max()
    
    print("FINAL maxNormErroA =",maxNormErroA)
    
    if dbugOpt.get('plotAErrFin',False) and (maxNormErroA > 0):
        plt.semilogy(normErroA)
        plt.grid()
        plt.title("ErroA")
        plt.show()

    if dbugOpt.get('plotLamErrFin',False):
        print("\nFINAL Lambda Error:")
        print("Cannot plot anymore. :( ")
        #optPlot['mode'] = 'states:LambdaError'
        #plotSol(sizes,t,erroLam,B,C,\
        #    constants,restrictions,optPlot)
        #maxNormErroLam = normErroLam.max()
    print("FINAL maxNormErroLam =",maxNormErroLam)

    if dbugOpt.get('plotLamErrFin',False) and (maxNormErroLam > 0):
        plt.semilogy(normErroLam)
        plt.grid()
        plt.title("ErroLam")
        plt.show()

    ##########################################
    
    #if (B>numpy.pi).any() or (B<-numpy.pi).any():
    #    print("\nProblems in grad: corrections will result in control overflow.")
    
#    if mustPlot:
        #optPlot['mode'] = 'var'
        #plotSol(sizes,t,A,B,C,constants,restrictions,optPlot)
        #optPlot['mode'] = 'proposed (states: lambda)'
        #plotSol(sizes,t,lam,B,C,constants,restrictions,optPlot)

    self.lam = lam
    self.mu = mu
    corr = {'x':A,'u':B,'pi':C}
    # Calculation of alfa
    alfa = self.calcStepGrad(corr,dbugOpt)

    self.aplyCorr(alfa,corr,dbugOpt)
    self.updtHistQ(alfa)
    
    # update P just to ensure proper restoration afterwards
    P,_,_ = self.calcP()
    self.P = P
    print("Leaving grad with alfa =",alfa)
    print("Delta pi = ",alfa*corr['pi'])