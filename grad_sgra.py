#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:27:55 2017

@author: levi
"""

import numpy
import matplotlib.pyplot as plt

#from utils_alt import ddt
from utils import ddt
from prob_rocket_sgra import calcGrads, plotSol, calcI
from rest_sgra import calcP
#from prob_test import calcGrads, plotSol
#from prob_pend import calcGrads, plotSol

def calcQ(sizes,x,u,pi,lam,mu,constants,restrictions,mustPlot=False):
    # Q expression from (15)

    N = sizes['N']
    n = sizes['n']
    m = sizes['m']
    p = sizes['p']
    dt = 1.0/(N-1)

    # get gradients
    Grads = calcGrads(sizes,x,u,pi,constants,restrictions)
    phix = Grads['phix']
    phiu = Grads['phiu']
    phip = Grads['phip']
    fx = Grads['fx']
    fu = Grads['fu']
    fp = Grads['fp']
    psix = Grads['psix']
    psip = Grads['psip']
    dlam = ddt(sizes,lam)
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
    print("Q = {:.4E}".format(Q)+": Qx = {:.4E}".format(Qx)+", Qu = {:.4E}".format(Qu)+", Qp = {:.4E}".format(Qp)+", Qt = {:.4E}".format(Qt))

    if mustPlot:
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
        
        n = sizes['n']; m = sizes['m']
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
    return Q

def calcStepGrad(sizes,x,u,pi,lam,mu,A,B,C,constants,boundary,restrictions):

    print("\nIn calcStepGrad.\n")
    
    I0 = calcI(sizes,x,u,pi,constants,restrictions)
    print("I0 = {:.4E}".format(I0))
    Q0 = calcQ(sizes,x,u,pi,lam,mu,constants,restrictions,True)
    P0,_,_ = calcP(sizes,x,u,pi,constants,boundary,restrictions,True)
    print("P0 = {:.4E}".format(P0))
#    print("In calcStepRest, P0 = {:.4E}".format(P0))
    
    alfa = .8
    print("\nalfa =",alfa)
    nx = x + alfa * A
    nu = u + alfa * B
    np = pi + alfa * C
    I1m = calcI(sizes,x,u,pi,constants,restrictions)
    print("I = {:.4E}".format(I1m))
    Q1m = calcQ(sizes,nx,nu,np,lam,mu,constants,restrictions,True)
    P1m,_,_ = calcP(sizes,nx,nu,np,constants,boundary,restrictions,True)
    print("P = {:.4E}".format(P1m))
    
    alfa = 1.0
    print("\nalfa =",alfa)
    nx = x + alfa * A
    nu = u + alfa * B
    np = pi + alfa * C
    I1 = calcI(sizes,x,u,pi,constants,restrictions)
    print("I = {:.4E}".format(I1))
    Q1 = calcQ(sizes,nx,nu,np,lam,mu,constants,restrictions,True)
    P1,_,_ = calcP(sizes,nx,nu,np,constants,boundary,restrictions,True)    
    print("P = {:.4E}".format(P1))

    alfa = 1.2
    print("\nalfa =",alfa)
    nx = x + alfa * A
    nu = u + alfa * B
    np = pi + alfa * C
    I1M = calcI(sizes,x,u,pi,constants,restrictions)
    print("I = {:.4E}".format(I1M))
    Q1M = calcQ(sizes,nx,nu,np,lam,mu,constants,restrictions,True)
    P1M,_,_ = calcP(sizes,nx,nu,np,constants,boundary,restrictions,True)
    print("P = {:.4E}".format(P1M))
        
    if Q1 >= Q1m or Q1 >= Q0:
        # alfa = 1.0 is too much. Reduce alfa.
        nQ = Q1; alfa=.8
        cont = 0; keepSearch = (nQ>Q0)
        while keepSearch and alfa > 1.0e-15:
            cont += 1
            Q = nQ
            alfa *= .8
            nx = x + alfa * A
            nu = u + alfa * B
            np = pi + alfa * C
            nQ = calcQ(sizes,nx,nu,np,lam,mu,constants,restrictions,True)
            print("\n alfa =",alfa,", Q = {:.4E}".format(nQ),\
                  " (Q0 = {:.4E})".format(Q0))
            if nQ < Q0:
                keepSearch = ((nQ-Q)/Q < -.05)
    else:
        
#        return 1.0
        
        if Q1 <= Q1M:
            # alfa = 1.0 is likely to be best value. 
            # Better not to waste time and return 1.0 
            return 1.0
        else:
            # There is still a descending gradient here. Increase alfa!
            nQ = Q1M
            cont = 0; keepSearch = True#(nPint>Pint1M)
            while keepSearch:
                cont += 1
                Q = nQ
                alfa *= 1.2
                nx = x + alfa * A
                nu = u + alfa * B
                np = pi + alfa * C
                nQ = calcQ(sizes,nx,nu,np,lam,mu,constants,restrictions,True)
                print("\n alfa =",alfa,", Q = {:.4E}".format(nQ),\
                      " (Q0 = {:.4E})".format(Q0))
                keepSearch = nQ<Q
                #if nPint < Pint0:
            alfa /= 1.2
    return alfa
    
#    # "Trissection" method
#    alfa = 1.0
#    nx = x + alfa * A
#    nu = u + alfa * B
#    np = pi + alfa * C
#
#    oldQ = calcQ(sizes,nx,nu,np,lam,mu,constants,restrictions,True)
#    nQ = .9*oldQ
##    print("Q =",Q)
#    alfaMin = 0.0
#    alfaMax = 1.0
#    cont = 0
#    while (nQ-oldQ)/oldQ < -.05 and cont < 5:
#        oldQ = nQ
#
#        dalfa = (alfaMax-alfaMin)/3.0
#        alfaList = numpy.array([alfaMin,alfaMin+dalfa,alfaMax-dalfa,alfaMax])
#        QList = numpy.empty(numpy.shape(alfaList))
#
#        for j in range(4):
#            alfa = alfaList[j]
#
#            nx = x + alfa * A
#            nu = u + alfa * B
#            np = pi + alfa * C
#
#            Q = calcQ(sizes,nx,nu,np,lam,mu,constants,restrictions)
#            QList[j] = Q
#        #
#        print("QList:",QList)
#        minQ = QList[0]
#        indxMinQ = 0
#
#        for j in range(1,4):
#            if QList[j] < minQ:
#                indxMinQ = j
#                minQ = QList[j]
#        #
#
#        alfa = alfaList[indxMinQ]
#        nQ = QList[indxMinQ]
#        print("nQ =",nQ)
#        if indxMinQ == 0:
#            alfaMin = alfaList[0]
#            alfaMax = alfaList[1]
#        elif indxMinQ == 1:
#            if QList[0] < QList[2]:
#                alfaMin = alfaList[0]
#                alfaMax = alfaList[1]
#            else:
#                alfaMin = alfaList[1]
#                alfaMax = alfaList[2]
#        elif indxMinQ == 2:
#            if QList[1] < QList[3]:
#                alfaMin = alfaList[1]
#                alfaMax = alfaList[2]
#            else:
#                alfaMin = alfaList[2]
#                alfaMax = alfaList[3]
#        elif indxMinQ == 3:
#            alfaMin = alfaList[2]
#            alfaMax = alfaList[3]
#
#        cont+=1
#    #
#
#    return .5*(alfaMin+alfaMax)

def grad(sizes,x,u,pi,t,Q0,constants,boundary,restrictions):
    print("In grad.")

    print("Q0 =",Q0)
    # get sizes
    N = sizes['N']
    n = sizes['n']
    m = sizes['m']
    p = sizes['p']
    q = sizes['q']

    # get gradients
    Grads = calcGrads(sizes,x,u,pi,constants,restrictions)

    phix = Grads['phix']
    phiu = Grads['phiu']
    phip = Grads['phip']
    fx = Grads['fx']
    fu = Grads['fu']
    fp = Grads['fp']
    psix = Grads['psix']
    psip = Grads['psip']
    dt = Grads['dt']

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

    optPlot = dict()
    ind1 = [1,2,0]
    ind2 = [2,0,1]
    for i in range(q+1):
        
        print("Integrating solution "+str(i+1)+" of "+str(q+1)+"...\n")
        
        mu = 0.0*mu
        if i<q:
            mu[i] = 1.0e-7
            #mu[i] = ((-1)**i)*1.0e-8#10#1.0#e-5#
            #mu[ind1[i]] = 1.0e-8
            #mu[ind2[i]] = -1.0e-8

        # integrate equation (38) backwards for lambda
        auxLam[0,:] = - psixTr.dot(mu)
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
            dlam = ddt(sizes,lam)
            erroLam = numpy.empty((N,n))
            normErroLam = numpy.empty(N)
            for k in range(N):
                erroLam[k,:] = dlam[k,:]+phix[k,:,:].transpose().dot(lam[k,:])-fx[k,:]
                normErroLam[k] = erroLam[k,:].transpose().dot(erroLam[k,:])                
            print("\nLambda Error:")
            optPlot['mode'] = 'states:LambdaError'
            plotSol(sizes,t,erroLam,numpy.zeros((N,m)),numpy.zeros(p),\
                    constants,restrictions,optPlot)

            maxNormErroLam = normErroLam.max()
            print("maxNormErroLam =",maxNormErroLam)
            if maxNormErroLam > 0:
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


        optPlot['mode'] = 'states:Lambda'
        plotSol(sizes,t,lam,B,C,constants,restrictions,optPlot)

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
        

        dA = ddt(sizes,A)        
        erroA = numpy.empty((N,n))
        normErroA = numpy.empty(N)  
        for k in range(N):
            erroA[k,:] = dA[k,:]-phix[k,:,:].dot(A[k,:]) -phiu[k,:,:].dot(B[k,:]) -phip[k,:,:].dot(C)
            normErroA[k] = erroA[k,:].dot(erroA[k,:])
        
        print("\nA Error:")
        optPlot['mode'] = 'states:AError'
        plotSol(sizes,t,erroA,B,C,\
            constants,restrictions,optPlot)            
        
        maxNormErroA = normErroA.max()
        print("maxNormErroA =",maxNormErroA)
        if maxNormErroA > 0:
            plt.semilogy(normErroA)
            plt.grid()
            plt.title("ErroA")
            plt.show()
        
        arrayA[i,:,:] = A
        arrayB[i,:,:] = B
        arrayC[i,:] = C
        arrayL[i,:,:] = lam
        arrayM[i,:] = mu
        
        optPlot['mode'] = 'var'
        plotSol(sizes,t,A,B,C,constants,restrictions,optPlot)

        M[1:,i] = psix.dot(A[N-1,:]) + psip.dot(C)
    #

    # Calculations of weights k:

    col = numpy.zeros(q+1)
    col[0] = 1.0
    
    print("M =",M)
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
    
    dlam = ddt(sizes,lam)
    dA = ddt(sizes,A)
    erroLam = numpy.empty((N,n))
    erroA = numpy.empty((N,n))
    normErroLam = numpy.empty(N)
    normErroA = numpy.empty(N)
    for k in range(N):
        erroLam[k,:] = dlam[k,:]+phix[k,:,:].transpose().dot(lam[k,:])-fx[k,:]
        normErroLam[k] = erroLam[k,:].transpose().dot(erroLam[k,:])                
        erroA[k,:] = dA[k,:]-phix[k,:,:].dot(A[k,:]) -phiu[k,:,:].dot(B[k,:]) -phip[k,:,:].dot(C)
        normErroA[k] = erroA[k,:].dot(erroA[k,:])
        
    print("\nFINAL A Error:")
    optPlot['mode'] = 'states:AError'
    plotSol(sizes,t,erroA,B,C,\
            constants,restrictions,optPlot)    
    maxNormErroA = normErroA.max()
    print("FINAL maxNormErroA =",maxNormErroA)
    if maxNormErroA > 0:
        plt.semilogy(normErroA)
        plt.grid()
        plt.title("ErroA")
        plt.show()


    print("\nFINAL Lambda Error:")
    optPlot['mode'] = 'states:LambdaError'
    plotSol(sizes,t,erroLam,B,C,\
            constants,restrictions,optPlot)
    maxNormErroLam = normErroLam.max()
    print("FINAL maxNormErroLam =",maxNormErroLam)
    if maxNormErroLam > 0:
        plt.semilogy(normErroLam)
        plt.grid()
        plt.title("ErroLam")
        plt.show()

    ##########################################
    
    #if (B>numpy.pi).any() or (B<-numpy.pi).any():
    #    print("\nProblems in grad: corrections will result in control overflow.")
    
    optPlot['mode'] = 'var'
    plotSol(sizes,t,A,B,C,constants,restrictions,optPlot)
    optPlot['mode'] = 'proposed (states: lambda)'
    plotSol(sizes,t,lam,B,C,constants,restrictions,optPlot)


    # Calculation of alfa
    alfa = calcStepGrad(sizes,x,u,pi,lam,mu,A,B,C,constants,boundary,restrictions)

    nx = x + alfa * A
    nu = u + alfa * B
    np = pi + alfa * C
    Q = calcQ(sizes,nx,nu,np,lam,mu,constants,restrictions)

    print("Leaving grad with alfa =",alfa)
    return nx,nu,np,lam,mu,Q
