#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:17:45 2017
rest_sgra: functions for performing restoration
@author: levi
"""

import numpy
import matplotlib.pyplot as plt

from prob_rocket_sgra import calcPhi,calcPsi,calcGrads,plotSol
#from prob_test import calcPhi,calcPsi,calcGrads,plotSol
#from utils_alt import ddt
from utils import ddt

def calcP(sizes,x,u,pi,constants,boundary,restrictions,mustPlot=False):
    #print("\nIn calcP.")
    N = sizes['N']

    dt = 1.0/(N-1)

    phi = calcPhi(sizes,x,u,pi,constants,restrictions)
    psi = calcPsi(sizes,x,boundary)
    dx = ddt(sizes,x)
    func = dx-phi
    vetP = numpy.empty(N)
    vetIP = numpy.empty(N)
    P = .5*(func[0,:].dot(func[0,:].transpose()))#norm(func[0,:])**2
    vetP[0] = P
    vetIP[0] = P

    for t in range(1,N-1):
        vetP[t] = func[t,:].dot(func[t,:].transpose())#norm(func[t,:])**2
        P += vetP[t]
        vetIP[t] = P
#        P += func[t,:].dot(func[t,:].transpose())

    
    vetP[N-1] = .5*(func[N-1,:].dot(func[N-1,:].transpose()))#norm(func[N-1,:])**2
    P += vetP[N-1]
    vetIP[N-1] = P
#    P += .5*(func[N-1,:].dot(func[N-1,:].transpose()))

    P *= dt
    vetP *= dt
    vetIP *= dt

    if mustPlot:
        tPlot = numpy.arange(0,1.0+dt,dt)
        
        plt.plot(tPlot,vetP)
        plt.grid(True)
        plt.title("Integrand of P")
        plt.show()
        
        # for zoomed version:
        indMaxP = vetP.argmax()
        ind1 = numpy.array([indMaxP-10,0]).max()
        ind2 = numpy.array([indMaxP+10,N-1]).min()
        plt.plot(tPlot[ind1:ind2],vetP[ind1:ind2],'o')
        plt.grid(True)
        plt.title("Integrand of P (zoom)")
        plt.show()
        
        n = sizes['n']; m = sizes['m']
        if n==4 and m==2:
            print("\nStates and controls on the region of maxP:")

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
        
        
        plt.plot(tPlot,vetIP)
        plt.grid(True)
        plt.title("Partially integrated P")
        plt.show()

    Pint = P
    Ppsi = psi.transpose().dot(psi)#norm(psi)**2
    P += Ppsi
    #print("P = {:.4E}".format(P)+": P_int = {:.4E}".format(Pint)+", P_psi = {:.4E}".format(Ppsi))
    return P,Pint,Ppsi

def calcStepRest(sizes,t,x,u,pi,A,B,C,constants,boundary,restrictions):

    P0,Pint0,Ppsi0 = calcP(sizes,x,u,pi,constants,boundary,restrictions)
#    print("In calcStepRest, P0 = {:.4E}".format(P0))
    
    alfa = .8
    #print("\nalfa =",alfa)
    nx = x + alfa * A
    nu = u + alfa * B
    np = pi + alfa * C
    P1m,Pint1m,Ppsi1m = calcP(sizes,nx,nu,np,constants,boundary,restrictions)
    
    alfa = 1.0
#    print("\nalfa =",alfa)
    nx = x + alfa * A
    nu = u + alfa * B
    np = pi + alfa * C
    P1,Pint1,Ppsi1 = calcP(sizes,nx,nu,np,constants,boundary,restrictions,False)
    
    alfa = 1.2
    #print("\nalfa =",alfa)
    nx = x + alfa * A
    nu = u + alfa * B
    np = pi + alfa * C
    P1M,Pint1M,Ppsi1M = calcP(sizes,nx,nu,np,constants,boundary,restrictions)
    
        
    if P1 >= P1m or P1 >= P0:
        # alfa = 1.0 is too much. Reduce alfa.
        # TODO: improve this!
        nP = P1; alfa=1
        cont = 0; keepSearch = (nP>P0)
        while keepSearch and alfa > 1.0e-15:
            cont += 1
            P = nP
            alfa *= .8
            nx = x + alfa * A
            nu = u + alfa * B
            np = pi + alfa * C
            nP,nPint,nPpsi = calcP(sizes,nx,nu,np,constants,boundary,\
                                   restrictions)
#            print("\n alfa =",alfa,", P = {:.4E}".format(nP),\
#                  " (P0 = {:.4E})".format(P0))
            if nP < P0:
                keepSearch = ((nP-P)/P < -.05)
    else:
        if P1 <= P1M:
            # alfa = 1.0 is likely to be best value. 
            # Better not to waste time and return 1.0 
            return 1.0
        else:
            # There is still a descending gradient here. Increase alfa!
            nP = P1M
            cont = 0; keepSearch = True#(nPint>Pint1M)
            while keepSearch:
                cont += 1
                P = nP
                alfa *= 1.2
                nx = x + alfa * A
                nu = u + alfa * B
                np = pi + alfa * C
                nP,nPint,nPpsi = calcP(sizes,nx,nu,np,constants,boundary,\
                                   restrictions)
 #               print("\n alfa =",alfa,", P = {:.4E}".format(nP),\
 #                     " (P0 = {:.4E})".format(P0))
                keepSearch = nP<P
                #if nPint < Pint0:
            alfa /= 1.2
    return alfa

def rest(sizes,x,u,pi,t,constants,boundary,restrictions):
    
    # Defining restoration "graphorragic" mode:
    RmustPlotLam = False
    RmustPlotLamZ = False
    RmustPlotPropLam = False
    RmustPlotErroLam = False
    RmustPlotErroLamZ = False
    RmustPlotVar = False
    RmustPlotVarT = False
    print("\nIn rest.")

    # get sizes
    N = sizes['N']
    n = sizes['n']
    m = sizes['m']
    p = sizes['p']
    q = sizes['q']

    #print("Calc phi...")
    # calculate phi and psi
    phi = calcPhi(sizes,x,u,pi,constants,restrictions)
    #print("Calc psi...")
    psi = calcPsi(sizes,x,boundary)

    # aux: phi - dx/dt
    err = phi - ddt(sizes,x)

    # get gradients
    #print("Calc grads...")
    Grads = calcGrads(sizes,x,u,pi,constants,restrictions)

    dt = Grads['dt']
    #dt6 = dt/6
    phix = Grads['phix']
    phiu = Grads['phiu']
    phip = Grads['phip']
    psix = Grads['psix']
    psip = Grads['psip']

    #print("Preparing matrices...")
    psixTr = psix.transpose()
    phixInv = phix.copy()
    phiuTr = numpy.empty((N,m,n))
    phipTr = numpy.empty((N,p,n))
    psipTr = psip.transpose()

    for k in range(N):
        phixInv[k,:,:] = phix[N-k-1,:,:].transpose()
        phiuTr[k,:,:] = phiu[k,:,:].transpose()
        phipTr[k,:,:] = phip[k,:,:].transpose()

    mu = numpy.zeros(q)

    # Matrix for linear system involving k's
    M = numpy.ones((q+1,q+1))

    # column vector for linear system involving k's    [eqs (88-89)]
    col = numpy.zeros(q+1)
    col[0] = 1.0 # eq (88)
    col[1:] = -psi # eq (89)

    arrayA = numpy.empty((q+1,N,n))
    arrayB = numpy.empty((q+1,N,m))
    arrayC = numpy.empty((q+1,p))
    arrayL = arrayA.copy()
    arrayM = numpy.empty((q+1,q))

    optPlot = dict()

    #print("Beginning loop for solutions...")
    for i in range(q+1):
#        print("\nIntegrating solution "+str(i+1)+" of "+str(q+1)+"...\n")
        mu = 0.0*mu
        if i<q:
            mu[i] = 1.0e-10

            # integrate equation (75-2) backwards
            auxLamInit = - psixTr.dot(mu)

            
            auxLam = numpy.empty((N,n))
            auxLam[0,:] = auxLamInit
            
            # Euler implicit
            I = numpy.eye(n)
            for k in range(N-1):
                auxLam[k+1,:] = numpy.linalg.solve(I-dt*phixInv[k+1,:],auxLam[k,:])
            
            # Euler's method
#            for k in range(N-1):
#                auxLam[k+1,:] = auxLam[k,:] + dt * phixInv[k,:,:].dot(auxLam[k,:])
            
            # Heun's method
            #for k in range(N-1):
            #    derk = phixInv[k,:,:].dot(auxLam[k,:])
            #    aux = auxLam[k,:] + dt*derk
            #    auxLam[k+1,:] = auxLam[k,:] + \
            #                    .5*dt*(derk + phixInv[k+1,:,:].dot(aux))

            # RK4 method (with interpolation...)
#            for k in range(N-1):
#               phixInv_k = phixInv[k,:,:]
#               phixInv_kp1 = phixInv[k+1,:,:]
#               phixInv_kpm = .5*(phixInv_k+phixInv_kp1)
#               k1 = phixInv_k.dot(auxLam[k,:])  
#               k2 = phixInv_kpm.dot(auxLam[k,:]+.5*dt*k1)  
#               k3 = phixInv_kpm.dot(auxLam[k,:]+.5*dt*k2)
#               k4 = phixInv_kp1.dot(auxLam[k,:]+dt*k3)  
#               auxLam[k+1,:] = auxLam[k,:] + dt6 * (k1+k2+k2+k3+k3+k4) 
    
            # equation for Bi (75-3)
            B = numpy.empty((N,m))
            lam = 0*x
            for k in range(N):
                lam[k,:] = auxLam[N-k-1,:]                    
                B[k,:] = phiuTr[k,:,:].dot(lam[k,:])
            
            
            ##################################################################
            # TESTING LAMBDA DIFFERENTIAL EQUATION
            
#            dlam = ddt(sizes,lam)
#            erroLam = numpy.empty((N,n))
#            normErroLam = numpy.empty(N)
#            for k in range(N):
#                erroLam[k,:] = dlam[k,:]+phix[k,:,:].transpose().dot(lam[k,:])
#                normErroLam[k] = erroLam[k,:].transpose().dot(erroLam[k,:])
#            if mustPlotErroLam:
#                print("\nLambda Error:")
#                optPlot['mode'] = 'states:LambdaError'
#                plotSol(sizes,t,erroLam,numpy.zeros((N,m)),numpy.zeros(p),\
#                        constants,restrictions,optPlot)
#                plt.semilogy(normErroLam)
#                plt.grid()
#                plt.title("ErroLam")
#                plt.show()
#            
#            if mustPlotErroLamZ:
#                optPlot['mode'] = 'states:LambdaError (zoom)'
#                N1 = 0#int(N/100)-10
#                N2 = 20##N1+20
#                plotSol(sizes,t[N1:N2],erroLam[N1:N2,:],numpy.zeros((N2-N1,m)),\
#                        numpy.zeros(p),constants,restrictions,optPlot)
#                plt.semilogy(normErroLam[N1:N2])
#                plt.grid()
#                plt.title("ErroLam (zoom)")
#                plt.show()
            
            ##################################################################            
            scal = 1.0/((numpy.absolute(B)).max())
            lam *= scal
            mu *= scal
            B *= scal
            
            
            # equation for Ci (75-4)
            C = numpy.zeros(p)
            for k in range(1,N-1):
                C += phipTr[k,:,:].dot(lam[k,:])
            C += .5*(phipTr[0,:,:].dot(lam[0,:]))
            C += .5*(phipTr[N-1,:,:].dot(lam[N-1,:]))
            C *= dt
            C -= -psipTr.dot(mu)
            
            if RmustPlotLam:            
                optPlot['mode'] = 'states:Lambda'
                plotSol(sizes,t,lam,B,C,constants,restrictions,optPlot)
                
            if RmustPlotLamZ:
                optPlot['mode'] = 'states:Lambda (zoom)'
                plotSol(sizes,t[N1:N2],lam[N1:N2,:],B[N1:N2,:],C,constants,restrictions,optPlot)
                        
            #print("Integrating ODE for A ["+str(i)+"/"+str(q)+"] ...")
            # integrate equation for A:                
            A = numpy.zeros((N,n))

            for k in range(N-1):
                derk = phix[k,:,:].dot(A[k,:]) + phiu[k,:,:].dot(B[k,:]) + \
                       phip[k,:,:].dot(C) + err[k,:]
                aux = A[k,:] + dt*derk
                A[k+1,:] = A[k,:] + .5*dt*(derk + \
                                           phix[k+1,:,:].dot(aux) + \
                                           phiu[k+1,:,:].dot(B[k+1,:]) + \
                                           phip[k+1,:,:].dot(C) + \
                                           err[k+1,:])   
#            for k in range(N-1):
#                phix_k = phix[k,:,:]
#                phix_kp1 = phix[k+1,:,:]
#                phix_kpm = .5*(phix_k+phix_kp1)
#                add_k = phiu[k,:,:].dot(B[k,:]) + phip[k,:,:].dot(C) + err[k,:]
#                add_kp1 = phiu[k+1,:,:].dot(B[k+1,:]) + phip[k+1,:,:].dot(C) + err[k+1,:]
#                add_kpm = .5*(add_k+add_kp1)
#
#                k1 = phix_k.dot(A[k,:]) + add_k  
#                k2 = phix_kpm.dot(A[k,:]+.5*dt*k1) + add_kpm 
#                k3 = phix_kpm.dot(A[k,:]+.5*dt*k2) + add_kpm
#                k4 = phix_kp1.dot(A[k,:]+dt*k3) + add_kp1 
#                A[k+1,:] = A[k,:] + dt6 * (k1+k2+k2+k3+k3+k4)

        else:
            # integrate equation (75-2) backwards
            lam *= 0.0
    
            # equation for Bi (75-3)
            B *= 0.0
            
            # equation for Ci (75-4)
            C *= 0.0

            #print("Integrating ODE for A ["+str(i)+"/"+str(q)+"] ...")                    
            # integrate equation for A:
            A = numpy.zeros((N,n))
            for k in range(N-1):
                derk = phix[k,:,:].dot(A[k,:]) + err[k,:]
                aux = A[k,:] + dt*derk
                A[k+1,:] = A[k,:] + .5*dt*(derk + \
                                           phix[k+1,:,:].dot(aux) + err[k+1,:])

#            for k in range(N-1):
#                phix_k = phix[k,:,:]
#                phix_kp1 = phix[k+1,:,:]
#                phix_kpm = .5*(phix_k+phix_kp1)
#                add_k = err[k,:]
#                add_kp1 = err[k+1,:]
#                add_kpm = .5*(add_k+add_kp1)
#
#                k1 = phix_k.dot(A[k,:]) + add_k  
#                k2 = phix_kpm.dot(A[k,:]+.5*dt*k1) + add_kpm 
#                k3 = phix_kpm.dot(A[k,:]+.5*dt*k2) + add_kpm
#                k4 = phix_kp1.dot(A[k,:]+dt*k3) + add_kp1 
#                A[k+1,:] = A[i,:] + dt6 * (k1+k2+k2+k3+k3+k4)
        #
                        
        # store solution in arrays
        arrayA[i,:,:] = A
        arrayB[i,:,:] = B
        arrayC[i,:] = C
        arrayL[i,:,:] = lam
        arrayM[i,:] = mu
        
        if RmustPlotVar:
            optPlot['mode'] = 'var'
            plotSol(sizes,t,A,B,C,constants,restrictions,optPlot)

        
        # Matrix for linear system (89)
        M[1:,i] = psix.dot(A[N-1,:]) + psip.dot(C)
    #

    # Calculations of weights k:
#    print("M =",M)
#    print("col =",col)
    K = numpy.linalg.solve(M,col)
#    print("K =",K)

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

    if RmustPlotVarT:
        optPlot['mode'] = 'var'
        plotSol(sizes,t,A,B,C,constants,restrictions,optPlot)
    
    if RmustPlotPropLam:   
        optPlot['mode'] = 'proposed (states: lambda)'
        plotSol(sizes,t,lam,B,C,constants,restrictions,optPlot)


    #print("Calculating step...")
    alfa = calcStepRest(sizes,t,x,u,pi,A,B,C,constants,boundary,restrictions)
    nx = x + alfa * A
    nu = u + alfa * B
    np = pi + alfa * C

#    print("Leaving rest with alfa =",alfa)
    return nx,nu,np,lam,mu

def calcStepOdeRest(sizes,t,x,u,pi,A,B,C,constants,boundary,restrictions):
    #print("\nIn calcStepOdeRest.\n")
    P0,Pint0,Ppsi0 = calcP(sizes,x,u,pi,constants,boundary,restrictions)
    
    alfa = .8
    #print("\nalfa =",alfa)
    nx = x + alfa * A
    nu = u + alfa * B
    np = pi + alfa * C
    P1m,Pint1m,Ppsi1m = calcP(sizes,nx,nu,np,constants,boundary,restrictions)
    
    alfa = 1.0
    #print("\nalfa =",alfa)
    nx = x + alfa * A
    nu = u + alfa * B
    np = pi + alfa * C
    P1,Pint1,Ppsi1 = calcP(sizes,nx,nu,np,constants,boundary,restrictions)
    
    alfa = 1.2
    #print("\nalfa =",alfa)
    nx = x + alfa * A
    nu = u + alfa * B
    np = pi + alfa * C
    P1M,Pint1M,Ppsi1M = calcP(sizes,nx,nu,np,constants,boundary,restrictions)
    
    if Pint1 >= Pint1m:
        # alfa = 1.0 is too much. Reduce alfa.
        # TODO: improve this!
        nPint = Pint1m
        cont = 0; keepSearch = (nPint>Pint0)
        while keepSearch and alfa > 1.0e-15:
            cont += 1
            Pint = nPint
            alfa *= .5
            nx = x + alfa * A
            nu = u + alfa * B
            np = pi + alfa * C
            nP,nPint,nPpsi = calcP(sizes,nx,nu,np,constants,boundary,\
                                   restrictions)
            #print("\n alfa =",alfa,", P_int = {:.4E}".format(nPint),\
            #      " (P_int0 = {:.4E})".format(Pint0))
            if nPint < Pint0:
                keepSearch = ((nPint-Pint)/Pint < -.05)
    else:
        if Pint1 <= Pint1M:
            # alfa = 1.0 is likely to be best value. 
            # Better not to waste time and return 1.0 
            return 1.0
        else:
            # There is still a descending gradient here. Increase alfa!
            nPint = Pint1M
            cont = 0; keepSearch = True#(nPint>Pint1M)
            while keepSearch:
                cont += 1
                Pint = nPint
                alfa *= 1.2
                nx = x + alfa * A
                nu = u + alfa * B
                np = pi + alfa * C
                nP,nPint,nPpsi = calcP(sizes,nx,nu,np,constants,boundary,\
                                   restrictions)
                #print("\n alfa =",alfa,", P_int = {:.4E}".format(nPint),\
                #      " (P_int0 = {:.4E})".format(Pint0))
                keepSearch = nPint<Pint
                #if nPint < Pint0:
            alfa /= 1.2

    return alfa

def oderest(sizes,x,u,pi,t,constants,boundary,restrictions):
    #print("\nIn oderest.")


    # get sizes
    N = sizes['N']
    n = sizes['n']
    m = sizes['m']
    p = sizes['p']
    q = sizes['q']

    #print("Calc phi...")
    # calculate phi and psi
    phi = calcPhi(sizes,x,u,pi,constants,restrictions)

    # err: phi - dx/dt
    err = phi - ddt(sizes,x)

    # get gradients
    #print("Calc grads...")
    Grads = calcGrads(sizes,x,u,pi,constants,restrictions)
    dt = Grads['dt']
    phix = Grads['phix']
    
    lam = 0*x; mu = numpy.zeros(q)        
    B = numpy.zeros((N,m))
    C = numpy.zeros(p)

    #print("Integrating ODE for A...")
    # integrate equation for A:
    A = numpy.zeros((N,n))
    for k in range(N-1):
        derk =  phix[k,:].dot(A[k,:]) + err[k,:]
        aux = A[k,:] + dt*derk
        A[k+1,:] = A[k,:] + .5*dt*( derk + phix[k+1,:].dot(aux) + err[k+1,:])

    optPlot = dict()    
    optPlot['mode'] = 'var'
    plotSol(sizes,t,A,B,C,constants,restrictions,optPlot)

    #print("Calculating step...")
    alfa = calcStepOdeRest(sizes,t,x,u,pi,A,B,C,constants,boundary,restrictions)
    nx = x + alfa * A

    print("Leaving oderest with alfa =",alfa)
    return nx,u,pi,lam,mu
