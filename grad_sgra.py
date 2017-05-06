#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:27:55 2017

@author: levi
"""

import numpy

from scipy.integrate import odeint
from numpy.linalg import norm
#from utils_alt import ddt
from utils import interpV, interpM, ddt
from prob_rocket_sgra import calcGrads
#from prob_test import calcGrads



def calcLamDotGrad(lam,t,tVec,fxInv,phixInv):
    fxt = interpV(t,tVec,fxInv)
    phixt = interpM(t,tVec,phixInv)

    return phixt.dot(lam) - fxt

def calcADotGrad(A,t,tVec,phixVec,phiuVec,phipVec,B,C):
    phixt = interpM(t,tVec,phixVec)
    phiut = interpM(t,tVec,phiuVec)
    phipt = interpM(t,tVec,phipVec)
    Bt = interpV(t,tVec,B)

    return phixt.dot(A) + phiut.dot(Bt) + phipt.dot(C)

def calcQ(sizes,x,u,pi,lam,mu,constants,restrictions):
    # Q expression from (15)

    N = sizes['N']
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
    #psip = Grads['psip']
    dlam = ddt(sizes,lam)
    Qx = 0.0
    Qu = 0.0
    Qp = 0.0
    Qt = 0.0
    Q = 0.0
    auxVecIntQ = numpy.zeros(p)
    isnan = 0
    for k in range(1,N-1):
#        dlam[k,:] = lam[k,:]-lam[k-1,:]
        Qx += norm(dlam[k,:] - fx[k,:] + phix[k,:,:].transpose().dot(lam[k,:]))**2
        Qu += norm(fu[k,:]-phiu[k,:,:].transpose().dot(lam[k,:]))**2
        auxVecIntQ += fp[k,:] - phip[k,:,:].transpose().dot(lam[k,:])
        if numpy.math.isnan(Qx) or numpy.math.isnan(Qu):
            isnan+=1
            if isnan == 1:
                print("k_nan=",k)
   #
    Qx += .5*(norm(dlam[0,:] - fx[0,:] + phix[0,:,:].transpose().dot(lam[0,:]))**2)
    Qx += .5*(norm(dlam[N-1,:] - fx[N-1,:] + phix[N-1,:,:].transpose().dot(lam[N-1,:]))**2)

    Qu += .5*norm(fu[0,:]-phiu[0,:,:].transpose().dot(lam[0,:]))**2
    Qu += .5*norm(fu[N-1,:]-phiu[N-1,:,:].transpose().dot(lam[N-1,:]))**2

    Qx *= dt
    Qu *= dt

    auxVecIntQ += .5*(fp[0,:] - phip[0,:,:].transpose().dot(lam[0,:]))
    auxVecIntQ += .5*(fp[N-1,:] - phip[N-1,:,:].transpose().dot(lam[N-1,:]))

    auxVecIntQ *= dt
    Qp = norm(auxVecIntQ)
    Qt = norm(lam[N-1,:] + psix.transpose().dot(mu))
#"Qx =",Qx)

    Q = Qx + Qu + Qp + Qt
    print("Q = {:.4E}".format(Q)+": Qx = {:.4E}".format(Qx)+", Qu = {:.4E}".format(Qu)+", Qp = {:.4E}".format(Qp)+", Qt = {:.4E}".format(Qt))

    return Q

def calcStepGrad(sizes,x,u,pi,lam,mu,A,B,C,constants,restrictions):

    # "Trissection" method
    alfa = 1.0
    nx = x + alfa * A
    nu = u + alfa * B
    np = pi + alfa * C

    oldQ = calcQ(sizes,nx,nu,np,lam,mu,constants,restrictions)
    nQ = .9*oldQ
#    print("Q =",Q)
    alfaMin = 0.0
    alfaMax = 1.0
    cont = 0
    while (nQ-oldQ)/oldQ < -.05 and cont < 5:
        oldQ = nQ

        dalfa = (alfaMax-alfaMin)/3.0
        alfaList = numpy.array([alfaMin,alfaMin+dalfa,alfaMax-dalfa,alfaMax])
        QList = numpy.empty(numpy.shape(alfaList))

        for j in range(4):
            alfa = alfaList[j]

            nx = x + alfa * A
            nu = u + alfa * B
            np = pi + alfa * C

            Q = calcQ(sizes,nx,nu,np,lam,mu,constants,restrictions)
            QList[j] = Q
        #
        print("QList:",QList)
        minQ = QList[0]
        indxMinQ = 0

        for j in range(1,4):
            if QList[j] < minQ:
                indxMinQ = j
                minQ = QList[j]
        #

        alfa = alfaList[indxMinQ]
        nQ = QList[indxMinQ]
        print("nQ =",nQ)
        if indxMinQ == 0:
            alfaMin = alfaList[0]
            alfaMax = alfaList[1]
        elif indxMinQ == 1:
            if QList[0] < QList[2]:
                alfaMin = alfaList[0]
                alfaMax = alfaList[1]
            else:
                alfaMin = alfaList[1]
                alfaMax = alfaList[2]
        elif indxMinQ == 2:
            if QList[1] < QList[3]:
                alfaMin = alfaList[1]
                alfaMax = alfaList[2]
            else:
                alfaMin = alfaList[2]
                alfaMax = alfaList[3]
        elif indxMinQ == 3:
            alfaMin = alfaList[2]
            alfaMax = alfaList[3]

        cont+=1
    #

    return .5*(alfaMin+alfaMax)

def grad(sizes,x,u,pi,t,Q0,constants,restrictions):
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
    M = numpy.ones((q+1,q+1))

    arrayA = numpy.empty((q+1,N,n))
    arrayB = numpy.empty((q+1,N,m))
    arrayC = numpy.empty((q+1,p))
    arrayL = arrayA.copy()
    arrayM = numpy.empty((q+1,q))

    for i in range(q+1):
        mu = 0.0*mu
        if i<q:
            mu[i] = 1.0

        #print("mu =",mu)
        # integrate equation (38) backwards for lambda
        auxLamInit = - psixTr.dot(mu)
        
        #auxLam = numpy.empty((N,n))
        #auxLam[0,:] = auxLamInit
        #for k in range(N-1):
        #    auxLam[k+1,:] = auxLam[k,:] + dt*(phixInv[k,:,:].dot(auxLam[k-1,:]))
        
        auxLam = odeint(calcLamDotGrad,auxLamInit,t,args=(t,fxInv, phixInv))

        # Calculate B
        B = -fu
        lam = auxLam.copy()
        for k in range(N):
            lam[k,:] = auxLam[N-k-1,:]
            B[k,:] += phiuTr[k,:,:].dot(lam[k,:])

        # Calculate C
        C = numpy.zeros(p)
        for k in range(1,N-1):
            C += fp[k,:] - phipTr[k,:,:].dot(lam[k,:])
        C += .5*(fp[0,:] - phipTr[0,:,:].dot(lam[0,:]))
        C += .5*(fp[N-1,:] - phipTr[N-1,:,:].dot(lam[N-1,:]))
        C *= -dt
        C -= -psipTr.dot(mu)

#        plt.plot(t,B)
#        plt.grid(True)
#        plt.xlabel("t")
#        plt.ylabel("B")
#        plt.show()
#
        # integrate diff equation for A
#        A = numpy.zeros((N,n))
#        for k in range(N-1):
#            A[k+1,:] = A[k,:] + dt*(phix[k,:,:].dot(A[k,:]) +  \
#                                    phiu[k,:,:].dot(B[k,:]) + \
#                                    phip[k,:].dot(C[k,:]))

        A = odeint(calcADotGrad,numpy.zeros(n),t, args=(t,phix,phiu,phip,B,C))

#        plt.plot(t,A)
#        plt.grid(True)
#        plt.xlabel("t")
#        plt.ylabel("A")
#        plt.show()

        arrayA[i,:,:] = A
        arrayB[i,:,:] = B
        arrayC[i,:] = C
        arrayL[i,:,:] = lam
        arrayM[i,:] = mu
        M[1:,i] = psix.dot(A[N-1,:]) + psip.dot(C)
    #

    # Calculations of weights k:

    col = numpy.zeros(q+1)
    col[0] = 1.0
    K = numpy.linalg.solve(M,col)
    print("K =",K)

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

    # Calculation of alfa
    alfa = calcStepGrad(x,u,pi,lam,mu,A,B,C,restrictions)

    nx = x + alfa * A
    nu = u + alfa * B
    np = pi + alfa * C
    Q = calcQ(sizes,nx,nu,np,lam,mu,constants,restrictions)

    print("Leaving grad with alfa =",alfa)

    return nx,nu,np,lam,mu,Q
