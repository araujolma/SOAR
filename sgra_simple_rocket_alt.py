# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:33:27 2016

@author: levi
"""

# a file for solving rocket single stage to orbit with L=0 and D=0


import numpy, time, datetime
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from numpy.linalg import norm
from utils import interpV, interpM, ddt
from prob_rocket_sgra import declProb, calcPhi, calcPsi, calcGrads, plotSol

#from scipy.integrate import quad

# ##################
# SOLUTION DOMAIN:
# ##################

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

def calcP(sizes,x,u,pi,constants,boundary,restrictions):
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

    
    vetP[N-1] = .5*(func[N-1,:].dot(func[N-1,:].transpose()))#norm(func)**2
    P += vetP[N-1]
    vetIP[N-1] = P

    P *= dt
    vetP *= dt
    vetIP *= dt

#    tPlot = numpy.arange(0,1.0+dt,dt)

#    plt.plot(tPlot,vetP)
#    plt.grid(True)
#    plt.title("Integrand of P")
#    plt.show()

#    plt.plot(tPlot,vetIP)
#    plt.grid(True)
#    plt.title("Partially integrated P")
#    plt.show()

    Pint = P
    Ppsi = norm(psi)**2
    P += Ppsi
    #print("P = {:.4E}".format(P)+": P_int = {:.4E}".format(Pint)+", P_psi = {:.4E}".format(Ppsi))
    return P,Pint,Ppsi

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

def calcStepGrad(x,u,pi,lam,mu,A,B,C,restrictions):

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

def grad(sizes,x,u,pi,t,Q0,restrictions):
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
    Q = calcQ(sizes,nx,nu,np,lam,mu,restrictions)

    print("Leaving grad with alfa =",alfa)

    return nx,nu,np,lam,mu,Q


def calcStepRest(sizes,t,x,u,pi,A,B,C,constants,boundary,restrictions):
    
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
    
    if P1 >= P1m:
        # alfa = 1.0 is too much. Reduce alfa.
        # TODO: improve this!
        nP = P1m
        cont = 0; keepSearch = (nP>P0)
        while keepSearch and alfa > 1.0e-15:
            cont += 1
            P = nP
            alfa *= .5
            nx = x + alfa * A
            nu = u + alfa * B
            np = pi + alfa * C
            nP,nPint,nPpsi = calcP(sizes,nx,nu,np,constants,boundary,\
                                   restrictions)
            #print("\n alfa =",alfa,", P = {:.4E}".format(nP),\
            #      " (P0 = {:.4E})".format(P0))
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
                #print("\n alfa =",alfa,", P = {:.4E}".format(nP),\
                #      " (P0 = {:.4E})".format(P0))
                keepSearch = nP<P
                #if nPint < Pint0:
            alfa /= 1.2

    return alfa

def rest(sizes,x,u,pi,t,constants,boundary,restrictions):
    #print("\nIn rest.")

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
    phix = Grads['phix']
    phiu = Grads['phiu']
    phip = Grads['phip']
    fx = Grads['fx']
    psix = Grads['psix']
    psip = Grads['psip']

    #print("Preparing matrices...")
    psixTr = psix.transpose()
    fxInv = fx.copy()
    phixInv = phix.copy()
    phiuTr = numpy.empty((N,m,n))
    phipTr = numpy.empty((N,p,n))
    psipTr = psip.transpose()

    for k in range(N):
        fxInv[k,:] = fx[N-k-1,:]
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

    #optPlot = dict()

    #print("Beginning loop for solutions...")
    for i in range(q+1):
        mu = 0.0*mu
        if i<q:
            mu[i] = 1.0

            # integrate equation (75-2) backwards
            auxLamInit = - psixTr.dot(mu)
            #auxLam = odeint(calcLamDotRest,auxLamInit,t,args=(t,phixInv))
    
            auxLam = numpy.empty((N,n))
            auxLam[0,:] = auxLamInit
            for k in range(N-1):
                derk = phixInv[k,:,:].dot(auxLam[k,:])
                aux = auxLam[k,:] + dt*derk
                auxLam[k+1,:] = auxLam[k,:] + \
                                .5*dt*(derk + phixInv[k+1,:,:].dot(aux))
    
            # equation for Bi (75-3)
            B = numpy.empty((N,m))
            lam = 0*x
            for k in range(N):
                lam[k,:] = auxLam[N-k-1,:]                    
                B[k,:] = phiuTr[k,:,:].dot(lam[k,:])
            
            scal = 1.0/((numpy.absolute(B)).max())
 #           lam
  #          while B.max() > numpy.pi or B.min() < - numpy.pi:
            lam *= scal
            mu *= scal
            B *= scal
      #          print("mu =",mu)
            
            # equation for Ci (75-4)
            C = numpy.zeros(p)
            for k in range(1,N-1):
                C += phipTr[k,:,:].dot(lam[k,:])
            C += .5*(phipTr[0,:,:].dot(lam[0,:]))
            C += .5*(phipTr[N-1,:,:].dot(lam[N-1,:]))
            C *= dt
            C -= -psipTr.dot(mu)
            
            #optPlot['mode'] = 'states:Lambda'
            #plotSol(sizes,t,lam,B,C,constants,restrictions,optPlot)
            
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
            #A = odeint(calcADotRest,numpy.zeros(n),t,args= (t,phix,phiu,phip,B,C,aux))
            

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
            #A = odeint(calcADotOdeRest,numpy.zeros(n),t,args= (t,phix,aux))
        #
                        
        # store solution in arrays
        arrayA[i,:,:] = A
        arrayB[i,:,:] = B
        arrayC[i,:] = C
        arrayL[i,:,:] = lam
        arrayM[i,:] = mu
        
        #optPlot['mode'] = 'var'
        #plotSol(sizes,t,A,B,C,constants,restrictions,optPlot)
        
        # Matrix for linear system (89)
        M[1:,i] = psix.dot(A[N-1,:]) + psip.dot(C)
    #

    # Calculations of weights k:

    K = numpy.linalg.solve(M,col)
    #print("K =",K)

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

    #optPlot['mode'] = 'var'
    #plotSol(sizes,t,A,B,C,constants,restrictions,optPlot)

    #print("Calculating step...")
    alfa = calcStepRest(sizes,t,x,u,pi,A,B,C,constants,boundary,restrictions)
    nx = x + alfa * A
    nu = u + alfa * B
    np = pi + alfa * C

    #print("Leaving rest with alfa =",alfa)
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
    
    lam = 0*x        
    B = numpy.zeros((N,m))
    C = numpy.zeros(p)

    #print("Integrating ODE for A...")
    # integrate equation for A:
    A = numpy.zeros((N,n))
    for k in range(N-1):
        derk =  phix[k,:].dot(A[k,:]) + err[k,:]
        aux = A[k,:] + dt*derk
        A[k+1,:] = A[k,:] + .5*dt*( derk + phix[k+1,:].dot(aux) + err[k+1,:])
    #A = odeint(calcADotOdeRest,numpy.zeros(n),t,args= (t,phix,aux))

    #optPlot = dict()    
    #optPlot['mode'] = 'var'
    #plotSol(sizes,t,A,B,C,constants,restrictions,optPlot)

    #print("Calculating step...")
    alfa = calcStepOdeRest(sizes,t,x,u,pi,A,B,C,constants,boundary,restrictions)
    nx = x + alfa * A

    #print("Leaving oderest with alfa =",alfa)
    return nx,u,pi,lam,mu

# ##################
# MAIN SEGMENT:
# ##################
if __name__ == "__main__":
    print('--------------------------------------------------------------------------------')
    print('\nThis is SGRA_SIMPLE_ROCKET_ALT.py!')
    print(datetime.datetime.now())
    print('\n')
    
    opt = dict()
    opt['initMode'] = 'extSol'#'default'#'extSol'

    # declare problem:
    sizes,t,x,u,pi,lam,mu,tol,constants,boundary,restrictions = declProb(opt)

    Grads = calcGrads(sizes,x,u,pi,constants,restrictions)
    phi = calcPhi(sizes,x,u,pi,constants,restrictions)
    psi = calcPsi(sizes,x,boundary)
#    phix = Grads['phix']
#    phiu = Grads['phiu']
#    psix = Grads['psix']
#    psip = Grads['psip']

    print("\nProposed initial guess:\n")

    P,Pint,Ppsi = calcP(sizes,x,u,pi,constants,boundary,restrictions)
    Q = calcQ(sizes,x,u,pi,lam,mu,constants,restrictions)
    optPlot = dict()
    optPlot['P'] = P
    optPlot['Q'] = Q
    optPlot['mode'] = 'sol'
    optPlot['dispP'] = True
    optPlot['dispQ'] = False
    plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)

#    for i in range(3):
#        print("\n Doing a small restoration... \n")
#        x,u,pi,lam,mu = rest(sizes,x,u,pi,t,constants,boundary,restrictions)
#        P,Pint,Ppsi = calcP(sizes,x,u,pi,constants,boundary,restrictions)
#        plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)

    tolP = tol['P']
    tolQ = tol['Q']

    MaxIterRest = 10000
    histP = numpy.zeros(MaxIterRest)
    histPint = histP.copy()
    histPpsi = histP.copy()
    uman = numpy.linspace(0,MaxIterRest,MaxIterRest+1) #um a n

    # first restoration rounds:
    NIterRest = 0
    #mustOdeRest = True
    nOdeRest = 0#10
    histP[0] = P; histPint[0] = Pint; histPpsi[0] = Ppsi
    print("\nBeginning first restoration rounds...\n")
    while P > tolP and NIterRest < MaxIterRest:
        NIterRest += 1

#        if Ppsi/Pint < 1e-5:
#            x,u,pi,lam,mu = rest(sizes,x,u,pi,t,constants,boundary,restrictions)
#        else:
#            x,u,pi,lam,mu = oderest(sizes,x,u,pi,t,constants,boundary,restrictions)
        
#        if nOdeRest>0: #Pint > 1e-5:#NIterRest % 3 == 0:
#            x,u,pi,lam,mu = oderest(sizes,x,u,pi,t,constants,boundary,restrictions)
#            nOdeRest-=1
#            if nOdeRest==0:
#                print("Back to normal restoration.\n")
#        else: 
#            x,u,pi,lam,mu = rest(sizes,x,u,pi,t,constants,boundary,restrictions)
#            if Ppsi/Pint < 1e-5:
#                nOdeRest = 100
#                print("Ready for fast restoration!\n")

        #x,u,pi,lam,mu = rest(sizes,x,u,pi,t,constants,boundary,restrictions)        
        x,u,pi,lam,mu = oderest(sizes,x,u,pi,t,constants,boundary,restrictions)                

        P,Pint,Ppsi = calcP(sizes,x,u,pi,constants,boundary,restrictions)
        optPlot['P'] = P
        histP[NIterRest] = P
        histPint[NIterRest] = Pint
        histPpsi[NIterRest] = Ppsi
        #plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)

        if NIterRest % 50 == 0:
            plt.semilogy(uman[0:(NIterRest+1)],histP[0:(NIterRest+1)])
            plt.hold(True)
            plt.semilogy(uman[0:(NIterRest+1)],histPint[0:(NIterRest+1)],'k')
            plt.semilogy(uman[0:(NIterRest+1)],histPpsi[0:(NIterRest+1)],'r')
            plt.grid()
            plt.title("Convergence of P. black: P_int, red: P_psi, blue: P")
            plt.ylabel("P")
            plt.xlabel("Iterations")
            plt.show()
        
 #           print("\a")
#            time.sleep(.2)
            print("P = {:.4E}".format(P)+", Pint = {:.4E}".format(Pint)+\
                  ", Ppsi = {:.4E}".format(Ppsi)+"\n")
            print(datetime.datetime.now())
            print("\a So far, so good?")
            time.sleep(0.2)
            print("\a")
            time.sleep(1)
        #input("Press any key to continue...")

    print("\nAfter first rounds of restoration:")
    Q = calcQ(sizes,x,u,pi,lam,mu,constants,restrictions)
    optPlot['Q'] = Q; optPlot['dispQ'] = True
    plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)

    # Gradient rounds:
    while Q > tolQ:
        while P > tolP:
            x,u,pi,lam,mu = rest(sizes,x,u,pi,t,constants,boundary,restrictions)
            P = calcP(sizes,x,u,pi,constants,boundary,restrictions)
            optPlot['P'] = P
            plotSol(sizes,t,x,u,pi,constants,restrictions, optPlot)
        #
        x,u,pi,lam,mu,Q = grad(sizes,x,u,pi,t,Q,restrictions)
        optPlot['Q'] = Q
        plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)
    #
#
