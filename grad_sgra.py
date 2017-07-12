#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:39:08 2017

@author: levi
"""

import numpy
from utils import ddt
import matplotlib.pyplot as plt

def calcQ(self):
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

    if self.dbugOptGrad['main']:
        print("\nDebug plots for this calcQ run:")
        
        if self.dbugOptGrad['plotQx']:
            plt.plot(self.t,normErrQx)
            plt.grid(True)
            plt.title("Integrand of Qx")
            plt.show()
        
        if self.dbugOptGrad['plotQu']:                
            plt.plot(self.t,normErrQu)
            plt.grid(True)
            plt.title("Integrand of Qu")
            plt.show()

        # for zoomed version:
        indMaxQx = normErrQx.argmax()
        ind1 = numpy.array([indMaxQx-20,0]).max()
        ind2 = numpy.array([indMaxQx+20,N-1]).min()
        
        if self.dbugOptGrad['plotQxZoom']:
            plt.plot(self.t[ind1:ind2],normErrQx[ind1:ind2],'o')
            plt.grid(True)
            plt.title("Integrand of Qx (zoom)")
            plt.show()
        
        if self.dbugOptGrad['plotSolQxMax']:
            print("\nSolution on the region of MaxQx:")
            self.plotSol(intv=numpy.arange(ind1,ind2,1,dtype='int'))
            
        # for zoomed version:
        indMaxQu = normErrQu.argmax()
        ind1 = numpy.array([indMaxQu-20,0]).max()
        ind2 = numpy.array([indMaxQu+20,N-1]).min()
        
        if self.dbugOptGrad['plotQuZoom']:
            plt.plot(self.t[ind1:ind2],normErrQu[ind1:ind2],'o')
            plt.grid(True)
            plt.title("Integrand of Qu (zoom)")
            plt.show()
        
        if self.dbugOptGrad['plotSolQuMax']:
            print("\nSolution on the region of MaxQu:")
            self.plotSol(intv=numpy.arange(ind1,ind2,1,dtype='int'))
        
        
#        if n==4 and m==2:
#            
#            plt.plot(tPlot[ind1:ind2],errQx[ind1:ind2,0])
#            plt.grid(True)
#            plt.ylabel("Qx_h")
#            plt.show()        
#    
#            plt.plot(tPlot[ind1:ind2],errQx[ind1:ind2,1],'g')
#            plt.grid(True)
#            plt.ylabel("Qx_V")
#            plt.show()
#    
#            plt.plot(tPlot[ind1:ind2],errQx[ind1:ind2,2],'r')
#            plt.grid(True)
#            plt.ylabel("Qx_gamma")
#            plt.show()
#    
#            plt.plot(tPlot[ind1:ind2],errQx[ind1:ind2,3],'m')
#            plt.grid(True)
#            plt.ylabel("Qx_m")
#            plt.show()
#            
#            print("\nStates, controls, lambda on the region of maxQx:")
#
#            plt.plot(tPlot[ind1:ind2],x[ind1:ind2,0])
#            plt.grid(True)
#            plt.ylabel("h [km]")
#            plt.show()        
#    
#            plt.plot(tPlot[ind1:ind2],x[ind1:ind2,1],'g')
#            plt.grid(True)
#            plt.ylabel("V [km/s]")
#            plt.show()
#    
#            plt.plot(tPlot[ind1:ind2],x[ind1:ind2,2]*180/numpy.pi,'r')
#            plt.grid(True)
#            plt.ylabel("gamma [deg]")
#            plt.show()
#    
#            plt.plot(tPlot[ind1:ind2],x[ind1:ind2,3],'m')
#            plt.grid(True)
#            plt.ylabel("m [kg]")
#            plt.show()
#    
#            plt.plot(tPlot[ind1:ind2],u[ind1:ind2,0],'k')
#            plt.grid(True)
#            plt.ylabel("u1 [-]")
#            plt.show()
#    
#            plt.plot(tPlot[ind1:ind2],u[ind1:ind2,1],'c')
#            plt.grid(True)
#            plt.xlabel("t")
#            plt.ylabel("u2 [-]")
#            plt.show()
#            
#            print("Lambda:")
#            
#            plt.plot(tPlot[ind1:ind2],lam[ind1:ind2,0])
#            plt.grid(True)
#            plt.ylabel("lam_h")
#            plt.show()        
#    
#            plt.plot(tPlot[ind1:ind2],lam[ind1:ind2,1],'g')
#            plt.grid(True)
#            plt.ylabel("lam_V")
#            plt.show()
#    
#            plt.plot(tPlot[ind1:ind2],lam[ind1:ind2,2],'r')
#            plt.grid(True)
#            plt.ylabel("lam_gamma")
#            plt.show()
#    
#            plt.plot(tPlot[ind1:ind2],lam[ind1:ind2,3],'m')
#            plt.grid(True)
#            plt.ylabel("lam_m")
#            plt.show()
#            
##            print("dLambda/dt:")
##            
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,0])
##            plt.grid(True)
##            plt.ylabel("dlam_h")
##            plt.show()        
##    
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,1])
##            plt.grid(True)
##            plt.ylabel("dlam_V")
##            plt.show()
##    
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,2],'r')
##            plt.grid(True)
##            plt.ylabel("dlam_gamma")
##            plt.show()
##    
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,3],'m')
##            plt.grid(True)
##            plt.ylabel("dlam_m")
##            plt.show()
##            
##            print("-phix*lambda:")
##            
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,0]-errQx[ind1:ind2,0])
##            plt.grid(True)
##            plt.ylabel("-phix*lambda_h")
##            plt.show()        
##    
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,1]-errQx[ind1:ind2,1],'g')
##            plt.grid(True)
##            plt.ylabel("-phix*lambda_V")
##            plt.show()
##    
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,2]-errQx[ind1:ind2,2],'r')
##            plt.grid(True)
##            plt.ylabel("-phix*lambda_gamma")
##            plt.show()
##    
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,3]-errQx[ind1:ind2,3],'m')
##            plt.grid(True)
##            plt.ylabel("-phix*lambda_m")
##            plt.show()
#
#            print("\nBlue: dLambda/dt; Black: -phix*lam")
#            
#            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,0])
#            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,0]-errQx[ind1:ind2,0],'k')
#            plt.grid(True)
#            plt.ylabel("z_h")
#            plt.show()        
#    
#            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,1])
#            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,1]-errQx[ind1:ind2,1],'k')
#            plt.grid(True)
#            plt.ylabel("z_V")
#            plt.show()
#    
#            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,2])
#            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,2]-errQx[ind1:ind2,2],'k')
#            plt.grid(True)
#            plt.ylabel("z_gamma")
#            plt.show()
#    
#            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,3])
#            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,3]-errQx[ind1:ind2,3],'k')
#            plt.grid(True)
#            plt.ylabel("z_m")
#            plt.show()

    return Q,Qx,Qu,Qp,Qt

def calcStepGrad(self,corr):
    print("\nIn calcStepGrad.\n")
    cont = 0
    # Get initial status (Q0, no correction applied)
    Q0,_,_,_,_ = self.calcQ(); cont += 1
    P0,_,_ = self.calcP()
    print("P0 = {:.4E}".format(P0))
    I0 = self.calcI()
    print("I0 = {:.4E}\n".format(I0))
    
    # Get status associated with integral correction (alfa=1.0)
    newSol = self.copy()
    newSol.aplyCorr(1.0,corr)
    Q1,_,_,_,_ = newSol.calcQ(); cont += 1
    P1,_,_ = newSol.calcP()    
    print("P1 = {:.4E}".format(P1))
    I1 = newSol.calcI()
    print("I1 = {:.4E}\n".format(I1))
    
    histQ = [Q1]
    histAlfa = [1.0]
    
    # Search for a better starting point for alfa, one that does not make Q
    # more than 10 times bigger.
#    Q = Q1; alfa = 1.0 #;  contGran = 0
#    while Q/Q0 >= 10.0:
#        alfa *= 0.1
#        newSol = self.copy()
#        newSol.aplyCorr(alfa,corr,dbugOpt)
#        Q,_,_,_,_ = newSol.calcQ(dbugOpt); cont += 1
#        print("alfa =",alfa,", Q = {:.6E}".format(Q),\
#              " (Q0 = {:.6E})\n".format(Q0))
#        histQ.append(Q)
#        histAlfa.append(alfa)
#        #contGran+=1

    Q = Q1; alfa = 1.0; keepLook = False; dAlfa = 1.0 #;  contGran = 0
    if Q/Q0 >= 10.0:
        print("Whoa! Going back to safe region of alphas...\n")
        keepLook = True
        dAlfa = 0.1
        cond = lambda nQ,Q: nQ/Q0>10.0
    elif Q<Q0:
        print("This seems boring. Going forward!\n")
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
        print("alfa =",alfa,", Q = {:.6E}".format(nQ),\
              " (Q0 = {:.6E})\n".format(Q0))
        histQ.append(nQ)
        histAlfa.append(alfa)
        keepLook = cond(nQ,Q)
    #
    if dAlfa > 1.0:
        alfa /= dAlfa
    
    # Now Q is not so much bigger than Q0. Start "bilateral analysis"
    print("Starting bilateral analysis...\n")
    alfa0 = alfa
    alfa = 1.2*alfa0 
    newSol = self.copy()
    newSol.aplyCorr(alfa,corr)
    QM,_,_,_,_ = newSol.calcQ(); cont += 1
    print("alfa =",alfa,", Q = {:.6E}".format(QM),\
          " (Q0 = {:.6E})\n".format(Q0))
    histQ.append(QM)
    histAlfa.append(alfa)
    
    alfa = .8*alfa0 
    newSol = self.copy()
    newSol.aplyCorr(alfa,corr)
    Qm,_,_,_,_ = newSol.calcQ(); cont += 1
    print("alfa =",alfa,", Q = {:.6E}".format(Qm),\
          " (Q0 = {:.6E})\n".format(Q0))
    histQ.append(Qm)
    histAlfa.append(alfa)
    
    # Start refined search
    
    if Qm < Q: 
        print("Beginning search for decreasing alfa...")
        # if the tendency is to still decrease alfa, do it...
        nQ = Q; keepSearch = True#(nQ<Q0)
        while keepSearch and alfa > 1.0e-15:
            Q = nQ
            alfa *= .8
            newSol = self.copy()
            newSol.aplyCorr(alfa,corr)
            nQ,_,_,_,_ = newSol.calcQ(); cont += 1
            print("alfa =",alfa,", Q = {:.6E}".format(nQ),\
                  " (Q0 = {:.6E})\n".format(Q0))
            histQ.append(nQ)
            histAlfa.append(alfa)
            print(Q0-nQ)
            if nQ < Q0:
                print("fact = ",(nQ-Q)/Q,"\n")
                keepSearch = ((nQ-Q)/Q < -.001)#nQ<Q#
        alfa /= 0.8
    else:
        if Q <= QM: 
            print("Apparently alfa =",alfa0,"is the best.")
            alfa = alfa0 # BRASIL
        else:
            print("Beginning search for increasing alfa...")
            # There still seems to be a negative gradient here. Increase alfa!
            nQ = QM
            alfa = 1.2*alfa0; keepSearch = True#(nPint>Pint1M)
            while keepSearch:
                Q = nQ
                alfa *= 1.2
                newSol = self.copy()
                newSol.aplyCorr(alfa,corr)
                nQ,_,_,_,_ = newSol.calcQ(); cont += 1
                print("alfa =",alfa,", Q = {:.4E}".format(nQ),\
                      " (Q0 = {:.4E})".format(Q0),"\n")
                histQ.append(nQ)
                histAlfa.append(alfa)
                keepSearch = nQ<Q
                #if nPint < Pint0:
            alfa /= 1.2
            
    # after all this analysis, plot the history of the tried alfas, and 
    # corresponding Q's        
    plt.loglog(histAlfa,histQ,'o')
    #plt.loglog(histAlfa[0,contGran,contGran+1],histQ[0,contGran,contGran+1],'ok')
    linhAlfa = numpy.array([min(histAlfa),max(histAlfa)])
    linQ0 = Q0 + 0.0*numpy.empty_like(linhAlfa)
    for k in range(len(histAlfa)):
        if abs(histAlfa[k]-alfa)<1e-14:
            break
    Q = histQ[k]
    plt.loglog(alfa,Q,'ok')
    plt.loglog(linhAlfa,linQ0,'--')
    plt.grid(True)
    plt.xlabel("alfa")
    plt.ylabel("Q")
    plt.title("Q versus Grad Step for current Grad run")
    plt.show()
    print("Chosen alfa = {:.4E}".format(alfa)+", Q = {:.4E}".format(Q))
    print("Number of calcQ evaluations:",cont)
    input("What now?")
      
    return alfa


def grad(self):
    
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
                
#            if dbugOpt.get('plotLamErr',False):
#                print("\nLambda Error:")
#                print("Cannot plot anymore. :( ")
                # TODO: include lambda error plotting mode
                
                #optPlot['mode'] = 'states:LambdaError'
                #plotSol(sizes,t,erroLam,numpy.zeros((N,m)),numpy.zeros(p),\
                #    constants,restrictions,optPlot)

            maxNormErroLam = normErroLam.max()
            print("maxNormErroLam =",maxNormErroLam)
#            if dbugOpt.get('plotLam',False) and (maxNormErroLam > 0):
#                plt.semilogy(normErroLam)
#                plt.grid()
#                plt.title("ErroLam")
#                plt.show()            
        
        ##################################################################            
                
        # Calculate C
        C = numpy.zeros(p)
        for k in range(1,N-1):
            C += fp[k,:] - phipTr[k,:,:].dot(lam[k,:])
        C += .5*(fp[0,:] - phipTr[0,:,:].dot(lam[0,:]))
        C += .5*(fp[N-1,:] - phipTr[N-1,:,:].dot(lam[N-1,:]))
        C *= -dt #yes, the minus sign is on purpose!
        C -= -psipTr.dot(mu)

#        if dbugOpt.get('plotLam',False):
#            print("Cannot plot lambda anymore... for now!")
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
        
#        if dbugOpt.get('plotAErr',False):
#            print("\nA Error:")
#            print("Cannot plot anymore. :( ")
            #optPlot['mode'] = 'states:AError'
            #plotSol(sizes,t,erroA,B,C,\
            #        constants,restrictions,optPlot)            
        
        maxNormErroA = normErroA.max()
        print("maxNormErroA =",maxNormErroA)
#        if dbugOpt.get('plotAErr',False) and (maxNormErroA > 0):
#            plt.semilogy(normErroA)
#            plt.grid()
#            plt.title("ErroA")
#            plt.show()
        
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
    
#    if dbugOpt.get('plotAErrFin',False):
#        print("\nFINAL A Error:")
#        print("Cannot plot anymore. :( ")
        #optPlot['mode'] = 'states:AError'
        #plotSol(sizes,t,erroA,B,C,\
        #        constants,restrictions,optPlot)    
    maxNormErroA = normErroA.max()
    
    print("FINAL maxNormErroA =",maxNormErroA)
    
#    if dbugOpt.get('plotAErrFin',False) and (maxNormErroA > 0):
#        plt.semilogy(normErroA)
#        plt.grid()
#        plt.title("ErroA")
#        plt.show()

#    if dbugOpt.get('plotLamErrFin',False):
#        print("\nFINAL Lambda Error:")
#        print("Cannot plot anymore. :( ")
        #optPlot['mode'] = 'states:LambdaError'
        #plotSol(sizes,t,erroLam,B,C,\
        #    constants,restrictions,optPlot)
        #maxNormErroLam = normErroLam.max()
    print("FINAL maxNormErroLam =",maxNormErroLam)

#    if dbugOpt.get('plotLamErrFin',False) and (maxNormErroLam > 0):
#        plt.semilogy(normErroLam)
#        plt.grid()
#        plt.title("ErroLam")
#        plt.show()

    ##########################################
    
    #if (B>numpy.pi).any() or (B<-numpy.pi).any():
    #    print("\nProblems in grad: corrections will result in control overflow.")
    
    if self.dbugOptGrad['plotCorr']:
        optPlot = {'mode':'var'}
        corrSol = self.copy()
        corrSol.x = A; corrSol.u = B; corrSol.pi = C
        corrSol.plotSol(opt=optPlot)
        #optPlot['mode'] = 'proposed (states: lambda)'
        #plotSol(sizes,t,lam,B,C,constants,restrictions,optPlot)

    self.lam = lam
    self.mu = mu
    corr = {'x':A,'u':B,'pi':C}
    # Calculation of alfa
    alfa = self.calcStepGrad(corr)

    self.aplyCorr(alfa,corr)
    self.updtHistQ(alfa)
    
    # update P just to ensure proper restoration afterwards
    P,_,_ = self.calcP()
    self.P = P
    print("Leaving grad with alfa =",alfa)
    print("Delta pi = ",alfa*corr['pi'])