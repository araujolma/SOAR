#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:36:59 2017

@author: levi
"""
import numpy
from utils import ddt
import matplotlib.pyplot as plt

def calcP(self):
    N = self.N
    dt = self.dt
    x = self.x
    u = self.u

    phi = self.calcPhi()
    psi = self.calcPsi()
    dx = ddt(x,N)
    
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
    
    vetP[N-1] = .5*(func[N-1,:].dot(func[N-1,:].transpose()))#norm(func[N-1,:])**2
    P += vetP[N-1]
    vetIP[N-1] = P

    P *= dt
    vetIP *= dt

    if self.dbugOptRest['main']:
        
        print("\nDebug plots for this calcP run:")
        
        indMaxP = vetP.argmax()
        ind1 = numpy.array([indMaxP-20,0]).max()
        ind2 = numpy.array([indMaxP+20,N]).min()
        
        if self.dbugOptRest['plotP_int']:
            plt.plot(self.t,vetP)
            plt.grid(True)
            plt.title("Integrand of P")
            plt.show()
    
        if self.dbugOptRest['plotIntP_int']:
            plt.plot(self.t,vetIP)
            plt.grid(True)
            plt.title("Partially integrated P")
            plt.show()

        # for zoomed version:
        if self.dbugOptRest['plotP_intZoom']:
            plt.plot(self.t[ind1:ind2],vetP[ind1:ind2],'o')
            plt.grid(True)
            plt.title("Integrand of P (zoom)")
            plt.show()
        
        if self.dbugOptRest['plotSolMaxP']:
            print("\nSolution on the region of MaxP:")
            self.plotSol(intv=numpy.arange(ind1,ind2,1,dtype='int'))
            
            # TODO: extend these debug plots
        if self.dbugOptRest['plotRsidMaxP']:
            
            print("\nResidual on the region of maxP:")
 
            if self.n==4 and self.m ==2:
                plt.plot(self.t[ind1:ind2],func[ind1:ind2,0])
                plt.grid(True)
                plt.ylabel("res_hDot [km/s]")
                plt.show()        
        
                plt.plot(self.t[ind1:ind2],func[ind1:ind2,1],'g')
                plt.grid(True)
                plt.ylabel("res_vDot [km/s/s]")
                plt.show()
        
                plt.plot(self.t[ind1:ind2],func[ind1:ind2,2]*180/numpy.pi,'r')
                plt.grid(True)
                plt.ylabel("res_gammaDot [deg/s]")
                plt.show()
        
                plt.plot(self.t[ind1:ind2],func[ind1:ind2,3],'m')
                plt.grid(True)
                plt.ylabel("res_mDot [kg/s]")
                plt.show()
            
    #            print("\nState time derivatives on the region of maxP:")
    #
    #            plt.plot(tPlot[ind1:ind2],dx[ind1:ind2,0])
    #            plt.grid(True)
    #            plt.ylabel("hDot [km/s]")
    #            plt.show()        
    #    
    #            plt.plot(tPlot[ind1:ind2],dx[ind1:ind2,1],'g')
    #            plt.grid(True)
    #            plt.ylabel("vDot [km/s/s]")
    #            plt.show()
    #    
    #            plt.plot(tPlot[ind1:ind2],dx[ind1:ind2,2]*180/numpy.pi,'r')
    #            plt.grid(True)
    #            plt.ylabel("gammaDot [deg/s]")
    #            plt.show()
    #    
    #            plt.plot(tPlot[ind1:ind2],dx[ind1:ind2,3],'m')
    #            plt.grid(True)
    #            plt.ylabel("mDot [kg/s]")
    #            plt.show()
    #            
    #            print("\nPHI on the region of maxP:")
    #
    #            plt.plot(tPlot[ind1:ind2],phi[ind1:ind2,0])
    #            plt.grid(True)
    #            plt.ylabel("hDot [km/s]")
    #            plt.show()        
    #    
    #            plt.plot(tPlot[ind1:ind2],phi[ind1:ind2,1],'g')
    #            plt.grid(True)
    #            plt.ylabel("vDot [km/s/s]")
    #            plt.show()
    #    
    #            plt.plot(tPlot[ind1:ind2],phi[ind1:ind2,2]*180/numpy.pi,'r')
    #            plt.grid(True)
    #            plt.ylabel("gammaDot [deg/s]")
    #            plt.show()
    #    
    #            plt.plot(tPlot[ind1:ind2],phi[ind1:ind2,3],'m')
    #            plt.grid(True)
    #            plt.ylabel("mDot [kg/s]")
    #            plt.show()
            else:
                 print("Not implemented (yet).")   
        #
    #        
        
    Pint = P
    Ppsi = psi.transpose().dot(psi)
    P += Ppsi
    return P,Pint,Ppsi    

def calcStepRest(self,corr):
    print("\nIn calcStepRest.\n")
    
    P0,_,_ = calcP(self)

    newSol = self.copy()
    newSol.aplyCorr(.8,corr)
    P1m,_,_ = newSol.calcP()

    newSol = self.copy()
    newSol.aplyCorr(1.0,corr)
    P1,_,_ = newSol.calcP()

    newSol = self.copy()
    newSol.aplyCorr(1.2,corr)
    P1M,_,_ = newSol.calcP()
            
    if P1 >= P1m or P1 >= P0:
        print("alfa=1.0 is too much.")
        # alfa = 1.0 is too much. Reduce alfa.
        nP = P1; alfa=1.0
        cont = 0; keepSearch = (nP>P0)
        while keepSearch and alfa > 1.0e-15:
            cont += 1
            P = nP
            alfa *= .8
            newSol = self.copy()
            newSol.aplyCorr(alfa,corr)
            nP,_,_ = newSol.calcP()
            if nP < P0:
                keepSearch = ((nP-P)/P < -.05)
    else:
        # no "overdrive!"
        #return 1.0
    
        if P1 <= P1M:
            # alfa = 1.0 is likely to be best value. 
            # Better not to waste time and return 1.0 
            return 1.0
        else:
            # There is still a descending gradient here. Increase alfa!
            nP = P1M
            cont = 0; keepSearch = True#(nPint>Pint1M)
            alfa = 1.2
            while keepSearch:
                cont += 1
                P = nP
                alfa *= 1.2
                newSol = self.copy()
                newSol.aplyCorr(alfa,corr)
                nP,_,_ = newSol.calcP()
                print("\n alfa =",alfa,", P = {:.4E}".format(nP),\
                      " (P0 = {:.4E})".format(P0))
                keepSearch = nP<P #( nP<P and alfa < 1.5)#2.0)#
                #if nPint < Pint0:
            alfa /= 1.2
    return alfa


def rest(self):
     
    print("\nIn rest, P0 = {:.4E}.".format(self.P))

    # get sizes
    N,n,m,p,q = self.N,self.n,self.m,self.p,self.q
    x = self.x
    
    # calculate phi and psi
    phi = self.calcPhi()
    psi = self.calcPsi()

    err = phi - ddt(x,N)

    # get gradients
    #print("Calc grads...")
    Grads = self.calcGrads()

    dt = 1.0/(N-1)
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

    #optPlot = dict()

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
            
            dlam = ddt(lam,N)
            erroLam = numpy.empty((N,n))
            normErroLam = numpy.empty(N)
            for k in range(N):
                erroLam[k,:] = dlam[k,:]+phix[k,:,:].transpose().dot(lam[k,:])
                normErroLam[k] = erroLam[k,:].transpose().dot(erroLam[k,:])
            maxNormErroLam = normErroLam.max()
            print("maxNormErroLam =",maxNormErroLam)
            #print("\nLambda Error:")
            #
            #optPlot['mode'] = 'states:LambdaError'
            #plotSol(sizes,t,erroLam,numpy.zeros((N,m)),numpy.zeros(p),\
            #        constants,restrictions,optPlot)
            #optPlot['mode'] = 'states:LambdaError (zoom)'
            #N1 = 0#int(N/100)-10
            #N2 = 20##N1+20
            #plotSol(sizes,t[N1:N2],erroLam[N1:N2,:],numpy.zeros((N2-N1,m)),\
            #        numpy.zeros(p),constants,restrictions,optPlot)

            #plt.semilogy(normErroLam)
            #plt.grid()
            #plt.title("ErroLam")
            #plt.show()
            
            #plt.semilogy(normErroLam[N1:N2])
            #plt.grid()
            #plt.title("ErroLam (zoom)")
            #plt.show()
            
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
            
#            optPlot['mode'] = 'states:Lambda'
#            plotSol(sizes,t,lam,B,C,constants,restrictions,optPlot)
#            
#            optPlot['mode'] = 'states:Lambda (zoom)'
#            plotSol(sizes,t[N1:N2],lam[N1:N2,:],B[N1:N2,:],C,constants,restrictions,optPlot)
            
            
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
        
        #optPlot['mode'] = 'var'
        #plotSol(sizes,t,A,B,C,constants,restrictions,optPlot)

        
        # Matrix for linear system (89)
        M[1:,i] = psix.dot(A[N-1,:]) + psip.dot(C)
    #

    # Calculations of weights k:
    print("M =",M)
    #print("col =",col)
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

    
    if self.dbugOptRest['plotCorr']:
        optPlot = {'mode':'var'}
        solCorr = self.copy()
        solCorr.x = A
        solCorr.u = B
        solCorr.pi = C
        solCorr.plotSol(opt=optPlot)
        optPlot['mode'] = 'proposed (states: lambda)'
        #plotSol(sizes,t,lam,B,C,constants,restrictions,optPlot)
    
    corr = {'x':A,
            'u':B,
            'pi':C}
    #print("Calculating step...")
    
    alfa = self.calcStepRest(corr)
    self.aplyCorr(alfa,corr)
    self.updtHistP(alfa)
    print("Leaving rest with alfa =",alfa)
    
    if self.dbugOptRest['pause']:
        input('Rest in debug mode. Press any key to continue...')