#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:07:35 2017

@author: levi
"""

import rest_sgra, grad_sgra, numpy, copy, pprint
import matplotlib.pyplot as plt
from utils import ddt

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
        
        self.tol = {'P':1e-7,'Q':1e-7}
        
        # Debugging options
        tf = True
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
        
    def plotCat(self,func,color='b'):
        
        s = self.s
        t = self.t
        
        pi = self.pi
        # Total time
        tTot = pi.sum()
        accAdimTime = 0.0

        for arc in range(s):
            adimTimeDur = (pi[arc]/tTot)
            plt.plot(accAdimTime + adimTimeDur * t, func[:,arc],color)
            # arc beginning with circle
            plt.plot(accAdimTime + adimTimeDur*t[0], \
                     func[0,arc],'o'+color)
            # arc end with square
            plt.plot(accAdimTime + adimTimeDur*t[-1], \
                     func[-1,arc],'s'+color)
            accAdimTime += adimTimeDur    
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
    
    def calcI(self,*args,**kwargs):
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
        print("calcQ: not implemented yet!")
        return 1.0,1.0,1.0,1.0,1.0
#        return grad_sgra.calcQ(self,*args,**kwargs)
    
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
        
#%% LMPBVP

    def LMPBVP(self,rho=0.0):
        
        # TIRAR ISSO DEPOIS!!
        #######################################################################
        numpy.set_printoptions(threshold=500)
        #######################################################################
        
        # get sizes
        Ns,N,n,m,p,q,s = self.Ns,self.N,self.n,self.m,self.p,self.q,self.s
        rho1 = rho-1.0
        
        # calculate phi and psi
        phi = self.calcPhi()
        psi = self.calcPsi()
        
        err = phi - ddt(self.x,N)
        
        #######################################################################
        print("\nThis is err:")
        for arc in range(s):
            plt.plot(self.t,err[:,0,arc])
            plt.ylabel("errPos")
            plt.grid(True)
            plt.show()
            
            if n>1:
                plt.plot(self.t,err[:,1,arc])
                plt.ylabel("errVel")
                plt.grid(True)
                plt.show()
        #######################################################################        
        
        # get gradients
        #print("Calc grads...")
        Grads = self.calcGrads()
        dt = 1.0/(N-1)
        #dt6 = dt/6
        phix = Grads['phix']
        phiu = Grads['phiu']
        phip = Grads['phip']
        psiy = Grads['psiy']
        psip = Grads['psip']
        fx = Grads['fx']
        fu = Grads['fu']
        fp = Grads['fp']
        
        #print("Preparing matrices...")
        phixTr = numpy.empty_like(phix)
        phiuTr = numpy.empty((N,m,n,s))
        phipTr = numpy.empty((N,p,n,s))
        
        phiuFu = numpy.empty((N,n,s))
        for arc in range(s):
            for k in range(N):
                phixTr[k,:,:,arc] = phix[k,:,:,arc].transpose()
                phiuTr[k,:,:,arc] = phiu[k,:,:,arc].transpose()
                phipTr[k,:,:,arc] = phip[k,:,:,arc].transpose()
                phiuFu[k,:,arc] = phiu[k,:,:,arc].dot(fu[k,:,arc])
        
        InitCondMat = numpy.eye(Ns,Ns+1)
        
        # Matrices Ctilde, Dtilde, Etilde
        Ct = numpy.empty((p,Ns+1))
        Dt = numpy.empty((2*n*s,Ns+1))
        Et = numpy.empty((2*n*s,Ns+1))
        
        # Dynamics matrix for propagating the LSODE:
        DynMat = numpy.zeros((N,2*n,2*n,s))
        for arc in range(s):
            for k in range(N):                
                DynMat[k,:n,:n,arc] = phix[k,:,:,arc]
                DynMat[k,:n,n:,arc] = phiu[k,:,:,arc].dot(phiuTr[k,:,:,arc])
                DynMat[k,n:,n:,arc] = -phixTr[k,:,:,arc]
        
        # Matrix for linear system involving k's
        M = numpy.zeros((Ns+q+1,Ns+q+1))
        M[0,:(Ns+1)] = numpy.ones(Ns+1) # eq (34d)
        M[(q+1):(q+1+p),(Ns+1):] = psip.transpose()
        M[(p+q+1):,(Ns+1):] = psiy.transpose()
        
        phiLamInt = numpy.zeros((p,Ns+1))
        # column vector for linear system involving k's    [eqs (34)]
        col = numpy.zeros(Ns+q+1)
        col[0] = 1.0 # eq (34d)
        col[1:(q+1)] = rho1 * psi

        sumIntFpi = numpy.zeros(p)
        if rho > 0.0:
            for arc in range(s):
                thisInt = numpy.zeros(p)
                for ind in range(p):
                    thisInt[ind] += fp[:,ind,arc].sum()
                    thisInt -= .5*(fp[0,:,arc] + fp[N-1,:,arc])
                    thisInt *= dt
                    sumIntFpi += thisInt
        
        col[(q+1):(q+p+1)] = -rho * sumIntFpi

        arrayA = numpy.empty((Ns+1,N,n,s))
        arrayB = numpy.empty((Ns+1,N,m,s))
        arrayC = numpy.empty((Ns+1,p))
        arrayL = numpy.empty((Ns+1,N,n,s))
        
        #optPlot = dict()
        
        print("\nBeginning loop for solutions...")
        for j in range(Ns+1):
            print("\nIntegrating solution "+str(j+1)+" of "+str(Ns+1)+"...\n")
            
            A = numpy.zeros((N,n,s))
            B = numpy.zeros((N,m,s))
            C = numpy.zeros((p,1))
            lam = numpy.zeros((N,n,s))
            
            Xi = numpy.zeros((N,2*n,s))
            # Initial conditions for LSODE:
            for arc in range(s):
                A[0,:,arc] = InitCondMat[2*n*arc:(2*n*arc+n) , j]
                lam[0,:,arc] = InitCondMat[(2*n*arc+n):(2*n*(arc+1)) , j]     
                Xi[0,:n,arc],Xi[0,n:,arc] = A[0,:,arc],lam[0,:,arc]
            C = InitCondMat[(2*n*s):,j]
            
            # Non-homogeneous term for LSODE:
            nonHom = numpy.empty((N,2*n,s))
            for arc in range(s):
                for k in range(N):
                    # minus sign in rho1 (rho-1) is on purpose!
                    nonHA = phip[k,:,:,arc].dot(C) + \
                                -rho1*err[k,:,arc] - rho*phiuFu[k,:,arc]
                    nonHL = rho * fx[k,:,arc]
                    nonHom[k,:n,arc] = nonHA#.copy()
                    nonHom[k,n:,arc] = nonHL#.copy()
                    
            # Integrate the LSODE:
            for arc in range(s):
                B[0,:,arc] = -rho*fu[0,:,arc] + \
                                    phiuTr[0,:,:,arc].dot(lam[0,:,arc])
                phiLamInt[:,j] += .5 * (phipTr[0,:,:,arc].dot(lam[0,:,arc]))
                for k in range(N-1):
                    derXik = DynMat[k,:,:,arc].dot(Xi[k,:,arc]) + \
                            nonHom[k,:,arc]
                    aux = Xi[k,:,arc] + dt * derXik
                    Xi[k+1,:,arc] = Xi[k,:,arc] + .5 * dt * (derXik + \
                                    DynMat[k+1,:,:,arc].dot(aux) + \
                                    nonHom[k+1,:,arc])
                    A[k+1,:,arc] = Xi[k+1,:n,arc]
                    lam[k+1,:,arc] = Xi[k+1,n:,arc]
                    B[k+1,:,arc] = -rho*fu[k+1,:,arc] + \
                                    phiuTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                    phiLamInt[:,j] += phipTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                #
                phiLamInt[:,j] -= .5*(phipTr[N-1,:,:,arc].dot(lam[N-1,:,arc]))
                # Put data into matrices Dtilde and Etilde
                Dt[(2*arc)*n   : (2*arc+1)*n, j] =    A[0,:,arc]   # eq (32a)
                Dt[(2*arc+1)*n : (2*arc+2)*n, j] =    A[N-1,:,arc] # eq (32a)
                Et[(2*arc)*n   : (2*arc+1)*n, j] = -lam[0,:,arc]   # eq (32b)
                Et[(2*arc+1)*n : (2*arc+2)*n, j] =  lam[N-1,:,arc] # eq (32b)      
            #
             
###############################################################################            
            print("\nHere are the corrections for iteration " + str(j+1) + \
                  " of " + str(Ns+1) + ":\n")
            for arc in range(s):
                print("> Corrections for arc =",arc)
                
                plt.plot(self.t,lam[:,0,arc])
                plt.grid(True)
                plt.ylabel('lam: pos')
                plt.show()
                
                if n>1:
                    plt.plot(self.t,lam[:,1,arc])
                    plt.grid(True)
                    plt.ylabel('lam: vel')
                    plt.show()
                
                plt.plot(self.t,A[:,0,arc])
                plt.grid(True)
                plt.ylabel('A: pos')
                plt.show()
                
                if n>1:
                    plt.plot(self.t,A[:,1,arc])
                    plt.grid(True)
                    plt.ylabel('A: vel')
                    plt.show()
                
                plt.plot(self.t,B[:,0,arc])
                plt.grid(True)
                plt.ylabel('B')
                plt.show()
                
                print("C[arc] =",C[arc])
#                
#            input(" > ")
###############################################################################

            # store solution in arrays
            arrayA[j,:,:,:] = A.copy()#[:,:,arc]
            arrayB[j,:,:,:] = B.copy()#[:,:,arc]
            arrayC[j,:] = C.copy()
            arrayL[j,:,:,:] = lam.copy()#[:,:,arc]
            Ct[:,j] = C.copy()
        #
        
        #######################################################################
        print("\nMatrices Ct, Dt, Et:\n")
        print("Ct =",Ct)
        print("Dt =",Dt)
        print("Et =",Et)
        #######################################################################
        
        
        #All integrations ready!
        phiLamInt *= dt
        
        # Finish assembly of matrix M
        M[1:(q+1),:(Ns+1)] = psiy.dot(Dt) + psip.dot(Ct) # from eq (34a)
        M[(q+1):(q+p+1),:(Ns+1)] = Ct - phiLamInt # from eq (34b)
        M[(q+p+1):,:(Ns+1)] = Et # from eq (34c)
        
        # Calculations of weights k:        
        print("M =",M)
        print("col =",col)
        KMi = numpy.linalg.solve(M,col)
        print("Residual:",M.dot(KMi)-col)
        K,mu = KMi[:(Ns+1)], KMi[(Ns+1):]
        print("K =",K)
        print("mu =",mu)
         
        # summing up linear combinations
        A *= 0.0
        B *= 0.0
        C *= 0.0
        lam *= 0.0
        for j in range(Ns+1):
            A += K[j] * arrayA[j,:,:,:]
            B += K[j] * arrayB[j,:,:,:]
            C += K[j] * arrayC[j]
            lam += K[j] * arrayL[j,:,:,:]
            
###############################################################################        
        print("\n------------------------------------------------------------")
        print("Final corrections:\n")
        for arc in range(s):
            print("> Corrections for arc =",arc)
            plt.plot(self.t,A[:,0,arc])
            plt.grid(True)
            plt.ylabel('A: pos')
            plt.show()

            if n>1:          
                plt.plot(self.t,A[:,1,arc])
                plt.grid(True)
                plt.ylabel('A: vel')
                plt.show()
            
            plt.plot(self.t,B[:,0,arc])
            plt.grid(True)
            plt.ylabel('B')
            plt.show()
            
            print("C[arc] =",C[arc])
                
#        input(" > ")
###############################################################################        
        
        return A,B,C,lam,mu