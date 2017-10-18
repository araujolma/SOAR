#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:07:35 2017

@author: levi
"""

import rest_sgra, grad_sgra, numpy, copy, pprint
import matplotlib.pyplot as plt
from utils import ddt
from multiprocessing import Pool

class binFlagDict(dict):
    """Class for binary flag dictionaries.
    Provides good grounding for any settings or options dictionary. """

    def __init__(self,inpDict={},inpName='options'):
        self.name = inpName
        
        for key in inpDict.keys():
            self[key] = inpDict[key]
        
    def setAll(self,tf=True,opt={}):
        for key in self.keys():
            self[key] = (tf and opt.get(key,True))
        
        print("\nSetting '"+self.name+"' as follows:")
        pprint.pprint(self)

class LMPBVPhelp():
    """Class for processing the Linear Multipoint Boundary Value Problem.
    The biggest advantage in using an object is that the parallelization of
    the computations for each solution is much easier."""
    
    def __init__(self,sol,rho):
        """Initialization method. Comprises all the "common" calculations for 
        each independent solution to be added over.
        
        According to the Miele (2003) convention, 
            rho = 0 for rest, and rho = 1 for grad.
        """
        
        # debug options...
        self.dbugOptGrad = sol.dbugOptGrad
        self.dbugOptRest = sol.dbugOptRest
        self.t = sol.t
        
        # get sizes
        Ns,N,m,n,p,q,s = sol.Ns,sol.N,sol.m,sol.n,sol.p,sol.q,sol.s
        self.Ns,self.N,self.m,self.n,self.p,self.q,self.s = Ns,N,m,n,p,q,s
        self.dt = 1.0/(N-1)
        self.rho = rho
        
        # calculate integration error (only if necessary)
        psi = sol.calcPsi()
        self.psi = psi
        if rho < .5:
            # calculate phi and psi
            phi = sol.calcPhi()
            err = phi - ddt(sol.x,N)
        else:
            err = numpy.zeros((N,n,s))
                
        self.err = err

        
        #######################################################################
        if rho < 0.5 and self.dbugOptRest['plotErr']:
            print("\nThis is err:")
            for arc in range(s):
                plt.plot(self.t,err[:,0,arc])
                plt.ylabel("errPos")
                plt.grid(True)
                plt.show()
                plt.clf()
                plt.close('all')

                if n>1:
                    plt.plot(self.t,err[:,1,arc])
                    plt.ylabel("errVel")
                    plt.grid(True)
                    plt.show()
                    plt.clf()
                    plt.close('all')
        #######################################################################        
        
        # Get gradients
        Grads = sol.calcGrads(calcCostTerm=(rho>0.5))
        #dt6 = dt/6
        phix = Grads['phix']
        phiu = Grads['phiu']
        phip = Grads['phip']
        psiy = Grads['psiy']
        psip = Grads['psip']
        fx = Grads['fx']
        fu = Grads['fu']
        fp = Grads['fp']
        
        self.phip = phip
        self.psiy = psiy
        self.psip = psip
        self.fx = fx
        self.fu = fu
        self.fp = fp
        
        # Prepare matrices with derivatives:
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
        self.phiuFu = phiuFu
        self.phiuTr = phiuTr
        self.phipTr = phipTr
        
        InitCondMat = numpy.eye(Ns,Ns+1)
        self.InitCondMat = InitCondMat
        
        # Dynamics matrix for propagating the LSODE:
        DynMat = numpy.zeros((N,2*n,2*n,s))
        for arc in range(s):
            for k in range(N):                
                DynMat[k,:n,:n,arc] = phix[k,:,:,arc]
                DynMat[k,:n,n:,arc] = phiu[k,:,:,arc].dot(phiuTr[k,:,:,arc])
                DynMat[k,n:,n:,arc] = -phixTr[k,:,:,arc]
        self.DynMat = DynMat

    def propagate(self,j):
        """This method computes each solution, via propagation of the 
        applicable Linear System of Ordinary Differential Equations."""
        
        # Load data (sizes, common matrices, etc)
        rho = self.rho
        rho1 = self.rho-1.0
        Ns,N,n,m,p,s = self.Ns,self.N,self.n,self.m,self.p,self.s
        dt = self.dt
        
        InitCondMat = self.InitCondMat
        phip = self.phip
        err = self.err
        phiuFu = self.phiuFu
        fx = self.fx
        if rho > .5:
            rhoFu = self.fu
        else:
            rhoFu = numpy.zeros((N,m,s))

        phiuTr = self.phiuTr
        phipTr = self.phipTr
        DynMat = self.DynMat
        I = numpy.eye(2*n)
        
        #if rho > 0.5:
        #    print("\nIntegrating solution "+str(j+1)+" of "+str(Ns+1)+"...\n")
        
        # Declare matrices for corrections
        phiLamIntCol = numpy.zeros(p)
        DtCol = numpy.empty(2*n*s)
        EtCol = numpy.empty(2*n*s)
        A = numpy.zeros((N,n,s))
        B = numpy.zeros((N,m,s))
        C = numpy.zeros((p,1))
        lam = numpy.zeros((N,n,s))
        
        # the vector that will be integrated is Xi = [A; lam]
        Xi = numpy.zeros((N,2*n,s))
        # Initial conditions for the LSODE:
        for arc in range(s):
            A[0,:,arc] = InitCondMat[2*n*arc:(2*n*arc+n) , j]
            lam[0,:,arc] = InitCondMat[(2*n*arc+n):(2*n*(arc+1)) , j]     
            Xi[0,:n,arc],Xi[0,n:,arc] = A[0,:,arc],lam[0,:,arc]
        C = InitCondMat[(2*n*s):,j]
        
        # Non-homogeneous terms for the LSODE:
        nonHom = numpy.empty((N,2*n,s))
        for arc in range(s):
            for k in range(N):
                # minus sign in rho1 (rho-1) is on purpose!
                nonHA = phip[k,:,:,arc].dot(C) + \
                            -rho1*err[k,:,arc] - rho*phiuFu[k,:,arc]
                nonHL = rho * fx[k,:,arc]
                nonHom[k,:n,arc] = nonHA#.copy()
                nonHom[k,n:,arc] = nonHL#.copy()
                
        # Integrate the LSODE (by Heun's method):
        for arc in range(s):
#            B[0,:,arc] = -rhoFu[0,:,arc] + \
#                                phiuTr[0,:,:,arc].dot(lam[0,:,arc])
#            phiLamIntCol += .5 * (phipTr[0,:,:,arc].dot(lam[0,:,arc]))
#            
#            # First point: simple propagation
#            derXik = DynMat[0,:,:,arc].dot(Xi[0,:,arc]) + \
#                        nonHom[0,:,arc]
#            Xi[1,:,arc] = Xi[0,:,arc] + dt * derXik
#            #A[1,:,arc] = Xi[1,:n,arc]
#            lam[1,:,arc] = Xi[1,n:,arc]
#            B[1,:,arc] = -rhoFu[1,:,arc] + \
#                            phiuTr[1,:,:,arc].dot(lam[1,:,arc])
#            phiLamIntCol += phipTr[1,:,:,arc].dot(lam[1,:,arc])
#            
#            # "Middle" points: original Heun propagation
#            for k in range(1,N-2):
#                derXik = DynMat[k,:,:,arc].dot(Xi[k,:,arc]) + \
#                        nonHom[k,:,arc]
#                aux = Xi[k,:,arc] + dt * derXik
#                Xi[k+1,:,arc] = Xi[k,:,arc] + .5 * dt * (derXik + \
#                                DynMat[k+1,:,:,arc].dot(aux) + \
#                                nonHom[k+1,:,arc])
#                #A[k+1,:,arc] = Xi[k+1,:n,arc]
#                lam[k+1,:,arc] = Xi[k+1,n:,arc]
#                B[k+1,:,arc] = -rhoFu[k+1,:,arc] + \
#                                phiuTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
#                phiLamIntCol += phipTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
#            #
#            
#            # Last point: simple propagation, but based on the last point
#            derXik = DynMat[N-1,:,:,arc].dot(Xi[N-2,:,arc]) + \
#                        nonHom[N-1,:,arc]
#            Xi[N-1,:,arc] = Xi[N-2,:,arc] + dt * derXik
#            #A[N-1,:,arc] = Xi[N-1,:n,arc]
#            lam[N-1,:,arc] = Xi[N-1,n:,arc]
#            B[N-1,:,arc] = -rhoFu[N-1,:,arc] + \
#                            phiuTr[N-1,:,:,arc].dot(lam[N-1,:,arc])
#            phiLamIntCol += .5*phipTr[N-1,:,:,arc].dot(lam[N-1,:,arc])
            
            # Integrate the SODE by Euler Backwards implicit
            B[0,:,arc] = -rhoFu[0,:,arc] + \
                                phiuTr[0,:,:,arc].dot(lam[0,:,arc])
            phiLamIntCol += .5 * (phipTr[0,:,:,arc].dot(lam[0,:,arc]))
            
            for k in range(N-1):
                Xi[k+1,:,arc] = numpy.linalg.solve(I - dt*DynMat[k+1,:,:,arc],\
                  Xi[k,:,arc] + dt*nonHom[k+1,:,arc])
                lam[k+1,:,arc] = Xi[k+1,n:,arc]
                B[k+1,:,arc] = -rhoFu[k+1,:,arc] + \
                                phiuTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                phiLamIntCol += phipTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                
            phiLamIntCol -= .5*phipTr[N-1,:,:,arc].dot(lam[N-1,:,arc])            
            

            # Get the A values from Xi
            A[:,:,arc] = Xi[:,:n,arc]

            # Put initial and final conditions of A and Lambda into matrices 
            # DtCol and EtCol, which represent the columns of Dtilde(Dt) and 
            # Etilde(Et) 
            DtCol[(2*arc)*n   : (2*arc+1)*n] =    A[0,:,arc]   # eq (32a)
            DtCol[(2*arc+1)*n : (2*arc+2)*n] =    A[N-1,:,arc] # eq (32a)
            EtCol[(2*arc)*n   : (2*arc+1)*n] = -lam[0,:,arc]   # eq (32b)
            EtCol[(2*arc+1)*n : (2*arc+2)*n] =  lam[N-1,:,arc] # eq (32b)      
        #

        # All integrations ready!
        phiLamIntCol *= dt
         
###############################################################################  
        if (rho > 0.5 and self.dbugOptGrad['plotCorr']) or \
           (rho < 0.5 and self.dbugOptRest['plotCorr']):          
            print("\nHere are the corrections for iteration " + str(j+1) + \
                  " of " + str(Ns+1) + ":\n")
            for arc in range(s):
                print("> Corrections for arc =",arc)
                plt.plot(self.t,A[:,0,arc])
                plt.grid(True)
                plt.ylabel('A: pos')
                plt.show()
                plt.clf()
                plt.close('all')

                
                plt.plot(self.t,lam[:,0,arc])
                plt.grid(True)
                plt.ylabel('lambda: pos')
                plt.show()
                plt.clf()
                plt.close('all')
    
                if n>1:          
                    plt.plot(self.t,A[:,1,arc])
                    plt.grid(True)
                    plt.ylabel('A: vel')
                    plt.show()
                    plt.clf()
                    plt.close('all')
                    
                    plt.plot(self.t,lam[:,1,arc])
                    plt.grid(True)
                    plt.ylabel('lambda: vel')
                    plt.show()
                    plt.clf()
                    plt.close('all')

                if n>2:          
                    plt.plot(self.t,A[:,2,arc])
                    plt.grid(True)
                    plt.ylabel('A: gama')
                    plt.show()
                    plt.clf()
                    plt.close('all')
                    
                    plt.plot(self.t,lam[:,2,arc])
                    plt.grid(True)
                    plt.ylabel('lambda: gamma')
                    plt.show()
                    plt.clf()
                    plt.close('all')

                
                if n>3:          
                    plt.plot(self.t,A[:,3,arc])
                    plt.grid(True)
                    plt.ylabel('A: m')
                    plt.show()
                    plt.clf()
                    plt.close('all')
                    
                    plt.plot(self.t,lam[:,3,arc])
                    plt.grid(True)
                    plt.ylabel('lambda: m')
                    plt.show()
                    plt.clf()
                    plt.close('all')

                
                plt.plot(self.t,B[:,0,arc])
                plt.grid(True)
                plt.ylabel('B0')
                plt.show()
                plt.clf()
                plt.close('all')

                if m>1:
                    plt.plot(self.t,B[:,1,arc])
                    plt.grid(True)
                    plt.ylabel('B1')
                    plt.show()
                    plt.clf()
                    plt.close('all')
                
                print("C[arc] =",C[arc])
                #input(" > ")
###############################################################################
        
        # All the outputs go to main output dictionary; the final solution is 
        # computed by the next method, 'getCorr'.
        outp = {'A':A,'B':B,'C':C,'L':lam,'Dt':DtCol,'Et':EtCol,
                'phiLam':phiLamIntCol}
    
        return outp
    
    
    def getCorr(self,res):
        """ Computes the actual correction for this grad/rest step, by linear
        combination of the solutions generated by method 'propagate'."""
        
        # Get sizes
        Ns,N,n,m,p,q,s = self.Ns,self.N,self.n,self.m,self.p,self.q,self.s
        rho1 = self.rho - 1.0
        
        # Declare matrices Ctilde, Dtilde, Etilde, and the integral term
        Ct = numpy.empty((p,Ns+1))
        Dt = numpy.empty((2*n*s,Ns+1))
        Et = numpy.empty((2*n*s,Ns+1))
        phiLamInt = numpy.empty((p,Ns+1))
        
        # Unpack outputs from 'propagate' into proper matrices Ct, Dt, etc.
        for j in range(Ns+1):
            Ct[:,j] = res[j]['C']
            Dt[:,j] = res[j]['Dt']
            Et[:,j] = res[j]['Et']
            phiLamInt[:,j] = res[j]['phiLam']

###############################################################################
#        if self.rho > 0.5:
#            print("\nMatrices Ct, Dt, Et:\n")
#            print("Ct =",Ct)
#            print("Dt =",Dt)
#            print("Et =",Et)
###############################################################################
        
        
        # Assembly of matrix M and column 'Col' for the linear system
        
        # Matrix for linear system involving k's and mu's
        M = numpy.zeros((Ns+q+1,Ns+q+1))
        # from eq (34d) - k term
        M[0,:(Ns+1)] = numpy.ones(Ns+1)
        # from eq (34b) - mu term
        M[(q+1):(q+1+p),(Ns+1):] = self.psip.transpose() 
        # from eq (34c) - mu term
        M[(p+q+1):,(Ns+1):] = self.psiy.transpose()
        # from eq (34a) - k term
        M[1:(q+1),:(Ns+1)] = self.psiy.dot(Dt) + self.psip.dot(Ct) 
        # from eq (34b) - k term
        M[(q+1):(q+p+1),:(Ns+1)] = Ct - phiLamInt 
        # from eq (34c) - k term
        M[(q+p+1):,:(Ns+1)] = Et 

        # column vector for linear system involving k's and mu's  [eqs (34)]
        col = numpy.zeros(Ns+q+1)
        col[0] = 1.0 # eq (34d)
        
        # Integral term
        if self.rho > 0.5:
            # eq (34b) - only applicable for grad
            sumIntFpi = numpy.zeros(p)
            for arc in range(s):
                for ind in range(p):
                    sumIntFpi[ind] += self.fp[:,ind,arc].sum()
                    sumIntFpi[ind] -= .5 * ( self.fp[0,ind,arc] + \
                             self.fp[-1,ind,arc])
            sumIntFpi *= self.dt
            col[(q+1):(q+p+1)] = -self.rho * sumIntFpi
        else:
            # eq (34a) - only applicable for rest
            col[1:(q+1)] = rho1 * self.psi

        
        # Calculations of weights k:        
#        print("M =",M)
#        print("col =",self.col)
        KMi = numpy.linalg.solve(M,col)
        Res = M.dot(KMi)-col
        print("Residual of the Linear System:",Res.transpose().dot(Res))
        K,mu = KMi[:(Ns+1)], KMi[(Ns+1):]
#        print("K =",K)
#        print("mu =",mu)   

        # summing up linear combinations
        A = numpy.zeros((N,n,s))
        B = numpy.zeros((N,m,s))
        C = numpy.zeros(p)
        lam = numpy.zeros((N,n,s))
        
        for j in range(Ns+1):
            A   += K[j] * res[j]['A']#self.arrayA[j,:,:,:]
            B   += K[j] * res[j]['B']#self.arrayB[j,:,:,:]
            C   += K[j] * res[j]['C']#self.arrayC[j,:]
            lam += K[j] * res[j]['L']#self.arrayL[j,:,:,:]
            
###############################################################################        
        if (self.rho > 0.5 and self.dbugOptGrad['plotCorrFin']) or \
           (self.rho < 0.5 and self.dbugOptRest['plotCorrFin']):
            print("\n------------------------------------------------------------")
            print("Final corrections:\n")
            for arc in range(s):
                print("> Corrections for arc =",arc)
                plt.plot(self.t,A[:,0,arc])
                plt.grid(True)
                plt.ylabel('A: pos')
                plt.show()
                plt.clf()
                plt.close('all')
    
                if n>1:          
                    plt.plot(self.t,A[:,1,arc])
                    plt.grid(True)
                    plt.ylabel('A: vel')
                    plt.show()
                    plt.clf()
                    plt.close('all')

                if n>2:          
                    plt.plot(self.t,A[:,2,arc])
                    plt.grid(True)
                    plt.ylabel('A: gama')
                    plt.show()
                    plt.clf()
                    plt.close('all')
                
                if n>3:          
                    plt.plot(self.t,A[:,3,arc])
                    plt.grid(True)
                    plt.ylabel('A: m')
                    plt.show()
                    plt.clf()
                    plt.close('all')
                
                plt.plot(self.t,B[:,0,arc])
                plt.grid(True)
                plt.ylabel('B0')
                plt.show()
                plt.clf()
                plt.close('all')

                if m>1:
                    plt.plot(self.t,B[:,1,arc])
                    plt.grid(True)
                    plt.ylabel('B1')
                    plt.show()
                    plt.clf()
                    plt.close('all')

                
                print("C[arc] =",C[arc])
                    
            #input(" > ")
###############################################################################        
        return A,B,C,lam,mu

class sgra():
    """Class for a general instance of the SGRA problem. 
    
    Here are all the methods and variables that are independent of a specific
    instance of a problem. 
    
    Each instance of an optimization problem must then inherit these methods 
    and properties. """
    
    probName = 'probSGRA'
    
    def __init__(self,parallel={}):
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
        MaxIterRest = 100000
        self.MaxIterRest = MaxIterRest
        self.NIterRest = 0
        self.histStepRest = numpy.zeros(MaxIterRest)
        self.histP = numpy.zeros(MaxIterRest)
        self.histPint = numpy.zeros(MaxIterRest)
        self.histPpsi = numpy.zeros(MaxIterRest)
        
        MaxIterGrad = 10000
        self.MaxIterGrad = MaxIterGrad
        self.NIterGrad = 0
        self.histStepGrad = numpy.zeros(MaxIterGrad)
        self.histQ = numpy.zeros(MaxIterGrad)
        self.histQx = numpy.zeros(MaxIterGrad)
        self.histQu = numpy.zeros(MaxIterGrad)
        self.histQp = numpy.zeros(MaxIterGrad)
        self.histQt = numpy.zeros(MaxIterGrad)

        self.histI = numpy.zeros(MaxIterGrad)
        self.histGRrate = numpy.zeros(MaxIterGrad)
        
        self.tol = {'P':1e-7,'Q':1e-7}
        
        # Gradient-Restoration EVent List (1-grad, 0-rest)
        self.GREvList = numpy.ones(MaxIterRest+MaxIterGrad,dtype='bool')
        self.GREvIndx = -1
        
        # Debugging options
        tf = False
        self.dbugOptRest = binFlagDict(inpDict={'pausRest':tf,
                            'pausCalcP':tf,
                            'plotP_int':tf,
                            'plotP_intZoom':tf,
                            'plotIntP_int':tf,
                            'plotSolMaxP':tf,
                            'plotRsidMaxP':tf,
                            'plotErr':tf,
                            'plotCorr':tf,
                            'plotCorrFin':tf},\
                                        inpName='Debug options for Rest')
        tf = False#True#
        self.dbugOptGrad = binFlagDict(inpDict={'pausGrad':tf,
                            'pausCalcQ':tf,
                            'prntCalcStepGrad':tf,
                            'plotCalcStepGrad': tf,
                            'pausCalcStepGrad':tf,
                            'plotQx':tf,
                            'plotQu':tf,
                            'plotLam':tf,
                            'plotQxZoom':tf,
                            'plotQuZoom':tf,
                            'plotQuComp':tf,
                            'plotQuCompZoom':tf,
                            'plotSolQxMax':tf,
                            'plotSolQuMax':tf,
                            'plotCorr':tf,
                            'plotCorrFin':tf,
                            'plotF':tf,
                            'plotFint':tf},\
                                        inpName='Debug options for Grad')

        # Solution plot saving status:
        self.save = binFlagDict(inpDict={'currSol':True,
                     'histP':True,
                     'histQ':True,
                     'histI':True,
                     'histGradStep':True,
                     'traj':True,
                     'comp':True},\
                                inpName='Plot saving options')

        # Paralellism options        
        self.isParallel = dict()
        self.isParallel['gradLMPBVP'] = parallel.get('gradLMPBVP',False) 
        self.isParallel['restLMPBVP'] = parallel.get('restLMPBVP',False)

#    def getAvrgGRrate(self,nGrads=self.NIterGrad):
#        
#        # find last gradient step index in event list
#        LastGradIndx = self.GREvIndx
#        while self.GREvList[LastGradIndx] == False:
#            LastGradIndx -= 1
    def updtGRrate(self):
#        print("\nIn updtGRrate.")
 
        # find last gradient step index in event list
        LastGradIndx = self.GREvIndx - 1
#        print(LastGradIndx)
        while self.GREvList[LastGradIndx] == False and LastGradIndx > 0:
            LastGradIndx -= 1
#            print(LastGradIndx)

        # number of restorations after last gradient step            
        if LastGradIndx == 0:
            nRest = self.GREvIndx-1
        else:
            nRest = self.GREvIndx - 1 - LastGradIndx
        
        self.histGRrate[self.NIterGrad] = nRest
#        print("\nUpdating GR rate...")
#        print("Writing nRest =",nRest,'at position #',self.NIterGrad)
        
    
    def copy(self):
        return copy.deepcopy(self)
    
    def aplyCorr(self,alfa,corr):
        print("\nApplying alfa =",alfa)
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
        
    def plotCat(self,func,mark='',color='b',labl='',piIsTime=True):
        """Plot a given function with several subarcs.
            Since this function should serve all the SGRA instances, the pi
            parameters (if exist!) are not necessarily the times for each 
            subarc. Hence, the optional parameter "piIsTime".
        """
        
#        s = self.s
#        t = self.t
#        pi = self.pi
#
#        # Total dimensional time
#        if piIsTime:
#            tTot = pi.sum()
#        else:
#            tTot = t[-1]*s
#            adimTimeDur = t[-1]
#        
#        accAdimTime = 0.0
#
#        for arc in range(s):
#            if piIsTime:
#                adimTimeDur = (pi[arc]/tTot)
#        
#            # Plot the function at each arc. Label only the first arc
#            if arc == 0:
#                plt.plot(accAdimTime + adimTimeDur * t, func[:,arc],\
#                         mark+color,label=labl)
#            else:
#                plt.plot(accAdimTime + adimTimeDur * t, func[:,arc],\
#                         mark+color)
#            # arc beginning with circle
#            plt.plot(accAdimTime + adimTimeDur*t[0], \
#                     func[0,arc],'o'+color)
#            # arc end with square
#            plt.plot(accAdimTime + adimTimeDur*t[-1], \
#                     func[-1,arc],'s'+color)
#            accAdimTime += adimTimeDur    
        
        s = self.s
        t = self.t
        pi = self.pi

        # Total dimensional time
        if not (piIsTime):
            TimeDur = t[-1]
        
        accTime = 0.0

        for arc in range(s):
            if piIsTime:
                TimeDur = pi[arc]
        
            # Plot the function at each arc. Label only the first arc
            if arc == 0:
                plt.plot(accTime + TimeDur * t, func[:,arc],\
                         mark+color,label=labl)
            else:
                plt.plot(accTime + TimeDur * t, func[:,arc],\
                         mark+color)
            # arc beginning with circle
            plt.plot(accTime + TimeDur*t[0], \
                     func[0,arc],'o'+color)
            # arc end with square
            plt.plot(accTime + TimeDur*t[-1], \
                     func[-1,arc],'s'+color)
            accTime += TimeDur
            
    def savefig(self,keyName='',fullName=''):
        if self.save.get(keyName,'False'):
            print('Saving ' + fullName + ' plot to ' + \
                  self.probName + '_'+keyName+'.pdf!')
            try:
                plt.savefig(self.probName+'_'+keyName+'.pdf',\
                            bbox_inches='tight', pad_inches=0.1)
            except:
                print("Sorry, pdf saving failed... Are you using Windows?")
                print("Anyway, you can always load the object and use some "+ \
                      "of its plotting methods later, I guess.")
        else:
            plt.show()
        #
        plt.clf()
        plt.close('all')
#%% Just for avoiding compatibilization issues with other problems
    # These methods are all properly implemented in probRock class.
    
    def plotTraj(self):
        print("plotTraj: unimplemented method.")
        pass
    
    def compWith(self,*args,**kwargs):
        print("compWith: unimplemented method.")
        pass
    
    def plotSol(self,*args,**kwargs):
        titlStr = "Current solution"
            
        plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
        Np = self.n + self.m

        
        # First state
        plt.subplot2grid((Np,1),(0,0))
        self.plotCat(self.x[:,0,:],piIsTime=False)
        plt.grid(True)
        plt.ylabel("State #1")
        plt.title(titlStr)

        ind = 1
        for i in range(1,self.n):
            plt.subplot2grid((Np,1),(ind,0))
            ind+=1
            self.plotCat(self.x[:,i,:],piIsTime=False)
            plt.grid(True)
            plt.ylabel("State #"+str(i+1))

        # Controls
        for i in range(self.m):
            plt.subplot2grid((Np,1),(ind,0))
            ind+=1
            self.plotCat(self.u[:,i,:],piIsTime=False)
            plt.grid(True)
            plt.ylabel("Control #"+str(i+1))
        
        self.savefig(keyName='currSol',fullName='solution')
    
    def calcI(self,*args,**kwargs):
        pass
    
#%% RESTORATION-WISE METHODS
    
    def rest(self,*args,**kwargs):
        rest_sgra.rest(self,*args,**kwargs)

    def calcStepRest(self,*args,**kwargs):
        return rest_sgra.calcStepRest(self,*args,**kwargs)        
        
    def calcP(self,*args,**kwargs):
        return rest_sgra.calcP(self,*args,**kwargs)   
    
    def updtHistP(self,alfa,mustPlotPint=False):
        
        NIterRest = self.NIterRest+1

        P,Pint,Ppsi = self.calcP(mustPlotPint=mustPlotPint)
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
        plt.title("Convergence report on P")
        plt.grid(True)
        plt.xlabel("Rest iterations")
        plt.ylabel("P values")
        plt.legend()
        
        self.savefig(keyName='histP',fullName='P')

#%% GRADIENT-WISE METHODS

    def grad(self,*args,**kwargs):
        grad_sgra.grad(self,*args,**kwargs)
        
    def calcStepGrad(self,*args,**kwargs):
        return grad_sgra.calcStepGrad(self,*args,**kwargs)

    def calcQ(self,*args,**kwargs):
        #print("calcQ: not implemented yet!")
        #return 1.0,1.0,1.0,1.0,1.0
        return grad_sgra.calcQ(self,*args,**kwargs)

    def plotQRes(self,args):
        return grad_sgra.plotQRes(self,args)
    
    def updtHistQ(self,alfa,mustPlotQs=False):
    
        
        NIterGrad = self.NIterGrad+1
        
        Q,Qx,Qu,Qp,Qt = self.calcQ(mustPlotQs=mustPlotQs)
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

        plt.plot(IterGrad,self.tol['Q']+0.0*IterGrad,'-.b',label='tolQ')
        plt.title("Convergence report on Q")
        plt.grid(True)
        plt.xlabel("Grad iterations")
        plt.ylabel("Q values")
        plt.legend()
        
        self.savefig(keyName='histQ',fullName='Q convergence history')
        
    def showHistI(self):
        IterGrad = numpy.arange(1,self.NIterGrad+1,1)
        
        plt.title("Convergence report on I")
        plt.plot(IterGrad,self.histI[IterGrad])
        plt.grid(True)
        plt.xlabel("Grad iterations")
        plt.ylabel("I values")
        
        self.savefig(keyName='histI',fullName='I convergence history')

    def showHistGradStep(self):
        IterGrad = numpy.arange(1,self.NIterGrad+1,1)
        
        plt.title("Gradient step history")
        plt.semilogy(IterGrad,self.histStepGrad[IterGrad])
        plt.grid(True)
        plt.xlabel("Grad iterations")
        plt.ylabel("Step values")
        
        self.savefig(keyName='histGradStep',fullName='GradStep convergence history')

        
#%% LMPBVP

    def LMPBVP(self,rho=0.0,isParallel=False):
        
        helper = LMPBVPhelp(self,rho)


        if isParallel:
            pool = Pool()
            res = pool.map(helper.propagate,range(self.Ns+1))
            pool.close()
            pool.join()
        else:
            if rho>0.5:
                print("\nRunning GRAD in sequential (non-parallel) mode...\n")
            else:
                print("\nRunning REST in sequential (non-parallel) mode...\n")
            res = list()
            for j in range(self.Ns+1):
                res.append(helper.propagate(j))
            
        A,B,C,lam,mu = helper.getCorr(res)
        
        return A,B,C,lam,mu