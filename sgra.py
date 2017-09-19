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
        each independent solution to be added over."""
        
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
        Grads = sol.calcGrads()
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
        fu = self.fu
        phiuTr = self.phiuTr
        phipTr = self.phipTr
        DynMat = self.DynMat
        
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
            B[0,:,arc] = -rho*fu[0,:,arc] + \
                                phiuTr[0,:,:,arc].dot(lam[0,:,arc])
            phiLamIntCol += .5 * (phipTr[0,:,:,arc].dot(lam[0,:,arc]))
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
                phiLamIntCol += phipTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
            #
            phiLamIntCol -= .5*(phipTr[N-1,:,:,arc].dot(lam[N-1,:,arc]))

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
            
        # store solution in arrays
        #print("Writing to arrays in j =",j)
        
        #self.arrayA[j,:,:,:] = A.copy()#[:,:,arc]
        #self.arrayB[j,:,:,:] = B.copy()#[:,:,arc]
        #self.arrayC[j,:] = C.copy()
        #self.arrayL[j,:,:,:] = lam.copy()#[:,:,arc]
        #Ct[:,j] = C.copy()
        
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
#            arrayA[j,:,:,:] = res[j]['A']#[:,:,arc]
#            arrayB[j,:,:,:] = res[j]['B']#[:,:,arc]
#            arrayC[j,:] = res[j]['C']
#            arrayL[j,:,:,:] = res[j]['L']#lam.copy()#[:,:,arc]
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
        
        
        # Assembly of matrix M and column 'Col'
        
        # Matrix for linear system involving k's and mu's
        M = numpy.zeros((Ns+q+1,Ns+q+1))
        M[0,:(Ns+1)] = numpy.ones(Ns+1) # eq (34d)
        M[(q+1):(q+1+p),(Ns+1):] = self.psip.transpose()
        M[(p+q+1):,(Ns+1):] = self.psiy.transpose()
        # from eq (34a):
        M[1:(q+1),:(Ns+1)] = self.psiy.dot(Dt) + self.psip.dot(Ct) 
        # from eq (34b):
        M[(q+1):(q+p+1),:(Ns+1)] = Ct - phiLamInt 
        # from eq (34c):
        M[(q+p+1):,:(Ns+1)] = Et 

        # column vector for linear system involving k's and mu's  [eqs (34)]
        col = numpy.zeros(Ns+q+1)
        col[0] = 1.0 # eq (34d)
        col[1:(q+1)] = rho1 * self.psi
        # Integral term
        sumIntFpi = numpy.zeros(p)
        if self.rho > 0.0:
            for arc in range(s):
                thisInt = numpy.zeros(p)
                for ind in range(p):
                    thisInt[ind] += self.fp[:,ind,arc].sum()
                    thisInt -= .5*(self.fp[0,:,arc] + self.fp[N-1,:,arc])
                    thisInt *= self.dt
                    sumIntFpi += thisInt
        
        col[(q+1):(q+p+1)] = -self.rho * sumIntFpi
        
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
            A += K[j] * res[j]['A']#self.arrayA[j,:,:,:]
            B += K[j] * res[j]['B']#self.arrayB[j,:,:,:]
            C += K[j] * res[j]['C']#self.arrayC[j,:]
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
        
        self.tol = {'P':1e-7,'Q':1e-7}
        
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
        
        self.isParallel = dict()
        self.isParallel['gradLMPBVP'] = parallel.get('gradLMPBVP',False) 
        self.isParallel['restLMPBVP'] = parallel.get('restLMPBVP',False)
    
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
        
    def plotCat(self,func,mark='',color='b',labl=''):
        s = self.s
        t = self.t
        pi = self.pi

        # Total dimensional time
        tTot = pi.sum()
        accAdimTime = 0.0

        for arc in range(s):
            adimTimeDur = (pi[arc]/tTot)
            plt.plot(accAdimTime + adimTimeDur * t, func[:,arc],mark+color,\
                     label=labl)
            # arc beginning with circle
            plt.plot(accAdimTime + adimTimeDur*t[0], \
                     func[0,arc],'o'+color)
            # arc end with square
            plt.plot(accAdimTime + adimTimeDur*t[-1], \
                     func[-1,arc],'s'+color)
            accAdimTime += adimTimeDur    
            
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

        plt.plot(IterGrad,self.tol['P']+0.0*IterGrad,'-.b',label='tolQ')
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
        
#        if rho>0.5:
#            print("\nBeginning loop for solutions...")

        # TODO: paralelize aqui!
        if isParallel:
            pool = Pool()
            res = pool.map(helper.propagate,range(self.Ns+1))
            pool.close()
            pool.join()
        else:
            res = list()
            for j in range(self.Ns+1):
                res.append(helper.propagate(j))
            
        A,B,C,lam,mu = helper.getCorr(res)
        
        return A,B,C,lam,mu