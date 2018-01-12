#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:40:31 2017

@author: araujolma

A module for the cart problem: 
get the cart from position 0 to position 1 in minimal time, 
subject to restrictions on maximum acceleration and deceleration.

There are two subarcs, connected through the "middle", equaling position and
velocity.

"""

import numpy
from sgra import sgra
import matplotlib.pyplot as plt

class prob(sgra):
    probName = 'probCart'
    
    def initGues(self,opt={}):
        # matrix sizes
        n = 2
        m = 1
        p = 2
        q = 6#8
        s = 2
        N = 2000+1#20000+1#2000+1

        self.N = N
        self.n = n
        self.m = m
        self.p = p
        self.q = q
        self.s = s
        self.Ns = 2*n*s + p
        
        dt = 1.0/(N-1)
        t = numpy.arange(0,1.0+dt,dt)
        self.dt = dt
        self.t = t
        
        #prepare tolerances
        tolP = 1.0e-4#7#8
        tolQ = 1.0e-6#8#5
        tol = dict()
        tol['P'] = tolP
        tol['Q'] = tolQ
        
        self.tol = tol

        
        # Get initialization mode
        
        x = numpy.zeros((N,n,s))
        u = numpy.zeros((N,m,s))#5.0*numpy.ones((N,m,s))
        
        #x[:,0,0] = t.copy()
        #x[:,0,0] = .5*t
        #x[:,0,1] = .5+.5*t
        
        lam = 0.0*x
        mu = numpy.zeros(q)
        pi = 10.0*numpy.ones(p)
        
        self.x = x
        self.u = u
        self.pi = pi
        self.lam = lam
        self.mu= mu
        
        solInit = self.copy()
        
        print("\nInitialization complete.\n")        
        return solInit
#%%

    def calcPhi(self):
        N = self.N
        n = self.n
        s = self.s
        phi = numpy.empty((N,n,s))
        x = self.x
        u = self.u
        pi = self.pi
        
        for arc in range(s):
            phi[:,0,arc] = pi[arc] * x[:,1,arc]
            phi[:,1,arc] = pi[arc] * numpy.tanh(u[:,0,arc])
    
        return phi

#%%

    def calcGrads(self,calcCostTerm=False):
        Grads = dict()
    
        N,n,m,p,q,s = self.N,self.n,self.m,self.p,self.q,self.s
        x,u,pi = self.x,self.u,self.pi
            
        # Pre-assign functions
        
        tanh = numpy.tanh
        array = numpy.array
        
        Grads['dt'] = 1.0/(N-1)

        phix = numpy.zeros((N,n,n,s))
        phiu = numpy.zeros((N,n,m,s))
                    
        if p>0:
            phip = numpy.zeros((N,n,p,s))
        else:
            phip = numpy.zeros((N,n,1,s))

        fx = numpy.zeros((N,n,s))
        fu = numpy.zeros((N,m,s))
        fp = numpy.empty((N,p,s))        
                
        #psiy = numpy.eye(q,2*n*s)
        psiy = numpy.zeros((q,2*n*s))
        psiy[0,0] = 1.0
        psiy[1,1] = 1.0
        psiy[2,2] = 1.0; psiy[2,4] = -1.0
        psiy[3,3] = 1.0; psiy[3,5] = -1.0
        psiy[4,6] = 1.0
        psiy[5,7] = 1.0
        
#        psiy[0,0] = 1.0
#        psiy[1,1] = 1.0
#        psiy[1,2] = -1.0
#        psiy[2,3] = 1.0
        
        psip = numpy.zeros((q,p))
        
        Idp = numpy.eye(p)
        
        tanh_u = tanh(u)
        DynMat = array([[0.0,1.0],[0.0,0.0]]) 
        for k in range(N):
            for arc in range(s):    
                
                phix[k,:,:,arc] = pi[arc] * DynMat
                phiu[k,:,:,arc] = pi[arc] * array([[0.0],\
                                                   [1.0-tanh_u[k,0,arc]**2]])
                phip[k,:,arc,arc] = array([x[k,1,arc],\
                                           tanh_u[k,0,arc]])
            fp[k,:] = Idp
        Grads['phix'] = phix
        Grads['phiu'] = phiu
        Grads['phip'] = phip
        Grads['fx'] = fx
        Grads['fu'] = fu
        Grads['fp'] = fp
    #    Grads['gx'] = gx
    #    Grads['gp'] = gp
        Grads['psiy'] = psiy
        Grads['psip'] = psip
        return Grads

#%%
    def calcPsi(self):
        x = self.x
        N = self.N
#        return numpy.array([x[0,0,0],x[0,1,0],x[N-1,0,0]-0.5,x[N-1,1,0],\
#                            x[0,0,1]-0.5,x[0,1,1],x[N-1,0,1]-1.0,x[N-1,1,1]])
        return numpy.array([x[0,0,0],x[0,1,0],\
                            x[N-1,0,0]-x[0,0,1],x[N-1,1,0]-x[0,1,1],\
                            x[N-1,0,1]-1.0,x[N-1,1,1]])
        
    def calcF(self):
        N,s = self.N,self.s
        f = numpy.empty((N,s))
        for arc in range(s):
            f[:,arc] = self.pi[arc] * numpy.ones(N)
    
        return f

    def calcI(self):
        N,s = self.N,self.s
        f = self.calcF()

        Ivec = numpy.empty(s)
        for arc in range(s):
            Ivec[arc] = .5*(f[0,arc]+f[N-1,arc])
            Ivec[arc] += f[1:(N-1),arc].sum()
            
        Ivec *= 1.0/(N-1)
        return Ivec.sum()
#%%
            
    def plotSol(self,opt={},intv=[]):

        x = self.x
        u = self.u
        pi = self.pi
        
#        if len(intv)==0:
#            intv = numpy.arange(0,self.N,1,dtype='int')
#        else:
#             intv = list(intv)   
    
        if len(intv)>0:       
            print("plotSol: Sorry, currently ignoring plotting range.")
    

        if opt.get('mode','sol') == 'sol':
            I = self.calcI()
            titlStr = "Current solution: I = {:.4E}".format(I) + \
            " P = {:.4E} ".format(self.P) + " Q = {:.4E} ".format(self.Q)
            
            plt.subplot2grid((4,1),(0,0),colspan=5)
            self.plotCat(x[:,0,:])
            plt.grid(True)
            plt.ylabel("Position")
            plt.title(titlStr)
            plt.subplot2grid((4,1),(1,0),colspan=5)
            self.plotCat(x[:,1,:],color='g')
            plt.grid(True)
            plt.ylabel("Speed")
            plt.subplot2grid((4,1),(2,0),colspan=5)
            self.plotCat(u[:,0,:],color='k')
            plt.grid(True)
            plt.ylabel("u1 [-]")
            plt.subplot2grid((4,1),(3,0),colspan=5)
            self.plotCat(numpy.tanh(u[:,0,:]),color='k')
            plt.grid(True)
            plt.ylabel('Acceleration')    
            plt.xlabel("Concat. adim. time [-]")
        
            plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)

            self.savefig(keyName='currSol',fullName='solution')
            
            print("pi =",pi,"\n")
        elif opt['mode'] == 'var':
            dx = opt['x']
            du = opt['u']
            dp = opt['pi']

            titlStr = "Proposed variations\n"+"Delta pi: "
            for i in range(self.p):
                titlStr += "{:.4E}, ".format(dp[i])
                #titlStr += str(dp[i])+", "
                        
            plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
        
            plt.subplot2grid((4,1),(0,0))
            self.plotCat(dx[:,0,:])
            plt.grid(True)
            plt.ylabel("Position")
            plt.title(titlStr)
            
            plt.subplot2grid((4,1),(1,0))
            self.plotCat(dx[:,1,:],color='g')
            plt.grid(True)
            plt.ylabel("Speed")
                        
            plt.subplot2grid((4,1),(2,0))
            self.plotCat(du[:,0,:],color='k')
            plt.grid(True)
            plt.ylabel("u1 [-]")
            
            new_u = self.u + du
            acc = numpy.tanh(self.u)
            new_acc = numpy.tanh(new_u)
            dacc = new_acc-acc
            plt.subplot2grid((4,1),(3,0))
            self.plotCat(dacc[:,0,:],color='r')
            plt.grid(True)
            plt.xlabel("t")
            plt.ylabel("Acceleration")    
            
            self.savefig(keyName='corr',fullName='corrections')
            
        elif opt['mode'] == 'lambda':
            titlStr = "Lambda for current solution"
            
            plt.subplot2grid((4,1),(0,0),colspan=5)
            self.plotCat(self.lam[:,0,:])
            plt.grid(True)
            plt.ylabel("lambda: Position")
            plt.title(titlStr)
            plt.subplot2grid((4,1),(1,0),colspan=5)
            self.plotCat(self.lam[:,1,:],color='g')
            plt.grid(True)
            plt.ylabel("lambda: Speed")
            plt.subplot2grid((4,1),(2,0),colspan=5)
            self.plotCat(u[:,0,:],color='k')
            plt.grid(True)
            plt.ylabel("u1 [-]")
            plt.subplot2grid((4,1),(3,0),colspan=5)
            self.plotCat(numpy.tanh(u[:,0,:]),color='k')
            plt.grid(True)
            plt.ylabel('Acceleration')    
            plt.xlabel("Time [s]")
        
            plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
            self.savefig(keyName='currLamb',fullName='lambdas')
            
            print("mu =",self.mu)
            
        else:
            titlStr = opt['mode']

  
        
    #
if __name__ == "__main__":
    print("\n\nRunning probCart.py!\n")
    exmpProb = prob()
    
    print("Initializing problem:")
    exmpProb = exmpProb.initGues()
    exmpProb.printPars()
    s = exmpProb.s
    
    print("Plotting current version of solution:")
    exmpProb.plotSol()
    
    print("Calculating f:")
    f = exmpProb.calcF()
    exmpProb.plotCat(f)
    plt.grid(True)
    plt.xlabel('Concat. adim. time')
    plt.ylabel('f')
    plt.show()
    
    print("Calculating grads:")
    Grads = exmpProb.calcGrads()
    for key in Grads.keys():
        print("Grads['",key,"'] = ",Grads[key])
    
    print("Calculating I:")
    I = exmpProb.calcI()
    print("I = ",I)