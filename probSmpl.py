#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 19:59:29 2017

@author: levi

A module for the simplest problem ever, for debugging msgra.

"""

import numpy
from sgra import sgra
import matplotlib.pyplot as plt

class prob(sgra):

    def initGues(self,opt={}):
        # matrix sizes
        n = 1
        m = 1
        p = 2
        s = 2
        q = 3#4
        N = 1000+1

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
        tolP = 1.0e-7#8
        tolQ = 1.0e-7#5
        tol = dict()
        tol['P'] = tolP
        tol['Q'] = tolQ
        
        self.tol = tol

        
        # Get initialization mode
        
        x = numpy.zeros((N,n,s))
        u = numpy.zeros((N,m,s))
        
        x[:,0,0] = t.copy() # very bad condition
        
        lam = numpy.zeros((N,n,s))
        mu = numpy.zeros(q)
        pi = numpy.array([1.0,1.0])
        
        self.x = x
        self.u = u
        self.pi = pi
        self.lam = lam
        self.mu = mu
        
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
            phi[:,0,arc] = pi[arc] * (numpy.tanh(u[:,0,arc])-x[:,0,arc])
    
        return phi

#%%

    def calcGrads(self):
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
                
#        psiy = numpy.eye(q,2*n*s)

        psiy = numpy.zeros((q,2*n*s))
        psiy[0,0] = 1.0
        psiy[1,1] = 1.0; psiy[1,2] = -1.0
        psiy[2,3] = 1.0
#        psiy[0,0] = 1.0
#        psiy[1,1] = 1.0
#        psiy[1,2] = -1.0
#        psiy[2,3] = 1.0
        
        psip = numpy.zeros((q,p))
        
        Idp = numpy.eye(p)
        
        tanh_u = tanh(u)
        
        for k in range(N):
            for arc in range(s):                
                phix[k,:,:,arc] = -array([[pi[arc]]])
                phiu[k,:,:,arc] = array([[pi[arc]*(1.0-tanh_u[k,0,arc]**2)]])
                # The rest of the derivatives is zero 
                # (time for other arcs do not interfere with current arc)
                phip[k,:,arc,arc] = array([[tanh_u[k,0,arc]-x[k,0,arc]]])

            fp[k,:,:] = Idp
     
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
        x,N = self.x,self.N
        
#        return numpy.array([x[0,0,0], \
#                            x[N-1,0,0]-0.5,\
#                            x[0,0,1]-0.5, \
#                            x[N-1,0,1] - 1.0])
        return numpy.array([x[0,0,0], \
                            x[N-1,0,0]-x[0,0,1], \
                            x[N-1,0,1] - .5])
        
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
        #t = self.t
        x = self.x
        u = self.u
        pi = self.pi
        
#        if len(intv)==0:
#            intv = numpy.arange(0,self.N,1,dtype='int')
#        else:
#             intv = list(intv)  
             
        if len(intv)>0:       
            print("plotSol: Sorry, currently ignoring plotting range.")
        
        plt.subplot2grid((8,4),(0,0),colspan=5)
        self.plotCat(x[:,0,:])
        
        if opt.get('mode','sol') == 'sol':
            I = self.calcI()
            titlStr = "Current solution: I = {:.4E}".format(I) + \
            " P = {:.4E} ".format(self.P) + " Q = {:.4E} ".format(self.Q)
        elif opt['mode'] == 'var':
            titlStr = "Proposed variations"
        else:
            titlStr = opt['mode']
        plt.title(titlStr)
        
        plt.grid(True)
        plt.ylabel('x')    
        plt.subplot2grid((8,4),(1,0),colspan=5)
        self.plotCat(u[:,0,:],color='k')
        plt.grid(True)
        plt.ylabel('u')    

        plt.subplot2grid((8,4),(2,0),colspan=5)
        self.plotCat(numpy.tanh(u[:,0,:]),color='r')
        plt.grid(True)
        plt.ylabel('Control')    
        plt.xlabel("Concat. adim. time [-]")
    
        plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
        plt.show()
        print("pi =",pi,"\n")
        
    #
    
#%%
if __name__ == "__main__":
    print("\n\nRunning probSmpl.py!\n")
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
    