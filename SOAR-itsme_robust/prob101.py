#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:40:31 2017

@author: araujolma

A module for the problem 10-1 from Miele (1970)
"""

import numpy
from sgra import sgra
import matplotlib.pyplot as plt

class prob(sgra):

    def initGues(self,opt={}):
        # matrix sizes
        n = 3
        m = 1
        p = 1
        q = 1
        N = 5000+1

        self.N = N
        self.n = n
        self.m = m
        self.p = p
        self.q = q
        
        dt = 1.0/(N-1)
        t = numpy.arange(0,1.0+dt,dt)
        self.dt = dt
        self.t = t
        
        # Payload mass
        #self.mPayl = 100.0

        #prepare tolerances
        tolP = 1.0e-7#8
        tolQ = 1.0e-7#5
        tol = dict()
        tol['P'] = tolP
        tol['Q'] = tolQ
        
        self.tol = tol

        
        # Get initialization mode
        
        x = numpy.zeros((N,n))
        u = numpy.ones((N,m))
        
        x[:,0] = t.copy()
        lam = 0.0*x.copy()
        mu = numpy.zeros(q)
        pi = numpy.array([1.0])
        
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
        phi = numpy.empty((N,n))
        x = self.x
        u = self.u
        pi = self.pi
        
        sinu = numpy.sin(u[:,0])
        phi[:,0] = pi[0] * x[:,2] * numpy.cos(u[:,0])
        phi[:,1] = pi[0] * x[:,2] * sinu
        phi[:,2] = pi[0] * sinu
    
        return phi

#%%

    def calcGrads(self):
        Grads = dict()
    
        N = self.N
        n = self.n
        m = self.m
        p = self.p
        #q = sizes['q']
        #N0 = sizes['N0']
    
        x = self.x
        u = self.u
        pi = self.pi
        
        # Pre-assign functions
        
        sin = numpy.sin
        cos = numpy.cos
        array = numpy.array
        
        Grads['dt'] = 1.0/(N-1)

        phix = numpy.zeros((N,n,n))
        phiu = numpy.zeros((N,n,m))
                    
        if p>0:
            phip = numpy.zeros((N,n,p))
        else:
            phip = numpy.zeros((N,n,1))

        fx = numpy.zeros((N,n))
        fu = numpy.zeros((N,m))
        fp = numpy.zeros((N,p))        
        
        for k in range(N):
                
            # Gradients from example 10.2:
            psix = numpy.array([[1.0,0.0,0.0]])        
            psip = numpy.array([0.0])  
            cosuk = cos(u[k,0])
            sinuk = sin(u[k,0])
            zk = x[k,2]
            phix[k,:,:] = pi[0]*array([[0.0,0.0,cosuk],[0.0,0.0,sinuk],[0.0,0.0,0.0]])
            phiu[k,:,:] = pi[0]*array([[-zk*sinuk],[zk*cosuk],[cosuk]])
            phip[k,:,:] = array([[zk*cosuk],[zk*sinuk],[sinuk]])
            fp[k,0] = 1.0  
     
        Grads['phix'] = phix
        Grads['phiu'] = phiu
        Grads['phip'] = phip
        Grads['fx'] = fx
        Grads['fu'] = fu
        Grads['fp'] = fp
    #    Grads['gx'] = gx
    #    Grads['gp'] = gp
        Grads['psix'] = psix
        Grads['psip'] = psip
        return Grads

#%%
    def calcPsi(self):
        x = self.x
        N = self.N
        return numpy.array([x[N-1,0]-1.0])
        
    def calcF(self):
        N = self.N
        f = self.pi[0]*numpy.ones(N)
    
        return f

    def calcI(self):
        return self.pi[0]
#%%
    def plotSol(self,opt={},intv=[]):
        t = self.t
        x = self.x
        u = self.u
        pi = self.pi
        
        if len(intv)==0:
            intv = numpy.arange(0,self.N,1,dtype='int')
        else:
             intv = list(intv)
        
        plt.subplot2grid((8,4),(0,0),colspan=5)
        plt.plot(t[intv],x[intv,0],)
        plt.grid(True)
        plt.ylabel("x")
        
        if opt.get('mode','sol') == 'sol':
            I = self.calcI()
            titlStr = "Current solution: I = {:.4E}".format(I) + \
            " P = {:.4E} ".format(self.P) + " Q = {:.4E} ".format(self.Q)
        elif opt['mode'] == 'var':
            titlStr = "Proposed variations"
        else:
            titlStr = opt['mode']
        plt.title(titlStr)

        plt.subplot2grid((8,4),(1,0),colspan=5)
        plt.plot(t[intv],x[intv,1],'g')
        plt.grid(True)
        plt.ylabel("y")
        
        plt.subplot2grid((8,4),(2,0),colspan=5)
        plt.plot(t[intv],x[intv,2],'r')
        plt.grid(True)
        plt.ylabel("z")
        
        plt.subplot2grid((8,4),(3,0),colspan=5)
        plt.plot(t[intv],u[intv,0],'k')
        plt.grid(True)
        plt.ylabel("u")
    
        plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
        plt.show()
        print("pi =",pi,"\n")
        
    #