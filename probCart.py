#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:40:31 2017

@author: araujolma

A module for the cart problem: 
get the cart from position 0 to position 1 in minimal time, 
subject to restrictions on maximum acceleration and deceleration.

"""

import numpy
from sgra import sgra
import matplotlib.pyplot as plt

class prob(sgra):

    def initGues(self,opt={}):
        # matrix sizes
        n = 2
        m = 1
        p = 1
        q = 2
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
        
        #prepare tolerances
        tolP = 1.0e-7#8
        tolQ = 1.0e-7#5
        tol = dict()
        tol['P'] = tolP
        tol['Q'] = tolQ
        
        self.tol = tol

        
        # Get initialization mode
        
        x = numpy.zeros((N,n))
        u = numpy.zeros((N,m))
        
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
        
        phi[:,0] = pi[0] * x[:,1]
        phi[:,1] = pi[0] * numpy.tanh(u[:,0])
    
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
        
        tanh = numpy.tanh
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
        fp = numpy.ones((N,p))        
        
        for k in range(N):
                
            # Gradients from example 10.2:
            psix = array([[1.0,0.0],[0.0,1.0]])   
            psip = array([[0.0],[0.0]])
            phix[k,:,:] = pi[0]*array([[0.0,1.0],[0.0,0.0]])
            phiu[k,:,:] = phiu[k,:,:] = pi[0]*array([[0.0],
                                                     [1.0-tanh(u[k])**2]])
            phip[k,:,:] = array([[x[k,1]],
                             [tanh(u[k])]])
     
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
        return numpy.array([x[N-1,0]-1.0,x[N-1,1]])
        
    def calcF(self):
        N = self.N
        f = self.pi[0]*numpy.ones(N)
    
        return f

    def calcI(self):
        N = self.N
        f = self.calcF()
        I = .5*(f[0]+f[N-1])
        I += f[1:(N-1)].sum()
        I *= 1.0/(N-1)

        return I
#%%
    def plotSol(self,opt={}):
        t = self.t
        x = self.x
        u = self.u
        pi = self.pi
        
        plt.subplot2grid((8,4),(0,0),colspan=5)
        plt.plot(t,x[:,0],)
        plt.grid(True)
        plt.ylabel("x")
        
        titlStr = "Current solution: I = {:.4E}".format(self.I)
        titlStr = titlStr + " P = {:.4E} ".format(self.P)
        titlStr = titlStr + " Q = {:.4E} ".format(self.Q)

        #
        plt.title(titlStr)
        plt.subplot2grid((8,4),(1,0),colspan=5)
        plt.plot(t,x[:,1],'g')
        plt.grid(True)
        plt.ylabel("y")
        plt.subplot2grid((8,4),(2,0),colspan=5)
        plt.plot(t,u[:,0],'k')
        plt.grid(True)
        plt.ylabel("u1 [-]")
        plt.subplot2grid((8,4),(3,0),colspan=5)
        plt.plot(t,numpy.tanh(u[:,0]),'r')
        plt.grid(True)
        plt.ylabel("contr [-]")
    
        plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
        plt.show()
        print("pi =",pi,"\n")
        
    #
