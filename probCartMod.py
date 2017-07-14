#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:40:31 2017

@author: araujolma

A module for the modified cart problem: 
get the cart from position 0 to position 1, and back to 0
(stopping a all these extremes)  in minimal time, 
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
        s = 2
        q = 2 + 2 + 2 + 2
        N = 5000+1

        self.N = N
        self.n = n
        self.m = m
        self.p = p
        self.q = q
        self.s = s
        
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
        
#        x[:,0,0] = t.copy()
        
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
            phi[:,0,arc] = pi[arc] * x[:,1,arc]
            phi[:,1,arc] = pi[arc] * numpy.tanh(u[:,0,arc])
    
        return phi

#%%

    def calcGrads(self):
        Grads = dict()
    
        N = self.N
        n = self.n
        m = self.m
        p = self.p
        s = self.s
        #q = sizes['q']
        #N0 = sizes['N0']
    
        x = self.x
        u = self.u
        pi = self.pi
        
        # Pre-assign functions
        
        tanh = numpy.tanh
        array = numpy.array
        
        Grads['dt'] = 1.0/(N-1)

        phix = numpy.zeros((N,n,n,s))
        phiu = numpy.zeros((N,n,m,s))
                    
        if p>0:
            phip = numpy.zeros((N,n,p))
        else:
            phip = numpy.zeros((N,n,1))

        fx = numpy.zeros((N,n,s))
        fu = numpy.zeros((N,m,s))
        fp = numpy.ones((N,p,s))        
        
        for arc in range(s):
            for k in range(N):
                
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
        x,N = self.x,self.N
        
        return numpy.array([x[0,0,0],         x[0,1,0],\
                            x[N-1,0,0] - 1.0, x[N-1,1,0],\
                            x[0,0,1] - 1.0,   x[0,1,1],\
                            x[N-1,0,1],       x[N-1,1,1]])
        
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
    def plotCat(self,func,color='b',nPlot=0,yLabl=''):
        
        s = self.s
        t = self.t
        
        pi = self.pi
        # Total time
        tTot = pi.sum()
        accAdimTime = 0.0

        plt.subplot2grid((8,4),(nPlot,0),colspan=5)
        
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

        plt.grid(True)
        plt.xlabel("Concat. adim. time [-]")
        plt.ylabel(yLabl)        
        
        
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
        
        self.plotCat(x[:,0,:],yLabl='Position')
        
        if opt.get('mode','sol') == 'sol':
            I = self.calcI()
            titlStr = "Current solution: I = {:.4E}".format(I) + \
            " P = {:.4E} ".format(self.P) + " Q = {:.4E} ".format(self.Q)
        elif opt['mode'] == 'var':
            titlStr = "Proposed variations"
        else:
            titlStr = opt['mode']
        plt.title(titlStr)
        
        self.plotCat(x[:,1,:],color='g',nPlot=1,yLabl='Speed')
        self.plotCat(u[:,0,:],color='k',nPlot=2,yLabl='u')
        self.plotCat(numpy.tanh(u[:,0,:]),color='r',nPlot=3,yLabl='Control')
    
        plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
        plt.show()
        print("pi =",pi,"\n")
        
    #
if __name__ == "__main__":
    print("\n\nRunning probCartMod.py!\n")
    exmpProb = prob()
    
    print("Initializing problem:")
    exmpProb = exmpProb.initGues()
    exmpProb.printPars()
    s = exmpProb.s
    
    
    print("Plotting current version of solution:")
    exmpProb.plotSol()
    
    # TODO: test all calculations here as well
    print("Calculating f")
    f = exmpProb.calcF()
    
    
    print("Calculating grads:")