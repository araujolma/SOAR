#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module for the problem 10-2 from Miele (1970)
"""

import numpy
from sgra import sgra
import matplotlib.pyplot as plt

class prob(sgra):
    probName = 'prob10_2'

    def initGues(self,opt={}):

        # The parameters that go here are the ones that cannot be simply
        # altered from an external configuration file... at least not
        # without a big increase in the complexity of the code...

        # matrix sizes
        n = 3
        m = 1
        p = 1
        q = 4
        s = 1


        self.n = n
        self.m = m
        self.p = p
        self.q = q
        self.s = s
        self.Ns = 2*n*s + p

        initMode = opt.get('initMode','default')
        if initMode == 'default':

            N = 5000+1
            self.N = N
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

            self.constants['gradStepSrchCte'] = 1.0e-4

            # Get initialization mode

            x = numpy.zeros((N,n,s))
            u = numpy.ones((N,m,s))

            x[:,0,0] = t.copy()
            lam = 0.0*x.copy()
            mu = numpy.zeros(q)
            pi = numpy.array([1.0])

            self.x = x
            self.u = u
            self.pi = pi
            self.lam = lam
            self.mu= mu

            solInit = self.copy()

            self.log.printL("\nInitialization complete.\n")
            return solInit

        elif initMode == 'extSol':
            inpFile = opt.get('confFile','')

            # Get parameters from file

            self.loadParsFromFile(file=inpFile)

            # The actual "initial guess"

            N,m,n,p,q,s = self.N,self.m,self.n,self.p,self.q,self.s
            x = numpy.zeros((N,n,s))
            u = numpy.ones((N,m,s))

            x[:,0,0] = self.t.copy()
            lam = 0.0*x.copy()
            mu = numpy.zeros(q)
            pi = numpy.array([1.0])

            self.x = x
            self.u = u
            self.pi = pi
            self.lam = lam
            self.mu= mu

            solInit = self.copy()

            self.log.printL("\nInitialization complete.\n")
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

        sinu = numpy.sin(u[:,0,0])
        phi[:,0,0] = pi[0] * x[:,2,0] * numpy.cos(u[:,0,0])
        phi[:,1,0] = pi[0] * x[:,2,0] * sinu
        phi[:,2,0] = pi[0] * sinu

        return phi

#%%

    def calcGrads(self,calcCostTerm=False):
        Grads = dict()

        N = self.N
        n = self.n
        m = self.m
        p = self.p
        q = self.q
        s = self.s
        #q = sizes['q']
        #N0 = sizes['N0']

        x = self.x
        u = self.u
        pi = self.pi

        # Pre-assign functions

        sin = numpy.sin
        cos = numpy.cos
        sin_u = sin(u[:,0,0])
        cos_u = cos(u[:,0,0])
        z = x[:,2,0]

        Grads['dt'] = 1.0/(N-1)

        phix = numpy.zeros((N,n,n,s))
        phiu = numpy.zeros((N,n,m,s))
        if p>0:
            phip = numpy.zeros((N,n,p,s))
        else:
            phip = numpy.zeros((N,n,1,s))

        phix[:,0,2,0] = pi[0] * cos_u
        phix[:,1,2,0] = pi[0] * sin_u
        phiu[:,0,0,0] = - pi[0] * z * sin_u
        phiu[:,1,0,0] = pi[0] * z * cos_u
        phiu[:,2,0,0] = pi[0] * cos_u
        phip[:,0,0,0] = z * cos_u
        phip[:,1,0,0] = z * sin_u
        phip[:,2,0,0] = sin_u

        psiy = numpy.zeros((q,2*n*s))

        psiy[0,0] = 1.0
        psiy[1,1] = 1.0
        psiy[2,2] = 1.0
        psiy[3,3] = 1.0
        psip = numpy.zeros((q,p))

        fx = numpy.zeros((N,n,s))
        fu = numpy.zeros((N,m,s))
        fp = numpy.ones((N,p,s))

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
        return numpy.array([x[0,0,0],\
                            x[0,1,0],\
                            x[0,2,0],\
                            x[N-1,0,0]-1.0])

    def calcF(self):
        N = self.N
        s = self.s
        f = self.pi[0]*numpy.ones((N,s))

        return f,f,numpy.zeros((N,s))

    def calcI(self):
        return self.pi[0],self.pi[0],0.0
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


        if opt.get('mode','sol') == 'sol':
            I,_,_ = self.calcI()
            titlStr = "Current solution: I = {:.4E}".format(I) + \
            " P = {:.4E} ".format(self.P) + " Q = {:.4E} ".format(self.Q)
            titlStr += "\n(grad iter #" + str(self.NIterGrad) + ")"

            plt.subplot2grid((8,4),(0,0),colspan=5)
            plt.plot(t[intv],x[intv,0,0],)
            plt.grid(True)
            plt.ylabel("x")
            plt.title(titlStr)

            plt.subplot2grid((8,4),(1,0),colspan=5)
            plt.plot(t[intv],x[intv,1,0],'g')
            plt.grid(True)
            plt.ylabel("y")

            plt.subplot2grid((8,4),(2,0),colspan=5)
            plt.plot(t[intv],x[intv,2,0],'r')
            plt.grid(True)
            plt.ylabel("z")

            plt.subplot2grid((8,4),(3,0),colspan=5)
            plt.plot(t[intv],u[intv,0,0],'k')
            plt.grid(True)
            plt.ylabel("u")

            plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
            self.savefig(keyName='currSol',fullName='solution')
            self.log.printL("pi ="+str(pi))
        elif opt['mode'] == 'var':
            dx = opt['x']
            du = opt['u']
            dp = opt['pi']

            plt.subplot2grid((8,4),(0,0),colspan=5)
            plt.plot(t[intv],dx[intv,0,0],)
            plt.grid(True)
            plt.ylabel("x")

            titlStr = "Proposed variations\n"+"Delta pi: "
            for i in range(self.p):
                titlStr += "{:.4E}, ".format(dp[i])
            titlStr += "\n(grad iter #" + str(self.NIterGrad) + ")"
            plt.title(titlStr)

            plt.subplot2grid((8,4),(1,0),colspan=5)
            plt.plot(t[intv],dx[intv,1,0],'g')
            plt.grid(True)
            plt.ylabel("y")

            plt.subplot2grid((8,4),(2,0),colspan=5)
            plt.plot(t[intv],dx[intv,2,0],'r')
            plt.grid(True)
            plt.ylabel("z")

            plt.subplot2grid((8,4),(3,0),colspan=5)
            plt.plot(t[intv],du[intv,0,0],'k')
            plt.grid(True)
            plt.ylabel("u")

            plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
            self.savefig(keyName='corr',fullName='corrections')

        else:
            titlStr = opt['mode']
    #