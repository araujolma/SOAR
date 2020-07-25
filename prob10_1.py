#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module for the problem 10-1 from Miele (1970)
"""

import numpy
from sgra import sgra
import matplotlib.pyplot as plt

class prob(sgra):
    probName = 'prob10_1'

    def initGues(self,opt={}):

        # The parameters that go here are the ones that cannot be simply
        # altered from an external configuration file... at least not
        # without a big increase in the complexity of the code...

        # matrix sizes
        n = 2
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
        else:
            raise(Exception("Unknown 'initMode' = {}.".format(initMode)))

        self.x = x
        self.u = u
        self.pi = pi
        self.lam = lam
        self.mu = mu

        # Setting up the exact (variational) analytical solution
        self.hasExactSol = True
        self.I_opt = .5 * numpy.pi
        self.x_opt = numpy.empty_like(x)
        self.u_opt = numpy.empty_like(u)
        self.pi_opt = self.I_opt
        self.x_opt[:, 0, 0] = numpy.sin(self.pi_opt * self.t)
        self.x_opt[:, 1, 0] = -.5 * numpy.sin(numpy.pi * self.t)
        self.u_opt[:, 0, 0] = numpy.cos(self.pi_opt * self.t)

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

        phi[:,0,0] = pi[0] * u[:,0,0]
        phi[:,1,0] = pi[0] * (x[:,0,0]**2 - u[:,0,0]**2)

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

        Grads['dt'] = 1.0/(N-1)

        phix = numpy.zeros((N,n,n,s))
        phiu = numpy.zeros((N,n,m,s))
        if p>0:
            phip = numpy.zeros((N,n,p,s))
        else:
            phip = numpy.zeros((N,n,1,s))

        phix[:,1,0,0] = 2.0 * pi[0] * x[:,0,0]
        phiu[:,0,0,0] = numpy.array([pi[0]])
        phiu[:,1,0,0] = - 2.0 * pi[0] * u[:,0,0]
        phip[:,0,0,0] = u[:,0,0]
        phip[:,1,0,0] = x[:,0,0]**2 - u[:,0,0]**2

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
                            x[N-1,0,0]-1.0,\
                            x[N-1,1,0]])

    def calcF(self):
        N = self.N
        s = self.s
        f = self.pi[0]*numpy.ones((N,s))

        return f, f, numpy.zeros((N,s))

    def calcI(self):
        return self.pi[0],self.pi[0],0.0
#%%
    def plotSol(self, opt={}, intv=None, piIsTime=True, mustSaveFig=True,
                subPlotAdjs={}):
        t = self.t
        x = self.x
        u = self.u
        pi = self.pi

        if intv is None:
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
            plt.plot(t[intv],u[intv,0,0],'k')
            plt.grid(True)
            plt.ylabel("u")

            plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
            self.savefig(keyName='currSol',fullName='solution')
            self.log.printL("pi = "+str(pi))
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
            plt.plot(t[intv],du[intv,0,0],'k')
            plt.grid(True)
            plt.ylabel("u")

            plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
            self.savefig(keyName='corr',fullName='corrections')

        else:
            titlStr = opt['mode']
    #

    def compWith(self,altSol,altSolLabl='altSol',piIsTime=True,
                 mustSaveFig=True,
                 subPlotAdjs={'left':0.0,'right':1.0,'bottom':0.0,
                     'top':2.1,'wspace':0.2,'hspace':0.4}):
        # TODO: Use PiIsTime!
        #  All plt.plot instances must be changed to self.plotCat...
        self.log.printL("\nComparing solutions...\n")
        pi = self.pi
        currSolLabl = 'Final solution'

        # Plotting the curves
        plt.subplots_adjust(**subPlotAdjs)

        plt.subplot2grid((3,1),(0,0))
        altSol.plotCat(altSol.x[:,0,:],mark='--',labl=altSolLabl)
        self.plotCat(self.x[:,0,:],color='c',labl=currSolLabl)
        plt.grid(True)
        plt.ylabel("x [-]")
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)
        #titlStr = "Comparing solutions: " + currSolLabl + " and " + \
        #          altSolLabl
        #titlStr += "\n(grad iter #" + str(self.NIterGrad) + ")"
#        plt.title(titlStr)
        plt.xlabel("Adimensional time")

        plt.subplot2grid((3,1),(1,0))
        altSol.plotCat(altSol.x[:,1,:],mark='--',labl=altSolLabl)
        self.plotCat(self.x[:,1,:],color='g',labl=currSolLabl)
        plt.grid(True)
        plt.ylabel("y [-]")
        plt.xlabel("Adimensional time")
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)

        plt.subplot2grid((3,1),(2,0))
        altSol.plotCat(altSol.u[:,0,:],mark='--',labl=altSolLabl)
        self.plotCat(self.u[:,0,:],color='k',labl=currSolLabl)
        plt.grid(True)
        plt.ylabel("u [-]")
        plt.xlabel("Adimensional time")
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)

        self.savefig(keyName='comp',fullName='comparisons')
        self.log.printL("pi = "+str(pi)+"\n")