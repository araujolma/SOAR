#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module for the problem 9-2 from Miele (1970)
"""

import numpy
from sgra import sgra
from utils import simp
import matplotlib.pyplot as plt

class prob(sgra):
    probName = 'prob9_2'

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
            u = numpy.zeros((N,m,s))

            x[:,1,0] = 0.3*t.copy()
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
            u = numpy.zeros((N,m,s))

            x[:,1,0] = 0.3*self.t.copy()
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

        phi[:,0,0] = 2*numpy.sin(u[:,0,0]) - numpy.ones(N)
        phi[:,1,0] = x[:,0,0]

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

        u = self.u

        # Pre-assign functions

        sin = numpy.sin
        cos = numpy.cos
        sin_u = sin(u[:,0,0])
        cos_u = cos(u[:,0,0])

        Grads['dt'] = 1.0/(N-1)

        phix = numpy.zeros((N,n,n,s))
        phiu = numpy.zeros((N,n,m,s))
        if p>0:
            phip = numpy.zeros((N,n,p,s))
        else:
            phip = numpy.zeros((N,n,1,s))

        phix[:,1,0,0] = numpy.ones(N)
        phiu[:,0,0,0] = 2.0 * cos_u

        psiy = numpy.zeros((q,2*n*s))

        psiy[0,0] = 1.0
        psiy[1,1] = 1.0
        psiy[2,2] = 1.0
        psiy[3,3] = 1.0
        if p>0:
            psip = numpy.zeros((q,p))
        else:
            psip = numpy.zeros((q,1))

        fx = numpy.zeros((N,n,s))
        fu = numpy.zeros((N,m,s))
        if p>0:
            fp = numpy.zeros((N,p,s))
        else:
            fp = numpy.zeros((N,1,s))

        fu[:,0,0] = 2.0 * sin_u

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
        return numpy.array([x[0,0,0],
                            x[0,1,0],
                            x[N-1,0,0],
                            x[N-1,1,0]-0.3])

    def calcF(self):
        N = self.N
        s = self.s
        u = self.u
        f =  2.0 - 2.0 * numpy.cos(u[:,0,:])

        return f, f, numpy.zeros((N,s))

    def calcI(self):
        f,_,_ = self.calcF()
        N = self.N
        I = 0.0

        for arc in range(self.s):
            I += simp(f[:,arc],N)

        return I, I, 0.0
#%%
    def plotSol(self,opt={},intv=[],piIsTime=True,mustSaveFig=True,
                subPlotAdjs={}):
        t = self.t
        x = self.x
        u = self.u
        pi = self.pi

        if len(intv)==0:
            intv = numpy.arange(0,self.N,1,dtype='int')
        else:
             intv = list(intv)

        if piIsTime:
            timeLabl = 't [s]'
        else:
            timeLabl = 'adim. t [-]'

        I, _, _ = self.calcI()
        titlStr = "Current solution: I = {:.4E}".format(I) + \
                  " P = {:.4E} ".format(self.P) + \
                  " Q = {:.4E} ".format(self.Q)
        titlStr += "\n(grad iter #" + str(self.NIterGrad) + ")"

        if opt.get('mode','sol') == 'sol':

            plt.subplot2grid((3,1),(0,0),colspan=5)
            plt.plot(t[intv],x[intv,0,0],)
            plt.grid(True)
            plt.ylabel("x")
            plt.xlabel(timeLabl)
            plt.title(titlStr)

            plt.subplot2grid((3,1),(1,0),colspan=5)
            plt.plot(t[intv],x[intv,1,0],'g')
            plt.grid(True)
            plt.ylabel("y")
            plt.xlabel(timeLabl)

            plt.subplot2grid((3,1),(2,0),colspan=5)
            plt.plot(t[intv],u[intv,0,0],'k')
            plt.grid(True)
            plt.ylabel("u")
            plt.xlabel(timeLabl)

            plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
            self.savefig(keyName='currSol',fullName='solution')
            self.log.printL("pi ="+str(pi))
        elif opt['mode'] == 'var':
            dx = opt['x']
            du = opt['u']
            dp = opt['pi']

            plt.subplot2grid((3,1),(0,0),colspan=5)
            plt.plot(t[intv],dx[intv,0,0],)
            plt.grid(True)
            plt.ylabel("x")
            plt.xlabel(timeLabl)

            titlStr = "Proposed variations\n"+"Delta pi: "
            for i in range(self.p):
                titlStr += "{:.4E}, ".format(dp[i])
            titlStr += "\n(grad iter #" + str(self.NIterGrad) + ")"

            plt.title(titlStr)

            plt.subplot2grid((3,1),(1,0),colspan=5)
            plt.plot(t[intv],dx[intv,1,0],'g')
            plt.grid(True)
            plt.ylabel("y")
            plt.xlabel(timeLabl)

            plt.subplot2grid((3,1),(2,0),colspan=5)
            plt.plot(t[intv],du[intv,0,0],'k')
            plt.grid(True)
            plt.ylabel("u")
            plt.xlabel(timeLabl)

            plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
            self.savefig(keyName='corr',fullName='corrections')
        elif opt['mode'] == 'lambda':
            titlStr = "Lambdas (grad iter #" + str(self.NIterGrad + 1) + ")"

            plt.subplot2grid((3, 1), (0, 0))
            self.plotCat(self.lam[:, 0, :], piIsTime=piIsTime, intv=intv)
            plt.grid(True)
            plt.ylabel("lam - x")
            plt.xlabel(timeLabl)
            plt.title(titlStr)

            plt.subplot2grid((3, 1), (1, 0))
            self.plotCat(self.lam[:, 1, :], color='g', piIsTime=piIsTime,
                         intv=intv)
            plt.grid(True)
            plt.ylabel("lam - y")
            plt.xlabel(timeLabl)

            plt.subplot2grid((3, 1), (2, 0))
            self.plotCat(u[:, 0, :], color='r', piIsTime=piIsTime, intv=intv)
            plt.grid(True)
            plt.ylabel("u")
            plt.xlabel(timeLabl)

        else:
            raise Exception("plotSol: Unknown mode '"+str(opt['mode'])+"'")
    #

    def compWith(self,altSol,altSolLabl='altSol',piIsTime=True,
                 mustSaveFig=True,subPlotAdjs={'left':0.0,'right':1.0,'bottom':0.0,
                     'top':2.1,'wspace':0.2,'hspace':0.4}):
        self.log.printL("\nComparing solutions...\n")
        pi = self.pi
        currSolLabl = 'Final solution'

        # Plotting the curves
        plt.subplots_adjust(**subPlotAdjs)

        plt.subplot2grid((3,1),(0,0))
        self.plotCat(self.x[:,0,:],color='c',piIsTime=piIsTime,
                     labl=currSolLabl)
        altSol.plotCat(altSol.x[:,0,:],mark='--',labl=altSolLabl)
        plt.grid(True)
        plt.ylabel("x [-]")
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)
        plt.xlabel("Time")

        plt.subplot2grid((3,1),(1,0))
        self.plotCat(self.x[:,1,:],color='g',piIsTime=piIsTime,
                     labl=currSolLabl)
        altSol.plotCat(altSol.x[:,1,:],mark='--',piIsTime=piIsTime,
                       labl=altSolLabl)
        plt.grid(True)
        plt.ylabel("y [-]")
        plt.xlabel("Time")
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)

        plt.subplot2grid((3,1),(2,0))
        self.plotCat(self.u[:,0,:],color='k',piIsTime=piIsTime,
                     labl=currSolLabl)
        altSol.plotCat(altSol.u[:,0,:],mark='--',piIsTime=piIsTime,
                       labl=altSolLabl)
        plt.grid(True)
        plt.ylabel("u [-]")
        plt.xlabel("Time")
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)

        self.savefig(keyName='comp',fullName='comparisons')
        self.log.printL("pi = "+str(pi)+"\n")