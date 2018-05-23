# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module for the Zermelo Problem, which is a classical minimum-time navigation
problem:
    Find the steering program of a boat that navigates from a given initial
position to a given terminal position in minimal time. The stream moves with a
constant velocity and the boat moves with a constant magnitude velocity
relative to the stream.
"""

import numpy
from sgra import sgra
from utils import simp
import matplotlib.pyplot as plt

class prob(sgra):
    probName = 'probZer'

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
            u = numpy.arctanh(0.5*numpy.ones((N,m,s)))

            x[:,0,0] = t.copy()
            x[:,1,0] = x[:,0,0]
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
            gama = numpy.zeros((N,m,s))#.25 * numpy.pi * numpy.ones((N,m,s))
            u = numpy.arctanh(gama / (numpy.pi))

            x[:,0,0] = 5.0 * self.t.copy()
            #x[:,1,0] = x[:,0,0]
            lam = 0.0 * x.copy()
            mu = numpy.zeros(q)
            pi = numpy.array([2.5])

            self.x = x
            self.u = u
            self.pi = pi
            self.lam = lam
            self.mu = mu

            solInit = self.copy()

            self.log.printL("\nInitialization complete.\n")
            return solInit

#%%

    def calcPhi(self):
        N = self.N
        n = self.n
        s = self.s
        phi = numpy.empty((N,n,s))
        u = self.u
        pi = self.pi
        gama = numpy.pi * numpy.tanh(u)

        phi[:,0,0] = pi[0] * (numpy.cos(gama[:,0,0]) + 1.0)
        phi[:,1,0] = pi[0] * numpy.sin(gama[:,0,0])
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
        u = self.u
        pi = self.pi
        gama = 0.5 * numpy.pi * numpy.tanh(u)
        # Pre-assign functions

        Grads['dt'] = 1.0/(N-1)

        phix = numpy.zeros((N,n,n,s))

        if p>0:
            phip = numpy.zeros((N,n,p,s))
        else:
            phip = numpy.zeros((N,n,1,s))

        phiu = numpy.zeros((N,n,m,s))

        sinGama = numpy.sin(gama[:,0,0])
        cosGama = numpy.cos(gama[:,0,0])
        dGama_du = (1.0-numpy.tanh(u[:,0,0])**2) * numpy.pi
        phiu[:,0,0,0] = - pi[0] * sinGama * dGama_du
        phiu[:,1,0,0] = pi[0] * cosGama * dGama_du
        phip[:,0,0,0] = cosGama + 1.0
        phip[:,1,0,0] = sinGama

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

        fp[:,0,0] = numpy.ones(N)

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
                            x[N-1,0,0]-5.0,\
                            x[N-1,1,0]-5.0])

    def calcF(self):
        N,s = self.N,self.s
        f = numpy.empty((N,s))
        for arc in range(s):
            f[:,arc] = self.pi[arc] * numpy.ones(N)

        return f, f, 0.0*f

    def calcI(self):
        N,s = self.N,self.s
        f, _, _ = self.calcF()

        Ivec = numpy.empty(s)
        for arc in range(s):
            Ivec[arc] += simp(f[:,arc],N)

        I = Ivec.sum()
        return I, I, 0.0
#%%
    def plotSol(self,opt={},intv=[]):
        t = self.t
        x = self.x
        u = self.u
        pi = self.pi
        gama = numpy.pi * numpy.tanh(u)

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
            plt.plot(t[intv],u[intv,0,0],'k')
            plt.grid(True)
            plt.ylabel("u")

            plt.subplot2grid((8,4),(3,0),colspan=5)
            plt.plot(t[intv],gama[intv,0,0]*180/numpy.pi,'y')
            plt.grid(True)
            plt.ylabel("gama")

            plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
            self.savefig(keyName='currSol',fullName='solution')
            self.log.printL("pi ="+str(pi))
        elif opt['mode'] == 'var':
            dx = opt['x']
            du = opt['u']
            dp = opt['pi']
            dgama = 0.5 * numpy.pi * numpy.tanh(du)

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

            plt.subplot2grid((8,4),(3,0),colspan=5)
            plt.plot(t[intv],dgama[intv,0,0]*180/numpy.pi,'y')
            plt.grid(True)
            plt.ylabel("gama")

            plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
            self.savefig(keyName='corr',fullName='corrections')

        else:
            titlStr = opt['mode']
    #

    def plotTraj(self,mustSaveFig=True,altSol=None,name=None):
        """Plot the trajectory of the sliding mass."""

        # TODO: these arguments altSol and "name" are not being used. Use them!

        X = self.x[:,0,0]
        Y = self.x[:,1,0]

        plt.plot(X,Y)
        plt.plot(X[0],Y[0],'o')
        plt.plot(X[-1],Y[-1],'s')
        plt.axis('equal')
        plt.grid(True)
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")

        titlStr = "Trajectory "
        titlStr += "(grad iter #" + str(self.NIterGrad) + ")\n"
        plt.title(titlStr)
        #plt.legend(loc="upper left", bbox_to_anchor=(1,1))

        if mustSaveFig:
            self.savefig(keyName='traj',fullName='trajectory')
        else:
            plt.show()
            plt.clf()
