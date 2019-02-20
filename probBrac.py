#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 08:23:37 2018

@author: araujolma

A module for the brachistochrone problem.
"""

import numpy
from sgra import sgra
import matplotlib.pyplot as plt

class prob(sgra):
    probName = 'probBrac'

    def initGues(self,opt={}):

        # The parameters that go here are the ones that cannot be simply
        # altered from an external configuration file... at least not
        # without a big increase in the complexity of the code...

        n = 3
        m = 1
        p = 1
        q = 5
        s = 1
        self.n = n
        self.m = m
        self.p = p
        self.q = q
        self.s = s
        self.Ns = 2*n*s + p

        initMode = opt.get('initMode','default')
        if initMode == 'default':
            # matrix sizes

            N = 2000+1#20000+1#2000+1

            self.N = N

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
            #u = numpy.zeros((N,m,s))#5.0*numpy.ones((N,m,s))
            u = numpy.arctanh(0.5*numpy.ones((N,m,s)))
            #x[:,0,0] = t.copy()
            #for i in range(N):
            #    x[i,1,0] = x[N-i-1,0,0]
            #x[:,2,0] = numpy.sqrt(20.0*x[:,0,0])
            pi = numpy.array([2.0/numpy.sqrt(10.0)])
            td = t * pi[0]
            x[:,0,0] = 2.5 * (td**2)
            x[:,1,0] = 1.0 - x[:,0,0]
            x[:,2,0] = numpy.sqrt(10.0 * x[:,0,0])

            #x[:,0,0] = .5*t
            #x[:,0,1] = .5+.5*t

            lam = 0.0*x
            mu = numpy.zeros(q)
            #pi = 10.0*numpy.ones(p)


            self.x = x
            self.u = u
            self.pi = pi
            self.lam = lam
            self.mu= mu

            self.constants['gradStepSrchCte'] = 1e-3

            solInit = self.copy()
            self.compWith(solInit,'Initial Guess')

            self.log.printL("\nInitialization complete.\n")
            return solInit

        elif initMode == 'extSol':
            inpFile = opt.get('confFile','')

            # Get parameters from file

            self.loadParsFromFile(file=inpFile)

            # The actual "initial guess"

            N,m,n,p,q,s = self.N,self.m,self.n,self.p,self.q,self.s

            x = numpy.zeros((N,n,s))
            #u = numpy.zeros((N,m,s))#5.0*numpy.ones((N,m,s))
            u = numpy.arctanh(0.5*numpy.ones((N,m,s)))
            #x[:,0,0] = t.copy()
            #for i in range(N):
            #    x[i,1,0] = x[N-i-1,0,0]
            #x[:,2,0] = numpy.sqrt(20.0*x[:,0,0])
            pi = numpy.array([2.0/numpy.sqrt(10.0)])
            td = self.t * pi[0]
            x[:,0,0] = 2.5 * (td**2)
            x[:,1,0] = 1.0 - x[:,0,0]
            x[:,2,0] = numpy.sqrt(10.0 * x[:,0,0])

            #x[:,0,0] = .5*t
            #x[:,0,1] = .5+.5*t

            lam = 0.0*x
            mu = numpy.zeros(q)
            #pi = 10.0*numpy.ones(p)

            self.x = x
            self.u = u
            self.pi = pi
            self.lam = lam
            self.mu= mu

            solInit = self.copy()
            self.compWith(solInit,'Initial Guess')

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

        gama = .5 * numpy.pi * numpy.tanh(u)

        for arc in range(s):
            phi[:,0,arc] = pi[arc] * x[:,2,arc] * numpy.cos(gama[:,0,arc])
            phi[:,1,arc] = -pi[arc] * x[:,2,arc] * numpy.sin(gama[:,0,arc])
            phi[:,2,arc] = pi[arc] * 10.0 * numpy.sin(gama[:,0,arc])

        return phi

#%%

    def calcGrads(self,calcCostTerm=False):
        Grads = dict()

        N,n,m,p,q,s = self.N,self.n,self.m,self.p,self.q,self.s
        pi = self.pi

        # Pre-assign functions
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
        psiy[0,0] = 1.0 # x(0) = 0
        psiy[1,1] = 1.0 # y(0) = 1
        psiy[2,2] = 1.0 # v(0) = 0
        psiy[3,3] = 1.0 # x(1) = 1
        psiy[4,4] = 1.0 # y(1) = 0

        psip = numpy.zeros((q,p))

        tanh_u = numpy.tanh(self.u)
        gama = .5 * numpy.pi * tanh_u
        dgdu = .5 * numpy.pi * (1.0 - tanh_u**2)
        sinGama, cosGama = numpy.sin(gama), numpy.cos(gama)

        for arc in range(s):
            phix[:,0,2,arc] = pi[arc] * cosGama[:,0,arc]
            phix[:,1,2,arc] = -pi[arc] * sinGama[:,0,arc]

            phiu[:,0,0,arc] = -pi[arc] * self.x[:,2,arc] * sinGama[:,0,arc] * dgdu[:,0,arc]
            phiu[:,1,0,arc] = -pi[arc] * self.x[:,2,arc] * cosGama[:,0,arc] * dgdu[:,0,arc]
            phiu[:,2,0,arc] = pi[arc] * 10.0 * cosGama[:,0,arc] * dgdu[:,0,arc]

            phip[:,0,arc,arc] = self.x[:,2,arc] * cosGama[:,0,arc]
            phip[:,1,arc,arc] = -self.x[:,2,arc] * sinGama[:,0,arc]
            phip[:,2,arc,arc] = 10.0 * sinGama[:,0,arc]

            fp[:,arc,arc] = 1.0
        #

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
        N = self.N
#        return numpy.array([x[0,0,0],x[0,1,0],x[N-1,0,0]-0.5,x[N-1,1,0],\
#                            x[0,0,1]-0.5,x[0,1,1],x[N-1,0,1]-1.0,x[N-1,1,1]])
        return numpy.array([self.x[0,0,0],\
                            self.x[0,1,0]-1.0,\
                            self.x[0,2,0],\
                            self.x[N-1,0,0]-3.0,\
                            self.x[N-1,1,0]])

    def calcF(self):
        N,s = self.N,self.s
        f = numpy.empty((N,s))
        for arc in range(s):
            f[:,arc] = self.pi[arc] * numpy.ones(N)

        return f, f, 0.0*f

    def calcI(self):
        #N,s = self.N,self.s
        #f, _, _ = self.calcF()

        Ivec = self.pi#numpy.empty(s)
#        for arc in range(s):
#            Ivec[arc] = .5*(f[0,arc]+f[N-1,arc])
#            Ivec[arc] += f[1:(N-1),arc].sum()
#        Ivec *= 1.0/(N-1)
        I = Ivec.sum()
        return I, I, 0.0
#%%
    def plotSol(self,opt={},intv=None,piIsTime=True,mustSaveFig=True,\
                subPlotAdjs={}):

        pi = self.pi
        r2d = 180.0/numpy.pi

        if piIsTime:
            timeLabl = 'Time [s]'
        else:
            timeLabl = 'Adim. time [-]'

#        if len(intv)==0:
#            intv = numpy.arange(0,self.N,1,dtype='int')
#        else:
#             intv = list(intv)


        if opt.get('mode','sol') == 'sol':
            I, _, _ = self.calcI()
            titlStr = "Current solution: I = {:.4E}".format(I) + \
            " P = {:.4E} ".format(self.P) + " Q = {:.4E} ".format(self.Q)
            titlStr += "\n(grad iter #" + str(self.NIterGrad) + ")"
            plt.subplot2grid((5,1),(0,0),colspan=5)
            self.plotCat(self.x[:,0,:],intv=intv)
            plt.grid(True)
            plt.ylabel("x [m]")
            plt.title(titlStr)
            plt.subplot2grid((5,1),(1,0),colspan=5)
            self.plotCat(self.x[:,1,:],intv=intv,color='g')
            plt.grid(True)
            plt.ylabel("y [m]")
            plt.subplot2grid((5,1),(2,0),colspan=5)
            self.plotCat(self.x[:,2,:],intv=intv,color='r')
            plt.grid(True)
            plt.ylabel("V [m/s]")
            plt.subplot2grid((5,1),(3,0),colspan=5)
            self.plotCat(self.u[:,0,:],intv=intv,color='k')
            plt.grid(True)
            plt.ylabel("u1 [-]")
            plt.subplot2grid((5,1),(4,0),colspan=5)
            gama = 0.5*numpy.pi*numpy.tanh(self.u)
            self.plotCat(r2d*gama[:,0,:],intv=intv,color='k')
            plt.grid(True)
            plt.ylabel('Inclination angle [deg]')
            plt.xlabel(timeLabl)

            plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)

            self.savefig(keyName='currSol',fullName='solution')

            self.log.printL("pi = "+str(pi)+"\n")
        elif opt['mode'] == 'var':
            dx = opt['x']
            du = opt['u']
            dp = opt['pi']

            titlStr = "Proposed variations (grad iter #" + \
                      str(self.NIterGrad+1) + ")\n"+"Delta pi: "
            for i in range(self.p):
                titlStr += "{:.4E}, ".format(dp[i])
                #titlStr += str(dp[i])+", "

            plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)

            plt.subplot2grid((5,1),(0,0))
            self.plotCat(dx[:,0,:],intv=intv)
            plt.grid(True)
            plt.ylabel("x [m]")
            plt.title(titlStr)

            plt.subplot2grid((5,1),(1,0))
            self.plotCat(dx[:,1,:],intv=intv,color='g')
            plt.grid(True)
            plt.ylabel("y [m]")

            plt.subplot2grid((5,1),(2,0))
            self.plotCat(dx[:,2,:],intv=intv,color='r')
            plt.grid(True)
            plt.ylabel("v [m/s]")

            plt.subplot2grid((5,1),(3,0))
            self.plotCat(du[:,0,:],intv=intv,color='k')
            plt.grid(True)
            plt.ylabel("u1 [-]")

            new_u = self.u + du
            gama = 0.5*numpy.pi*numpy.tanh(self.u)
            new_gama = 0.5*numpy.pi*numpy.tanh(new_u)
            dgama = new_gama - gama
            plt.subplot2grid((5,1),(4,0))
            self.plotCat(r2d*dgama[:,0,:],intv=intv,color='r')
            plt.grid(True)
            plt.xlabel("t")
            plt.ylabel("Inclination angle [deg]")

            self.savefig(keyName='corr',fullName='corrections')

        elif opt['mode'] == 'lambda':
            titlStr = "Lambda for current solution"

            plt.subplot2grid((5,1),(0,0),colspan=5)
            self.plotCat(self.lam[:,0,:],intv=intv)
            plt.grid(True)
            plt.ylabel("lambda: x")
            plt.title(titlStr)
            plt.subplot2grid((5,1),(1,0),colspan=5)
            self.plotCat(self.lam[:,1,:],intv=intv,color='g')
            plt.grid(True)
            plt.ylabel("lambda: y")
            plt.subplot2grid((5,1),(2,0),colspan=5)
            self.plotCat(self.lam[:,2,:],intv=intv,color='g')
            plt.grid(True)
            plt.ylabel("lambda: Speed")
            plt.subplot2grid((5,1),(3,0),colspan=5)
            self.plotCat(self.u[:,0,:],intv=intv,color='k')
            plt.grid(True)
            plt.ylabel("u1 [-]")
            plt.subplot2grid((5,1),(4,0),colspan=5)
            gama = 0.5*numpy.pi*numpy.tanh(self.u)
            self.plotCat(r2d*gama[:,0,:],intv=intv,color='k')
            plt.grid(True)
            plt.ylabel('Inclination angle [deg]')
            plt.xlabel(timeLabl)

            plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
            self.savefig(keyName='currLamb',fullName='lambdas')

            self.log.printL("mu = "+str(self.mu))

        else:
            titlStr = opt['mode']

    def compWith(self,altSol,altSolLabl='altSol',mustSaveFig=True,\
                 piIsTime=True,\
                 subPlotAdjs={'left':0.0,'right':1.0,'bottom':0.0,
                              'top':3.2,'wspace':0.2,'hspace':0.45}):
        self.log.printL("\nComparing solutions...\n")
        pi = self.pi
        r2d = 180.0/numpy.pi
        currSolLabl = 'Final solution'
        if piIsTime:
            timeLabl = 't [s]'
        else:
            timeLabl = 'adim. t [-]'


        # Plotting the curves
        plt.subplots_adjust(**subPlotAdjs)

        plt.subplot2grid((5,1),(0,0))
        altSol.plotCat(altSol.x[:,0,:],mark='--',labl=altSolLabl,\
                       piIsTime=piIsTime)
        self.plotCat(self.x[:,0,:],color='c',labl=currSolLabl,\
                     piIsTime=piIsTime)
        plt.grid(True)
        plt.ylabel("x [m]")
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)
        titlStr = "Comparing solutions: " + currSolLabl + " and " + \
                  altSolLabl
        titlStr += "\n(grad iter #" + str(self.NIterGrad) + ")"
        plt.title(titlStr)
        plt.xlabel(timeLabl)

        plt.subplot2grid((5,1),(1,0))
        altSol.plotCat(altSol.x[:,1,:],mark='--',labl=altSolLabl,\
                       piIsTime=piIsTime)
        self.plotCat(self.x[:,1,:],color='g',labl=currSolLabl,\
                     piIsTime=piIsTime)
        plt.grid(True)
        plt.ylabel("y [m]")
        plt.xlabel(timeLabl)
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)

        plt.subplot2grid((5,1),(2,0))
        altSol.plotCat(altSol.x[:,2,:],mark='--',labl=altSolLabl,\
                       piIsTime=piIsTime)
        self.plotCat(self.x[:,2,:],color='r',\
                     labl=currSolLabl,piIsTime=piIsTime)
        plt.grid(True)
        plt.ylabel("V [m/s]")
        plt.xlabel(timeLabl)
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)

        plt.subplot2grid((5,1),(3,0))
        altSol.plotCat(altSol.u[:,0,:],mark='--',labl=altSolLabl,\
                       piIsTime=piIsTime)
        self.plotCat(self.u[:,0,:],color='k',labl=currSolLabl,\
                     piIsTime=piIsTime)
        plt.grid(True)
        plt.ylabel("u1 [-]")
        plt.xlabel(timeLabl)
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)

        plt.subplot2grid((5,1),(4,0))
        altgama = 0.5*numpy.pi*numpy.tanh(altSol.u)
        gama = 0.5*numpy.pi*numpy.tanh(self.u)
        altSol.plotCat(r2d*altgama[:,0,:],mark='--',labl=altSolLabl,\
                       piIsTime=piIsTime)
        self.plotCat(r2d*gama[:,0,:],color='k',labl=currSolLabl,\
                     piIsTime=piIsTime)
        plt.grid(True)
        plt.ylabel('Inclination angle [deg]')
        plt.xlabel(timeLabl)
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)

        self.savefig(keyName='comp',fullName='comparisons')
        self.log.printL("pi = "+str(pi)+"\n")

    def plotTraj(self,compare=False,altSol=None,altSolLabl='altSol', \
                 mustSaveFig=True):
        """Plot the trajectory of the sliding mass."""

        X = self.x[:,0,0]
        Y = self.x[:,1,0]
        currSolLabl = 'Final solution'

        if compare:
            if altSol is None:
                self.log.printL("plotTraj: comparing mode is set to True," + \
                                " but no solution was given to which " + \
                                "compare. Ignoring...")
                compare=False
            else:
                X_alt = altSol.x[:,0,0];
                Y_alt = altSol.x[:,1,0];
                plt.plot(X_alt,Y_alt,'b--',label=altSolLabl)
                plt.plot(X_alt[0],Y_alt[0],'o')
                plt.plot(X_alt[-1],Y_alt[-1],'s')
                plt.plot(X,Y,'k',label=currSolLabl)
                plt.plot(X[0],Y[0],'o')
                plt.plot(X[-1],Y[-1],'s')
                plt.axis('equal')
                plt.grid(True)
                plt.ylabel('y [m]')
                plt.xlabel("x [m]")
   #             titlStr = "Comparing trajectory solutions: " + currSolLabl + " and " + \
  #                altSolLabl
   #             titlStr += "\n(grad iter #" + str(self.NIterGrad) + ")\n"
                plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)
        else:
            plt.plot(X,Y)
            plt.plot(X[0],Y[0],'o')
            plt.plot(X[-1],Y[-1],'s')
            plt.axis('equal')
            plt.grid(True)
            plt.ylabel('y [m]')
            plt.xlabel("x [m]")
 #           titlStr = "Trajectory: " + currSolLabl
 #           titlStr += "\n(grad iter #" + str(self.NIterGrad) + ")\n"

 #       plt.title(titlStr)

        if mustSaveFig:
            self.savefig(keyName='traj',fullName='trajectory')
        else:
            plt.show()
            plt.clf()
    #
if __name__ == "__main__":
    print("\n\nRunning probBrac.py!\n")
    exmpProb = prob()

    print("Initializing problem:")
    exmpProb = exmpProb.initGues()
    exmpProb.printPars()
    s = exmpProb.s

    print("Plotting current version of solution:")
    exmpProb.plotSol()

    print("Calculating f:")
    f,_,_ = exmpProb.calcF()
    exmpProb.plotCat(f)
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('f')
    plt.show()

    print("Calculating grads:")
    Grads = exmpProb.calcGrads()
    for key in Grads.keys():
        print("Grads['",key,"'] = ",Grads[key])

    print("Calculating I:")
    I = exmpProb.calcI()
    print("I = ",I)