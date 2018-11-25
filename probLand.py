#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 12:21:03 2018

@author: levi

A module for testing the MSGRA formulation with a rocket-like problem: the
rocket lander.
"""

import numpy
from sgra import sgra
from itsme import problemConfigurationSGRA
from utils import simp
import matplotlib.pyplot as plt

class problemConfigurationSGRA2(problemConfigurationSGRA):
#    def dyn(self):
#        """Dynamics parameters."""
#
#        for key in ['a11', 'a12', 'a21', 'a22', 'b1', 'b2']:
#            self.con[key] = self.config.getfloat('dyn', key)

    def restr(self):
        """Restriction parameters."""

        for key in ['h', 'V']:
            self.con[key] = self.config.getfloat('restr', key)

    def vehicle(self):
        """Vehicle parameters."""

        for key in ['Mu', 'Mp', 'Isp', 'efes', 'T']:
            self.con[key] = self.config.getfloat('vehicle', key)

    def env(self):
        """Environment parameters."""

        for key in ['GM', 'R', 'g0']:
            self.con[key] = self.config.getfloat('env', key)

    def cost(self):
        """Cost function parameters."""

        for key in ['contCostWeig', 'timeCostWeig', \
                    'sttCostWeig11','sttCostWeig12', 'sttCostWeig22']:
            self.con[key] = self.config.getfloat('cost',key)


class prob(sgra):
    probName = 'probLand'

    def loadParsFromFile2(self,file):
        pConf = problemConfigurationSGRA2(fileAdress=file)
        #pConf.dyn()
        pConf.restr()
        pConf.vehicle()
        pConf.env()
        #pConf.cost()

        for key in ['Mu', 'Mp', 'Isp', 'efes', 'T', 'GM', 'R', 'g0']:
            self.constants[key] = pConf.con[key]

        self.constants['g'] = self.constants['GM']/(self.constants['R']**2)

        for key in ['h','V']:
            self.restrictions[key] = pConf.con[key]
        self.restrictions['M'] = self.constants['Mu'] + \
                                 self.constants['Mp']/self.constants['efes']

    def initGues(self,opt={}):

        # The parameters that go here are the ones that cannot be simply
        # altered from an external configuration file... at least not
        # without a big increase in the complexity of the code...

        n = 3
        m = 1
        s = 4
        q = n + (n-1) + n * (s-1)
        p = s
        self.n = n
        self.m = m
        self.p = p
        self.q = q
        self.s = s
        self.Ns = 2*n*s + p

        initMode = opt.get('initMode','default')
        if initMode == 'default':
            # matrix sizes
            self.log.printL("InitGues in default: Not implemented yet!")
#            N = 2000+1#20000+1#2000+1
#
#            self.N = N
#
#            dt = 1.0/(N-1)
#            t = numpy.arange(0,1.0+dt,dt)
#            self.dt = dt
#            self.t = t
#
#            #prepare tolerances
#            tolP = 1.0e-4#7#8
#            tolQ = 1.0e-6#8#5
#            tol = dict()
#            tol['P'] = tolP
#            tol['Q'] = tolQ
#
#            self.tol = tol
#
#
#            # Get initialization mode
#
#            x = numpy.zeros((N,n,s))
#            #u = numpy.zeros((N,m,s))#5.0*numpy.ones((N,m,s))
#            u = numpy.arctanh(0.5*numpy.ones((N,m,s)))
#            #x[:,0,0] = t.copy()
#            #for i in range(N):
#            #    x[i,1,0] = x[N-i-1,0,0]
#            #x[:,2,0] = numpy.sqrt(20.0*x[:,0,0])
#            pi = numpy.array([2.0/numpy.sqrt(10.0)])
#            td = t * pi[0]
#            x[:,0,0] = 2.5 * (td**2)
#            x[:,1,0] = 1.0 - x[:,0,0]
#            x[:,2,0] = numpy.sqrt(10.0 * x[:,0,0])
#
#            #x[:,0,0] = .5*t
#            #x[:,0,1] = .5+.5*t
#
#            lam = 0.0*x
#            mu = numpy.zeros(q)
#            #pi = 10.0*numpy.ones(p)
#
#
#            self.x = x
#            self.u = u
#            self.pi = pi
#            self.lam = lam
#            self.mu= mu
#
#            self.constants['gradStepSrchCte'] = 1e-3
#
#            solInit = self.copy()
#
#            self.log.printL("\nInitialization complete.\n")
#            return solInit

        elif initMode == 'extSol':
            inpFile = opt.get('confFile','')

            # Get parameters from file

            self.loadParsFromFile(file=inpFile)
            self.loadParsFromFile2(file=inpFile)

            # The actual "initial guess"

            N,m,n,p,q,s = self.N,self.m,self.n,self.p,self.q,self.s

            M0 = self.restrictions['M']
            Mp = self.constants['Mp']
            T = self.constants['T']

            psi0 = T / M0 / self.constants['g']
            phi0 = Mp / M0
            self.constants['phi0'] = phi0

            self.log.printL("Initial mass = {:.4E} kg".format(M0))
            self.log.printL("Propellant mass = {:.4E} kg".format(Mp) + \
                  ", phi0 = {:.4G}".format(phi0))
            self.log.printL("MaxThrust = {:.4E} kN".format(T) + \
                  ", psi0 = {:.4G}".format(psi0))
            self.log.printL("\nUnder constant specific thrust conditions:")
            vMax = ((psi0-1.)/psi0) * \
                    (self.constants['g0'] * self.constants['Isp']) * \
                    (-numpy.log(1.-phi0))
            self.log.printL(" Spacecraft vMax = {:.4E} km/s".format(vMax))

#            h0 = self.restrictions['h']
#            V0 = self.restrictions['V']
#            M0 = self.restrictions['M']
#            g = self.constants['g']
#            a = g - .5 * self.constants['T'] / M0
#            m = .5 * self.constants['T'] / \
#                  (self.constants['g0'] * self.constants['Isp'])
#
#            tF = (V0 + numpy.sqrt(V0**2 + 2. * a * h0))/a
#            t = tF * numpy.linspace(0.,1.,num=N)
#            x[:,2,0] = M0 - m * t
#
#
#            x[:,0,0] = h0 + V0 * t - .5 * a * t**2
#            x[:,1,0] = V0 - a * t
#
#            u[:,0,0] = -5.

            self.lander()

#            self.x = numpy.zeros((N,n,s))
#            self.u = numpy.zeros((N,m,s))
#            self.pi = numpy.array([100.])

            self.lam = 0.0 * self.x
            self.mu = numpy.zeros(q)

#            self.Kpf = 10.0
#            self.uLim = 1.0

            solInit = self.copy()

            self.log.printL("\nInitialization complete.\n")
            return solInit

#%%

    def calcPhi(self):
        N = self.N
        n = self.n
        s = self.s
        phi = numpy.empty((N,n,s))
        g0Isp = self.constants['g0'] * self.constants['Isp']
        g = self.constants['g']

        for arc in range(s):
            Thrust = self.constants['T'] * \
                    (.5 + .5 * numpy.tanh(self.u[:,0,arc]))
            phi[:,0,arc] = self.pi[arc] * self.x[:,1,arc]
            phi[:,1,arc] = self.pi[arc] * (-g + Thrust/self.x[:,2,arc])
            phi[:,2,arc] = - self.pi[arc] * Thrust / g0Isp

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
        fp = numpy.zeros((N,p,s))

        #psiy = numpy.eye(q,2*n*s)
        psiy = numpy.zeros((q,2*n*s))

        if s == 1:
            psiy[0, 0] = 1.0  # h(0) = h0
            psiy[1, 1] = 1.0  # V(0) = V0
            psiy[2, 2] = 1.0  # m(0) = m0
            psiy[3, 3] = 1.0  # h(1) = 0.0
            psiy[4, 4] = 1.0  # V(1) = 0.0
        elif s == 2:
            # psi = numpy.array([self.x[0, 0, 0] - h0,
            #                    self.x[0, 1, 0] - V0,
            #                    self.x[0, 2, 0] - M0,
            #                    self.x[0, 0, 1] - self.x[N - 1, 0, 0],
            #                    self.x[0, 1, 1] - self.x[N - 1, 1, 0],
            #                    self.x[0, 2, 1] - self.x[N - 1, 2, 0],
            #                    self.x[N - 1, 0, 1],
            #                    self.x[N - 1, 1, 1]])
            psiy[0, 0] = 1.0  # h(0) = h0
            psiy[1, 1] = 1.0  # V(0) = V0
            psiy[2, 2] = 1.0  # m(0) = m0
            psiy[3, 3] = -1.0
            psiy[3, 6] = 1.0
            psiy[4, 4] = -1.0
            psiy[4, 7] = 1.0
            psiy[5, 5] = -1.0
            psiy[5, 8] = 1.0
            psiy[6, 9] = 1.0
            psiy[7, 10] = 1.0
        elif s % 2 == 0:
            psiy[0, 0] = 1.0  # h(0) = h0
            psiy[1, 1] = 1.0  # V(0) = V0
            psiy[2, 2] = 1.0  # m(0) = m0

            # para s = 4
            # psi = numpy.array([self.x[0, 0, 0] - h0,
            #                    self.x[0, 1, 0] - V0,
            #                    self.x[0, 2, 0] - M0,
            #                    self.x[0, 0, 1] - self.x[N - 1, 0, 0],
            #                    self.x[0, 1, 1] - self.x[N - 1, 1, 0],
            #                    self.x[0, 2, 1] - self.x[N - 1, 2, 0],
            #                    self.x[0, 0, 2] - self.x[N - 1, 0, 1],
            #                    self.x[0, 1, 2] - self.x[N - 1, 1, 1],
            #                    self.x[0, 2, 2] - self.x[N - 1, 2, 1],
            #                    self.x[0, 0, 3] - self.x[N - 1, 0, 2],
            #                    self.x[0, 1, 3] - self.x[N - 1, 1, 2],
            #                    self.x[0, 2, 3] - self.x[N - 1, 2, 2],
            #                    self.x[N - 1, 0, 3],
            #                    self.x[N - 1, 1, 3]])

            jm, jp = 3, 6
            for arc in range(s-1):
                for k in range(n):
                    # psiy[n * (arc+1) + i, n * arc + i] = -1.0
                    # psiy[n * (arc+1) + i, n * (arc+1) + i] = 1.0
                    # ind = n * (arc + 1) + i
                    # psiy[ind, ind] = -1.0
                    # psiy[ind, ind + n] = 1.0
                    i = n * (arc+1) + k
                    psiy[i,jp+k] = 1.
                    psiy[i,jm+k] = -1.
                #
                jp += 2 * n
                jm += 2 * n

            psiy[q-2, 2*n*s-3] = 1.0 # hFinal = 0
            psiy[q-1, 2*n*s-2] = 1.0 # vFinal = 0
            #print("psiy =\n"+str(psiy))
        else:
            msg = "Psi gradients calculation only compatible" + \
                  "with s == 1 or 2."
            raise Exception(msg)

        psip = numpy.zeros((q,p))

        g0Isp = self.constants['g0'] * self.constants['Isp']
        g = self.constants['g']
        for arc in range(s):
            Thrust = .5 * self.constants['T'] * \
                     (1. + numpy.tanh(self.u[:,0,arc]))
            dTdu = .5 * self.constants['T'] * \
                     (1. - ( numpy.tanh(self.u[:,0,arc]) )**2  )

            phix[:,0,1,arc] = pi[arc]
            phix[:,1,2,arc] = - pi[arc] * Thrust / (self.x[:,2,arc]**2)

            phiu[:,1,0,arc] = pi[arc] * dTdu / self.x[:,2,arc]
            phiu[:,2,0,arc] = - pi[arc] * dTdu / g0Isp

            phip[:,0,arc,arc] = self.x[:,1,arc]
            phip[:,1,arc,arc] = Thrust/self.x[:,2,arc] - g
            phip[:,2,arc,arc] = - Thrust / g0Isp

            fp[:,arc,arc] = Thrust / g0Isp
#            fp[:,arc,arc] = sttCostWeig11 * (ex1**2) + \
#                            sttCostWeig22 * (ex2**2) + \
#                            2. * sttCostWeig12 * ex1 * ex2 + \
#                            contCostWeig * (self.u[:,0,arc]**2) + \
#                            timeCostWeig + \
#                            self.Kpf * ((self.u[:,0,arc]>self.uLim) * \
#                            (self.u[:,0,arc]-self.uLim)**2 + \
#                                (self.u[:,0,arc]<-self.uLim) * \
#                                (self.u[:,0,arc]+self.uLim)**2 )

#            fu[:,0,arc] = 2.0 * contCostWeig * self.u[:,0,arc] * pi[arc] +\
#                          2.0 * self.Kpf * pi[arc] * \
#                          ( (self.u[:,0,arc]-self.uLim) *  \
#                           (self.u[:,0,arc]>self.uLim) + \
#                           (self.u[:,0,arc]+self.uLim) *  \
#                           (self.u[:,0,arc]<-self.uLim) )
            fu[:,0,arc] = self.pi[arc] * dTdu / g0Isp
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
        N, q, s = self.N,  self.q, self.s

        h0 = self.restrictions['h']
        V0 = self.restrictions['V']
        M0 = self.restrictions['M']

        if s == 1:
            psi = numpy.array([self.x[0,0,0] - h0,
                                self.x[0,1,0] - V0,
                                self.x[0,2,0] - M0,
                                self.x[N-1,0,0],
                                self.x[N-1,1,0]])
        elif s == 2:
            psi = numpy.array([self.x[0,0,0] - h0,
                               self.x[0,1,0] - V0,
                               self.x[0,2,0] - M0,
                               self.x[0,0,1] - self.x[N-1,0,0],
                               self.x[0,1,1] - self.x[N-1,1,0],
                               self.x[0,2,1] - self.x[N-1,2,0],
                               self.x[N-1,0,1],
                               self.x[N-1,1,1]])
        elif s % 2 == 0:
            psi = numpy.empty(q)
            psi[0] = self.x[0, 0, 0] - h0
            psi[1] = self.x[0, 1, 0] - V0
            psi[2] = self.x[0, 2, 0] - M0

            i = 2
            for arc in range(s-1):
                #print("s = ",s,", arc = ",arc)
                i+=1
                psi[i] = self.x[0,0,arc+1] - self.x[N-1,0,arc]
                i+=1
                psi[i] = self.x[0,1,arc+1] - self.x[N-1,1,arc]
                i+=1
                psi[i] = self.x[0,2,arc+1] - self.x[N-1,2,arc]

            psi[q-2] = self.x[N - 1, 0, -1]
            psi[q-1] = self.x[N - 1, 1, -1]

        else:
            msg = "Psi calculation only compatible" + \
                  "with s even or s == 1."
            raise Exception(msg)
        return psi

    def calcF(self):
        N,s = self.N,self.s
        fOrig = numpy.empty((N,s))
        fPF = numpy.zeros((N,s))

#        sttEndP1 = self.restrictions['finish1']
#        sttEndP2 = self.restrictions['finish2']
#        contCostWeig = self.constants['contCostWeig']
#        timeCostWeig = self.constants['timeCostWeig']
#        sttCostWeig11 = self.constants['sttCostWeig11']
#        sttCostWeig12 = self.constants['sttCostWeig12']
#        sttCostWeig22 = self.constants['sttCostWeig22']
        g0Isp = self.constants['g0'] * self.constants['Isp']
        for arc in range(s):
            fOrig[:,arc] = (self.pi[arc] * self.constants['T'] / g0Isp) * \
                           .5 * (1. + numpy.tanh(self.u[:,0,arc]))
#            fPF[:,arc] = self.pi[arc] * self.Kpf * \
#                        ((self.u[:,0,arc]>self.uLim) * \
#                          (self.u[:,0,arc]- self.uLim)**2 + \
#                         (self.u[:,0,arc]<-self.uLim) * \
#                          (self.u[:,0,arc]+self.uLim)**2)

        return fOrig+fPF, fOrig, fPF

    def calcI(self):
#        N,s = self.N,self.s
#        _, fOrig, fPF = self.calcF()

#        IvecOrig = numpy.empty(s)
#        IvecPF = numpy.empty(s)
#        for arc in range(s):
#            IvecOrig[arc] = simp(fOrig[:,arc],N)
#            IvecPF[arc] = simp(fPF[:,arc],N)
#
#        IOrig, IPF = IvecOrig.sum(), IvecPF.sum()
#        return IOrig+IPF, IOrig, IPF

        IOrig = self.x[0,2,0] - self.x[-1,2,-1]
        return IOrig, IOrig, 0.0
#%%
    def plotSol(self,opt={},intv=[],piIsTime=True,mustSaveFig=True,
                subPlotAdjs={}):

        pi = self.pi

#        if len(intv)==0:
#            intv = numpy.arange(0,self.N,1,dtype='int')
#        else:
#             intv = list(intv)

        if len(intv)>0:
            self.log.printL("plotSol: Sorry, ignoring plotting range.")


        if opt.get('mode','sol') == 'sol':
            I, _, _ = self.calcI()
            titlStr = "Current solution: I = {:.4E}".format(I) + \
            " P = {:.4E} ".format(self.P) + " Q = {:.4E} ".format(self.Q)
            titlStr += "\n(grad iter #" + str(self.NIterGrad) + ")"
            ng = 5
            plt.subplot2grid((ng,1),(0,0),colspan=ng)
            self.plotCat(self.x[:,0,:])
            plt.grid(True)
            plt.ylabel("h [km]")
            plt.title(titlStr)
            plt.subplot2grid((ng,1),(1,0),colspan=ng)
            self.plotCat(self.x[:,1,:],color='g')
            plt.grid(True)
            plt.ylabel("V [km/s]")
            plt.subplot2grid((ng,1),(2,0),colspan=ng)
            self.plotCat(self.x[:,2,:],color='r')
            plt.grid(True)
            plt.ylabel("M [kg]")

            plt.subplot2grid((ng,1),(3,0),colspan=ng)
            self.plotCat(self.u[:,0,:],color='k')
            plt.grid(True)
            plt.ylabel("u1 [-]")
            plt.xlabel("Time [s]")
            plt.subplot2grid((ng,1),(4,0),colspan=ng)
            Thrust = .5 * self.constants['T'] * (1.+numpy.tanh(self.u[:,0,:]))
            self.plotCat(Thrust,color='k')
            plt.grid(True)
            plt.ylabel("Thrust [kN]")
            plt.xlabel("Time [s]")


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

            ng = 5
            plt.subplot2grid((ng,1),(0,0),colspan=ng)
            self.plotCat(dx[:,0,:])
            plt.grid(True)
            plt.ylabel("h [km]")
            plt.title(titlStr)
            plt.subplot2grid((ng,1),(1,0),colspan=ng)
            self.plotCat(dx[:,1,:],color='g')
            plt.grid(True)
            plt.ylabel("V [km/s]")
            plt.subplot2grid((ng,1),(2,0),colspan=ng)
            self.plotCat(dx[:,2,:],color='r')
            plt.grid(True)
            plt.ylabel("M [kg]")

            plt.subplot2grid((ng,1),(3,0),colspan=ng)
            self.plotCat(du[:,0,:],color='k')
            plt.grid(True)
            plt.ylabel("u1 [-]")
            plt.xlabel("Time [s]")
            plt.subplot2grid((ng,1),(4,0),colspan=ng)
            Thr = .5 * self.constants['T'] * (1.+numpy.tanh(self.u[:,0,:]))
            NewThr = .5 * self.constants['T'] * \
                (1.+numpy.tanh(self.u[:,0,:]+du[:,0,:]))
            self.plotCat(NewThr-Thr,color='k')
            plt.grid(True)
            plt.ylabel("Thrust [kN]")
            plt.xlabel("Time [s]")

            self.savefig(keyName='corr',fullName='corrections')

        elif opt['mode'] == 'lambda':
            titlStr = "Lambda for current solution"

            ng = 5
            plt.subplot2grid((ng,1),(0,0),colspan=ng)
            self.plotCat(self.lam[:,0,:])
            plt.grid(True)
            plt.ylabel("lambda: h")
            plt.title(titlStr)
            plt.subplot2grid((ng,1),(1,0),colspan=ng)
            self.plotCat(self.lam[:,1,:],color='g')
            plt.grid(True)
            plt.ylabel("lambda: v")
            plt.subplot2grid((ng,1),(2,0),colspan=ng)
            self.plotCat(self.lam[:,2,:],color='r')
            plt.grid(True)
            plt.ylabel("lambda: m")
            plt.subplot2grid((ng,1),(3,0),colspan=ng)
            self.plotCat(self.u[:,0,:],color='k')
            plt.grid(True)
            plt.ylabel("u1 [-]")
            plt.xlabel("Time [s]")
            plt.subplot2grid((ng,1),(4,0),colspan=ng)
            Thrust = .5 * self.constants['T'] * (1.+numpy.tanh(self.u[:,0,:]))
            self.plotCat(Thrust,color='k')
            plt.grid(True)
            plt.ylabel("Thrust [kN]")
            plt.xlabel("Time [s]")

            plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
            self.savefig(keyName='currLamb',fullName='lambdas')

            self.log.printL("mu = "+str(self.mu))

        else:
            titlStr = opt['mode']

#    def plotTraj(self,mustSaveFig=True,altSol=None,name=None):
#        """Plot the trajectory on the state space."""
#
#        X = self.x[:,0,0]
#        Y = self.x[:,1,0]
#
#        plt.plot(X,Y)
#        plt.plot(X[0],Y[0],'o')
#        plt.plot(X[-1],Y[-1],'s')
#        plt.axis('equal')
#        plt.grid(True)
#        plt.xlabel("x1")
#        plt.ylabel("x2")
#
#        titlStr = "Trajectory "
#        titlStr += "(grad iter #" + str(self.NIterGrad) + ")\n"
#        plt.title(titlStr)
#        #plt.legend(loc="upper left", bbox_to_anchor=(1,1))
#
#        if mustSaveFig:
#            self.savefig(keyName='traj',fullName='trajectory')
#        else:
#            plt.show()
#            plt.clf()

    def lander(self,opt=None):
        """ Proposes an initial guess for the lander problem.
        P
        FP
        FPFP
        FPFPFP

        "General" case: as many propulsive stages as required,
        but without having to specify too many parameters

        s = 2 * Np 'stages'; Np equal falls,
        1 first stopping thrust to rest, then
        Np-1 equal stopping thrusts from the same initial speed to rest,


        Only one parameter is necessary, either the falling time
        or the limit speed. How to provide boundaries for these parameters?


        Maximum tFall = ?
        0 = h + V * t - .5 * g * t² ->
        t = (-V +- sqrt(V²+2gh))/-g = V/g +- sqrt(V²+2gh)/g = (V+sqrt(V²+2gh)/g
        so, maximum tFall = (V+sqrt(V²+2gh))/g.
        under these conditions, Vfinal = V - g * tFall = -sqrt(V²+2gh).
        """

        self.log.printL("\nIn lander.\n")

        g0Isp = self.constants['g0'] * self.constants['Isp']
        g = self.constants['g']
        Tmax = self.constants['T']
        phi0 = self.constants['phi0']

        N, n, m, s = self.N, self.n, self.m, self.s
        h, V, M = self.restrictions['h'], self.restrictions['V'], \
                    self.restrictions['M']

        # TODO: this should become a parameter in the .its file!
        psi = .25 * Tmax / M / g

        # calculate maximum velocity limit for starting propulsive phase,
        # so that all the propellant is used at the moment that s/c stops.
        vLimProp = (1.-1./psi) * g0Isp * abs(numpy.log(1.-phi0))
        self.log.printL("Propellant-based vLim = {:.4E} km/s".format(vLimProp))

        vLim = numpy.sqrt((V*V + 2. * g * h) * (2./s) * (1.-1./psi))

        tFall = vLim / g
        tStop = tFall / (psi-1.)
        t1 = tFall + V / g
        #h1 = (vLim*vLim - V*V) / 2. / g

        # dh = vLim * vLim / 2. / g / (psi-1.)
        # if dh > h:
        #     hLim = (V*V + 2. * g * h) / 2. / g / psi
        #     vLim = numpy.sqrt(V*V + 2. * g * (h - hLim))
        #     self.log.printL("Since the spacecraft crashes before reaching" + \
        #                     " that speed, recalculating...\n" + \
        #                     " vLim = {:.4E} km/s".format(vLim))
        # else:
        #     msg = "Error: too little propellant to stop rocket,\n" + \
        #           "at least under current configurations." + \
        #           "\n Unfeasible problem."
        #     raise Exception(msg)

        self.x = numpy.zeros((N, n, s))
        self.u = numpy.zeros((N, m, s))

        if s == 1:
            tTot = tStop + tFall
            hLim = (V * V + 2. * g * h) / 2. / g / psi
            self.pi = numpy.array([tTot])
            dt = tTot/(N-1)
            kT = int(tFall * N / tTot)
            tFallVec = numpy.linspace(0.,1.,num=kT) * (tFall-dt)


            self.x[:kT,0,0] = h + V * tFallVec - .5 * g * tFallVec ** 2.
            self.x[:kT,1,0] = V - g * tFallVec
            self.x[:kT,2,0] = M + tFallVec * 0.
            self.u[:kT,0,0] = -6.

            tStopVec = numpy.linspace(0.,1.,num=(N-kT)) * tStop
            self.x[kT:,0,0] = hLim - vLim * tStopVec + \
                                    .5 * g * (psi-1.) * tStopVec ** 2.
            self.x[kT:,1,0] = -vLim + g * (psi - 1.) * tStopVec
            self.x[kT:,2,0] = M * numpy.exp(-psi * tStopVec * g / g0Isp)

            thr = psi * M * numpy.exp(-psi * tStopVec * g / g0Isp) * g
            self.u[kT:,0,0] = numpy.arctanh(2. * thr / Tmax - 1.)
        elif s == 2:
            self.pi = numpy.array([tFall,tStop])
            tFallVec = numpy.linspace(0., 1., num=N) * tFall
            hLim = (V * V + 2. * g * h) / 2. / g / psi
            self.x[:, 0, 0] = h + V * tFallVec - .5 * g * tFallVec ** 2.
            self.x[:, 1, 0] = V - g * tFallVec
            self.x[:, 2, 0] = M + tFallVec * 0.
            self.u[:, 0, 0] = -6.

            tStopVec = numpy.linspace(0., 1., num=N) * tStop
            self.x[:, 0, 1] = hLim - vLim * tStopVec + \
                                .5 * g * (psi - 1.) * tStopVec ** 2.
            self.x[:, 1, 1] = -vLim + g * (psi - 1.) * tStopVec
            self.x[:, 2, 1] = M * numpy.exp(-psi * tStopVec * g / g0Isp)

            thr = psi * M * numpy.exp(-psi * tStopVec * g / g0Isp) * g
            self.u[:, 0, 1] = numpy.arctanh(2. * thr / Tmax - 1.)
        elif s % 2 == 0:
            self.pi = numpy.zeros(s)
            s2 = int(s / 2)
            self.pi[0] = t1
            t1Vec = numpy.linspace(0., 1., num=N) * t1
            self.x[:, 0, 0] = h + V * t1Vec - .5 * g * t1Vec ** 2.
            self.x[:, 1, 0] = V - g * t1Vec
            self.x[:, 2, 0] = M
            self.u[:, 0, 0] = -6.

            tStopVec = numpy.linspace(0., 1., num=N) * tStop
            tFallVec = numpy.linspace(0., 1., num=N) * tFall

            arc = 0
            for k in range(s2):
                h, M = self.x[-1, 0, arc], self.x[-1, 2, arc]
                arc += 1
                self.pi[arc] = tStop
                x0StopVec = h - vLim * tStopVec + \
                            .5 * g * (psi - 1.) * tStopVec ** 2.
                x1StopVec = -vLim + g * (psi - 1.) * tStopVec
                x2StopVec = M * numpy.exp(-psi * tStopVec * g / g0Isp)
                thr = psi * M * numpy.exp(-psi * tStopVec * g / g0Isp) * g
                uStopVec = numpy.arctanh(2. * thr / Tmax - 1.)
                self.x[:, 0, arc] = x0StopVec
                self.x[:, 1, arc] = x1StopVec
                self.x[:, 2, arc] = x2StopVec
                self.u[:, 0, arc] = uStopVec


                h, M = self.x[-1, 0, arc], self.x[-1, 2, arc]
                arc += 1
                if arc == s:
                    break
                self.pi[arc] = tFall
                x0FallVec = h + V * tFallVec - .5 * g * tFallVec ** 2.
                x1FallVec = - g * tFallVec
                x2FallVec = M + tFallVec * 0.
                self.x[:, 0, arc] = x0FallVec
                self.x[:, 1, arc] = x1FallVec
                self.x[:, 2, arc] = x2FallVec
                self.u[:, 0, arc] = -6.

        else:
            msg = "\nLander method undefined for s = " + str(s)
            raise Exception(msg)
        #


        self.log.printL("\nFinal mass = {:.1F} kg.".format(M))

    #


if __name__ == "__main__":
    print("\n\nRunning probLand.py!\n")
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