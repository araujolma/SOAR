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
from utils import simp, avoidRepCalc
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

    def accel(self):
        """Acceleration limitation parameters."""

        for key in ['acc_max', 'acc_max_relTol', 'PFtol']:
            self.con[key] = self.config.getfloat('accel',key)

        self.con['PFmode'] = self.config.get('accel', 'PFmode')

class prob(sgra):
    probName = 'probLand'

    def loadParsFromFile2(self,file):
        pConf = problemConfigurationSGRA2(fileAdress=file)
        #pConf.dyn()
        pConf.restr()
        pConf.vehicle()
        pConf.env()
        pConf.accel()

        for key in ['Mu', 'Mp', 'Isp', 'efes', 'T', 'GM', 'R', 'g0']:
            self.constants[key] = pConf.con[key]

        self.constants['g'] = self.constants['GM']/(self.constants['R']**2)

        for key in ['h','V']:
            self.restrictions[key] = pConf.con[key]
        self.restrictions['M'] = self.constants['Mu'] + \
                                 self.constants['Mp']/self.constants['efes']
        #self.restrictions['hList'] = []

        acc_max = pConf.con['acc_max'] * self.constants['g']  # in km/s²
        self.restrictions['acc_max'] = acc_max
        # Penalty function settings

        # This approach to Kpf is that if, in any point during flight, the
        # acceleration exceeds the limit by acc_max_tol, then the penalty
        # function at that point is PFtol times greater than the maximum
        # value of the original cost functional.

        PFmode = pConf.con['PFmode']
        costFuncVals = pConf.con['T'] / pConf.con['g0'] / pConf.con['Isp']
        maxCost = costFuncVals

        PFtol = pConf.con['PFtol']
        acc_max_relTol = pConf.con['acc_max_relTol']
        acc_max_tol = acc_max_relTol * acc_max
        if PFmode == 'lin':
            Kpf = PFtol * maxCost / acc_max_tol
        elif PFmode == 'quad':
            Kpf = PFtol * maxCost / (acc_max_tol ** 2)
        elif PFmode == 'tanh':
            Kpf = PFtol * maxCost / numpy.tanh(acc_max_relTol)
        else:
            self.log.printL('Error: unknown PF mode "' + str(PFmode) + '"')
            raise KeyError
        self.log.printL("Kpf = " + str(Kpf) + "\n")
        self.log.prom("Please confirm Kpf value. ")
        self.constants['PFmode'] =  PFmode
        self.constants['Kpf'] = Kpf


    def initGues(self,opt={}):

        # The parameters that go here are the ones that cannot be simply
        # altered from an external configuration file... at least not
        # without a big increase in the complexity of the code...

        n = 3
        m = 1
        s = 2#4
        # This q expression works for s=1 only:
        #q = n + (n-1) + n * (s-1)
        # This is more general:
        q = n + (n-1) + (n+1) * (s-1)
        p = s
        self.n = n
        self.m = m
        self.p = p
        self.q = q
        self.s = s
        self.Ns = 2*n*s + p
        self.ctrlPar = 'tanh'#'sin' #

        # omission configurations
        if s == 1:
            # psi = numpy.array([self.x[0,0,0] - h0,   ---> 0, omitted
            #                     self.x[0,1,0] - V0,  ---> 1, omitted
            #                     self.x[0,2,0] - M0,  ---> 2, omitted
            #                     self.x[N-1,0,0],
            #                     self.x[N-1,1,0]])
            mat = numpy.eye(self.q)
            # matrix for omitting equations
            self.omitEqMat = mat[[3, 4], :]
            # list of variations after omission
            self.omitVarList = [           # states for 1st arc (all omitted)
                                3, 4, 5,   # Lambdas for 1st arc
                                6,         # pi for 1st arc
                                7]         # final variation
            self.omit = True
        elif s == 2:
            # hBound = self.restrictions['hList'][0]
            # psi = numpy.array([self.x[0,0,0] - h0,               ---> 0, omitted
            #                    self.x[0,1,0] - V0,               ---> 1, omitted
            #                    self.x[0,2,0] - M0,               ---> 2, omitted
            #                    self.x[N - 1, 0, 0] - hBound,
            #                    self.x[0,0,1] - hBound,           ---> 4, omitted
            #                    self.x[0,1,1] - self.x[N-1,1,0],
            #                    self.x[0,2,1] - self.x[N-1,2,0],
            #                    self.x[N-1,0,1],
            #                    self.x[N-1,1,1]])
            # list of variations after omission
            mat = numpy.eye(self.q)
            # matrix for omitting equations
            self.omitEqMat = mat[[3, 5, 6, 7, 8], :]
            # list of variations after omission
            self.omitVarList = [           # states for 1st arc (all omitted)
                                3, 4, 5,   # Lambdas for 1st arc
                                7, 8,      # states for 2nd arc (height omitted)
                                9, 10, 11, # Lambdas for 2nd arc
                                12, 13,    # pi's, 1st and 2nd arc
                                14]        # final variation
            self.omit = True
        elif s % 2 == 0:
            # psi = numpy.empty(q)
            # psi[0] = self.x[0, 0, 0] - h0
            # psi[1] = self.x[0, 1, 0] - V0
            # psi[2] = self.x[0, 2, 0] - M0
            #
            # hList = self.restrictions['hList']
            # i = 2
            # for arc in range(s-1):
            #     i += 1
            #     psi[i] = self.x[N - 1, 0, arc] - hList[arc]
            #     i += 1
            #     psi[i] = self.x[0, 0, arc + 1] - hList[arc]
            #     i += 1
            #     psi[i] = self.x[0, 1, arc + 1] - self.x[N - 1, 1, arc]
            #     i += 1
            #     psi[i] = self.x[0, 2, arc + 1] - self.x[N - 1, 2, arc]
            #
            # psi[q-2] = self.x[N - 1, 0, -1]
            # psi[q-1] = self.x[N - 1, 1, -1]
            self.log.printL("\nNo omission available yet for this value of s. Sorry.")
            self.omit = False
        else:
            # Not even a psi calculation for this configuration.
            self.omit = False

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

    def calcDimCtrl(self,ext_u = None,driv=False):
        """Calculate beta (thrust), or its derivative, from either
        the object's own control (self.u) or external control
        (additional parameter needed).
        driv parameter determines if the mode is for derivative or not."""

        restrictions = self.restrictions
        if ext_u is None:
            u = self.u
        else:
            u = ext_u

        if self.ctrlPar == 'tanh':
            if driv:
                beta = .5 * (1. - numpy.tanh(u[:,0,:])**2)
            else:
                beta = .5 * (1. + numpy.tanh(u[:,0,:]))
        elif self.ctrlPar == 'sin':
            if driv:
                beta = .5 * numpy.cos(u[:,0,:])
            else:
                beta = .5 * (1. + numpy.sin(u[:,0,:]))
        else:
            #beta = 0. * u
            raise(Exception("calcDimCtrl: Unknown control parametrization."))

        return beta

    def calcAdimCtrl(self, beta):
        """Calculate non-dimensional control from a given thrust profile
        (beta), which is an external array."""

        # sh = beta.shape
        # if len(sh) == 2:
        #     Nu, su = sh
        # else:
        #     Nu = sh
        #     su = 1
        #
        # print(Nu)
        # print(su)
        Nu = len(beta)

        if self.ctrlPar == 'tanh':
            # Basic saturation

            sat = 0.99999

            lowL = .5 * (1-sat)
            highL = .5 * (1+sat)
            for k in range(Nu):
                if beta[k] > highL:
                    beta[k] = highL
                if beta[k] < lowL:
                    beta[k] = lowL
            #
            u = numpy.arctanh(2.*beta-1.)


        elif self.ctrlPar == 'sin':
            # Basic saturation
            for k in range(Nu):
                if beta[k] > 1.:
                    beta[k] = 1.
                if beta[k] < -1.:
                    beta[k] = -1.
            #
            u = numpy.arcsin(2.*beta-1.)

        else:
            raise (Exception("calcAdimCtrl: Unknown control parametrization."))

        return u

    @avoidRepCalc(fieldsTuple=('phi',))
    def calcPhi(self):
        N = self.N
        n = self.n
        s = self.s
        phi = numpy.empty((N,n,s))
        g0Isp = self.constants['g0'] * self.constants['Isp']
        g = self.constants['g']

        for arc in range(s):
            Thrust = self.constants['T'] * self.calcDimCtrl()
            phi[:,0,arc] = self.pi[arc] * self.x[:,1,arc]
            phi[:,1,arc] = self.pi[arc] * (-g + Thrust[:,arc]/self.x[:,2,arc])
            phi[:,2,arc] = - self.pi[arc] * Thrust[:,arc] / g0Isp

        return phi

#%%
    def calcAcc(self):
        """Calculate acceleration "felt" by the rocket."""

        acc = numpy.empty((self.N, self.s))

        # Calculate acceleration
        phi = self.calcPhi()
        for arc in range(self.s):
            acc[:, arc] = phi[:, 1, arc] / self.pi[arc]

        # add a 'g' so that it returns the acc. "felt" inside the rocket
        return acc + self.constants['g']

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
            psiy[0, 0] = 1.0  # h(0,0) = h0          (0)
            psiy[1, 1] = 1.0  # V(0,0) = V0          (1)
            psiy[2, 2] = 1.0  # m(0,0) = m0          (2)
            psiy[3, 3] = 1.0  # h(1,0) = hBound      (3)
            psiy[4, 6] = 1.0  # h(0,1) = hBound      (4)
            psiy[5, 4] = -1.0 # V(0,1) - V(1,0) = 0  (5)
            psiy[5, 7] = 1.0  # V(0,1) - V(1,0) = 0  (5)
            psiy[6, 5] = -1.0 # M(0,1) - M(1,0) = 0  (6)
            psiy[6, 8] = 1.0  # M(0,1) - M(1,0) = 0  (6)
            psiy[7, 9] = 1.0  # h(1,1) = 0           (7)
            psiy[8, 10] = 1.0 # V(1,1) = 0           (8)
            # self.x[0, 0, 0] - h0,
            # self.x[0, 1, 0] - V0,
            # self.x[0, 2, 0] - M0,
            # self.x[N - 1, 0, 0] - hBound,
            # self.x[0, 0, 1] - hBound,
            # self.x[0, 1, 1] - self.x[N - 1, 1, 0],
            # self.x[0, 2, 1] - self.x[N - 1, 2, 0],
            # self.x[N - 1, 0, 1],
            # self.x[N - 1, 1, 1]
        elif s % 2 == 0:
            psiy[0, 0] = 1.0  # h(0) = h0
            psiy[1, 1] = 1.0  # V(0) = V0
            psiy[2, 2] = 1.0  # m(0) = m0

            # jm, jp = 3, 6
            # for arc in range(s-1):
            #     for k in range(n):
            #         # psiy[n * (arc+1) + i, n * arc + i] = -1.0
            #         # psiy[n * (arc+1) + i, n * (arc+1) + i] = 1.0
            #         # ind = n * (arc + 1) + i
            #         # psiy[ind, ind] = -1.0
            #         # psiy[ind, ind + n] = 1.0
            #         i = n * (arc+1) + k
            #         psiy[i,jp+k] = 1.
            #         psiy[i,jm+k] = -1.
            #     #
            #     jp += 2 * n
            #     jm += 2 * n

            for arc in range(s-1):
                i = n + arc * (n+1)
                j = n + arc * 2 * n
                psiy[i,j] = 1.;       psiy[i+1,j+n] = 1.  # separate height
                psiy[i+2,j+n+1] = 1.; psiy[i+2,j+1] = -1. # continuity in speed
                psiy[i+3,j+n+2] = 1.; psiy[i+3,j+2] = -1. # continuity in mass

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
        Thrust = self.constants['T'] * self.calcDimCtrl()
        dTdu = self.constants['T'] * self.calcDimCtrl(driv=True)

        # penalty function terms
        acc = self.calcAcc()
        acc_max = self.restrictions['acc_max']
        Kpf = self.constants['Kpf']
        PFmode = self.constants['PFmode']
        PenaltyIsTrue = (acc >= acc_max)
        if PFmode == 'lin':
            K2dAPen = Kpf * PenaltyIsTrue
        elif PFmode == 'quad':
            K2dAPen = 2.0 * Kpf * (acc - acc_max) * PenaltyIsTrue
        elif PFmode == 'tanh':
            tanh = numpy.tanh
            K2dAPen = Kpf * PenaltyIsTrue * \
                      (1.0 - tanh(acc / acc_max - 1.0) ** 2) * (1.0 / acc_max)
        else:
            self.log.printL("Unknown PFmode '"+str(PFmode)+"'.")
            raise KeyError
        #self.log.printL("\nK2dAPen = "+str(K2dAPen))

        for arc in range(s):

            phix[:,0,1,arc] = pi[arc]
            phix[:,1,2,arc] = - pi[arc] * Thrust[:,arc] / (self.x[:,2,arc]**2)

            phiu[:,1,0,arc] = pi[arc] * dTdu[:,arc] / self.x[:,2,arc]
            phiu[:,2,0,arc] = - pi[arc] * dTdu[:,arc] / g0Isp

            phip[:,0,arc,arc] = self.x[:,1,arc]
            phip[:,1,arc,arc] = Thrust[:,arc] / self.x[:,2,arc] - g
            phip[:,2,arc,arc] = - Thrust[:,arc] / g0Isp

            # acc = self.pi[arc] * (-g + Thrust[:,arc]/self.x[:,2,arc])
            #fp[:, arc, arc] = Thrust[:, arc] / g0Isp
            fp[:,arc,arc] = Thrust[:,arc] / g0Isp + \
                            Kpf * PenaltyIsTrue[:,arc] * (acc[:,arc]-acc_max)**2

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
            fu[:,0,arc] = self.pi[arc] * dTdu[:,arc] * \
                          (1./g0Isp + K2dAPen[:,arc]/self.x[:,2,arc])

            fx[:,2,arc] = K2dAPen[:,arc] * phix[:,1,2,arc]
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
            hBound = self.restrictions['hList'][0]
            psi = numpy.array([self.x[0,0,0] - h0,
                               self.x[0,1,0] - V0,
                               self.x[0,2,0] - M0,
                               self.x[N - 1, 0, 0] - hBound,
                               self.x[0,0,1] - hBound,
                               self.x[0,1,1] - self.x[N-1,1,0],
                               self.x[0,2,1] - self.x[N-1,2,0],
                               self.x[N-1,0,1],
                               self.x[N-1,1,1]])
        elif s % 2 == 0:
            psi = numpy.empty(q)
            psi[0] = self.x[0, 0, 0] - h0
            psi[1] = self.x[0, 1, 0] - V0
            psi[2] = self.x[0, 2, 0] - M0

            # i = 2
            # for arc in range(s-1):
            #     #print("s = ",s,", arc = ",arc)
            #     i+=1
            #     psi[i] = self.x[0,0,arc+1] - self.x[N-1,0,arc]
            #     i+=1
            #     psi[i] = self.x[0,1,arc+1] - self.x[N-1,1,arc]
            #     i+=1
            #     psi[i] = self.x[0,2,arc+1] - self.x[N-1,2,arc]

            hList = self.restrictions['hList']
            i = 2
            for arc in range(s-1):
                i += 1
                psi[i] = self.x[N - 1, 0, arc] - hList[arc]
                i += 1
                psi[i] = self.x[0, 0, arc + 1] - hList[arc]
                i += 1
                psi[i] = self.x[0, 1, arc + 1] - self.x[N - 1, 1, arc]
                i += 1
                psi[i] = self.x[0, 2, arc + 1] - self.x[N - 1, 2, arc]


            psi[q-2] = self.x[N - 1, 0, -1]
            psi[q-1] = self.x[N - 1, 1, -1]

        else:
            msg = "Psi calculation only compatible" + \
                  "with s even or s == 1."
            raise Exception(msg)
        return psi

    @avoidRepCalc(fieldsTuple=('f', 'fOrig', 'f_pf'))
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
        Thr = self.constants['T'] * self.calcDimCtrl()

        acc = self.calcAcc()
        acc_max = self.restrictions['acc_max']
        for arc in range(s):
            fOrig[:,arc] = (self.pi[arc] / g0Isp) * Thr[:,arc]

            fPF[:,arc] = self.pi[arc] * self.constants['Kpf'] * \
                        ((acc[:,arc]>=acc_max) * (acc[:,arc]-acc_max)**2)

        return fOrig+fPF, fOrig, fPF

    @avoidRepCalc(fieldsTuple=('I', 'Iorig', 'I_pf'))
    def calcI(self):
        N,s = self.N,self.s
        _, fOrig, fPF = self.calcF()

        IvecOrig = numpy.empty(s)
        IvecPF = numpy.empty(s)
        for arc in range(s):
            IvecOrig[arc] = simp(fOrig[:,arc],N)
            IvecPF[arc] = simp(fPF[:,arc],N)
#
        IOrig, IPF = IvecOrig.sum(), IvecPF.sum()
        IOrig = self.x[0, 2, 0] - self.x[-1, 2, -1]
        return IOrig+IPF, IOrig, IPF

#        IOrig = self.x[0,2,0] - self.x[-1,2,-1]
#        return IOrig, IOrig, 0.0
#%%
    def plotSol(self,opt={},intv=None,piIsTime=True,mustSaveFig=True,
                subPlotAdjs={}):

        pi = self.pi


        if piIsTime:
            timeLabl = 't [s]'
            tVec = [0.0, self.pi.sum()]
        else:
            timeLabl = 'adim. t [-]'
            tVec = [0.0, float(self.s)]


        if opt.get('mode','sol') == 'sol':
            I, _, _ = self.calcI()
            titlStr = "Current solution: I = {:.4E}".format(I) + \
            " P = {:.4E} ".format(self.P) + " Q = {:.4E} ".format(self.Q)
            titlStr += "\n(grad iter #" + str(self.NIterGrad) + ")"
            ng = 7
            plt.subplot2grid((ng,1),(0,0),colspan=ng)
            self.plotCat(self.x[:,0,:],intv=intv,piIsTime=piIsTime)
            plt.grid(True)
            plt.ylabel("h [km]")
            plt.title(titlStr)
            plt.subplot2grid((ng,1),(1,0),colspan=ng)
            self.plotCat(self.x[:,1,:],intv=intv,color='g',piIsTime=piIsTime)
            plt.grid(True)
            plt.ylabel("V [km/s]")
            plt.subplot2grid((ng,1),(2,0),colspan=ng)
            self.plotCat(self.x[:,2,:],intv=intv,color='r',piIsTime=piIsTime)
            plt.grid(True)
            plt.ylabel("M [kg]")

            plt.subplot2grid((ng,1),(3,0),colspan=ng)
            self.plotCat(self.u[:,0,:],intv=intv,color='k',piIsTime=piIsTime)
            plt.grid(True)
            plt.ylabel("u1 [-]")

            plt.subplot2grid((ng,1),(4,0),colspan=ng)
            Thrust = self.constants['T'] * self.calcDimCtrl()
            Weight = self.constants['g'] * self.x[:,2,:]
            self.plotCat(Thrust,intv=intv,color='r',labl='Thrust',
                         piIsTime=piIsTime)
            self.plotCat(Weight,intv=intv,color='k',labl='Weight',
                         piIsTime=piIsTime)
            plt.grid(True)
            plt.ylabel("Force [kN]")
            plt.legend()

            plt.subplot2grid((ng, 1), (5, 0), colspan=ng)
            acc = self.calcAcc()
            self.plotCat(acc/self.constants['g'], intv=intv, labl='acc',
                         piIsTime=piIsTime)
            plt.plot(tVec, self.restrictions['acc_max'] * \
                     numpy.array([1.0, 1.0])/self.constants['g'], '--',
                     label='acc_lim')
            plt.grid(True)
            plt.legend()
            plt.ylabel("Acceleration [g]")
            plt.xlabel(timeLabl)

            ax = plt.subplot2grid((ng, 1), (6, 0))
            s = self.s
            position = numpy.arange(s)
            stages = numpy.arange(1, s + 1)
            width = 0.4
            ax.bar(position, pi, width, color='b')
            # Put the values of the arc lengths on the bars...
            for arc in range(s):
                coord = (float(arc) - .25 * width, pi[arc] + 10.)
                ax.annotate("{:.1E}".format(pi[arc]), xy=coord, xytext=coord)
            ax.set_xticks(position)
            ax.set_xticklabels(stages)
            plt.grid(True, axis='y')
            plt.xlabel("Arcs")
            plt.ylabel("Duration [s]")

            plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.45)
            if mustSaveFig:
                self.savefig(keyName='currSol',fullName='solution')
            else:
                plt.show()

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
            titlStr += "\nDelta pi (%): "
            for i in range(self.p):
                titlStr += "{:.1F}, ".format(100.0 * dp[i] / self.pi[i])

            plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)

            ng = 5
            plt.subplot2grid((ng,1),(0,0),colspan=ng)
            self.plotCat(dx[:,0,:],intv=intv,piIsTime=piIsTime)
            plt.grid(True)
            plt.ylabel("h [km]")
            plt.title(titlStr)
            plt.subplot2grid((ng,1),(1,0),colspan=ng)
            self.plotCat(dx[:,1,:],intv=intv,color='g',piIsTime=piIsTime)
            plt.grid(True)
            plt.ylabel("V [km/s]")

            plt.subplot2grid((ng,1),(2,0),colspan=ng)
            self.plotCat(dx[:,2,:],intv=intv,color='r',piIsTime=piIsTime)
            plt.grid(True)
            plt.ylabel("M [kg]")


            plt.subplot2grid((ng,1),(3,0),colspan=ng)
            self.plotCat(du[:,0,:],intv=intv,color='k',piIsTime=piIsTime)
            plt.grid(True)
            plt.ylabel("u1 [-]")

            plt.subplot2grid((ng,1),(4,0),colspan=ng)
            Thr = self.constants['T'] * self.calcDimCtrl()
            new_u = self.u + du
            NewThr = self.constants['T'] * self.calcDimCtrl(ext_u=new_u)
            #deltaThr = NewThr-Thr
            self.plotCat(NewThr-Thr,intv=intv,color='k',piIsTime=piIsTime)
            plt.grid(True)
            plt.ylabel("Thrust [kN]")
            plt.xlabel(timeLabl)

            self.savefig(keyName='corr',fullName='corrections')

        elif opt['mode'] == 'lambda':
            titlStr = "Lambda for current solution"

            ng = 5
            plt.subplot2grid((ng,1),(0,0),colspan=ng)
            self.plotCat(self.lam[:,0,:],intv=intv,piIsTime=piIsTime)
            plt.grid(True)
            plt.ylabel("lambda: h")
            plt.title(titlStr)
            plt.subplot2grid((ng,1),(1,0),colspan=ng)
            self.plotCat(self.lam[:,1,:],intv=intv,color='g',
                         piIsTime=piIsTime)
            plt.grid(True)
            plt.ylabel("lambda: v")
            plt.subplot2grid((ng,1),(2,0),colspan=ng)
            self.plotCat(self.lam[:,2,:],intv=intv,color='r',
                         piIsTime=piIsTime)
            plt.grid(True)
            plt.ylabel("lambda: m")
            plt.subplot2grid((ng,1),(3,0),colspan=ng)
            self.plotCat(self.u[:,0,:],intv=intv,color='k',
                         piIsTime=piIsTime)
            plt.grid(True)
            plt.ylabel("u1 [-]")
            plt.xlabel("Time [s]")
            plt.subplot2grid((ng,1),(4,0),colspan=ng)
            Thrust = self.constants['T'] * self.calcDimCtrl()
            self.plotCat(Thrust,intv=intv,color='k',piIsTime=piIsTime)
            plt.grid(True)
            plt.ylabel("Thrust [kN]")
            plt.xlabel(timeLabl)

            plt.subplots_adjust(0.0125,0.0,0.9,3.5,0.2,0.2)
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

        s = 2 * Np 'stages'; Np equal falls (still thrusting, with psi<1),
        1 first (thrusting) fall to vLim, then
        Np-1 equal stopping thrusts (with psi>1) from the same initial speed
       (vLim) to rest.

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
        psiHigh = 0.5 * Tmax / M / g
        psiLow = 0.#0.9#
        msg = "\nThrust to weight ratios used:\n" + \
              '"Stopping" phase: {:.2F}\n'.format(psiHigh) + \
              '"Falling" phase: {:.2F}\n'.format(psiLow)
        self.log.printL(msg)

        # calculate maximum velocity limit for starting propulsive phase,
        # so that all the propellant is used at the moment that s/c stops.
        vLimProp = (1.-1./psiHigh) * g0Isp * abs(numpy.log(1.-phi0))
        self.log.printL("Propellant-based vLim = {:.4E} km/s".format(vLimProp))

        #vLim = numpy.sqrt((V*V + 2. * g * h) * (2./s) * (1.-1./psiHigh))
        g_ = g * (1.-psiLow)
        vLim = numpy.sqrt((V*V + 2. * g_ * h) * (2./s) / \
                          ((psiHigh - psiLow)/(psiHigh - 1.)) )

        tBase = vLim / g
        tFall = tBase / (1.-psiLow)
        tStop = tBase / (psiHigh-1.)
        t1 = tFall + V / g_
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

        self.x = numpy.zeros((N, n, s)); self.u = numpy.zeros((N, m, s))
        hList = numpy.empty(s - 1)
        if s == 1:
            tTot = tStop + tFall
            psi = psiHigh
            hLim = (V * V + 2. * g * h) / 2. / g / psi
            self.pi = numpy.array([tTot])
            dt = tTot/(N-1)
            kT = int(tFall * N / tTot)
            #print("\nkT =",kT)
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
            self.u[kT:,0,0] = self.calcAdimCtrl(thr/Tmax)
            #numpy.arctanh(2. * thr / Tmax - 1.)
        elif s == 2:
            self.pi = numpy.array([tFall,tStop])
            tFallVec = numpy.linspace(0., 1., num=N) * tFall
            psi = psiHigh
            hLim = (V * V + 2. * g * h) / 2. / g / psi
            self.x[:, 0, 0] = h + V * tFallVec - .5 * g * tFallVec ** 2.
            self.x[:, 1, 0] = V - g * tFallVec
            self.x[:, 2, 0] = M + tFallVec * 0.
            self.u[:, 0, 0] = self.calcAdimCtrl(tFallVec*0.)

            hList[0] = hLim
            tStopVec = numpy.linspace(0., 1., num=N) * tStop
            self.x[:, 0, 1] = hLim - vLim * tStopVec + \
                                .5 * g * (psi - 1.) * tStopVec ** 2.
            self.x[:, 1, 1] = -vLim + g * (psi - 1.) * tStopVec
            self.x[:, 2, 1] = M * numpy.exp(-psi * tStopVec * g / g0Isp)

            thr = psi * M * numpy.exp(-psi * tStopVec * g / g0Isp) * g
            self.u[:, 0, 1] = self.calcAdimCtrl(thr/Tmax)
            #numpy.arctanh(2. * thr / Tmax - 1.)
        elif s % 2 == 0:
            self.pi = numpy.zeros(s)
            s2 = int(s / 2)
            self.pi[0] = t1
            t1Vec = numpy.linspace(0., 1., num=N) * t1
            g_ = g * (1. - psiLow)
            self.x[:, 0, 0] = h + V * t1Vec - .5 * g_ * t1Vec ** 2.
            self.x[:, 1, 0] = V - g_ * t1Vec
            self.x[:, 2, 0] = M * numpy.exp(-psiLow * t1Vec * g / g0Isp)
            thr = psiLow * self.x[:, 2, 0] * g
            self.u[:, 0, 0] = self.calcAdimCtrl(thr/Tmax)
            #numpy.arctanh(2. * thr / Tmax - 1.)

            tStopVec = numpy.linspace(0., 1., num=N) * tStop
            tFallVec = numpy.linspace(0., 1., num=N) * tFall

            arc = 0
            for k in range(s2):
                h, M = self.x[-1, 0, arc], self.x[-1, 2, arc]
                hList[arc] = h
                arc += 1
                self.pi[arc] = tStop
                psi = psiHigh
                x0StopVec = h - vLim * tStopVec + \
                            .5 * g * (psi - 1.) * tStopVec ** 2.
                x1StopVec = -vLim + g * (psi - 1.) * tStopVec
                x2StopVec = M * numpy.exp(-psi * tStopVec * g / g0Isp)
                thr = psi * x2StopVec * g
                #uStopVec = numpy.arctanh(2. * thr / Tmax - 1.)
                self.x[:, 0, arc] = x0StopVec
                self.x[:, 1, arc] = x1StopVec
                self.x[:, 2, arc] = x2StopVec
                self.u[:, 0, arc] = self.calcAdimCtrl(thr/Tmax)


                if arc == s-1:
                    break

                h, M = self.x[-1, 0, arc], self.x[-1, 2, arc]
                hList[arc] = h
                arc += 1

                self.pi[arc] = tFall
                g_ = g * (1.-psiLow)
                x0FallVec = h + V * tFallVec - .5 * g_ * tFallVec ** 2.
                x1FallVec = - g_ * tFallVec
                x2FallVec = M * numpy.exp(-psiLow * tFallVec * g / g0Isp)
                thr = psiLow * x2FallVec * g
                #uFallVec = numpy.arctanh(2. * thr / Tmax - 1.)
                self.x[:, 0, arc] = x0FallVec
                self.x[:, 1, arc] = x1FallVec
                self.x[:, 2, arc] = x2FallVec
                self.u[:, 0, arc] = self.calcAdimCtrl(thr/Tmax)

        else:
            msg = "\nLander method undefined for s = " + str(s)
            raise Exception(msg)
        #
        self.restrictions['hList'] = hList
        self.log.printL("\nHeight list: " + str(hList))
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