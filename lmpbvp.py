#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:09:52 2018

@author: levi
"""

import numpy
import matplotlib.pyplot as plt
from utils import simp
from scipy.linalg import expm

class LMPBVPhelp():
    """Class for processing the Linear Multipoint Boundary Value Problem.
    The biggest advantage in using an object is that the parallelization of
    the computations for each solution is much easier."""

    def __init__(self,sol,rho):
        """Initialization method. Comprises all the "common" calculations for
        each independent solution to be added over.

        According to the Miele (2003) convention,
            rho = 0 for rest, and rho = 1 for grad.
        """

        # debug options...
        self.dbugOptGrad = sol.dbugOptGrad
        self.dbugOptRest = sol.dbugOptRest
        self.t = sol.t

        # get sizes
        Ns,N,m,n,p,q,s = sol.Ns,sol.N,sol.m,sol.n,sol.p,sol.q,sol.s
        self.Ns,self.N,self.m,self.n,self.p,self.q,self.s = Ns,N,m,n,p,q,s
        self.dt = 1.0/(N-1)
        self.rho = rho

        # calculate integration error (only if necessary)
        psi = sol.calcPsi()
        self.psi = psi
        if rho < .5:
            err = sol.calcErr()
        else:
            err = numpy.zeros((N,n,s))

        self.err = err

        # Solver
        # TODO: put the solver as an external option
        self.solver = 'trap'#sol.solver

        #######################################################################
        if rho < 0.5 and self.dbugOptRest['plotErr']:
            print("\nThis is err:")
            for arc in range(s):
                plt.plot(self.t,err[:,0,arc])
                plt.ylabel("errPos")
                plt.grid(True)
                plt.show()
                plt.clf()
                plt.close('all')

                if n>1:
                    plt.plot(self.t,err[:,1,arc])
                    plt.ylabel("errVel")
                    plt.grid(True)
                    plt.show()
                    plt.clf()
                    plt.close('all')
        #######################################################################

        # Get gradients
        Grads = sol.calcGrads(calcCostTerm=(rho>0.5))
        #dt6 = dt/6
        phix = Grads['phix']
        phiu = Grads['phiu']
        phip = Grads['phip']
        psiy = Grads['psiy']
        psip = Grads['psip']
        fx = Grads['fx']
        fu = Grads['fu']
        fp = Grads['fp']

        self.phip = phip
        self.psiy = psiy
        self.psip = psip
        self.fx = fx
        self.fu = fu
        self.fp = fp

        # Prepare matrices with derivatives:
        phixTr = numpy.empty_like(phix)
        phiuTr = numpy.empty((N,m,n,s))
        phipTr = numpy.empty((N,p,n,s))
        phiuFu = numpy.empty((N,n,s))
        for arc in range(s):
            for k in range(N):
                phixTr[k,:,:,arc] = phix[k,:,:,arc].transpose()
                phiuTr[k,:,:,arc] = phiu[k,:,:,arc].transpose()
                phipTr[k,:,:,arc] = phip[k,:,:,arc].transpose()
                phiuFu[k,:,arc] = phiu[k,:,:,arc].dot(fu[k,:,arc])
        self.phiuFu = phiuFu
        self.phiuTr = phiuTr
        self.phipTr = phipTr

        InitCondMat = numpy.eye(Ns,Ns+1)
        self.InitCondMat = InitCondMat

        # Dynamics matrix for propagating the LSODE:
        DynMat = numpy.zeros((N,2*n,2*n,s))
        for arc in range(s):
            for k in range(N):
                DynMat[k,:n,:n,arc] = phix[k,:,:,arc]
                DynMat[k,:n,n:,arc] = phiu[k,:,:,arc].dot(phiuTr[k,:,:,arc])
                DynMat[k,n:,n:,arc] = -phixTr[k,:,:,arc]
        self.DynMat = DynMat

        #self.showEig(N,n,s)

    def showEig(self,N,n,s,mustShow=False):
        #print("\nLá vem os autovalores!")

        plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
        #plt.subplot2grid((s,1),(0,0))

        eigen = numpy.empty((N,n,s),dtype=complex)

        for arc in range(s):
            eigen[:,:,arc] = numpy.linalg.eigvals(self.DynMat[:,:n,:n,arc])
            plt.subplot2grid((s,1),(arc,0))
            for i in range(n):
                plt.plot(eigen[:,i,arc].real,eigen[:,i,arc].imag,'o-',\
                         label='eig #' + str(i+1) + ", start @(" + \
                         str(eigen[0,i,arc])+")")
            #
            plt.grid(True)
            plt.legend()
            plt.xlabel('Real')
            plt.ylabel('Imag')
            if arc == 0:
                plt.title('Eigenvalues for each arc')

        if mustShow:
            plt.show()


    def propagate(self,j):
        """This method computes each solution, via propagation of the
        applicable Linear System of Ordinary Differential Equations."""

        # Load data (sizes, common matrices, etc)
        rho = self.rho
        rho1 = self.rho-1.0
        Ns,N,n,m,p,s = self.Ns,self.N,self.n,self.m,self.p,self.s
        dt = self.dt

        InitCondMat = self.InitCondMat
        phip = self.phip
        err = self.err
        phiuFu = self.phiuFu
        fx = self.fx
        if rho > .5:
            rhoFu = self.fu
        else:
            rhoFu = numpy.zeros((N,m,s))

        phiuTr = self.phiuTr
        phipTr = self.phipTr
        DynMat = self.DynMat
        I = numpy.eye(2*n)

        # Declare matrices for corrections
        phiLamIntCol = numpy.zeros(p)
        DtCol = numpy.empty(2*n*s)
        EtCol = numpy.empty(2*n*s)
        A = numpy.zeros((N,n,s))
        B = numpy.zeros((N,m,s))
        C = numpy.zeros((p,1))
        lam = numpy.zeros((N,n,s))

        # the vector that will be integrated is Xi = [A; lam]
        Xi = numpy.zeros((N,2*n,s))
        # Initial conditions for the LSODE:
        for arc in range(s):
            A[0,:,arc] = InitCondMat[2*n*arc:(2*n*arc+n) , j]
            lam[0,:,arc] = InitCondMat[(2*n*arc+n):(2*n*(arc+1)) , j]
            Xi[0,:n,arc],Xi[0,n:,arc] = A[0,:,arc],lam[0,:,arc]
        C = InitCondMat[(2*n*s):,j]

        # Non-homogeneous terms for the LSODE:
        nonHom = numpy.empty((N,2*n,s))
        for arc in range(s):
            for k in range(N):
                # minus sign in rho1 (rho-1) is on purpose!
                nonHA = phip[k,:,:,arc].dot(C) + \
                            -rho1*err[k,:,arc] - rho*phiuFu[k,:,arc]
                nonHL = rho * fx[k,:,arc]
                nonHom[k,:n,arc] = nonHA#.copy()
                nonHom[k,n:,arc] = nonHL#.copy()

        # This command probably broke the compatibility with other integration
        # methods. They weren't working anyway, so...
        coefList = simp([],N,onlyCoef=True)

        for arc in range(s):
            if self.solver == 'heun':
###############################################################################
                # Integrate the LSODE (by Heun's method):
                B[0,:,arc] = -rhoFu[0,:,arc] + \
                                    phiuTr[0,:,:,arc].dot(lam[0,:,arc])
                phiLamIntCol += .5 * (phipTr[0,:,:,arc].dot(lam[0,:,arc]))

                # First point: simple propagation
                derXik = DynMat[0,:,:,arc].dot(Xi[0,:,arc]) + \
                            nonHom[0,:,arc]
                Xi[1,:,arc] = Xi[0,:,arc] + dt * derXik
                #A[1,:,arc] = Xi[1,:n,arc]
                lam[1,:,arc] = Xi[1,n:,arc]
                B[1,:,arc] = -rhoFu[1,:,arc] + \
                                phiuTr[1,:,:,arc].dot(lam[1,:,arc])
                phiLamIntCol += phipTr[1,:,:,arc].dot(lam[1,:,arc])

                # "Middle" points: original Heun propagation
                for k in range(1,N-2):
                    derXik = DynMat[k,:,:,arc].dot(Xi[k,:,arc]) + \
                            nonHom[k,:,arc]
                    aux = Xi[k,:,arc] + dt * derXik
                    Xi[k+1,:,arc] = Xi[k,:,arc] + .5 * dt * (derXik + \
                                    DynMat[k+1,:,:,arc].dot(aux) + \
                                    nonHom[k+1,:,arc])
                    #A[k+1,:,arc] = Xi[k+1,:n,arc]
                    lam[k+1,:,arc] = Xi[k+1,n:,arc]
                    B[k+1,:,arc] = -rhoFu[k+1,:,arc] + \
                                    phiuTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                    phiLamIntCol += phipTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                #

                # Last point: simple propagation, but based on the last point
                derXik = DynMat[N-1,:,:,arc].dot(Xi[N-2,:,arc]) + \
                            nonHom[N-1,:,arc]
                Xi[N-1,:,arc] = Xi[N-2,:,arc] + dt * derXik
                #A[N-1,:,arc] = Xi[N-1,:n,arc]
                lam[N-1,:,arc] = Xi[N-1,n:,arc]
                B[N-1,:,arc] = -rhoFu[N-1,:,arc] + \
                                phiuTr[N-1,:,:,arc].dot(lam[N-1,:,arc])
                phiLamIntCol += .5*phipTr[N-1,:,:,arc].dot(lam[N-1,:,arc])
###############################################################################
            elif self.solver == 'trap':
                # Integrate the LSODE by trapezoidal (implicit) method
                B[0,:,arc] = -rhoFu[0,:,arc] + \
                                    phiuTr[0,:,:,arc].dot(lam[0,:,arc])
                phiLamIntCol += coefList[0] * \
                                (phipTr[0,:,:,arc].dot(lam[0,:,arc]))

                for k in range(N-1):
                    Xi[k+1,:,arc] = numpy.linalg.solve(I - .5 * dt * DynMat[k+1,:,:,arc],\
                      (I + .5 * dt * DynMat[k,:,:,arc]).dot(Xi[k,:,arc]) + \
                      .5 * dt * (nonHom[k+1,:,arc]+nonHom[k,:,arc]))
                    lam[k+1,:,arc] = Xi[k+1,n:,arc]
                    B[k+1,:,arc] = -rhoFu[k+1,:,arc] + \
                                    phiuTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                    phiLamIntCol += coefList[k+1] * \
                                    phipTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                #

###############################################################################
            elif self.solver == 'BEI':
            # Integrate the LSODE by "original" Euler Backwards implicit
                B[0,:,arc] = -rhoFu[0,:,arc] + \
                                    phiuTr[0,:,:,arc].dot(lam[0,:,arc])
                phiLamIntCol += .5*(phipTr[0,:,:,arc].dot(lam[0,:,arc]))

                for k in range(N-1):
                    Xi[k+1,:,arc] = numpy.linalg.solve(I - dt*DynMat[k+1,:,:,arc],\
                      Xi[k,:,arc] + dt*nonHom[k+1,:,arc])
                    lam[k+1,:,arc] = Xi[k+1,n:,arc]
                    B[k+1,:,arc] = -rhoFu[k+1,:,arc] + \
                                    phiuTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                    phiLamIntCol += phipTr[k+1,:,:,arc].dot(lam[k+1,:,arc])

                phiLamIntCol -= .5*phipTr[N-1,:,:,arc].dot(lam[N-1,:,arc])
###############################################################################
            if self.solver == 'leapfrog':
                # Integrate the LSODE by "leapfrog" with special start and end

                # with special 1st step...
                B[0,:,arc] = -rhoFu[0,:,arc] + \
                                    phiuTr[0,:,:,arc].dot(lam[0,:,arc])
                phiLamIntCol += coefList[0] * (phipTr[0,:,:,arc].dot(lam[0,:,arc]))

                Xi[1,:,arc] = Xi[0,:,arc] + dt * \
                              (DynMat[0,:,:,arc].dot(Xi[0,:,arc])+nonHom[0,:,arc])
                lam[1,:,arc] = Xi[1,n:,arc]
                B[1,:,arc] = -rhoFu[1,:,arc] + \
                                    phiuTr[1,:,:,arc].dot(lam[1,:,arc])
                phiLamIntCol += coefList[1] * \
                                    phipTr[1,:,:,arc].dot(lam[1,:,arc])

                for k in range(1,N-2):

                    Xi[k+1,:,arc] = Xi[k-1,:,arc] + 2. * dt * \
                        (DynMat[k,:,:,arc].dot(Xi[k,:,arc]) + nonHom[k,:,arc])
                    lam[k+1,:,arc] = Xi[k+1,n:,arc]
                    B[k+1,:,arc] = -rhoFu[k+1,:,arc] + \
                                    phiuTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                    phiLamIntCol += coefList[k+1] * \
                                    phipTr[k+1,:,:,arc].dot(lam[k+1,:,arc])

    #            # with special last step...
                Xi[N-1,:,arc] = numpy.linalg.solve(I - dt*DynMat[N-1,:,:,arc],\
                      Xi[N-2,:,arc] + dt*nonHom[N-1,:,arc])

    #            # with special last step...
    #            Xi[N-1,:,arc] = Xi[N-2,:,arc] + dt * \
    #                          (DynMat[N-1,:,:,arc].dot(Xi[N-2,:,arc])+nonHom[N-1,:,arc])

                # with special last step...
    #            derXik = DynMat[N-2,:,:,arc].dot(Xi[N-2,:,arc]) + \
    #                        nonHom[N-2,:,arc]
    #            aux = Xi[N-2,:,arc] + dt * derXik
    #            Xi[N-1,:,arc] = Xi[N-2,:,arc] + .5 * dt * (derXik + \
    #                                DynMat[N-1,:,:,arc].dot(aux) + \
    #                                nonHom[N-1,:,arc])

                lam[N-1,:,arc] = Xi[N-1,n:,arc]
                B[N-1,:,arc] = -rhoFu[N-1,:,arc] + \
                                    phiuTr[N-1,:,:,arc].dot(lam[N-1,:,arc])
                phiLamIntCol += coefList[N-1] * \
                                    phipTr[N-1,:,:,arc].dot(lam[N-1,:,arc])
###############################################################################
            if self.solver == 'BEI_spec':
                # Integrate the LSODE by Euler Backwards implicit,
                # with special 1st step...
                B[0,:,arc] = -rhoFu[0,:,arc] + \
                                    phiuTr[0,:,:,arc].dot(lam[0,:,arc])
                phiLamIntCol += coefList[0] * (phipTr[0,:,:,arc].dot(lam[0,:,arc]))

                Xi[1,:,arc] = Xi[0,:,arc] + dt * \
                              (DynMat[0,:,:,arc].dot(Xi[0,:,arc])+nonHom[0,:,arc])
                lam[1,:,arc] = Xi[1,n:,arc]
                B[1,:,arc] = -rhoFu[1,:,arc] + \
                                    phiuTr[1,:,:,arc].dot(lam[1,:,arc])
                phiLamIntCol += coefList[1] * \
                                    phipTr[1,:,:,arc].dot(lam[1,:,arc])

                for k in range(1,N-1):
                    Xi[k+1,:,arc] = numpy.linalg.solve(I - dt*DynMat[k+1,:,:,arc],\
                      Xi[k,:,arc] + dt*nonHom[k+1,:,arc])
                    lam[k+1,:,arc] = Xi[k+1,n:,arc]
                    B[k+1,:,arc] = -rhoFu[k+1,:,arc] + \
                                    phiuTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                    phiLamIntCol += coefList[k+1] * \
                                    phipTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
###############################################################################
            if self.solver == 'hamming_mod':
                # Integrate the LSODE by Hamming's mod predictor-corrector method
                B[0,:,arc] = -rhoFu[0,:,arc] + \
                                    phiuTr[0,:,:,arc].dot(lam[0,:,arc])
                phiLamIntCol += coefList[0] * (phipTr[0,:,:,arc].dot(lam[0,:,arc]))

                # First points: RKF4
                Xi[1,:,arc] = Xi[0,:,arc] + dt * \
                            (DynMat[0,:,:,arc].dot(Xi[0,:,arc]) + nonHom[0,:,arc])
                lam[1,:,arc] = Xi[1,n:,arc]
                B[1,:,arc] = -rhoFu[1,:,arc] + \
                                    phiuTr[1,:,:,arc].dot(lam[1,:,arc])
                phiLamIntCol += coefList[1] * \
                                    phipTr[1,:,:,arc].dot(lam[1,:,arc])

                for k in range(1,3):
                    Xik = Xi[k,:,arc]
                    DM13 = DynMat[k,:,:,arc]*(2./3.) + DynMat[k+1,:,:,arc]*(1./3.)
                    NH13 = nonHom[k,:,arc] * (2./3.) + nonHom[k+1,:,arc] * (1./3.)
                    DM23 = DynMat[k,:,:,arc]*(1./3.) + DynMat[k+1,:,:,arc]*(2./3.)
                    NH23 = nonHom[k,:,arc] * (1./3.) + nonHom[k+1,:,arc] * (2./3.)

                    f1 = DynMat[k,:,:,arc].dot(Xi[k,:,arc]) + nonHom[k,:,arc]
                    f2 = DM13.dot(Xik+(1./3.)*dt*f1) + NH13
                    f3 = DM23.dot(Xik + dt*(-(1./3.)*f1 + f2)) + NH23
                    f4 = DynMat[k+1,:,:,arc].dot(Xik + dt*(f1-f2+f3)) + nonHom[k+1,:,arc]

                    Xi[k+1,:,arc] = Xik + dt * (f1 + 3.*f2 + 3.*f3 + f4)/8.

    #                Xik = Xi[k,:,arc]
    #                DM14 = DynMat[k,:,:,arc]*.75 + DynMat[k+1,:,:,arc]*.25
    #                NH14 = nonHom[k,:,arc] * .75 + nonHom[k+1,:,arc] * .25
    #                DM38 = DynMat[k,:,:,arc]*.625 + DynMat[k+1,:,:,arc]*.375
    #                NH38 = nonHom[k,:,arc] * .625 + nonHom[k+1,:,arc] * .375
    #                DM12 = DynMat[k,:,:,arc]*.5 + DynMat[k+1,:,:,arc]*.5
    #                NH12 = nonHom[k,:,arc] * .5 + nonHom[k+1,:,arc] * .5
    #                DM1213 = DynMat[k,:,:,arc]*(1./13.) + DynMat[k+1,:,:,arc]*(12./13.)
    #                NH1213 = nonHom[k,:,arc] * (1./13.) + nonHom[k+1,:,arc] * (1./13.)
    #                f1 = DynMat[k,:,:,arc].dot(Xi[k,:,arc]) + nonHom[k,:,arc]
    #                f2 = DM14.dot(Xik+.25*dt*f1) + NH14
    #                f3 = DM38.dot(Xik + dt*(3.*f1 + 9.*f2)/32.) + NH38
    #                f4 = DM1213.dot(Xik + dt*(1932.*f1 - 7200.*f2 + 7296.)/2197.) + NH1213
    #                f5 = DynMat[k+1,:,:,arc].dot(Xik + dt*((439./216.)*f1-8.*f2+(3680./513.)*f3-(845./4104.)*f4)) + nonHom[k+1,:,arc]
    #                f6 = DM12.dot(Xik+dt*(-(8./27.)*f1 + 2.*f2 -(3544./2565.)*f3 +(1859./4104.)*f4 -(11./40.)*f5)) + NH12
    #
    #                Xi[k+1,:,arc] = Xik + dt * ((16./135.)*f1 + \
    #                                            (6656./12825.)*f3 + \
    #                                            (28561./56430.)*f4 + \
    #                                            -(9./50.)*f5 + \
    #                                            (2./55.)*f6)

                    lam[k+1,:,arc] = Xi[k+1,n:,arc]
                    B[k+1,:,arc] = -rhoFu[k+1,:,arc] + \
                                    phiuTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                    phiLamIntCol += coefList[k+1] * \
                                    phipTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                #
                # Now, Hamming's...
                pk = numpy.zeros_like(Xi[0,:,arc])
                ck = pk
                for k in range(3,N-1):
                    fk = DynMat[k,:,:,arc].dot(Xi[k,:,arc]) + nonHom[k,:,arc]
                    fkm1 = DynMat[k-1,:,:,arc].dot(Xi[k-1,:,arc]) + nonHom[k-1,:,arc]
                    fkm2 = DynMat[k-2,:,:,arc].dot(Xi[k-2,:,arc]) + nonHom[k-2,:,arc]
                    pkp1 = Xi[k-3,:,arc] + (4.0*dt/3.0) * \
                            (2.0 * fk - fkm1 + 2.0 * fkm2)
                    mkp1 = pkp1 + (112.0/121.0) * (ck - pk)
                    rdkp1  = DynMat[k+1,:,:,arc].dot(mkp1)
                    ckp1 = (9.0/8.0) * Xi[k,:,arc] -(1.0/8.0) * Xi[k-2,:,arc] + \
                           (3.0*dt/8.0) * (rdkp1 + 2.0 * fk - fkm1)
                    Xi[k+1,:,arc] = ckp1 + (9.0/121.0) * (pkp1 - ckp1)

                    lam[k+1,:,arc] = Xi[k+1,n:,arc]
                    B[k+1,:,arc] = -rhoFu[k+1,:,arc] + \
                                    phiuTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                    phiLamIntCol += coefList[k+1] * \
                                    phipTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                    pk, ck = pkp1, ckp1
                #
###############################################################################
            if self.solver == 'hamming':
                # Integrate the LSODE by Hamming's predictor-corrector method
                B[0,:,arc] = -rhoFu[0,:,arc] + \
                                    phiuTr[0,:,:,arc].dot(lam[0,:,arc])
                phiLamIntCol += coefList[0] * (phipTr[0,:,:,arc].dot(lam[0,:,arc]))

                # first point: simple propagation?
    #            Xi[1,:,arc] = numpy.linalg.solve(I - dt*DynMat[1,:,:,arc],\
    #                  Xi[0,:,arc] + dt*nonHom[1,:,arc])

                # First points: Heun...
                for k in range(3):
                    Xi[k+1,:,arc] = numpy.linalg.solve(I - .5*dt*DynMat[k+1,:,:,arc],\
                                    Xi[k,:,arc] + .5 * dt * \
                                    (DynMat[k,:,:,arc].dot(Xi[k,:,arc]) + \
                                    nonHom[k,:,arc] + nonHom[k+1,:,arc]))

    #                derXik = DynMat[k,:,:,arc].dot(Xi[k,:,arc]) + \
    #                            nonHom[k,:,arc]
    #                aux = Xi[k,:,arc] + dt * derXik
    #                Xi[k+1,:,arc] = Xi[k,:,arc] + .5 * dt * (derXik + \
    #                                DynMat[k+1,:,:,arc].dot(aux) + \
    #                                nonHom[k+1,:,arc])

                    lam[k+1,:,arc] = Xi[k+1,n:,arc]
                    B[k+1,:,arc] = -rhoFu[k+1,:,arc] + \
                                    phiuTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                    phiLamIntCol += coefList[k+1] * \
                                    phipTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                #
                # Now, Hamming's...
                for k in range(3,N-1):
                    fk = DynMat[k,:,:,arc].dot(Xi[k,:,arc]) + nonHom[k,:,arc]
                    fkm1 = DynMat[k-1,:,:,arc].dot(Xi[k-1,:,arc]) + nonHom[k-1,:,arc]
                    fkm2 = DynMat[k-2,:,:,arc].dot(Xi[k-2,:,arc]) + nonHom[k-2,:,arc]
                    Xikp1 = Xi[k-3,:,arc] + (4.0*dt/3.0) * \
                            (2.0 * fk - fkm1 + 2.0 * fkm2)
                    fkp1 = DynMat[k+1,:,:,arc].dot(Xikp1) + nonHom[k+1,:,arc]
                    Xi[k+1,:,arc] = (9.0/8.0) * Xi[k,:,arc] + \
                                    -(1.0/8.0) * Xi[k-2,:,arc] + (3.0*dt/8.0) * \
                                    (fkp1 - fkm1 + 2.0 * fk)

                    lam[k+1,:,arc] = Xi[k+1,n:,arc]
                    B[k+1,:,arc] = -rhoFu[k+1,:,arc] + \
                                    phiuTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                    phiLamIntCol += coefList[k+1] * \
                                    phipTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                #
###############################################################################
            if self.solver == 'expm':
                # Integrate the LSODE by matrix exponentiation
                B[0,:,arc] = -rhoFu[0,:,arc] + \
                                    phiuTr[0,:,:,arc].dot(lam[0,:,arc])
                phiLamIntCol += .5 * (phipTr[0,:,:,arc].dot(lam[0,:,arc]))
                for k in range(N-1):
                    expDM = expm(DynMat[k,:,:,arc]*dt)
                    NHterm = expDM.dot(nonHom[k,:,arc]) - nonHom[k,:,arc]
                    Xi[k+1,:,arc] = expDM.dot(Xi[k,:,arc]) + \
                                    numpy.linalg.solve(DynMat[k,:,:,arc], NHterm)
                    lam[k+1,:,arc] = Xi[k+1,n:,arc]
                    B[k+1,:,arc] = -rhoFu[k+1,:,arc] + \
                                    phiuTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                    phiLamIntCol += phipTr[k+1,:,:,arc].dot(lam[k+1,:,arc])
                #

                phiLamIntCol -= .5*phipTr[N-1,:,:,arc].dot(lam[N-1,:,arc])
###############################################################################

            # Get the A values from Xi
            A[:,:,arc] = Xi[:,:n,arc]

            # Put initial and final conditions of A and Lambda into matrices
            # DtCol and EtCol, which represent the columns of Dtilde(Dt) and
            # Etilde(Et)
            DtCol[(2*arc)*n   : (2*arc+1)*n] =    A[0,:,arc]   # eq (32a)
            DtCol[(2*arc+1)*n : (2*arc+2)*n] =    A[N-1,:,arc] # eq (32a)
            EtCol[(2*arc)*n   : (2*arc+1)*n] = -lam[0,:,arc]   # eq (32b)
            EtCol[(2*arc+1)*n : (2*arc+2)*n] =  lam[N-1,:,arc] # eq (32b)
        #
        # All integrations ready!
        # no longer used, because coefList from simp already includes dt
        #phiLamIntCol *= dt

###############################################################################
        if (rho > 0.5 and self.dbugOptGrad['plotCorr']) or \
           (rho < 0.5 and self.dbugOptRest['plotCorr']):
            print("\nHere are the corrections for iteration " + str(j+1) + \
                  " of " + str(Ns+1) + ":\n")
            for arc in range(s):
                print("> Corrections for arc =",arc)
                plt.plot(self.t,A[:,0,arc])
                plt.grid(True)
                plt.ylabel('A: pos')
                plt.show()
                plt.clf()
                plt.close('all')


                plt.plot(self.t,lam[:,0,arc])
                plt.grid(True)
                plt.ylabel('lambda: pos')
                plt.show()
                plt.clf()
                plt.close('all')

                if n>1:
                    plt.plot(self.t,A[:,1,arc])
                    plt.grid(True)
                    plt.ylabel('A: vel')
                    plt.show()
                    plt.clf()
                    plt.close('all')

                    plt.plot(self.t,lam[:,1,arc])
                    plt.grid(True)
                    plt.ylabel('lambda: vel')
                    plt.show()
                    plt.clf()
                    plt.close('all')

                if n>2:
                    plt.plot(self.t,A[:,2,arc])
                    plt.grid(True)
                    plt.ylabel('A: gama')
                    plt.show()
                    plt.clf()
                    plt.close('all')

                    plt.plot(self.t,lam[:,2,arc])
                    plt.grid(True)
                    plt.ylabel('lambda: gamma')
                    plt.show()
                    plt.clf()
                    plt.close('all')


                if n>3:
                    plt.plot(self.t,A[:,3,arc])
                    plt.grid(True)
                    plt.ylabel('A: m')
                    plt.show()
                    plt.clf()
                    plt.close('all')

                    plt.plot(self.t,lam[:,3,arc])
                    plt.grid(True)
                    plt.ylabel('lambda: m')
                    plt.show()
                    plt.clf()
                    plt.close('all')


                plt.plot(self.t,B[:,0,arc])
                plt.grid(True)
                plt.ylabel('B0')
                plt.show()
                plt.clf()
                plt.close('all')

                if m>1:
                    plt.plot(self.t,B[:,1,arc])
                    plt.grid(True)
                    plt.ylabel('B1')
                    plt.show()
                    plt.clf()
                    plt.close('all')

                print("C[arc] =",C[arc])
                #input(" > ")
###############################################################################

        # All the outputs go to main output dictionary; the final solution is
        # computed by the next method, 'getCorr'.
        outp = {'A':A,'B':B,'C':C,'L':lam,'Dt':DtCol,'Et':EtCol,
                'phiLam':phiLamIntCol}

        return outp


    def getCorr(self,res,log):
        """ Computes the actual correction for this grad/rest step, by linear
        combination of the solutions generated by method 'propagate'."""

        # Get sizes
        Ns,N,n,m,p,q,s = self.Ns,self.N,self.n,self.m,self.p,self.q,self.s
        rho1 = self.rho - 1.0

        # Declare matrices Ctilde, Dtilde, Etilde, and the integral term
        Ct = numpy.empty((p,Ns+1))
        Dt = numpy.empty((2*n*s,Ns+1))
        Et = numpy.empty((2*n*s,Ns+1))
        phiLamInt = numpy.empty((p,Ns+1))

        # Unpack outputs from 'propagate' into proper matrices Ct, Dt, etc.
        for j in range(Ns+1):
            Ct[:,j] = res[j]['C']
            Dt[:,j] = res[j]['Dt']
            Et[:,j] = res[j]['Et']
            phiLamInt[:,j] = res[j]['phiLam']

        # Assembly of matrix M and column 'Col' for the linear system

        # Matrix for linear system involving k's and mu's
        M = numpy.zeros((Ns+q+1,Ns+q+1))
        # from eq (34d) - k term
        M[0,:(Ns+1)] = numpy.ones(Ns+1)
        # from eq (34b) - mu term
        M[(q+1):(q+1+p),(Ns+1):] = self.psip.transpose()
        # from eq (34c) - mu term
        M[(p+q+1):,(Ns+1):] = self.psiy.transpose()
        # from eq (34a) - k term
        M[1:(q+1),:(Ns+1)] = self.psiy.dot(Dt) + self.psip.dot(Ct)
        # from eq (34b) - k term
        M[(q+1):(q+p+1),:(Ns+1)] = Ct - phiLamInt
        # from eq (34c) - k term
        M[(q+p+1):,:(Ns+1)] = Et

        # column vector for linear system involving k's and mu's  [eqs (34)]
        col = numpy.zeros(Ns+q+1)
        col[0] = 1.0 # eq (34d)

        # Integral term
        if self.rho > 0.5:
            # eq (34b) - only applicable for grad

#            sumIntFpi = numpy.zeros(p)
#            for arc in range(s):
#                for ind in range(p):
#                    sumIntFpi[ind] += self.fp[:,ind,arc].sum()
#                    sumIntFpi[ind] -= .5 * ( self.fp[0,ind,arc] + \
#                             self.fp[-1,ind,arc])
#            sumIntFpi *= self.dt

            sumIntFpi = numpy.zeros(p)
            for arc in range(s):
                for ind in range(p):
                    sumIntFpi[ind] += simp(self.fp[:,ind,arc],N)
                #
            #
            col[(q+1):(q+p+1)] = -self.rho * sumIntFpi
        else:
            # eq (34a) - only applicable for rest
            col[1:(q+1)] = rho1 * self.psi


        # Calculations of weights k:
        KMi = numpy.linalg.solve(M,col)
        Res = M.dot(KMi)-col
        log.printL("Residual of the Linear System: " + \
                   str(Res.transpose().dot(Res)))
        K,mu = KMi[:(Ns+1)], KMi[(Ns+1):]

        # summing up linear combinations
        A = numpy.zeros((N,n,s))
        B = numpy.zeros((N,m,s))
        C = numpy.zeros(p)
        lam = numpy.zeros((N,n,s))

        for j in range(Ns+1):
            A   += K[j] * res[j]['A']#self.arrayA[j,:,:,:]
            B   += K[j] * res[j]['B']#self.arrayB[j,:,:,:]
            C   += K[j] * res[j]['C']#self.arrayC[j,:]
            lam += K[j] * res[j]['L']#self.arrayL[j,:,:,:]

###############################################################################
        if (self.rho > 0.5 and self.dbugOptGrad['plotCorrFin']) or \
           (self.rho < 0.5 and self.dbugOptRest['plotCorrFin']):
            log.printL("\n------------------------------------------------------------")
            log.printL("Final corrections:\n")
            for arc in range(s):
                log.printL("> Corrections for arc =",arc)
                plt.plot(self.t,A[:,0,arc])
                plt.grid(True)
                plt.ylabel('A: pos')
                plt.show()
                plt.clf()
                plt.close('all')

                if n>1:
                    plt.plot(self.t,A[:,1,arc])
                    plt.grid(True)
                    plt.ylabel('A: vel')
                    plt.show()
                    plt.clf()
                    plt.close('all')

                if n>2:
                    plt.plot(self.t,A[:,2,arc])
                    plt.grid(True)
                    plt.ylabel('A: gama')
                    plt.show()
                    plt.clf()
                    plt.close('all')

                if n>3:
                    plt.plot(self.t,A[:,3,arc])
                    plt.grid(True)
                    plt.ylabel('A: m')
                    plt.show()
                    plt.clf()
                    plt.close('all')

                plt.plot(self.t,B[:,0,arc])
                plt.grid(True)
                plt.ylabel('B0')
                plt.show()
                plt.clf()
                plt.close('all')

                if m>1:
                    plt.plot(self.t,B[:,1,arc])
                    plt.grid(True)
                    plt.ylabel('B1')
                    plt.show()
                    plt.clf()
                    plt.close('all')


                log.printL("C[arc] =",C[arc])

            #input(" > ")
###############################################################################
        return A,B,C,lam,mu