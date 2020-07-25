#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:09:52 2018

@author: levi
"""

import numpy
import matplotlib.pyplot as plt
from scipy.linalg import expm

useCython = False

if useCython:
    # this refers to the cython code
    import pyximport
    pyximport.install(setup_args={"include_dirs":numpy.get_include()},
                      reload_support=True)
    from utils_c import simp, prepMat, propagate as _propagate
else:
    from utils import simp

class LMPBVPhelp():
    """Class for processing the Linear Multipoint Boundary Value Problem.
    The biggest advantage in using an object is that the parallelization of
    the computations for each solution is much easier."""

    def __init__(self,sol,rho):
        """Initialization method. Comprises all the "common" calculations for
        each independent solution to be added over.

        According to the Miele and Wang (2003) convention,
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

        # get omission status from sol
        self.omit = sol.omit
        if self.omit:
            self.omitEqMat = sol.omitEqMat
            self.omitVarList = sol.omitVarList
        # calculate psi
        self.psi = sol.calcPsi()

        if rho < .5:
            # Restoration. Override the omit
            self.omit = False
            # calculate integration error (only if necessary)
            err = sol.calcErr()
        else:
            err = numpy.zeros((N,n,s))

        self.err = err

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
        phix, phiu, phip = Grads['phix'], Grads['phiu'], Grads['phip']
        psiy, psip = Grads['psiy'], Grads['psip']
        fx, fu, fp = Grads['fx'], Grads['fu'], Grads['fp']

        self.phip, self.psiy, self.psip = phip, psiy, psip
        self.fx, self.fu, self.fp = fx, fu, fp

        if useCython:
            sizes = {'N': self.N, 'Ns': self.Ns, 'n': self.n,
                     'm': self.m, 'p': self.p, 's': self.s}
            outp = prepMat(sizes, fu, phip, phiu, phix)
            self.DynMat, self.InvDynMat = outp['DynMat'], outp['InvDynMat']
            self.phipTr = outp['phipTr']
            self.phiuFu = outp['phiuFu']
            self.phiuTr = outp['phiuTr']
            self.InitCondMat = outp['InitCondMat']
        else:
            # Prepare matrices with derivatives:
            I = numpy.eye(2 * n)
            mdt = 0.5 * self.dt
            # this computes phiu * fu, for all times and arcs
            self.phiuFu = numpy.einsum('nijs,njs->nis',phiu,fu)
            self.phiu = phiu
            self.InitCondMat = numpy.eye(Ns,Ns+1)

            # Dynamics matrix for propagating the LSODE:
            # originally, DynMat was the matrix, D, with phix, phiu*phiu^T, etc;
            # but it is more efficient to store (I + .5 * dt * D) instead.
            DynMat_ = numpy.zeros((N,2*n,2*n,s))
            DynMat_[:,:n,:n,:] = phix
            # phiu * phiu^T:
            DynMat_[:, :n, n:, :] = numpy.einsum('nijs,nkjs->niks',phiu,phiu)
            # - phix^T
            DynMat_[:, n:, n:, :] = -phix.swapaxes(1,2)
            #DynMat[k, :, :, arc] = I + mdt * DynMat_[k, :, :, arc]
            DynMat = mdt * DynMat_[:,:,:,:]
            # TODO: is this the best way to do it?
            for arc in range(s):
                DynMat[:, 0:(2*n), 0:(2*n), arc] += I
            self.DynMat = DynMat

            # This is a strategy for lowering the cost of the trapezoidal solver.
            # Instead of solving (N-1) * s * (2ns+p) linear systems of order 2n,
            # resulting in a cost in the order of
            #         (N-1) * s * (2ns+p) * 4n² ;
            # it is  better to pre-invert (N-1) * s matrices of order 2n,
            # resulting in a cost in the order of
            #         (N-1) * s * 8n³.
            # This should always result in an increased performance because
            #         (N-1) * s * 8n³ < (N-1) * s * (2ns+p) * 4n²

            InvDynMat = -mdt * DynMat_#numpy.zeros((N, s, 2 * n, 2 * n))
            for arc in range(s):
                InvDynMat[:,:,:,arc] += I
            # the inversion works in this vectorized form, but only in the last two
            # axes. Let's swap them back and forth, then!
            InvDynMat = numpy.linalg.inv(InvDynMat.
                             swapaxes(2,3).swapaxes(1,2)).swapaxes(1,2).swapaxes(2,3)
            self.InvDynMat = InvDynMat
            # this is the main propagation matrix. The differential equation is
            # propagated by:
            # Xi[k+1,:,arc] = MainPropMat[k,:,:,arc] * Xi[k,:,arc] + genNonHom[k,:,arc]
            # for every k and every arc.
            self.MainPropMat = numpy.einsum('nijs,njks->niks',InvDynMat[1:,:,:,:],
                                                        DynMat[:(N-1),:,:,:])
            #self.BigMat = numpy.empty((s, N * n * 2, N * n * 2))
            #self.prepBigMat()
        #self.showEig(N,n,s)

    def prepBigMat(self):
        """
        Prepare the "big matrix" for Xi propagation.

        This function assembles the "big matrix" (s x 2*n*N x 2*n*N !) for propagating
        the Xi differential equation in a single step.
        :return: None
        """

        n2 = 2 * self.n
        BigMat = numpy.empty((self.s,self.N * n2,self.N * n2))
        I = numpy.eye(n2*self.N)
        for arc in range(self.s):
            BigMat[arc,:,:] = I
        for k in range(1,self.N):
            kn2 = k * n2
            BigMat[:, kn2:(kn2+n2) , 0:kn2] = numpy.einsum('ijs,sjk->sik',
                                                self.MainPropMat[k-1,:,:,:],
                                                BigMat[:,(kn2-n2):k*n2, 0:kn2])
        self.BigMat = BigMat

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

    def propXi(self,Xi,genNonHom):
        # Integrate the LSODE by trapezoidal (implicit) method
        for k in range(self.N - 1):
            # this computes
            # Xi[k+1,:,arc] = MainPropMat[k,:,:,arc] * Xi[k,:,arc]
            #                 + genNonHom[k,:, arc] ,
            # for each arc
            Xi[k + 1, :, :] = numpy.einsum('ijs,js->is', self.MainPropMat[k, :, :, :],
                                           Xi[k, :, :]) + genNonHom[k, :, :]

        # n2 = self.n * 2
        # BigCol = numpy.empty((n2 * self.N, self.s))
        # BigCol[:n2, :] = Xi[0, :, :]
        # BigCol[n2:, :] = genNonHom.reshape((n2 * (self.N - 1), self.s))
        # Xi = numpy.einsum('sij,js->is', self.BigMat, BigCol).reshape(
        #     (self.N, n2, self.s))
        return Xi

    def propagate(self,j):
        """This method computes each solution, via propagation of the
        applicable Linear System of Ordinary Differential Equations."""

        if useCython:
            sizes = {'N': self.N, 'Ns': self.Ns, 'n': self.n,
                     'm': self.m, 'p': self.p, 's': self.s}
            outp = _propagate(j, sizes, self.DynMat, self.err, self.fu, self.fx,
                self.InitCondMat, self.InvDynMat, self.phip, self.phipTr,
                self.phiuFu, self.phiuTr, grad=(self.rho>.5))
        else:
            # Load data (sizes, common matrices, etc)
            rho = self.rho
            Ns,N,n,m,p,s = self.Ns,self.N,self.n,self.m,self.p,self.s
            dt = self.dt
            mdt = .5 * dt
            InitCondMat = self.InitCondMat
            phip, phiu, phiuFu = self.phip, self.phiu, self.phiuFu
            err = self.err
            fx = self.fx
            grad = (rho>.5) # rho = 1: grad = True; rho = 0: grad = False

            # Declare matrices for corrections
            DtCol, EtCol  = numpy.empty(2*n*s), numpy.empty(2*n*s)
            #A, B, C, lam = numpy.zeros((N,n,s)), "(N,m,s), "(p,1), "(N,n,s)

            # the vector that will be integrated is Xi = [A; lam]
            Xi = numpy.zeros((N,2*n,s))
            # Initial conditions for the LSODE:
            # TODO: this can probably be done faster!
            for arc in range(s):
                Xi[0, :n, arc] = InitCondMat[2*n*arc:(2*n*arc+n) , j]
                Xi[0, n:, arc] = InitCondMat[(2*n*arc+n):(2*n*(arc+1)) , j]
            C = InitCondMat[(2*n*s):,j]

            # Non-homogeneous terms for the LSODE:
            # nonHom = [[phip*C - rho * phiu * fu + (1-rho) * err],
            #           [rho * fx]],
            #           for each time and arc
            nonHom = numpy.zeros((N,2*n,s))
            if numpy.any(C > 0.5):  # C is NOT composed of zeros!
                nonHom[:,:n,:] = numpy.einsum('nijs,j->nis',phip,C)

            if grad:
                nonHom[:, :n, :] -= phiuFu  # "A" terms
                nonHom[:, n:, :] = fx  # "lambda" terms
            else:
                nonHom[:, :n, :] += err  # "A" terms

            # Pre-multiplying with 0.5*dt yields better performance
            nonHom *= mdt


            # This command probably broke the compatibility with other integration
            # methods. They weren't working anyway, so...
            coefList = simp([],N,onlyCoef=True)

            genNonHom = numpy.einsum('nijs,njs->nis',self.InvDynMat[1:,:,:,:],
                                     nonHom[1:,:,:] + nonHom[:(N-1),:,:])

            # TODO: no need for a function here, this is just for profiling
            Xi = self.propXi(Xi,genNonHom) # most of the cost is right here!

            # Get the A values from Xi
            A, lam = Xi[:,:n,:], Xi[:,n:,:]
            if grad:
                # this calculates phiu^T * lam - fu, for all times and arcs
                B = numpy.einsum('njis,njs->nis', phiu, lam) - self.fu
            else:
                # this calculates phiu^T * lam, for all times and arcs
                B = numpy.einsum('njis,njs->nis', phiu, lam)
            # this calculates the product phip * lam, for all times and arcs...
            phiLamIntCol = numpy.einsum('njis,njs->nis', phip, lam)
            # ...and this, the integral of the previous term, w.r.t. the non-dim time
            phiLamIntCol = numpy.einsum('nis,n->i', phiLamIntCol, coefList)

            for arc in range(s):
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

        # Declare matrices Ctilde, Dtilde, Etilde, and the integral term
        if self.omit:
            # Grad and omit: proceed to omit
            NsRed = len(self.omitVarList)
            q_ = q + NsRed - Ns - 1
            omit = self.omitEqMat
        else:
            # nothing is omitted
            NsRed = Ns + 1
            q_ = q

        Ct = numpy.empty((p,NsRed))
        Dt = numpy.empty((2*n*s,NsRed))
        Et = numpy.empty((2*n*s,NsRed))
        phiLamInt = numpy.empty((p,NsRed))

        # Unpack outputs from 'propagate' into proper matrices Ct, Dt, etc.
        for j in range(NsRed):
            Ct[:,j] = res[j]['C']
            Dt[:,j] = res[j]['Dt']
            Et[:,j] = res[j]['Et']
            phiLamInt[:,j] = res[j]['phiLam']

        # Assembly of matrix M and column 'Col' for the linear system

        # Matrix for linear system involving k's and mu's
        M = numpy.zeros((NsRed+q,NsRed+q))
        # from eq (34d) - k term
        M[0,:NsRed] = numpy.ones(NsRed)
        # from eq (34b) - mu term
        M[(q_+1):(q_+1+p),NsRed:] = self.psip.transpose()
        # from eq (34c) - mu term
        M[(p+q_+1):,NsRed:] = self.psiy.transpose()
        # from eq (34a) - k term
        if self.omit:
            # under omission, less equations are needed
            M[1:(q_+1), :NsRed] = omit.dot(self.psiy.dot(Dt) + self.psip.dot(Ct))
        else:
            # no omission, all the equations are needed
            M[1:(q_ + 1), :NsRed] = self.psiy.dot(Dt) + self.psip.dot(Ct)
        # from eq (34b) - k term
        M[(q_+1):(q_+p+1),:NsRed] = Ct - phiLamInt
        # from eq (34c) - k term
        M[(q_+p+1):,:NsRed] = Et

        # column vector for linear system involving k's and mu's  [eqs (34)]
        col = numpy.zeros(NsRed+q)
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
            col[(q_+1):(q_+p+1)] = - sumIntFpi
        else:
            # eq (34a) - only applicable for rest
            col[1:(q+1)] = - self.psi


        # Calculations of weights k:
        KMi = numpy.linalg.solve(M,col)
        Res = M.dot(KMi)-col
        log.printL("LMPBVP: Residual of the Linear System: " + \
                   str(Res.transpose().dot(Res)))
        K,mu = KMi[:NsRed], KMi[NsRed:]
        log.printL("LMPBVP: coefficients of particular solutions: " + \
                   str(K))

        # summing up linear combinations
        A = numpy.zeros((N,n,s))
        B = numpy.zeros((N,m,s))
        C = numpy.zeros(p)
        lam = numpy.zeros((N,n,s))

        for j in range(NsRed):
            A   += K[j] * res[j]['A']#self.arrayA[j,:,:,:]
            B   += K[j] * res[j]['B']#self.arrayB[j,:,:,:]
            C   += K[j] * res[j]['C']#self.arrayC[j,:]
            lam += K[j] * res[j]['L']#self.arrayL[j,:,:,:]

###############################################################################
        if (self.rho > 0.5 and self.dbugOptGrad['plotCorrFin']) or \
           (self.rho < 0.5 and self.dbugOptRest['plotCorrFin']):
            log.printL("\n-----------------------------"
                       "-------------------------------")
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