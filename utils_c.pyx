# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 18:53:01 2016

@author: levi
"""
import numpy

cimport cython
cimport numpy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def simp(vec, int N, onlyCoef=False):
    """ Simple integration of array according to Simpson's method.
    It can also only yield the coefficients if one wants to do the integration
    by oneself (maybe in an optimized loop)."""


#    coefList = numpy.ones(N)
#    coefList[0] = 17.0/48.0; coefList[N-1] = coefList[0]
#    coefList[1] = 59.0/48.0; coefList[N-2] = coefList[1]
#    coefList[2] = 43.0/48.0; coefList[N-3] = coefList[2]
#    coefList[3] = 49.0/48.0; coefList[N-4] = coefList[3]
#    coefList *= 1.0/(N-1)

    #cdef double[:]

    coefList = numpy.empty(N, dtype=numpy.double)
    cdef Py_ssize_t k
    cdef double oneN = 1.0 / (3.0 * N)
    cdef double twoN = 2.0 * oneN
    cdef double fourN = 2.0 * twoN

    for k in range(1,N-1):
        if k % 2 == 0:
            coefList[k] = twoN
        else:
            coefList[k] = fourN
    #
    coefList[0] = oneN
    coefList[N-1] = oneN

    if onlyCoef:
        return numpy.asarray(coefList)
    else:
        return numpy.asarray(coefList.dot(vec))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def prepMat(sizes, fu, phip, phiu, phix):
    return c_prepMat(sizes, fu, phip, phiu, phix)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef dict c_prepMat(dict sizes, fu, phip, phiu, phix):
    """ Prepare matrices for propagation. """


    cdef int Ns, N, n, m, p, s
    cdef Py_ssize_t arc, k#, kp1, Nm1
    Ns, N, n, m, p, s = sizes['Ns'], sizes['N'], sizes['n'], \
                        sizes['m'],  sizes['p'], sizes['s']
    #Nm1 = N-1
    cdef double dt = 1.0 / (N-1)#Nm1
    cdef double mdt = 0.5 * dt
    cdef double mhdt = - 0.5 * dt

    # Prepare matrices with derivatives:
    phiuTr = numpy.empty((N,m,n,s))
    phipTr = numpy.empty((N,p,n,s))
    phiuFu = numpy.empty((N,n,s))

    # Initial conditions
    InitCondMat = numpy.eye(Ns,Ns+1)

    # Dynamics matrix for propagating the LSODE:
    # originally, DynMat was the matrix, D,  with phix, phiu*phiu^T, etc;
    # but it is more efficient to store (I + .5 * dt * D) instead.
    DynMat_ = numpy.zeros((N,2*n,2*n,s))
    DynMat = numpy.empty((N,2*n,2*n,s))
    I = numpy.eye(2 * n)

    for arc in range(s):
        for k in range(N):
            phiuTr[k,:,:,arc] = phiu[k,:,:,arc].transpose()
            phipTr[k,:,:,arc] = phip[k,:,:,arc].transpose()
            phiuFu[k,:,arc] = phiu[k,:,:,arc].dot(fu[k,:,arc])

            # TODO: this can be moved to a single command as well
            DynMat_[k,:n,:n,arc] = phix[k,:,:,arc]

            DynMat_[k,:n,n:,arc] = phiu[k,:,:,arc].dot(phiuTr[k,:,:,arc])
            DynMat_[k,n:,n:,arc] = -phix[k,:,:,arc].transpose()

            # TODO: maybe it is even better to do this operation in a single
            # command...
            DynMat[k, :, :, arc] = I + mdt * DynMat_[k,:,:,arc]


    # This is a strategy for lowering the cost of the trapezoidal solver.
    # Instead of solving (N-1) * s * (2ns+p) linear systems of order 2n,
    # resulting in a cost in the order of (N-1) * s * (2ns+p) * 4n² ; it is
    # better to pre-invert (N-1) * s matrices of order 2n, resulting in a cost
    # in the order of (N-1) * s * 8n³. This should always result in an
    # increased performance because
    #         (N-1) * s * 8n³ < (N-1) * s * (2ns+p) * 4n²

    InvDynMat = numpy.zeros((N, 2 * n, 2 * n, s))
    for arc in range(s):
        for k in range(1,N):
             #InvDynMat[k, :, :, arc] = numpy.linalg.inv(DynMat[k, :, :, arc])
             InvDynMat[k, :, :, arc] = numpy.linalg.inv(
                 I + mhdt * DynMat_[k, :, :, arc])


    outp = {'DynMat': DynMat, 'InitCondMat': InitCondMat,
            'InvDynMat': InvDynMat,
            'phipTr':phipTr, 'phiuFu': phiuFu, 'phiuTr': phiuTr}
    return outp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def propagate(j, sizes, DynMat, err, fu, fx, InitCondMat, InvDynMat,
              phip, phipTr, phiuFu, phiuTr, grad=True):
    return c_propagate(j, sizes, DynMat, err, fu, fx, InitCondMat, InvDynMat,
              phip, phipTr, phiuFu, phiuTr, grad=grad)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef dict c_propagate(int j, dict sizes, DynMat, err, fu, fx, InitCondMat,
                      InvDynMat, phip, phipTr, phiuFu, phiuTr, int grad):

    # Load data (sizes, common matrices, etc)
    cdef int Ns, N, n, m, p, s
    cdef Py_ssize_t arc, k
    Ns, N, n, m, p, s = sizes['Ns'], sizes['N'], sizes['n'], \
                        sizes['m'],  sizes['p'], sizes['s']

    cdef double dt = 1.0 / (N-1)
    cdef double mdt = 0.5 * dt

    I = numpy.eye(2*n)

    # Declare matrices for corrections
    phiLamIntCol = numpy.zeros(p)
    A = numpy.zeros((N,n,s))
    B = numpy.zeros((N,m,s))
    #C = numpy.zeros((p,1))
    DtCol = numpy.empty(2*n*s)
    EtCol = numpy.empty(2*n*s)
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


    # TODO: this can be further optimized: the C multiplication can be avioded
    # in most cases, and err / phiuFu can be done in a single command!
    if grad:
        for arc in range(s):
            for k in range(N):
                # "A" terms
                nonHom[k,:n,arc] = phip[k,:,:,arc].dot(C) - phiuFu[k,:,arc]
        nonHom[:,n:,:] = fx # "lambda" terms

    else:
        for arc in range(s):
            for k in range(N):
                # "A" terms
                nonHom[k,:n,arc] = phip[k,:,:,arc].dot(C) + err[k,:,arc]
        nonHom[:,n:,:] = numpy.zeros((N,n,s)) # "lambda" terms

    # get the coefficients for the phiLam integration
    coefList = simp([],N,onlyCoef=True)

    lam_kp1 = numpy.empty(n)
    if grad:
        for arc in range(s):
            # Integrate the LSODE by trapezoidal (implicit) method
            B[0,:,arc] = -fu[0,:,arc] + \
                                phiuTr[0,:,:,arc].dot(lam[0,:,arc])
            phiLamIntCol += coefList[0] * \
                            (phipTr[0,:,:,arc].dot(lam[0,:,arc]))

            # is it better to index in k+1 ?
            for k in range(N-1):
                Xi[k+1,:,arc] = InvDynMat[k+1,:,:,arc].dot(
                  DynMat[k,:,:,arc].dot(Xi[k,:,arc]) + \
                  mdt * (nonHom[k+1,:,arc]+nonHom[k,:,arc]) )

                lam_kp1 = Xi[k+1,n:,arc]
                B[k+1,:,arc] = -fu[k+1,:,arc] + \
                                phiuTr[k+1,:,:,arc].dot(lam_kp1)
                phiLamIntCol += coefList[k+1] * \
                                phipTr[k+1,:,:,arc].dot(lam_kp1)
    else:
        for arc in range(s):
            # Integrate the LSODE by trapezoidal (implicit) method
            B[0,:,arc] = phiuTr[0,:,:,arc].dot(lam[0,:,arc])
            phiLamIntCol += coefList[0] * \
                            (phipTr[0,:,:,arc].dot(lam[0,:,arc]))

            # is it better to index in k+1 ?
            for k in range(N-1):
                Xi[k+1,:,arc] = InvDynMat[k+1,:,:,arc].dot(
                   DynMat[k,:,:,arc].dot(Xi[k,:,arc]) + \
                   mdt * (nonHom[k+1,:,arc]+nonHom[k,:,arc]) )

                lam_kp1 = Xi[k+1,n:,arc]
                B[k+1,:,arc] = phiuTr[k+1,:,:,arc].dot(lam_kp1)
                phiLamIntCol += coefList[k+1] * \
                                phipTr[k+1,:,:,arc].dot(lam_kp1)


    for arc in range(s):
        # Get the A and lam values from Xi
        A[:,:,arc] = Xi[:,:n,arc]
        lam[:,:,arc] = Xi[:,n:,arc]

        # Put initial and final conditions of A and Lambda into matrices
        # DtCol and EtCol, which represent the columns of Dtilde(Dt) and
        # Etilde(Et)
        DtCol[(2*arc)*n   : (2*arc+1)*n] =    A[0,:,arc]   # eq (32a)
        DtCol[(2*arc+1)*n : (2*arc+2)*n] =    A[N-1,:,arc] # eq (32a)
        EtCol[(2*arc)*n   : (2*arc+1)*n] = -lam[0,:,arc]   # eq (32b)
        EtCol[(2*arc+1)*n : (2*arc+2)*n] =  lam[N-1,:,arc] # eq (32b)

    # All the outputs go to main output dictionary; the final solution is
    # computed by the next method, 'getCorr'.
    outp = {'A':A,'B':B,'C':C,'L':lam,'Dt':DtCol,'Et':EtCol,
            'phiLam':phiLamIntCol}
    return outp
