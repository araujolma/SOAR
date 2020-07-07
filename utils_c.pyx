# cython: profile = True
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 18:53:01 2016

put this up there for profiling # cython: profile = True

@author: levi
"""

import numpy

cimport cython
cimport numpy
#from libc.stdlib cimport malloc


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
    cdef double oneN = 1.0 / (3.0 * N)

    coefList[0] = oneN
    coefList[1:N:2] = 4.0 * oneN # set odd coefficientes to 4/3N
    coefList[2:N:2] = 2.0 * oneN # set even coefficientes to 2/3N
    coefList[N-1] = oneN

    if onlyCoef:
        return numpy.asarray(coefList)
    else:
        return numpy.asarray(coefList.dot(vec))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef c_simp(int N):
    """ Coefficientes for simple integration of array according to Simpson's
    method. """


    #cdef double[:]

    coefList = numpy.empty(N, dtype=numpy.double)
    cdef:
        double oneN = 1.0 / (3.0 * N)
        #double *

    coefList[0] = oneN
    coefList[1:N:2] = 4.0 * oneN # set odd coefficientes to 4/3N
    coefList[2:N:2] = 2.0 * oneN # set even coefficientes to 2/3N
    coefList[N-1] = oneN

    return coefList


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


    cdef:
        int Ns = sizes['Ns']
        int N = sizes['N']
        int n = sizes['n']
        int m = sizes['m']
        int p = sizes['p']
        int s = sizes['s']

        Py_ssize_t arc, k
        double dt = 1.0 / (N-1)
        double mdt = 0.5 * dt
        double mhdt = - 0.5 * dt

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
              phip, phipTr, phiuFu, phiuTr, grad=True, isCnull=False):
    return c_propagate(j, sizes, DynMat, err, fu, fx, InitCondMat, InvDynMat,
              phip, phipTr, phiuFu, phiuTr, grad)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef dummy1(int grad, int n, int s, int N, double mdt, C, err, fx, phip,
            phiuFu):
    """  Non-homogeneous terms for the LSODE """

    nonHom = numpy.empty((N,2*n,s))
    cdef Py_ssize_t arc, k


    #print("\nPropagating with j = {}, C = {}".format(j,C))
    if numpy.any(C > 0.5): # C is NOT composed of zeros!
        #print("Doing the proper math...")
        for arc in range(s):
            for k in range(N):
                nonHom[k,:n,arc] = phip[k,:,:,arc].dot(C)
    else:
        #print("Skipped the math!")
        nonHom[:,:n,:] = numpy.zeros((N,n,s))

    if grad:
        nonHom[:,:n,:] -= phiuFu # "A" terms
        nonHom[:,n:,:] = fx      # "lambda" terms
    else:
        nonHom[:,:n,:] += err                 # "A" terms
        nonHom[:,n:,:] = numpy.zeros((N,n,s)) # "lambda" terms

    # Pre-multiplying with 0.5*dt should yield better perfomance
    nonHom *= mdt
    return nonHom


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef dummy2(int grad, int n, int s, int N, double mdt, coefList, fu, lam,
            nonHom, phipTr, phiuTr, phiLamIntCol, B, DynMat, InvDynMat, Xi):
    """Perform the actual integration """
    lam_kp1 = numpy.empty(n)
    cdef Py_ssize_t arc, k

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
                  nonHom[k+1,:,arc]+nonHom[k,:,arc] )

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
                   nonHom[k+1,:,arc]+nonHom[k,:,arc] )

                lam_kp1 = Xi[k+1,n:,arc]
                B[k+1,:,arc] = phiuTr[k+1,:,:,arc].dot(lam_kp1)
                phiLamIntCol += coefList[k+1] * \
                                phipTr[k+1,:,:,arc].dot(lam_kp1)
    return phiLamIntCol, B, Xi


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef dummy3(int n, int s, int N, lam, A, DtCol, EtCol, Xi):
    """Assemble A, lam, Dt and Et."""
    cdef Py_ssize_t arc

    A = Xi[:,:n,:]
    lam = Xi[:,n:,:]
    for arc in range(s):
        # Get the A and lam values from Xi
        #A[:,:,arc] = Xi[:,:n,arc]
        #lam[:,:,arc] = Xi[:,n:,arc]

        # Put initial and final conditions of A and Lambda into matrices
        # DtCol and EtCol, which represent the columns of Dtilde(Dt) and
        # Etilde(Et)
        DtCol[(2*arc)*n   : (2*arc+1)*n] =    A[0,:,arc]   # eq (32a)
        DtCol[(2*arc+1)*n : (2*arc+2)*n] =    A[N-1,:,arc] # eq (32a)
        EtCol[(2*arc)*n   : (2*arc+1)*n] = -lam[0,:,arc]   # eq (32b)
        EtCol[(2*arc+1)*n : (2*arc+2)*n] =  lam[N-1,:,arc] # eq (32b)
    return lam, A, DtCol, EtCol


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef dict c_propagate(int j, dict sizes, DynMat,
                      err, fu, fx, InitCondMat, InvDynMat,
                      phip, phipTr, phiuFu, phiuTr, int grad):
#cdef dict c_propagate(int j, dict sizes, DynMat,
#                      double[:,:,:] err, fu, fx, double[:,:] InitCondMat,
#                      InvDynMat, phip, phipTr, phiuFu, phiuTr, int grad):



    # Load data (sizes, common matrices, etc)
    cdef:
        int Ns = sizes['Ns']
        int N = sizes['N']
        int n = sizes['n']
        int m = sizes['m']
        int p = sizes['p']
        int s = sizes['s']

        Py_ssize_t arc, k
        double dt = 1.0 / (N-1)
        double mdt = 0.5 * dt


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

    # non-homogenous terms
    nonHom = dummy1(grad, n, s, N, mdt, C, err, fx, phip, phiuFu)


    # get the coefficients for the phiLam integration
    coefList = c_simp(N)

    # perform the integration
    phiLamIntCol, B, Xi = dummy2(grad, n, s, N, mdt, coefList, fu, lam,
            nonHom, phipTr, phiuTr, phiLamIntCol, B, DynMat, InvDynMat, Xi)

    # assemble final arrays
    lam, A, DtCol, EtCol = dummy3(n, s, N, lam, A, DtCol, EtCol, Xi)


    # All the outputs go to main output dictionary; the final solution is
    # computed by the next method, 'getCorr'.
    outp = {'A':A,'B':B,'C':C,'L':lam,'Dt':DtCol,'Et':EtCol,
            'phiLam':phiLamIntCol}
    return outp

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cdef dict c_propagate(int j, dict sizes, DynMat,
#                       err, fu, fx, InitCondMat, InvDynMat,
#                       phip, phipTr, phiuFu, phiuTr, int grad):
# #cdef dict c_propagate(int j, dict sizes, DynMat,
# #                      double[:,:,:] err, fu, fx, double[:,:] InitCondMat,
# #                      InvDynMat, phip, phipTr, phiuFu, phiuTr, int grad):



#     # Load data (sizes, common matrices, etc)
#     cdef:
#         int Ns = sizes['Ns']
#         int N = sizes['N']
#         int n = sizes['n']
#         int m = sizes['m']
#         int p = sizes['p']
#         int s = sizes['s']

#         Py_ssize_t arc, k
#         double dt = 1.0 / (N-1)
#         double mdt = 0.5 * dt


#     I = numpy.eye(2*n)

#     # Declare matrices for corrections
#     phiLamIntCol = numpy.zeros(p)
#     A = numpy.zeros((N,n,s))
#     B = numpy.zeros((N,m,s))
#     #C = numpy.zeros((p,1))
#     DtCol = numpy.empty(2*n*s)
#     EtCol = numpy.empty(2*n*s)
#     lam = numpy.zeros((N,n,s))

#     # the vector that will be integrated is Xi = [A; lam]
#     Xi = numpy.zeros((N,2*n,s))
#     # Initial conditions for the LSODE:

#     for arc in range(s):
#         A[0,:,arc] = InitCondMat[2*n*arc:(2*n*arc+n) , j]
#         lam[0,:,arc] = InitCondMat[(2*n*arc+n):(2*n*(arc+1)) , j]
#         Xi[0,:n,arc],Xi[0,n:,arc] = A[0,:,arc],lam[0,:,arc]


#     C = InitCondMat[(2*n*s):,j]

#     # Non-homogeneous terms for the LSODE:
#     nonHom = numpy.empty((N,2*n,s))


#     # TODO: this can be further optimized: the C multiplication can be avioded
#     # in most cases, and err / phiuFu can be done in a single command!
#     #print("\nPropagating with j = {}, C = {}".format(j,C))
#     if sum(C) > 0.5: # C is NOT composed of zeros!
#         #print("Doing the proper math...")
#         for arc in range(s):
#             for k in range(N):
#                 nonHom[k,:n,arc] = phip[k,:,:,arc].dot(C)
#     else:
#         #print("Skipped the math!")
#         nonHom[:,:n,:] = numpy.zeros((N,n,s))

#     if grad:
#         nonHom[:,:n,:] -= phiuFu # "A" terms
#         nonHom[:,n:,:] = fx      # "lambda" terms
#     else:
#         nonHom[:,:n,:] += err                 # "A" terms
#         nonHom[:,n:,:] = numpy.zeros((N,n,s)) # "lambda" terms

#     # Pre-multiplying with 0.5*dt should yield better perfomance
#     nonHom *= mdt

#     # get the coefficients for the phiLam integration
#     coefList = c_simp(N)

#     lam_kp1 = numpy.empty(n)
#     if grad:
#         for arc in range(s):
#             # Integrate the LSODE by trapezoidal (implicit) method
#             B[0,:,arc] = -fu[0,:,arc] + \
#                                 phiuTr[0,:,:,arc].dot(lam[0,:,arc])
#             phiLamIntCol += coefList[0] * \
#                             (phipTr[0,:,:,arc].dot(lam[0,:,arc]))

#             # is it better to index in k+1 ?
#             for k in range(N-1):
#                 Xi[k+1,:,arc] = InvDynMat[k+1,:,:,arc].dot(
#                   DynMat[k,:,:,arc].dot(Xi[k,:,arc]) + \
#                   nonHom[k+1,:,arc]+nonHom[k,:,arc] )

#                 lam_kp1 = Xi[k+1,n:,arc]
#                 B[k+1,:,arc] = -fu[k+1,:,arc] + \
#                                 phiuTr[k+1,:,:,arc].dot(lam_kp1)
#                 phiLamIntCol += coefList[k+1] * \
#                                 phipTr[k+1,:,:,arc].dot(lam_kp1)
#     else:
#         for arc in range(s):
#             # Integrate the LSODE by trapezoidal (implicit) method
#             B[0,:,arc] = phiuTr[0,:,:,arc].dot(lam[0,:,arc])
#             phiLamIntCol += coefList[0] * \
#                             (phipTr[0,:,:,arc].dot(lam[0,:,arc]))

#             # is it better to index in k+1 ?
#             for k in range(N-1):
#                 Xi[k+1,:,arc] = InvDynMat[k+1,:,:,arc].dot(
#                    DynMat[k,:,:,arc].dot(Xi[k,:,arc]) + \
#                    nonHom[k+1,:,arc]+nonHom[k,:,arc] )

#                 lam_kp1 = Xi[k+1,n:,arc]
#                 B[k+1,:,arc] = phiuTr[k+1,:,:,arc].dot(lam_kp1)
#                 phiLamIntCol += coefList[k+1] * \
#                                 phipTr[k+1,:,:,arc].dot(lam_kp1)


#     for arc in range(s):
#         # Get the A and lam values from Xi
#         A[:,:,arc] = Xi[:,:n,arc]
#         lam[:,:,arc] = Xi[:,n:,arc]

#         # Put initial and final conditions of A and Lambda into matrices
#         # DtCol and EtCol, which represent the columns of Dtilde(Dt) and
#         # Etilde(Et)
#         DtCol[(2*arc)*n   : (2*arc+1)*n] =    A[0,:,arc]   # eq (32a)
#         DtCol[(2*arc+1)*n : (2*arc+2)*n] =    A[N-1,:,arc] # eq (32a)
#         EtCol[(2*arc)*n   : (2*arc+1)*n] = -lam[0,:,arc]   # eq (32b)
#         EtCol[(2*arc+1)*n : (2*arc+2)*n] =  lam[N-1,:,arc] # eq (32b)

#     # All the outputs go to main output dictionary; the final solution is
#     # computed by the next method, 'getCorr'.
#     outp = {'A':A,'B':B,'C':C,'L':lam,'Dt':DtCol,'Et':EtCol,
#             'phiLam':phiLamIntCol}
#     return outp
