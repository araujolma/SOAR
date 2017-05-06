#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:23:20 2017
MAIN_SGRA: the module (script?) for running the gradient-restoration algorithm
@author: levi
"""

import numpy, time, datetime
import matplotlib.pyplot as plt

#from utils_alt import ddt

from prob_rocket_sgra import declProb, calcPhi, calcPsi, calcGrads, plotSol
#from prob_test import declProb, calcPhi, calcPsi, calcGrads, plotSol
from rest_sgra import calcP, rest, oderest
from grad_sgra import calcQ, grad

# ##################
# MAIN SEGMENT:
# ##################
if __name__ == "__main__":
    print('--------------------------------------------------------------------------------')
    print('\nThis is MAIN_SGRA.py!')
    print(datetime.datetime.now())
    print('\n')
    
    opt = dict()
    opt['initMode'] = 'extSol'#'default'#'extSol'

    # declare problem:
    sizes,t,x,u,pi,lam,mu,tol,constants,boundary,restrictions = declProb(opt)

    Grads = calcGrads(sizes,x,u,pi,constants,restrictions)
    phi = calcPhi(sizes,x,u,pi,constants,restrictions)
    psi = calcPsi(sizes,x,boundary)
#    phix = Grads['phix']
#    phiu = Grads['phiu']
#    psix = Grads['psix']
#    psip = Grads['psip']

    print("\nProposed initial guess:\n")

    P,Pint,Ppsi = calcP(sizes,x,u,pi,constants,boundary,restrictions,True)
    print("P = {:.4E}".format(P)+", Pint = {:.4E}".format(Pint)+\
              ", Ppsi = {:.4E}".format(Ppsi)+"\n")
    Q = calcQ(sizes,x,u,pi,lam,mu,constants,restrictions)
    optPlot = dict()
    optPlot['P'] = P
    optPlot['Q'] = Q
    optPlot['mode'] = 'sol'
    optPlot['dispP'] = True
    optPlot['dispQ'] = False
    plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)

    psi = calcPsi(sizes,x,boundary)
    print("psi =",psi)
    #input("Everything ok?")

    tolP = tol['P']
    tolQ = tol['Q']

    MaxIterRest = 10000
    histP = numpy.zeros(MaxIterRest)
    histPint = histP.copy()
    histPpsi = histP.copy()
    uman = numpy.linspace(0,MaxIterRest,MaxIterRest+1) #um a n

    # first restoration rounds:
    NIterRest = 0
    #mustOdeRest = True
    nOdeRest = 0#10
    histP[0] = P; histPint[0] = Pint; histPpsi[0] = Ppsi
    print("\nBeginning first restoration rounds...\n")
    while P > tolP and NIterRest < MaxIterRest:
        NIterRest += 1

#        if Ppsi/Pint < 1e-5:
#            x,u,pi,lam,mu = rest(sizes,x,u,pi,t,constants,boundary,restrictions)
#        else:
#            x,u,pi,lam,mu = oderest(sizes,x,u,pi,t,constants,boundary,restrictions)
        
#        if nOdeRest>0: #Pint > 1e-5:#NIterRest % 3 == 0:
#            x,u,pi,lam,mu = oderest(sizes,x,u,pi,t,constants,boundary,restrictions)
#            nOdeRest-=1
#            if nOdeRest==0:
#                print("Back to normal restoration.\n")
#        else: 
#            x,u,pi,lam,mu = rest(sizes,x,u,pi,t,constants,boundary,restrictions)
#            if Ppsi/Pint < 1e-5:
#                nOdeRest = 100
#                print("Ready for fast restoration!\n")

        x,u,pi,lam,mu = rest(sizes,x,u,pi,t,constants,boundary,restrictions)        
#        x,u,pi,lam,mu = oderest(sizes,x,u,pi,t,constants,boundary,restrictions)                

        P,Pint,Ppsi = calcP(sizes,x,u,pi,constants,boundary,restrictions,True)
        optPlot['P'] = P
        histP[NIterRest] = P
        histPint[NIterRest] = Pint
        histPpsi[NIterRest] = Ppsi

        #if NIterRest % 50 == 0:
        plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)
        
        plt.semilogy(uman[0:(NIterRest+1)],histP[0:(NIterRest+1)])
        plt.hold(True)
        plt.semilogy(uman[0:(NIterRest+1)],histPint[0:(NIterRest+1)],'k')
        plt.semilogy(uman[0:(NIterRest+1)],histPpsi[0:(NIterRest+1)],'r')
        plt.grid()
        plt.title("Convergence of P. black: P_int, red: P_psi, blue: P")
        plt.ylabel("P")
        plt.xlabel("Iterations")
        plt.show()
    
 #           print("\a")
#            time.sleep(.2)
        print("P = {:.4E}".format(P)+", Pint = {:.4E}".format(Pint)+\
              ", Ppsi = {:.4E}".format(Ppsi)+"\n")
        psi = calcPsi(sizes,x,boundary)
        print("psi =",psi)
        #    print(datetime.datetime.now())
        print("\a So far, so good?")
#            time.sleep(0.2)
        #    print("\a")
        #
        time.sleep(5.0)
       # input("\nPress any key to continue...")

    print("\nAfter first rounds of restoration:")
    Q = calcQ(sizes,x,u,pi,lam,mu,constants,restrictions)
    optPlot['Q'] = Q; optPlot['dispQ'] = True
    plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)

    # Gradient rounds:
    while Q > tolQ:
        while P > tolP:
            x,u,pi,lam,mu = rest(sizes,x,u,pi,t,constants,boundary,restrictions)
            P = calcP(sizes,x,u,pi,constants,boundary,restrictions)
            optPlot['P'] = P
            plotSol(sizes,t,x,u,pi,constants,restrictions, optPlot)
        #
        x,u,pi,lam,mu,Q = grad(sizes,x,u,pi,t,Q,restrictions)
        optPlot['Q'] = Q
        plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)
    #
#