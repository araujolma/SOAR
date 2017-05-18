#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:23:20 2017
MAIN_SGRA: the module (script?) for running the gradient-restoration algorithm
@author: levi
"""

import numpy, datetime#, time
import matplotlib.pyplot as plt

#from utils_alt import ddt

#from prob_rocket_sgra import declProb, calcPhi, calcPsi, calcGrads, plotSol
from prob_test import declProb, calcPhi, calcPsi, calcGrads, plotSol
from rest_sgra import calcP, rest#, oderest
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

    print("##################################################################")
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
    print("##################################################################")
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
    while P > tolP and NIterRest < (MaxIterRest-1):
        NIterRest += 1

#        if Ppsi/Pint > 1e-15:
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
#                nOdeRest = 10
#                print("Ready for fast restoration!\n")

        x,u,pi,lamR,muR = rest(sizes,x,u,pi,t,constants,boundary,restrictions)        
#        x,u,pi,lamR,muR = oderest(sizes,x,u,pi,t,constants,boundary,restrictions)                

        P,Pint,Ppsi = calcP(sizes,x,u,pi,constants,boundary,restrictions)#,True)
        print("> P = {:.4E}".format(P)+", Pint = {:.4E}".format(Pint)+\
          ", Ppsi = {:.4E}".format(Ppsi)+"\n")
        optPlot['P'] = P
        histP[NIterRest] = P
        histPint[NIterRest] = Pint
        histPpsi[NIterRest] = Ppsi  
        plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)         
        input("What now?")
            
    print("\nConvergence report:")
    
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

    print("\nAfter first rounds of restoration:")
    Q = calcQ(sizes,x,u,pi,lam,mu,constants,restrictions,True)
    optPlot['Q'] = Q; optPlot['dispQ'] = True
    plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)
    print("P = {:.4E}".format(P)+", Pint = {:.4E}".format(Pint)+\
          ", Ppsi = {:.4E}".format(Ppsi)+"\n")
    psi = calcPsi(sizes,x,boundary)
    print("psi =",psi)

    print("\n################################################################")
    print("\a")
    
    plt.plot(u[0:241,0])
    plt.grid(True)
    plt.title('This is the beginning of the problem...')
    plt.show()
    #input("ok?")
    
    print("\nBeginning gradient rounds...")
    
    # Gradient rounds:
    NIterGrad = 0
    histQ = histP*0.0; histQ[0] = Q
    while Q > tolQ:
        
#        plt.plot(u[0:241,0])
#        plt.grid(True)
#        plt.title('This is the beginning of the problem...')
#        plt.show()
       
        while P > tolP:
            print("\nPerforming restoration...")
            x,u,pi,lamR,muR = rest(sizes,x,u,pi,t,constants,boundary,restrictions)
            NIterRest+=1
            P,Pint,Psi = calcP(sizes,x,u,pi,constants,boundary,restrictions)#,True)
            optPlot['P'] = P
            histP[NIterRest] = P
            histPint[NIterRest] = Pint
            histPpsi[NIterRest] = Ppsi
            #plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)
            print("P = {:.4E}".format(P)+", Pint = {:.4E}".format(Pint)+\
                  ", Ppsi = {:.4E}".format(Ppsi)+"\n")

#            plt.plot(u[0:241,0])
#            plt.grid(True)
#            plt.title('This is the beginning of the problem...')
#            plt.show()
            #input("what now?")
        #
        print("\nRestoration report:")
        plt.semilogy(uman[0:(NIterRest+1)],histP[0:(NIterRest+1)])
        plt.hold(True)
        plt.semilogy(uman[0:(NIterRest+1)],histPint[0:(NIterRest+1)],'k')
        plt.semilogy(uman[0:(NIterRest+1)],histPpsi[0:(NIterRest+1)],'r')
        plt.grid()
        plt.title("Convergence of P. black: P_int, red: P_psi, blue: P")
        plt.ylabel("P")
        plt.xlabel("Iterations")
        plt.show()

        plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)
        
        x,u,pi,lam,mu,Q = grad(sizes,x,u,pi,t,Q,constants,restrictions)
        NIterGrad+=1
        optPlot['Q'] = Q; histQ[NIterGrad] = Q
        print("After grad:\n")
        P,Pint,Ppsi = calcP(sizes,x,u,pi,constants,boundary,restrictions,True)
        print("P = {:.4E}".format(P)+", Pint = {:.4E}".format(Pint)+\
          ", Ppsi = {:.4E}".format(Ppsi)+"\n")
        optPlot['P'] = P
        plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)
        
        plt.semilogy(uman[0:(NIterGrad+1)],histQ[0:(NIterGrad+1)])
        plt.grid()
        plt.title("Convergence of Q.")
        plt.ylabel("Q")
        plt.xlabel("Iterations")
        plt.show()
        print("\a")
        input("So far so good?")
    #
    print("\a")
    print("\n\n\nDONE!\n\n")
    print("\a")
    print("This is the final solution:")
    print("\a")
    plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)
    print("\a")
#