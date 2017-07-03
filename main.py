#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:19:46 2017

@author: levi
"""

import datetime#, time
#import matplotlib.pyplot as plt

from interf import ITman
import probRock as prob

# MODOS: full debug (esperar)
# Gradiente: 
    # full debug: imprime tudo, espera comando para seguir
    # autonomo: imprime nada, passa pra próxima sozinho
    # soneca: imprime nada, conta algumas iterações passando sozinho depois pergunta
    # print: imprime tudo

#    

#%%

            
#%%
            
# ##################
# MAIN SEGMENT:
# ##################
if __name__ == "__main__":
    print('--------------------------------------------------------------------------------')
    print('\nRunning main.py!')
    print(datetime.datetime.now())
    print('\n')
    
    sol = prob.prob()#probRock.probRock()
    #GradStat = GradStat()

    ITman = ITman()
    ITman.greet()
    
    if ITman.isNewSol:
        # declare problem:
        #opt = {'initMode': 'extSol'}#'crazy'#'default'#'extSol'
        sol.initGues({'initMode':ITman.initOpt})
        ITman.saveSol(sol,'solInit.pkl')
    else:
        # load previously prepared solution
        sol = ITman.loadSol()
    #
    
    ITman.checkPars(sol)

    #Grads = sol.calcGrads()
    #phi = calcPhi(sizes,x,u,pi,constants,restrictions)
    #psi = calcPsi(sizes,x,boundary)
#    phix = Grads['phix']
#    phiu = Grads['phiu']
#    psix = Grads['psix']
#    psip = Grads['psip']

    #input("E aí?")

#%%
    
    print("##################################################################")
    print("\nProposed initial guess:\n")
#
    
    P,Pint,Ppsi = sol.calcP()
    print("P = {:.4E}".format(P)+", Pint = {:.4E}".format(Pint)+\
              ", Ppsi = {:.4E}".format(Ppsi)+"\n")
    sol.histP[sol.NIterRest] = P
    sol.histPint[sol.NIterRest] = Pint
    sol.histPpsi[sol.NIterRest] = Ppsi
    
    Q,Qx,Qu,Qp,Qt = sol.calcQ()
    #print("Q = {:.4E}".format(Q)+", Qx = {:.4E}".format(Qx)+\
    #      ", Qu = {:.4E}".format(Qu)+", Qp = {:.4E}".format(Qp)+\
    #      ", Qt = {:.4E}".format(Qt)+"\n")
    
    
#    Q = calcQ(sizes,x,u,pi,lam,mu,constants,restrictions)
#    optPlot = dict()
#    optPlot['P'] = P
#    optPlot['Q'] = Q
#    optPlot['mode'] = 'sol'
#    optPlot['dispP'] = True
#    optPlot['dispQ'] = False
#    plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)
#
#    psi = calcPsi(sizes,x,boundary)
#    print("psi =",psi)
#    print("##################################################################")
#    #input("Everything ok?")
#
#    tolP = tol['P']
#    tolQ = tol['Q']
#
#
#    # first restoration rounds:
#    NIterRest = 0
#    #mustOdeRest = True
#    nOdeRest = 0#10
#    histP[0] = P; histPint[0] = Pint; histPpsi[0] = Ppsi
#    print("\nBeginning first restoration rounds...\n")
    while sol.P > sol.tol['P']:
        sol.rest()
        P,Pint,Ppsi = sol.calcP()
        print("P = {:.4E}".format(P)+", Pint = {:.4E}".format(Pint)+\
              ", Ppsi = {:.4E}".format(Ppsi)+"\n")
        sol.plotSol()
     
    #print("\nConvergence report:")
    #sol.showHistP()
    

#
# #           print("\a")
##            time.sleep(.2)
#
#    print("\nAfter first rounds of restoration:")
#    Q = calcQ(sizes,x,u,pi,lam,mu,constants,restrictions,True)
#    optPlot['Q'] = Q; optPlot['dispQ'] = True
#    plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)
#    print("P = {:.4E}".format(P)+", Pint = {:.4E}".format(Pint)+\
#          ", Ppsi = {:.4E}".format(Ppsi)+"\n")
#    psi = calcPsi(sizes,x,boundary)
#    print("psi =",psi)
#
#    print("\n################################################################")
#    print("\n################################################################")
#    print("\n################################################################")
#    print("\n################################################################")
#    print("\a")
#    
#    print("\nProceed to gradient rounds?")
#    input()
    print("Beginning gradient rounds...")
    sol.Q,_,_,_,_ = sol.calcQ()
    while sol.Q > sol.tol['Q']:

        sol.rest()
        while sol.P > sol.tol['P']:
            sol.rest()
        sol.showHistP()

        sol.grad()
        sol.showHistQ()

        if sol.NIterGrad%10==0:
            input("So far so good?")
        
   

#    # Gradient rounds:
#    NIterGrad = 0
#    histQ = histP*0.0; histQ[0] = Q

        #contP = 3
        #while contP > 0 and sol.P > sol.tol['P']:#P > tolP:
#            print("\nPerforming restoration...")
#            sol.rest()
#            x,u,pi,lamR,muR = rest(sizes,x,u,pi,t,constants,boundary,restrictions)
#            NIterRest+=1
#            P,Pint,Psi = calcP(sizes,x,u,pi,constants,boundary,restrictions,\
#                               GradStat.mustPlotRest)
#            optPlot['P'] = P
#            histP[NIterRest] = P
#            histPint[NIterRest] = Pint
#            histPpsi[NIterRest] = Ppsi
#            #plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)
#            print("P = {:.4E}".format(P)+", Pint = {:.4E}".format(Pint)+\
#                  ", Ppsi = {:.4E}".format(Ppsi)+"\n")
            #contP-=1
#            #input("what now?")
#        #
#        print("\nRestoration report:")
#        plt.semilogy(uman[0:(NIterRest+1)],histP[0:(NIterRest+1)])
#        plt.semilogy(uman[0:(NIterRest+1)],histPint[0:(NIterRest+1)],'k')
#        plt.semilogy(uman[0:(NIterRest+1)],histPpsi[0:(NIterRest+1)],'r')
#        plt.grid()
#        plt.title("Convergence of P. black: P_int, red: P_psi, blue: P")
#        plt.ylabel("P")
#        plt.xlabel("Iterations")
#        plt.show()
#
#        plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)
#        
#        x,u,pi,lam,mu,Q = grad(sizes,x,u,pi,t,Q,constants,boundary,\
#                               restrictions,GradStat.mustPlotGrad)
#        NIterGrad+=1
#        optPlot['Q'] = Q; histQ[NIterGrad] = Q
#        print("\nAfter grad:\n")
#        P,Pint,Ppsi = calcP(sizes,x,u,pi,constants,boundary,restrictions,\
#                            GradStat.mustPlotRest)
#        print("P = {:.4E}".format(P)+", Pint = {:.4E}".format(Pint)+\
#          ", Ppsi = {:.4E}".format(Ppsi)+"\n")
#        optPlot['P'] = P
#        if GradStat.mustPlotSol:
#            plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)
#        print("mu =",mu,"\n")
#        
#        plt.semilogy(uman[1:(NIterGrad+1)],histQ[1:(NIterGrad+1)])
#        plt.grid()
#        plt.title("Convergence of Q.")
#        plt.ylabel("Q")
#        plt.xlabel("Iterations")
#        plt.show()
#        print("\a")
#        
#        GradStat.endOfLoop()
#        sol.showHistQ()
#        input("So far so good?")
#    #
#    
#    while P > tolP:
#        print("\nPerforming final restoration...")
#        x,u,pi,lamR,muR = rest(sizes,x,u,pi,t,constants,boundary,restrictions,False)
#        NIterRest+=1
#        P,Pint,Psi = calcP(sizes,x,u,pi,constants,boundary,restrictions)#,True)
#        optPlot['P'] = P
#        histP[NIterRest] = P
#        histPint[NIterRest] = Pint
#        histPpsi[NIterRest] = Ppsi
#        #plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)
#        print("P = {:.4E}".format(P)+", Pint = {:.4E}".format(Pint)+\
#              ", Ppsi = {:.4E}".format(Ppsi)+"\n")
#    
#    print("\a")
#    print("\n\n\nDONE!\n\n")
#    print("\a")
#    print("This is the final solution:")
#    print("\a")
#    plotSol(sizes,t,x,u,pi,constants,restrictions,optPlot)
#    print("\a")
##