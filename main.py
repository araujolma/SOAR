#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:19:46 2017

@author: levi
"""

import datetime, time
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
            
# ##################
# MAIN SEGMENT:
# ##################
if __name__ == "__main__":
    print('------------------------------------------------------------------')
    print('\nRunning main.py!')
    print(datetime.datetime.now())
    print('\n')
    
    sol = prob.prob()#probRock.probRock()
    
    #GradStat = GradStat()

    ITman = ITman()
    ITman.greet()
    
    start_time = time.time()
    sol,solInit = ITman.setInitSol(sol)
    
    sol.plotTraj()  
    sol = ITman.frstRestRnds(sol)

    sol = ITman.gradRestCycl(sol,solInit)
    
    sol = ITman.restRnds(sol)
    
    print("\n\n")
    print("##################################################################")
    print("                   THIS IS THE FINAL SOLUTION:                    ")
    print("##################################################################")
    
    sol.plotSol()

    print("\n################################################################")
    print("=== First Guess + SGRA execution: %s seconds ===\n" % \
          (time.time() - start_time))
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