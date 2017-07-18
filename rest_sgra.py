#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:36:59 2017

@author: levi
"""
import numpy
from utils import ddt
import matplotlib.pyplot as plt

def calcP(self):
    N,s,dt = self.N,self.s,self.dt
    x = self.x

    phi = self.calcPhi()
    psi = self.calcPsi()
    dx = ddt(x,N)
    
    func = dx-phi
    vetP = numpy.empty((N,s))
    vetIP = numpy.empty((N,s))
    
    for arc in range(s):
        P = .5*(func[0,:,arc].dot(func[0,:,arc].transpose()))
        vetP[0,arc] = P
        vetIP[0,arc] = P

        for t in range(1,N-1):  
            vetP[t,arc] = func[t,:,arc].dot(func[t,:,arc].transpose())
            P += vetP[t,arc]
            vetIP[t,arc] = P
    
        vetP[N-1,arc] = .5*(func[N-1,:,arc].dot(func[N-1,:,arc].transpose()))
        P += vetP[N-1,arc]
        vetIP[N-1,arc] = P
    
        P *= dt

    vetIP *= dt

    somePlot = False
    for key in self.dbugOptRest.keys():
        if ('plot' in key) or ('Plot' in key):
            if self.dbugOptRest[key]:
                somePlot = True
                break
    if somePlot:
        print("\nDebug plots for this calcP run:")
        
        indMaxP = numpy.argmax(vetP, axis=0)
        print(indMaxP)
        for arc in range(s):
            print("\nArc =",arc,"\n")
            ind1 = numpy.array([indMaxP[arc]-20,0]).max()
            ind2 = numpy.array([indMaxP[arc]+20,N]).min()
    
            if self.dbugOptRest['plotP_int']:
                plt.plot(self.t,vetP[:,arc])
                plt.grid(True)
                plt.title("Integrand of P")
                plt.show()
            
            if self.dbugOptRest['plotIntP_int']:
                plt.plot(self.t,vetIP[:,arc])
                plt.grid(True)
                plt.title("Partially integrated P")
                plt.show()

            #for zoomed version:
            if self.dbugOptRest['plotP_intZoom']:
                plt.plot(self.t[ind1:ind2],vetP[ind1:ind2,arc],'o')
                plt.grid(True)
                plt.title("Integrand of P (zoom)")
                plt.show()
            
            if self.dbugOptRest['plotSolMaxP']:
                print("rest_sgra: plotSol @ MaxP region: not implemented yet!")
                #print("\nSolution on the region of MaxP:")
                #self.plotSol(intv=numpy.arange(ind1,ind2,1,dtype='int'))
        
#        # TODO: extend these debug plots
#    if self.dbugOptRest['plotRsidMaxP']:
#        
#        print("\nResidual on the region of maxP:")
# 
#        if self.n==4 and self.m ==2:
#            plt.plot(self.t[ind1:ind2],func[ind1:ind2,0])
#            plt.grid(True)
#            plt.ylabel("res_hDot [km/s]")
#            plt.show()        
#    
#            plt.plot(self.t[ind1:ind2],func[ind1:ind2,1],'g')
#            plt.grid(True)
#            plt.ylabel("res_vDot [km/s/s]")
#            plt.show()
#    
#            plt.plot(self.t[ind1:ind2],func[ind1:ind2,2]*180/numpy.pi,'r')
#            plt.grid(True)
#            plt.ylabel("res_gammaDot [deg/s]")
#            plt.show()
#    
#            plt.plot(self.t[ind1:ind2],func[ind1:ind2,3],'m')
#            plt.grid(True)
#            plt.ylabel("res_mDot [kg/s]")
#            plt.show()
#            
#    #            print("\nState time derivatives on the region of maxP:")
#    #
#    #            plt.plot(tPlot[ind1:ind2],dx[ind1:ind2,0])
#    #            plt.grid(True)
#    #            plt.ylabel("hDot [km/s]")
#    #            plt.show()        
#    #    
#    #            plt.plot(tPlot[ind1:ind2],dx[ind1:ind2,1],'g')
#    #            plt.grid(True)
#    #            plt.ylabel("vDot [km/s/s]")
#    #            plt.show()
#    #    
#    #            plt.plot(tPlot[ind1:ind2],dx[ind1:ind2,2]*180/numpy.pi,'r')
#    #            plt.grid(True)
#    #            plt.ylabel("gammaDot [deg/s]")
#    #            plt.show()
#    #    
#    #            plt.plot(tPlot[ind1:ind2],dx[ind1:ind2,3],'m')
#    #            plt.grid(True)
#    #            plt.ylabel("mDot [kg/s]")
#    #            plt.show()
#    #            
#    #            print("\nPHI on the region of maxP:")
#    #
#    #            plt.plot(tPlot[ind1:ind2],phi[ind1:ind2,0])
#    #            plt.grid(True)
#    #            plt.ylabel("hDot [km/s]")
#    #            plt.show()        
#    #    
#    #            plt.plot(tPlot[ind1:ind2],phi[ind1:ind2,1],'g')
#    #            plt.grid(True)
#    #            plt.ylabel("vDot [km/s/s]")
#    #            plt.show()
#    #    
#    #            plt.plot(tPlot[ind1:ind2],phi[ind1:ind2,2]*180/numpy.pi,'r')
#    #            plt.grid(True)
#    #            plt.ylabel("gammaDot [deg/s]")
#    #            plt.show()
#    #    
#    #            plt.plot(tPlot[ind1:ind2],phi[ind1:ind2,3],'m')
#    #            plt.grid(True)
#    #            plt.ylabel("mDot [kg/s]")
#    #            plt.show()
#    
#    
#    #        else:
#    #             print("Not implemented (yet).")   
#        #
#    #        
        
    Pint = vetIP[N-1,:].sum()#P
    Ppsi = psi.transpose().dot(psi)
    P = Ppsi + Pint
    print("P = {:.6E}".format(P)+", Pint = {:.6E}".format(Pint)+\
          ", Ppsi = {:.6E}.".format(Ppsi))
    return P,Pint,Ppsi    

def calcStepRest(self,corr):
    print("\nIn calcStepRest.\n")
    
    P0,_,_ = calcP(self)

    newSol = self.copy()
    newSol.aplyCorr(.8,corr)
    P1m,_,_ = newSol.calcP()

    newSol = self.copy()
    newSol.aplyCorr(1.0,corr)
    P1,_,_ = newSol.calcP()

    newSol = self.copy()
    newSol.aplyCorr(1.2,corr)
    P1M,_,_ = newSol.calcP()
            
    if P1 >= P1m or P1 >= P0:
        print("alfa=1.0 is too much.")
        # alfa = 1.0 is too much. Reduce alfa.
        nP = P1; alfa=1.0
        cont = 0; keepSearch = (nP>P0)
        while keepSearch and alfa > 1.0e-15:
            cont += 1
            P = nP
            alfa *= .8
            newSol = self.copy()
            newSol.aplyCorr(alfa,corr)
            nP,_,_ = newSol.calcP()
            if nP < P0:
                keepSearch = (nP>P)#((nP-P)/P < -.01)#((nP-P)/P < -.05)
        alfa /= 0.8
    else:
        # no "overdrive!"
        #return 1.0
    
        if P1 <= P1M:
            # alfa = 1.0 is likely to be best value. 
            # Better not to waste time and return 1.0 
            return 1.0
        else:
            # There is still a descending gradient here. Increase alfa!
            nP = P1M
            cont = 0; keepSearch = True#(nPint>Pint1M)
            alfa = 1.2
            while keepSearch:
                cont += 1
                P = nP
                alfa *= 1.2
                newSol = self.copy()
                newSol.aplyCorr(alfa,corr)
                nP,_,_ = newSol.calcP()
                print("\n alfa =",alfa,", P = {:.4E}".format(nP),\
                      " (P0 = {:.4E})".format(P0))
                keepSearch = nP<P #( nP<P and alfa < 1.5)#2.0)#
                #if nPint < Pint0:
            alfa /= 1.2
    return alfa


def rest(self):
     
    print("\nIn rest, P0 = {:.4E}.".format(self.P))

    A,B,C,lam,mu = self.LMPBVP(rho=0.0)
    
    # TODO: must rethink the correction plot, 
    # current design makes no sense because C=0 is a possibility
#    if self.dbugOptRest['plotCorr']:
#        solCorr = self.copy()
#        solCorr.x = A; solCorr.u = B; solCorr.pi = C
#        solCorr.printPars()
#        solCorr.plotSol(opt={'mode':'var'})
#        #optPlot['mode'] = 'proposed (states: lambda)'
#        #plotSol(sizes,t,lam,B,C,constants,restrictions,optPlot)
    
    corr = {'x':A,
            'u':B,
            'pi':C}
    #print("Calculating step...")
    
    alfa = self.calcStepRest(corr)
    self.aplyCorr(alfa,corr)
    self.updtHistP(alfa)
    print("Leaving rest with alfa =",alfa)
    
    if self.dbugOptRest['pausRest']:
        self.plotSol()
        input('Rest in debug mode. Press any key to continue...')