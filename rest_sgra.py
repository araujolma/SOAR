#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:36:59 2017

@author: levi
"""
import numpy
from utils import ddt
import matplotlib.pyplot as plt

def calcP(self,mustPlotPint=False):
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
    
    #P *= dt

    vetIP *= dt
    
    # Look for some debug plot
    someDbugPlot = False
    for key in self.dbugOptRest.keys():
        if ('plot' in key) or ('Plot' in key):
            if self.dbugOptRest[key]:
                someDbugPlot = True
                break
    if someDbugPlot:
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
    self.P = P
    
    if mustPlotPint:
#        plt.subplots_adjust(wspace=1.0,hspace=1.0)
        plt.subplots_adjust(hspace=.5)
        plt.subplot2grid((2,1),(0,0))
        self.plotCat(vetP,piIsTime=False)
        plt.grid(True)
        plt.title("P_int: Integrand and accumulated value "+\
                  "(rest. iter. #"+str(self.NIterRest+1)+")\n"+\
                   "P = {:.4E}, ".format(P)+\
                   "P_int = {:.4E}, ".format(Pint)+\
                   "P_psi = {:.4E}".format(Ppsi))
        plt.ylabel('Integrand')
        
        for arc in range(1,s):
            vetIP[:,arc] += vetIP[-1,arc-1]
        plt.subplot2grid((2,1),(1,0))
        self.plotCat(vetIP,piIsTime=False)
        plt.grid(True)
        plt.ylabel('Accum.')
        
        self.savefig(keyName='Pint',fullName='integrand of P')
    
    
    return P,Pint,Ppsi    

def calcStepRest(self,corr):
    print("\nIn calcStepRest.\n")
    
    newSol = self.copy()
    newSol.aplyCorr(1.0,corr)
    P1,_,_ = newSol.calcP()
    
    # if applying alfa = 1.0 already meets the tolerance requirements, 
    # why waste time decreasing alfa?
    if P1 < self.tol['P']:
        return 1.0
    
    #P0,_,_ = calcP(self)
    P0 = self.P

    newSol = self.copy()
    newSol.aplyCorr(.8,corr)
    P1m,_,_ = newSol.calcP()

            
    if P1 >= P1m or P1 >= P0:
        print("\nalfa = 1.0 is too much.")
        # alfa = 1.0 is too much. Reduce alfa.
        nP = P1m; alfa=.8#1.0
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
        if cont>0:
            alfa /= 0.8
    else:
        # no "overdrive!"
        return 1.0
    
    
        # with "overdrive":
        newSol = self.copy()
        newSol.aplyCorr(1.2,corr)
        P1M,_,_ = newSol.calcP()

    
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


def rest(self,parallelOpt={}):
     
    print("\nIn rest, P0 = {:.4E}.".format(self.P))

    isParallel = parallelOpt.get('restLMPBVP',False)
    A,B,C,_,_ = self.LMPBVP(rho=0.0,isParallel=isParallel)
    
    
    corr = {'x':A,
            'u':B,
            'pi':C}
    #print("Calculating step...")
    
    alfa = self.calcStepRest(corr)
    self.aplyCorr(alfa,corr)
    self.updtHistP(alfa,mustPlotPint=True)
    
    # update Gradient-Restoration event list
    self.GREvIndx += 1
    self.GREvList[self.GREvIndx] = False
#    print("\nUpdating GREvList.")
#    print("Writing False in position",self.GREvIndx)
#    print("GREvList =",self.GREvList[:(self.GREvIndx+1)])
    
    print("Leaving rest with alfa =",alfa)

    
    if self.dbugOptRest['pausRest']:
        self.plotSol()
        input('Rest in debug mode. Press any key to continue...')