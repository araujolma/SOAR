#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:36:59 2017

@author: levi
"""
import numpy
from utils import simp
import matplotlib.pyplot as plt

def calcP(self,mustPlotPint=False):
    N, s = self.N, self.s

    psi = self.calcPsi()
    #print("psi = "+str(psi))
    func = self.calcErr()

    vetP = numpy.empty((N,s))
    vetIP = numpy.empty((N,s))


    for arc in range(s):
        for t in range(N):
            vetP[t,arc] = func[t,:,arc].dot(func[t,:,arc].transpose())
    coefList = simp([],N,onlyCoef=True)

    for arc in range(s):
        vetIP[0,arc] = coefList[0] * vetP[0,arc]
        for t in range(1,N):
            vetIP[t,arc] = vetIP[t-1,arc] + coefList[t] * vetP[t,arc]
    #

    #vetIP *= self.dt # THIS IS WRONG!! REMOVE IT!!

    # TEST FOR VIOLATIONS!
    PiCondVio = False
    piLowLim = self.restrictions['pi_min']
    piHighLim = self.restrictions['pi_max']
    for i in range(self.s):
        # violated here in lower limit condition
        if piLowLim[i] is not None and self.pi[i] < piLowLim[i]:
            PiCondVio = True; break # already violated, no need to continue
        # violated here in upper limit condition
        if piHighLim[i] is not None and self.pi[i] > piHighLim[i]:
            PiCondVio = True; break # already violated, no need to continue
    #
    if PiCondVio:
        vetIP *= 1e300


#    for arc in range(s):
#        vetIP[0,arc] = (17.0/48.0) * vetP[0,arc]
#        vetIP[1,arc] = vetIP[0,arc] + (59.0/48.0) * vetP[1,arc]
#        vetIP[2,arc] = vetIP[1,arc] + (43.0/48.0) * vetP[2,arc]
#        vetIP[3,arc] = vetIP[2,arc] + (49.0/48.0) * vetP[3,arc]
#        for t in range(4,N-4):
#            vetIP[t] = vetIP[t-1,arc] + vetP[t,arc]
#        vetIP[N-4,arc] = vetIP[N-5,arc] + (49.0/48.0) * vetP[N-4,arc]
#        vetIP[N-3,arc] = vetIP[N-4,arc] + (43.0/48.0) * vetP[N-3,arc]
#        vetIP[N-2,arc] = vetIP[N-3,arc] + (59.0/48.0) * vetP[N-2,arc]
#        vetIP[N-1,arc] = vetIP[N-2,arc] + (17.0/48.0) * vetP[N-1,arc]

    # for arc in range(s):
    #     vetIP[0,arc] = coefList[0] * vetP[0,arc]
    #     for t in range(1,N):
    #         vetIP[t] = vetIP[t-1,arc] + coefList[t] * vetP[t,arc]

    #vetIP *= dt

    # Look for some debug plot
#    someDbugPlot = False
#    for key in self.dbugOptRest.keys():
#        if ('plot' in key) or ('Plot' in key):
#            if self.dbugOptRest[key]:
#                someDbugPlot = True
#                break
#    if someDbugPlot:
#        self.log.printL("\nDebug plots for this calcP run:")
#
#        indMaxP = numpy.argmax(vetP, axis=0)
#        self.log.printL(indMaxP)
#        for arc in range(s):
#            self.log.printL("\nArc =",arc,"\n")
#            ind1 = numpy.array([indMaxP[arc]-20,0]).max()
#            ind2 = numpy.array([indMaxP[arc]+20,N]).min()
#
#            if self.dbugOptRest['plotP_int']:
#                plt.plot(self.t,vetP[:,arc])
#                plt.grid(True)
#                plt.title("Integrand of P")
#                plt.show()
#
#            if self.dbugOptRest['plotIntP_int']:
#                plt.plot(self.t,vetIP[:,arc])
#                plt.grid(True)
#                plt.title("Partially integrated P")
#                plt.show()
#
#            #for zoomed version:
#            if self.dbugOptRest['plotP_intZoom']:
#                plt.plot(self.t[ind1:ind2],vetP[ind1:ind2,arc],'o')
#                plt.grid(True)
#                plt.title("Integrand of P (zoom)")
#                plt.show()
#
#            if self.dbugOptRest['plotSolMaxP']:
#                self.log.printL("rest_sgra: plotSol @ MaxP region: not implemented yet!")
#                #self.log.printL("\nSolution on the region of MaxP:")
#                #self.plotSol(intv=numpy.arange(ind1,ind2,1,dtype='int'))

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

    Pint = vetIP[N-1,:].sum()
    Ppsi = psi.transpose().dot(psi)
    P = Ppsi + Pint
    strPs = "P = {:.6E}".format(P)+", Pint = {:.6E}".format(Pint)+\
          ", Ppsi = {:.6E}.".format(Ppsi)
    self.log.printL(strPs)
    self.P = P

    if mustPlotPint:
        #plt.subplots_adjust(wspace=1.0,hspace=1.0)
        #plt.subplots_adjust(hspace=.5)
        plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
        Np = self.n + 2

        for arc in range(1,s):
            vetIP[:,arc] += vetIP[-1,arc-1]
        plt.subplot2grid((Np,1),(0,0))
        self.plotCat(vetIP,piIsTime=False)
        plt.grid(True)
        plt.title("P_int: Accumulated value, integrand and error components\n"+\
                  "P = {:.4E}, ".format(P)+\
                  "P_int = {:.4E}, ".format(Pint)+\
                  "P_psi = {:.4E}".format(Ppsi)+\
                  "\n(event #" + str(int((self.EvntIndx+1)/2)) + \
                  ", rest. iter. #"+str(self.NIterRest+1)+")\n")
        plt.ylabel('Accum.')

        plt.subplot2grid((Np,1),(1,0))
        self.plotCat(vetP,piIsTime=False,color='k')
        plt.grid(True)
        plt.ylabel('Integrand')

        colorList = ['b','g','r','m']
        for i in range(self.n):
            plt.subplot2grid((Np,1),(i+2,0))
            self.plotCat(func[:,i,:],piIsTime=False,color=colorList[i%4])
            plt.grid(True)
            plt.ylabel('State '+str(i))

        self.savefig(keyName='Pint',fullName='integrand of P')


    return P,Pint,Ppsi

def calcStepRest(self,corr):
    self.log.printL("\nIn calcStepRest.\n")

    newSol = self.copy()
    newSol.aplyCorr(1.0,corr)
    P1,_,_ = newSol.calcP()

    # if applying alfa = 1.0 already meets the tolerance requirements,
    # why waste time decreasing alfa?
    if P1 < self.tol['P']:
        return 1.0

    #P0,_,_ = calcP(self)
    P0 = self.P
    dalfa = 0.9
    newSol = self.copy()
    newSol.aplyCorr(dalfa,corr)
    P1m,_,_ = newSol.calcP()

    if P1 >= P1m or P1 >= P0:
        self.log.printL("\nalfa = 1.0 is too much.")
        # alfa = 1.0 is too much. Reduce alfa.
        nP = P1m; alfa = dalfa#1.0
        cont = 0; keepSearch = (nP>P0)
        while keepSearch and alfa > 1.0e-15:
            cont += 1
            P = nP
            alfa *= dalfa
            newSol = self.copy()
            newSol.aplyCorr(alfa,corr)
            nP,_,_ = newSol.calcP()
            if P < P0:
                keepSearch = (nP>P)#((nP-P)/P < -.01)#((nP-P)/P < -.05)
        if cont>0:
            alfa /= dalfa
    else:
        # no "overdrive!"
        self.log.printL("Leaving rest with alfa = 1.")
        self.log.printL("Delta pi = "+str(corr['pi']))
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
                self.log.printL("\n alfa =",alfa,", P = {:.4E}".format(nP),\
                      " (P0 = {:.4E})".format(P0))
                keepSearch = nP<P #( nP<P and alfa < 1.5)#2.0)#
                #if nPint < Pint0:
            alfa /= 1.2
        #
    #

    self.log.printL("Leaving rest with alfa = "+str(alfa))
    self.log.printL("Delta pi = "+str(alfa*corr['pi']))
    return alfa


def rest(self,parallelOpt={}):

    self.log.printL("\nIn rest, P0 = {:.4E}.".format(self.P))

    isParallel = parallelOpt.get('restLMPBVP',False)
    corr,_,_ = self.LMPBVP(rho=0.0,isParallel=isParallel)

    #A, B, C = corr['x'], corr['u'], corr['pi']
    #self.plotSol(opt={'mode':'var','x':A,'u':B,'pi':C})
#    input("rest_sgra: Olha lá a correção!")

    alfa = self.calcStepRest(corr)
#    self.plotSol(opt={'mode':'var','x':alfa*A,'u':alfa*B,'pi':alfa*C})
#    input("rest_sgra: Olha lá a correção ponderada!")
    self.aplyCorr(alfa,corr)

    self.updtEvntList('rest')
    self.updtHistP()#mustPlotPint=True)
    self.updtHistRest(alfa)

    self.log.printL("Leaving rest with alfa = "+str(alfa))
    if self.dbugOptRest['pausRest']:
        self.plotSol()
        input('Rest in debug mode. Press any key to continue...')

