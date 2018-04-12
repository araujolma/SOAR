#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 18:24:24 2018

@author: levi
"""

import numpy
import matplotlib.pyplot as plt

def updtEvntList(self,evnt):
    """ Update the event list.
    This happens at every restoration, grad-reject or grad-accept.
    Each event is represented by two bits:
        00: restoration step
        01: the first gradient step (begin of the first GR cycle)
        10: gradient step rejected, trying again
        11: gradient step accepted, starting new GR cycle"""

    self.EvntIndx += 2
    if evnt == 'rest': #00
        # No alteration needed
        pass
    elif evnt == 'init': #01
        self.EvntList[self.EvntIndx] = True
    else:
        self.EvntList[self.EvntIndx-1] = True
        if evnt == 'gradOK': #11
            self.EvntList[self.EvntIndx] = True
        else: #10
            # Rejected gradient. This entry stays false
            pass

def updtGRrate(self):
        pass
#        # find last gradient step index in event list
#        LastGradIndx = self.GREvIndx - 1
#
#        while self.GREvList[LastGradIndx] == False and LastGradIndx > 0:
#            LastGradIndx -= 1
#
#        # number of restorations after last gradient step
#        if LastGradIndx == 0:
#            nRest = self.GREvIndx-1
#        else:
#            nRest = self.GREvIndx - 1 - LastGradIndx
#
#        self.histGRrate[self.NIterGrad] = nRest

def updtHistP(self,mustPlotPint=False):
    """Update the P histories.
       This happens at every restoration, grad-reject or grad-accept."""

    P,Pint,Ppsi = self.calcP(mustPlotPint=mustPlotPint)
    self.P = P
    thisIndx = int((self.EvntIndx+1)/2)
    self.histP[thisIndx] = P
    self.histPint[thisIndx] = Pint
    self.histPpsi[thisIndx] = Ppsi

def updtHistRest(self,alfa):
    """Update the restoration histories.
        Currently only the restoration step is used."""
    NIterRest = self.NIterRest+1

    self.histStepRest[NIterRest] = alfa
    self.NIterRest = NIterRest

def showHistP(self):
    """ Show the history of P, Pint and Ppsi."""

    NEvnt = int((self.EvntIndx+1)/2) + 1
    print("NEvnt =",NEvnt)
    Evnt = numpy.arange(0,NEvnt,1)
    print("Evnt =",Evnt)

    if self.histP[Evnt].any() > 0:
        plt.semilogy(Evnt,self.histP[Evnt],'b',label='P')

    if self.histPint[Evnt].any() > 0:
        plt.semilogy(Evnt,self.histPint[Evnt],'k',label='P_int')

    if self.histPpsi[Evnt].any() > 0:
        plt.semilogy(Evnt,self.histPpsi[Evnt],'r',label='P_psi')

    GradAcptIndxList = []
    GradRjecIndxList = []
    GradInitIndx = []
    for i in range(1,NEvnt):
        if self.EvntList[2*i-1] == 0:
            if self.EvntList[2*i] == 1:
                GradInitIndx = [i]
        else:
            if self.EvntList[2*i] == 1:
                GradAcptIndxList.append(i)
            else:
                GradRjecIndxList.append(i)
    if len(GradInitIndx)>0:
        plt.semilogy(Evnt[GradInitIndx],self.histP[GradInitIndx],'*',\
                 label='Init')
    if len(GradAcptIndxList)>0:
        plt.semilogy(Evnt[GradAcptIndxList], \
                     self.histP[Evnt[GradAcptIndxList]], \
                     'ob', label='GradAccept')
    if len(GradRjecIndxList)>0:
        plt.semilogy(Evnt[GradRjecIndxList], \
                     self.histP[Evnt[GradRjecIndxList]], \
                     'xb', label='GradReject')

    plt.plot(Evnt,self.tol['P']+0.0*Evnt,'-.b',label='tolP')
    plt.title("Convergence report on P")
    plt.grid(True)
    plt.xlabel("Events")
    plt.ylabel("P values")
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))

    self.savefig(keyName='histP',fullName='P')

    print("HistP: ",self.histP[Evnt])
    input("showHistP: aguardando...")

def updtHistGrad(self,alfa,mustPlotQs=False):
    """ Updates the gradient histories.
        This includes the Qs, Is, Js and gradStep. """

    NIterGrad = self.NIterGrad+1

    Q,Qx,Qu,Qp,Qt = self.calcQ(mustPlotQs=mustPlotQs)
    self.Q = Q
    self.histQ[NIterGrad] = Q
    self.histQx[NIterGrad] = Qx
    self.histQu[NIterGrad] = Qu
    self.histQp[NIterGrad] = Qp
    self.histQt[NIterGrad] = Qt

    self.histStepGrad[NIterGrad] = alfa

    J, J_Lint, J_Lpsi, I, Iorig, Ipf = self.calcJ()
    self.I = I
    self.J = J
    self.histJ[NIterGrad] = J
    self.histJLint[NIterGrad] = J_Lint
    self.histJLpsi[NIterGrad] = J_Lpsi
    self.histI[NIterGrad] = I
    self.histIorig[NIterGrad] = Iorig
    self.histIpf[NIterGrad] = Ipf

    self.NIterGrad = NIterGrad


def showHistQ(self):
    IterGrad = numpy.arange(1,self.NIterGrad+1,1)

    if self.histQ[IterGrad].any() > 0:
        plt.semilogy(IterGrad,self.histQ[IterGrad],'b',label='Q')

    if self.histQx[IterGrad].any() > 0:
        plt.semilogy(IterGrad,self.histQx[IterGrad],'k',label='Qx')

    if self.histQu[IterGrad].any() > 0:
        plt.semilogy(IterGrad,self.histQu[IterGrad],'r',label='Qu')

    if self.histQp[IterGrad].any() > 0:
        plt.semilogy(IterGrad,self.histQp[IterGrad],'g',label='Qp')

    if self.histQt[IterGrad].any() > 0:
        plt.semilogy(IterGrad,self.histQt[IterGrad],'y',label='Qt')

    plt.plot(IterGrad,self.tol['Q']+0.0*IterGrad,'-.b',label='tolQ')
    plt.title("Convergence report on Q")
    plt.grid(True)
    plt.xlabel("Grad iterations")
    plt.ylabel("Q values")
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))

    self.savefig(keyName='histQ',fullName='Q convergence history')

def showHistI(self):
    IterGrad = numpy.arange(1,self.NIterGrad+1,1)

    plt.title("Convergence report on I")
    plt.semilogy(IterGrad,self.histI[IterGrad],label='I')
    plt.semilogy(IterGrad,self.histIorig[IterGrad],label='Iorig')
    plt.semilogy(IterGrad,self.histIpf[IterGrad],label='Ipf')
    plt.grid(True)
    plt.xlabel("Grad iterations")
    plt.ylabel("I values")
    plt.legend()

    self.savefig(keyName='histI',fullName='I convergence history')

def showHistGradStep(self):
    IterGrad = numpy.arange(1,self.NIterGrad+1,1)

    plt.title("Gradient step history")
    plt.semilogy(IterGrad,self.histStepGrad[IterGrad])
    plt.grid(True)
    plt.xlabel("Grad iterations")
    plt.ylabel("Step values")

    self.savefig(keyName='histGradStep',fullName='GradStep convergence history')

def showHistGRrate(self):
    pass
#    IterGrad = numpy.arange(1,self.NIterGrad+1,1)
#
#    if self.histGRrate[IterGrad].any() > 0:
#        plt.title("Gradient-restoration rate history")
#        plt.semilogy(IterGrad,self.histGRrate[IterGrad])
#        plt.grid(True)
#        plt.xlabel("Grad iterations")
#        plt.ylabel("Step values")
#
#        self.savefig(keyName='histGRrate',fullName='Grad-Rest rate history')
#    else:
#        self.log.printL("showHistGRrate: No positive values. Skipping...")

