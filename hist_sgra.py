#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 18:24:24 2018

@author: levi
"""

import numpy
import matplotlib.pyplot as plt

def declHist(self):
    """A method for declaring all the histories. """

    MaxIterRest = 100000
    MaxIterGrad = 10000
    MaxEvnt = 2*MaxIterRest + MaxIterGrad
    # Gradient-Restoration EVent List
    # (00-rest, 10-denied grad, 11-accepted grad)
    self.MaxEvnt = MaxEvnt
    self.EvntList = numpy.zeros(MaxEvnt,dtype='bool')
    self.EvntIndx = 0
    self.ContRest = 0
    #self.GREvIndx = -1

    # Basic maximum number of iterations for grad/rest.
    # May be overriden in the problem definition

    self.MaxIterRest = MaxIterRest
    self.NIterRest = 0
    self.histStepRest = numpy.zeros(MaxIterRest)
    MaxHistP = int((MaxEvnt+1)/2)
    self.histP = numpy.zeros(MaxHistP)
    self.histPint = numpy.zeros(MaxHistP)
    self.histPpsi = numpy.zeros(MaxHistP)

    self.MaxIterGrad = MaxIterGrad
    self.NIterGrad = 0

    self.histStepGrad = numpy.zeros(MaxIterGrad)
    self.histQ = numpy.zeros(MaxIterGrad)
    self.histQx = numpy.zeros(MaxIterGrad)
    self.histQu = numpy.zeros(MaxIterGrad)
    self.histQp = numpy.zeros(MaxIterGrad)
    self.histQt = numpy.zeros(MaxIterGrad)

    self.histI = numpy.zeros(MaxIterGrad)
    self.histIorig = numpy.zeros(MaxIterGrad)
    self.histIpf = numpy.zeros(MaxIterGrad)
    self.histJ = numpy.zeros(MaxIterGrad)
    self.histJLint = numpy.zeros(MaxIterGrad)
    self.histJLpsi = numpy.zeros(MaxIterGrad)

    self.histGRrate = numpy.zeros(MaxIterGrad)
    self.histGaRrate = numpy.zeros(MaxIterGrad)
    self.histGaGrRate = numpy.zeros(MaxIterGrad)

def updtEvntList(self,evnt):
    """ Update the event list.
    This happens at every restoration, grad-reject or grad-accept.
    Each event is represented by two bits:
        00: restoration step
        01: the first gradient step (begin of the first GR cycle)
        10: gradient step rejected, trying again
        11: gradient step accepted, starting new GR cycle"""

    self.EvntIndx += 2
    self.log.printL("\nhist_sgra: new event ("+str(int(self.EvntIndx/2)) + \
                    ") -- " + evnt)
    if evnt == 'rest': #00
        # No alteration needed in the bits
        self.ContRest += 1
    elif evnt == 'init': #01
        # second bit becomes true
        self.EvntList[self.EvntIndx] = True
        self.histGRrate[1] = self.ContRest
        self.ContRest = 0
    else:
        self.EvntList[self.EvntIndx-1] = True
        if evnt == 'gradOK': #11
            # both bits become true
            self.EvntList[self.EvntIndx] = True
        else: #10
            # Rejected gradient. Second bit stays false
            pass
        self.histGRrate[self.NIterGrad+1] = self.ContRest
        self.ContRest = 0

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
    self.NIterRest = NIterRest
    self.histStepRest[NIterRest] = alfa


def showHistP(self):
    """ Show the history of P, Pint and Ppsi."""

    # Total number of events
    NEvnt = int((self.EvntIndx+1)/2) + 1
    # array for plotting (x-axis)
    Evnt = numpy.arange(0,NEvnt,1)

    # the actual plots of P, Pint, Ppsi
    if self.histP[Evnt].any() > 0:
        plt.semilogy(Evnt,self.histP[Evnt],'b',label='P')

    if self.histPint[Evnt].any() > 0:
        plt.semilogy(Evnt,self.histPint[Evnt],'k',label='P_int')

    if self.histPpsi[Evnt].any() > 0:
        plt.semilogy(Evnt,self.histPpsi[Evnt],'r',label='P_psi')

    # Assemble lists for the indexes (relative to the event list)
    # where there is a gradient (init, accept, reject)

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

    # with the lists assembled, mark the points where there is a gradient
    # (init, accept, reject)

    if len(GradInitIndx)>0:
        plt.semilogy(GradInitIndx,self.histP[GradInitIndx],'*',\
                 label='GradInit')
    if len(GradAcptIndxList)>0:
        plt.semilogy(GradAcptIndxList, \
                     self.histP[GradAcptIndxList], \
                     'ob', label='GradAccept')
    if len(GradRjecIndxList)>0:
        plt.semilogy(GradRjecIndxList, \
                     self.histP[GradRjecIndxList], \
                     'xb', label='GradReject')

    # Finally, plot the P tolerance
    plt.plot(Evnt,self.tol['P']+0.0*Evnt,'-.b',label='tolP')
    plt.title("Convergence report on P")
    plt.grid(True)
    plt.xlabel("Events")
    plt.ylabel("P values")
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))

    self.savefig(keyName='histP',fullName='P convergence history')


#def updtGradCont(self,alfa):
#    """ Updates the gradient counter, as well as gradStep."""
#    self.log.printL("\nhist_sgra: Updating grad counters.")
#    NIterGrad = self.NIterGrad+1
#    self.NIterGrad = NIterGrad
#
#    self.histStepGrad[NIterGrad] = alfa
#
#    self.ContRest = 0
#
#def updtHistGrad(self,mustPlotQs=False):
#    """ Updates the other gradient histories.
#        This includes the Qs, Is and Js. """
#
#    self.log.printL("\nhist_sgra: Updating Q,I,J histories.")
#    NIterGrad = self.NIterGrad
#
#    Q,Qx,Qu,Qp,Qt = self.calcQ(mustPlotQs=mustPlotQs)
#    self.Q = Q
#    self.histQ[NIterGrad] = Q
#    self.histQx[NIterGrad] = Qx
#    self.histQu[NIterGrad] = Qu
#    self.histQp[NIterGrad] = Qp
#    self.histQt[NIterGrad] = Qt
#
#    J, J_Lint, J_Lpsi, I, Iorig, Ipf = self.calcJ()
#    self.I = I
#    self.J = J
#    self.histJ[NIterGrad] = J
#    self.histJLint[NIterGrad] = J_Lint
#    self.histJLpsi[NIterGrad] = J_Lpsi
#    self.histI[NIterGrad] = I
#    self.histIorig[NIterGrad] = Iorig
#    self.histIpf[NIterGrad] = Ipf

def updtHistGrad(self,alfa,mustPlotQs=False):
    """ Updates the gradient histories.
        This includes the Qs, Is, Js and gradStep. """

    NIterGrad = self.NIterGrad+1
    self.log.printL("\nhist_sgra: Updating histGrad.")

    Q,Qx,Qu,Qp,Qt = self.calcQ(mustPlotQs=mustPlotQs)
    self.Q = Q
    self.histQ[NIterGrad] = Q
    self.histQx[NIterGrad] = Qx
    self.histQu[NIterGrad] = Qu
    self.histQp[NIterGrad] = Qp
    self.histQt[NIterGrad] = Qt

    J, J_Lint, J_Lpsi, I, Iorig, Ipf = self.calcJ()
    self.I = I
    self.J = J
    self.histJ[NIterGrad] = J
    self.histJLint[NIterGrad] = J_Lint
    self.histJLpsi[NIterGrad] = J_Lpsi
    self.histI[NIterGrad] = I
    self.histIorig[NIterGrad] = Iorig
    self.histIpf[NIterGrad] = Ipf

    self.histStepGrad[NIterGrad] = alfa
    self.NIterGrad = NIterGrad
    self.ContRest = 0


def showHistQ(self):
    """ Show the Q, Qx, Qu, Qp, Qt histories."""

    # Assemble the plotting array (x-axis)
    IterGrad = numpy.arange(1,self.NIterGrad+1,1)

    # Perform the plots
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

    # Assemble lists for the indexes (relative to the gradient event list)
    # where there is each gradient (init, accept, reject)

    GradAcptIndxList = []
    GradRjecIndxList = []
    GradInitIndx = []

    iGrad = 0
    NEvnt = int((self.EvntIndx+1)/2) + 1
    for i in range(1,NEvnt):
        if self.EvntList[2*i-1] == 0:
            if self.EvntList[2*i] == 1:
                iGrad += 1
                GradInitIndx = [iGrad]
        else:
            if self.EvntList[2*i] == 1:
                iGrad += 1
                GradAcptIndxList.append(iGrad)
            else:
                iGrad += 1
                GradRjecIndxList.append(iGrad)

    if len(GradInitIndx)>0:
        plt.semilogy(GradInitIndx,self.histQ[GradInitIndx],'*',\
                 label='GradInit')
    if len(GradAcptIndxList)>0:
        plt.semilogy(GradAcptIndxList, \
                     self.histQ[GradAcptIndxList], \
                     'ob', label='GradAccept')
    if len(GradRjecIndxList)>0:
        plt.semilogy(GradRjecIndxList, \
                     self.histQ[GradRjecIndxList], \
                     'xb', label='GradReject')


    plt.plot(IterGrad,self.tol['Q']+0.0*IterGrad,'-.b',label='tolQ')
    plt.title("Convergence report on Q")
    plt.grid(True)
    plt.xlabel("Grad iterations")
    plt.ylabel("Q values")
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))

    self.savefig(keyName='histQ',fullName='Q convergence history')

def showHistI(self):
    IterGrad = numpy.arange(0,self.NIterGrad+1,1)

    plt.title("Convergence report on I")
    plt.plot(IterGrad,self.histI[IterGrad],label='I')
    plt.plot(IterGrad,self.histIorig[IterGrad],label='Iorig')
    plt.plot(IterGrad,self.histIpf[IterGrad],label='Ipf')
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

    self.savefig(keyName='histGradStep',fullName='GradStep history')
#    self.log.printL("showHistGradStep: these are the gradSteps: " + \
#          str(self.histStepGrad[IterGrad]))

def showHistGRrate(self):
    """Show the history of the GR rate (rests per grad).
        It must begin at 1, there is no such thing as "initial GR rate". """

    IterGrad = numpy.arange(1,self.NIterGrad+1,1)

    #if self.histGRrate[IterGrad].any() > 0:
    plt.title("Gradient-restoration rate history")
    plt.plot(IterGrad,self.histGRrate[IterGrad])
    plt.grid(True)
    plt.xlabel("Grad iterations")
    plt.ylabel("Restorations per grad")

    self.savefig(keyName='histGRrate',fullName='Grad-Rest rate history')
    #else:
    #    self.log.printL("showHistGRrate: No positive values. Skipping...")

def copyHistFrom(self,sol_from):
    self.EvntList = sol_from.EvntList
    self.EvntIndx = sol_from.EvntIndx
    self.ContRest = sol_from.ContRest

    self.histP = sol_from.histP
    self.histPint = sol_from.histPint
    self.histPpsi = sol_from.histPpsi
    self.histStepRest = sol_from.histStepRest
    self.NIterRest = sol_from.NIterRest

    self.histQ = sol_from.histQ
    self.histQx = sol_from.histQx
    self.histQu = sol_from.histQu
    self.histQp = sol_from.histQp
    self.histQt = sol_from.histQt

    self.histI = sol_from.histI
    self.histIorig = sol_from.histIorig
    self.histIpf = sol_from.histIpf
    self.histJ = sol_from.histJ
    self.histJLint = sol_from.histJLint
    self.histJLpsi = sol_from.histJLpsi

    self.histGRrate = sol_from.histGRrate
    self.histStepGrad = sol_from.histStepGrad
    self.NIterGrad = sol_from.NIterGrad
