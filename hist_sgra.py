#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 18:24:24 2018

@author: levi
"""

import numpy
import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker

def declHist(self,MaxIterGrad=10000):
    """A method for declaring all the histories. """

    MaxIterRest = 100000
    MaxEvnt = 2*MaxIterRest + MaxIterGrad
    # Gradient-Restoration EVent List
    # (00-rest, 10-denied grad, 11-accepted grad)
    self.MaxEvnt = MaxEvnt
    self.EvntList = numpy.zeros(MaxEvnt,dtype=numpy.bool_)#'bool')
    self.EvntIndx = 0
    self.ContRest = 0
    #self.GREvIndx = -1

    # Basic maximum number of iterations for grad/rest.
    # May be overridden in the problem definition

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
    self.histGSSstopMotv = numpy.zeros(MaxIterGrad,dtype=numpy.int8)
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
    self.histObjEval = numpy.zeros(MaxIterGrad,dtype=numpy.int16)

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
        plt.semilogy(GradInitIndx,self.histP[GradInitIndx],'*',
                 label='GradInit')
    if len(GradAcptIndxList)>0:
        plt.semilogy(GradAcptIndxList,
                     self.histP[GradAcptIndxList],
                     'ob', label='GradAccept')
    if len(GradRjecIndxList)>0:
        plt.semilogy(GradRjecIndxList,
                     self.histP[GradRjecIndxList],
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

def updtHistGrad(self,alfa,GSSstopMotv,mustPlotQs=False):
    """ Updates the gradient histories.
        This includes the Qs, Is, Js and gradStep. """

    NIterGrad = self.NIterGrad+1
    self.log.printL("\nhist_sgra: Updating histGrad.")

    #Q,Qx,Qu,Qp,Qt = self.calcQ(mustPlotQs=mustPlotQs)
    self.histQ[NIterGrad] = self.Q
    self.histQx[NIterGrad] = self.Qx; self.histQu[NIterGrad] = self.Qu
    self.histQp[NIterGrad] = self.Qp; self.histQt[NIterGrad] = self.Qt

    #J, J_Lint, J_Lpsi, I, Iorig, Ipf = self.calcJ()

    self.histJ[NIterGrad] = self.J
    self.histJLint[NIterGrad] = self.J_Lint
    self.histJLpsi[NIterGrad] = self.J_Lpsi
    self.histI[NIterGrad] = self.I
    self.histIorig[NIterGrad] = self.Iorig
    self.histIpf[NIterGrad] = self.Ipf

    self.histStepGrad[NIterGrad] = alfa
    # "Stop motive" codes:  0 - step rejected
    #                       1 - local min found
    #                       2 - step limit hit
    #                       3 - too many evals
    self.histGSSstopMotv[NIterGrad] = GSSstopMotv
    self.NIterGrad = NIterGrad
    self.ContRest = 0

def showHistQ(self,tolZoom=True,nptsMark=40):
    """ Show the Q, Qx, Qu, Qp, Qt histories.
    There is a huge part of this function dedicated to marking the events (grad init,
    grad accept, grad reject) on top of the Q curve. The basic idea there is to try to
    only mark events (points) until the limit 'nptsMark' supplied by the user is met.
    If there are too many events to mark, the code omits "redundant" events, that is,
    events equal to the events in between. In this way, an acceptance that occurred
    exactly between two acceptances or a rejection that occurred between rejections
    may not be marked. """

    # Assemble the plotting array (x-axis)
    IterGrad = numpy.arange(1,self.NIterGrad+1,1)

    # Perform the plots
    if self.histQ[IterGrad].any() > 0:
        # only the first point, to ensure the 'Q' label is the first
        plt.semilogy(IterGrad[0],self.histQ[IterGrad[0]],'b',label='Q')

    if self.histQx[IterGrad].any() > 0:
        plt.semilogy(IterGrad-1,self.histQx[IterGrad],'k',label='Qx')

    if self.histQu[IterGrad].any() > 0:
        plt.semilogy(IterGrad-1,self.histQu[IterGrad],'r',label='Qu')

    if self.histQp[IterGrad].any() > 0:
        plt.semilogy(IterGrad-1,self.histQp[IterGrad],'g',label='Qp')

    if self.histQt[IterGrad].any() > 0:
        plt.semilogy(IterGrad-1,self.histQt[IterGrad],'y',label='Qt')

    # Plot the tolerance line
    plt.plot(IterGrad-1, self.tol['Q'] + 0.0 * IterGrad, '-.b', label='tolQ')

    if self.histQ[IterGrad].any() > 0:
        # plot it again, now to ensure it stays on top of the other plots!
        plt.semilogy(IterGrad-1, self.histQ[IterGrad], 'b')

    # Assemble lists for the indexes (relative to the gradient event list)
    # where there is each gradient (init, accept, reject)

    GradAcptIndxList, GradRjecIndxList, GradInitIndx = [], [], []
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

    # Mark the first point(s?)
    if len(GradInitIndx)>0:
        plt.semilogy(numpy.array(GradInitIndx)-1, self.histQ[GradInitIndx],
                     '*C0', label='GradInit')

    # get number of acceptance and rejection iterations, figure out what events to mark
    nGA, nGR = len(GradAcptIndxList), len(GradRjecIndxList)
    isDecim = True #variable for indication of decimation (omission of points)
    # preparing the arrays for removing some elements
    newGradAcptIndxList = GradAcptIndxList.copy()
    newGradRjecIndxList = GradRjecIndxList.copy()
    if nGA + nGR <= nptsMark:
        # not enough events for taking the trouble to omit anything. Carry on
        isDecim = False
    else:
        # calculate "overload rate"
        over = (nGA+nGR)/nptsMark
        # theoretically, this "period" works for decimation
        # (only mark 1 point every 'per')

        #msg = "\nNumber of events = {} ({};{}), limit = {}".format(nGA+nGR,nGA,nGR,
        #                                                           nptsMark)
        #msg += "\nOverload level: {}... Decimation time!".format(over)
        #msg += "\nThese are the lists:"
        #msg += "\nThis is GradAcptIndxList: " + str(GradAcptIndxList)
        #msg += "\nThis is GradRjecIndxList: " + str(GradRjecIndxList)
        #self.log.printL(msg)

        # Search in Acceptance events
        blockListA = [] # empty block list
        blockA = [0,0] # first default block
        nremA = 0 # number of possible removals in acceptance events
        for ind in range(1,len(GradAcptIndxList)):
            #self.log.printL("ind = "+str(ind))
            if GradAcptIndxList[ind] == GradAcptIndxList[blockA[1]]+1:
                # ind belongs to the same block: grow it
                blockA[1] = ind
                #self.log.printL("Block growing! New block: "+str(blockA))
            else:
                # ind does not belong to the same block. End it and begin another
                #self.log.printL("Block is finished.")
                # if it is a meaningful block (3 or more equal events),
                # add to block list
                if blockA[1]-blockA[0] > 1:
                    blockListA.append(blockA.copy())
                    nremA += blockA[1]-blockA[0]-1
                # create new block
                blockA[0], blockA[1] = ind, ind
                #self.log.printL("Creating new block: " + str(blockA) +
                #                ", event: " + str(GradAcptIndxList[blockA[0]]))

        #self.log.printL("Ok, search is finished.")
        # final block check
        if blockA[1] - blockA[0] > 1:
            blockListA.append(blockA.copy())
            nremA += blockA[1] - blockA[0] - 1

        # Search in Rejection events
        blockListR = []  # empty block list
        blockR = [0, 0]  # first default block
        nremR = 0  # number of possible removals in rejection events
        for ind in range(1, len(GradRjecIndxList)):
            #self.log.printL("ind = ", str(ind))
            if GradRjecIndxList[ind] == GradRjecIndxList[blockR[1]] + 1:
                # ind belongs to the same block: grow it
                blockR[1] = ind
                #self.log.printL("Block growing! New block: " + str(blockR))
            else:
                # ind does not belong to the same block. End it and begin another
                #self.log.printL("Block is finished.")
                # if it is a meaningful block  (3 or more equal events),
                # add to block list
                if blockR[1] - blockR[0] > 1:
                    blockListR.append(blockR.copy())
                    nremR += blockR[1] - blockR[0] - 1
                blockR[0], blockR[1] = ind, ind
                #self.log.printL("Creating new block: " + str(blockR) +
                #                ", event: " + str(GradRjecIndxList[blockR[0]]))
        #self.log.printL("Ok, search is finished.")
        # final block check
        if blockR[1] - blockR[0] > 1:
            blockListR.append(blockR.copy())
            nremR += blockR[1] - blockR[0] - 1

        # Decision about which points to be actually removed. The objective of this
        # section
        # is to define a remove function rmFunc to remove or not each point from the
        # plot
        nrem = nremA + nremR #total number of points that can be omitted
        #self.log.printL("\nThese are the acceptance blocks:\n"+str(blockListA))
        #self.log.printL("\nThese are the rejection blocks:\n" + str(blockListR))
        #self.log.printL("Total removable points: "+str(nrem))

        if nGA + nGR - nrem >= nptsMark:
            # If the nptsMark target cannot be met even if all removable points are
            # removed, no choice but to remove all of them...
            #self.log.printL("All inner elements will be removed...")
            rmFunc = lambda x,y: True
        else:
            # Otherwise, find the minimum periodicity to ensure proper spacing of the
            # dots in the plot.
            # in order to avoid too few points, or concentrated and sparse regions,
            # try to decimate the points (mark 1 acceptance for each 'per').
            perMin = int(numpy.ceil(nremA/(nptsMark+nrem-nGA-nGR)))
            #self.log.printL("Minimum periodicty: "+str(perMin))
            rmFunc = lambda x,y : (x - y + 1) % perMin > 0

        # actual removal of points (acceptance)
        # Inner points are removed according to the rmFunc set above
        for blockA in blockListA:
             #self.log.printL("Removing inner block elements from acceptance list.")
             for k in range(blockA[0] + 1, blockA[1]):
                 if rmFunc(k,blockA[0]):
                    newGradAcptIndxList.remove(GradAcptIndxList[k])
                    nGA -= 1
        # actual removal of points (rejection). All inner points are removed
        for blockR in blockListR:
             #self.log.printL("Removing inner block elements from rejection list.")
             for k in range(blockR[0] + 1, blockR[1]):
                 newGradRjecIndxList.remove(GradRjecIndxList[k])
                 nGR -= 1

    # FINALLY, the plots of the event points themselves
    if nGA>0:
        # mark just the first acceptance event, for proper labeling
        plt.semilogy(newGradAcptIndxList[0]-1,
                     self.histQ[newGradAcptIndxList[0]], 'ob',
                     label='GradAccept')
        for k in range(1,nGA):
            # mark the remaining acceptance events
            plt.semilogy(newGradAcptIndxList[k]-1,
                         self.histQ[newGradAcptIndxList[k]],'ob')

    if nGR>0:
        # mark just the first rejection event, for proper labeling
        plt.semilogy(GradRjecIndxList[0]-1, self.histQ[GradRjecIndxList[0]],
                     'xC0', label='GradReject')
        for k in range(1, nGR):
            # mark the remaining rejection events
            plt.semilogy(newGradRjecIndxList[k]-1,
                         self.histQ[newGradRjecIndxList[k]],'xC0')

    # make sure the star is visible, by marking it again!
    if len(GradInitIndx)>0:
        plt.semilogy(numpy.array(GradInitIndx)-1,
                     self.histQ[GradInitIndx], '*C0')

    if isDecim:
        plt.title("Convergence report on Q\n(some redundant events not shown)")
    else:
        plt.title("Convergence report on Q")

    # If applicable, remove from the plot everything that is way too small
    if tolZoom:
        plt.ylim(ymin=(self.tol['Q'])**2)
    plt.grid(True)
    plt.xlabel("Gradient iterations")
    plt.ylabel("Q values")
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))

    self.savefig(keyName='histQ',fullName='Q convergence history')

def showHistI(self,tolZoom=True):
    IterGrad = numpy.arange(0,self.NIterGrad+1,1)
    # Get first and final values of I
    # Ok, technically it is the second value of I because the first one may
    # not be significant due to a (possibly) high P value associated with the
    # first guess.
    I0, I = self.histI[1], self.histI[self.NIterGrad]
    Iorig = self.histIorig[self.NIterGrad]
    Ipf = self.histIpf[self.NIterGrad]
    titl = "Convergence report on I\nFinal values: (Iorig,Ipf) = " + \
           "({:.4E}, {:.4E})".format(Iorig,Ipf)
    if self.histIpf[IterGrad].any() > 0.:
        plt.plot(IterGrad, self.histI[IterGrad], label='I')
        plt.plot(IterGrad,self.histIorig[IterGrad],label='Iorig')
        plt.plot(IterGrad,self.histIpf[IterGrad],label='Ipf')
    else:
        titl += "\n(Ipf omitted: identically zero)"
        plt.plot(IterGrad, self.histI[IterGrad], label='I')
    # If applicable, remove from the plot everything that is way too small
    if tolZoom:
        plt.ylim(ymin=min([min(self.histI[IterGrad]), I-(I0-I)]))
    plt.grid(True)
    plt.title(titl)
    plt.xlabel("Gradient iterations")
    plt.ylabel("I values")
    plt.legend()

    self.savefig(keyName='histI',fullName='I convergence history')

def showHistQvsI(self, tolZoom=True, nptsMark=10):
    """Plot a curve of Q against I
    (which should be strictly descending, hence a proper variable for plotting...)"""

    # Assemble the plotting array (x-axis)
    IterGrad = numpy.arange(1,self.NIterGrad+1,1)
    # I reduction, %, w.r.t. the first actual value of I.
    # It is best to use I[1] instead of I[0] because the latter could be a very low
    # value due to a possibly high P value.
    Ired = 100*(self.histI[1] - self.histI)/self.histI[1]

    # Perform the plots
    if self.histQ[IterGrad].any() > 0:
        plt.semilogy(Ired[IterGrad],self.histQ[IterGrad],'b',label='Q')

    if self.histQx[IterGrad].any() > 0:
        plt.semilogy(Ired[IterGrad],self.histQx[IterGrad],'k',label='Qx')

    if self.histQu[IterGrad].any() > 0:
        plt.semilogy(Ired[IterGrad],self.histQu[IterGrad],'r',label='Qu')

    if self.histQp[IterGrad].any() > 0:
        plt.semilogy(Ired[IterGrad],self.histQp[IterGrad],'g',label='Qp')

    if self.histQt[IterGrad].any() > 0:
        plt.semilogy(Ired[IterGrad],self.histQt[IterGrad],'y',label='Qt')

    # mark points (use supplied value of the number of points themselves)
    if nptsMark > self.NIterGrad:
        nptsMark = self.NIterGrad
    per = self.NIterGrad // nptsMark # period
    # mark the first point
    plt.semilogy(Ired[1], self.histQ[1], 'ob')
    # mark each point
    for k in range(1, nptsMark + 1):
        plt.semilogy(Ired[k*per],self.histQ[k*per],'ob')
    # mark the last point
    plt.semilogy(Ired[self.NIterGrad],self.histQ[self.NIterGrad],'ob')

    # Draw the Q tolerance line
    plt.plot(Ired[IterGrad],self.tol['Q']+0.0*IterGrad,'-.b',label='tolQ')

    msg = "Convergence: Q vs I behavior\nCircles mark points every " + \
          "{} grad iterations".format(per)
    plt.title(msg)
    # If applicable, remove from the plot everything that is way too small
    if tolZoom:
        plt.ylim(ymin=(self.tol['Q'])**2)
    plt.grid(True)
    plt.xlabel("I reduction, % (w.r.t. I0 = {:.4E})".format(self.histI[1]))
    plt.ylabel("Q values")
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))

    self.savefig(keyName='histQvsI',fullName='Q vs. I convergence history')

def showHistGradStep(self):
    IterGrad = numpy.arange(1,self.NIterGrad,1)

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel("Gradient iterations")
    ax1.set_ylabel('Step values',color=color)
    ax1.semilogy(IterGrad,self.histStepGrad[IterGrad],color=color,
                 label='Step values')
    ax1.tick_params(axis='y',labelcolor=color)
    ax1.grid(True)
    ax1.set_title("Gradient step history")

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Step search stop motive',color=color)
    labl = '"Stop motive" codes:\n' + \
           '0 - step rejected\n' + \
           '1 - local min found\n' + \
           '2 - step limit hit\n' + \
           '3 - too many evals'
    ax2.plot(IterGrad,self.histGSSstopMotv[IterGrad],color=color,
                 label=labl)
    ax2.tick_params(axis='y',labelcolor=color)
    #ax2.yaxis.set_major_locator(ticker.IndexLocator(base=1., offset=1.))
    ax2.grid(True)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.1,1))

    fig.tight_layout()

    self.savefig(keyName='histGradStep',fullName='GradStep history')

def showHistGRrate(self):
    """Show the history of the GR rate (rests per grad).
        It must begin at 1, there is no such thing as an "initial GR rate". """

    IterGrad = numpy.arange(1,self.NIterGrad+1,1)

    #if self.histGRrate[IterGrad].any() > 0:
    plt.title("Gradient-restoration rate history")
    plt.plot(IterGrad-1,self.histGRrate[IterGrad])
    plt.grid(True)
    plt.xlabel("Gradient iterations")
    plt.ylabel("Restorations per gradient")

    self.savefig(keyName='histGRrate',fullName='Grad-Rest rate history')
    #else:
    #    self.log.printL("showHistGRrate: No positive values. Skipping...")

def showHistObjEval(self,onlyAcpt=True):
    """ Show the history of object evaluations"""

    # The next calculations assume a minimum number of iterations.
    # If this number is not met, leave now.
    if self.NIterGrad < 2:
        self.log.printL("showHistObjEval: no gradient iterations, leaving.")
        self.GSStotObjEval = 0
        self.GSSavgObjEval = 0
        return False

    # "onlyAcpt": supress the items corresponding to the rejections
    if onlyAcpt:

        indPlot = list()
        noRejc = False
        for ind in range(1, self.NIterGrad):
            if self.histGSSstopMotv[ind] == 0:
                # rejection!
                if noRejc:
                    # new rejection! Flag it
                    noRejc = False
            else:
                # no rejection, record previous index
                indPlot.append(ind - 1)
                if not noRejc: # recovery from rejection, reset flag
                    noRejc = True

        # remove the index 0
        indPlot.remove(0)
        # include the last index
        indPlot.append(self.NIterGrad-1)

        # Now plot the evaluations
        plt.plot(range(1,len(indPlot)+1), self.histObjEval[indPlot],
                 label = 'Total evaluations')
        totEval = sum(self.histObjEval[indPlot])
        avgEval = totEval / len(indPlot)

        # Mark the points with rejections
        KeepLabl = True
        for i in range(1,len(indPlot)):
            if indPlot[i] > indPlot[i-1]+1:
                if KeepLabl:
                    # label only the first point
                    plt.plot(i+1, self.histObjEval[indPlot[i]], 'xr',
                             label='Rejections included')
                    KeepLabl = False
                else:
                    plt.plot(i+1, self.histObjEval[indPlot[i]], 'xr')

        plt.xlabel("Accepted gradient iterations")
        plt.ylabel("Obj. evaluations per gradient")
    else:
        IterGrad = numpy.arange(1, self.NIterGrad, 1)
        plt.plot(IterGrad, self.histObjEval[IterGrad], label='Obj. evaluations')
        totEval = sum(self.histObjEval[IterGrad])
        avgEval = totEval / (self.NIterGrad-1)
        KeepLabl = True

        for ind in range(1, self.NIterGrad):
            if self.histGSSstopMotv[ind] == 0:
                if KeepLabl:
                    plt.plot(ind-1,self.histObjEval[ind-1],'xr',
                             label='Rejections')
                    KeepLabl = False # label only the first one
                else:
                    plt.plot(ind-1,self.histObjEval[ind-1],'xr')

        plt.grid(True)
        plt.xlabel("Gradient iterations")
        plt.ylabel("Obj. evaluations per gradient")
    plt.title("Objective function evaluation history\n"
              "Total evaluations = {}, average = {:.3G}".format(totEval,avgEval))
    plt.grid(True)
    # No need for legend if there is just the first curve...
    if not KeepLabl:
        plt.legend()

    # put the statistics in the 'sol' object for easier post-processing
    self.GSStotObjEval = totEval
    self.GSSavgObjEval = avgEval

    # proceed to save the figure
    self.savefig(keyName='histObjEval',
                 fullName='Objective function evaluation history')


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
    self.histGSSstopMotv = sol_from.histGSSstopMotv
    self.NIterGrad = sol_from.NIterGrad
