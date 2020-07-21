#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:36:59 2017

@author: levi
"""
import numpy
from utils import simp, avoidRepCalc
import matplotlib.pyplot as plt

@avoidRepCalc(fieldsTuple=('P','Pint','Ppsi'))
def calcP(self,mustPlotPint=False):
    #self.log.printL("\nIn calcP.")
    N, s = self.N, self.s

    psi = self.calcPsi()
    #print("psi = "+str(psi))
    func = self.calcErr()
    # this computes vetP[k,arc] = |func[k,:,arc]|² for all k's and arcs.
    vetP = numpy.einsum('nis,nis->ns',func,func)

    # TODO: this test for violations could (theoretically) be removed, since
    #  calcStepGrad already checks for it...
    PiCondVio = False
    piLowLim = self.restrictions['pi_min']
    piHighLim = self.restrictions['pi_max']
    for i in range(self.s):
        # violated here in lower limit condition
        if piLowLim[i] is not None and self.pi[i] < piLowLim[i]:
            PiCondVio = True
            break  # already violated, no need to continue
        # violated here in upper limit condition
        if piHighLim[i] is not None and self.pi[i] > piHighLim[i]:
            PiCondVio = True
            break  # already violated, no need to continue
    #
    if PiCondVio:
        vetP *= 1e300

    coefList = simp([],N,onlyCoef=True)

    if mustPlotPint:
        vetIP = numpy.empty((N, s))
        for arc in range(s):
            vetIP[0,arc] = coefList[0] * vetP[0,arc]
            for t in range(1,N):
                vetIP[t,arc] = vetIP[t-1,arc] + coefList[t] * vetP[t,arc]
        Pint = vetIP[N-1,:].sum()
    else:
        Pint = numpy.einsum('n,ns->',coefList,vetP)
    Ppsi = psi.transpose().dot(psi)
    P = Ppsi + Pint
    strPs = "P = {:.6E}, Pint = {:.6E}, Ppsi = {:.6E}.".format(P,Pint,Ppsi)
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

def getPvalue(self,step,corr,mustPlotPint=False):
    """Abbreviation function for testing step values."""

    newSol = self.copy()
    newSol.aplyCorr(step,corr)
    P,_,_ = newSol.calcP(mustPlotPint=mustPlotPint)
    return P

def calcStepRest(self,corr):
    """Calculate the restoration step (referred to as  "alfa" or "alpha").

    The idea is to search for a value that reduces the value of the P
    functional, so that eventually P < tolP. It may be possible to meet that
    condition with a single restoration, if not, sequential restorations
    will be performed. """

    self.log.printL("\nIn calcStepRest.\n")
    plotPint = False

    # Get P value for "full restoration"
    P1 = getPvalue(self,1.0,corr,mustPlotPint=plotPint)

    # if applying alfa = 1.0 already meets the tolerance requirements,
    # why waste time decreasing alfa?
    if P1 < self.tol['P']:
        msg = "Unitary step already satisfies tolerances.\n" + \
              "Leaving rest with alfa = 1.\nDelta pi = " + str(corr['pi'])
        self.log.printL(msg)
        return 1.0

    # Avoid a new P calculation by loading P value from the sol object
    P0 = self.P

    # If alfa = 1 already lowers the P value, it is best to stop here
    # and return alfa = 1.
    if P1 < P0:
        msg = "Unitary step lowers P.\n" + \
              "Leaving rest with alfa = 1.\nDelta pi = " + str(corr['pi'])
        self.log.printL(msg)
        return 1.0

    # Beginning actual search: get the maximum value of alfa so that P<=P0
    self.log.printL("\nSearching for proper step...")
    # Perform a simple bisection monitoring the "error" P-P0:
    # alfaLow is maximum step so that P<=P0,
    # alfaHigh is minimum step so that P>=P0,
    alfaLow, alfa, alfaHigh = 0., 0.5,  1.

    # Stop conditions on "error" (1% of P0) and step variation (also 1%)
    #while abs(P-P0) > tolSrch and (alfaHigh-alfaLow) > 1e-2 * alfa:

    # Stop condition on step variation (1%)
    while (alfaHigh - alfaLow) > 1e-2 * alfa:
        # Try P in the middle
        alfa = .5 * (alfaLow + alfaHigh)
        P = getPvalue(self,alfa,corr,mustPlotPint=plotPint)
        if P < P0:
            # Negative error: go forward
            alfaLow = alfa
        else:
            # Positive error: go backwards
            alfaHigh = alfa
    # Get a step so that P<P0, just to be sure
    alfa = alfaLow
    msg =  "\nLeaving rest with alfa = {:.4E}".format(alfa) + \
           "\nDelta pi = " + str(alfa * corr['pi'])
    self.log.printL(msg)
    return alfa

    # Manual input of step
    # alfa = 1.
    # promptMsg = "Please enter new value of step to be used, or hit " + \
    #             "'enter' to finish:\n>> "
    # while True:
    #     inp = input(promptMsg)
    #
    #     if inp == '':
    #         break
    #
    #     try:
    #         alfa = float(inp)
    #         P = getPvalue(self,alfa,corr,mustPlotPint=plotPint)
    #         msg = "alfa = {:.4E}, P = {:.4E}, P0 = {:.4E}, tolP = {:.1E}\n".format(alfa,P,P0,self.tol['P'])
    #         self.log.printL(msg)
    #
    #     except KeyboardInterrupt:
    #         self.log.printL("Okay then.")
    #         raise
    #     except:
    #         msg = "\nSorry, could not cast '{}' to float.\n".format(inp)
    #         self.log.printL(msg)
    #
    # self.log.printL("Leaving rest with alfa = " + str(alfa))
    # self.log.printL("Delta pi = " + str(alfa * corr['pi']))
    # return alfa

    #


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
        self.log.prom('Rest in debug mode. Press any key to continue...')

