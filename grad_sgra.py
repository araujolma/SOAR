#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:39:08 2017

@author: levi
"""

import numpy
#from utils import testAlgn
from utils import simp, testAlgn
import matplotlib.pyplot as plt

class stepMngr():
    """ Class for the step manager, the object used to calculate the step
    value for the gradient phase. The step is chosen through (rudimentary)
    minimization of an objective function "Obj".

    For now, the objective function looks only into J, the extended cost
    functional."""

    def __init__(self, log, ctes, corr, prntCond=False):
        self.cont = -1
        self.histStep = list()
        self.histI = list()
        self.histP = list()
        self.histObj = list()

        #Overall parameters
        self.tolP = ctes['tolP'] #from sgra already
        self.limP = ctes['limP'] #1e5 * tolP seems ok
        # for stopping conditions
        self.stopStepLimTol = ctes['stopStepLimTol'] # been using 1e-4
        self.stopObjDerTol = ctes['stopObjDerTol'] # been using 1e-4
        self.stopNEvalLim = ctes['stopNEvalLim'] # been using 100
        # for findStepLim
        self.findLimStepTol = ctes['findLimStepTol'] # been using 1e-2
        self.piLowLim = ctes['piLowLim']
        self.piHighLim = ctes['piHighLim']

        self.corr = corr
        self.log = log
        self.mustPrnt = prntCond
        self.best = {'step':0.0,
                     'obj': None}
        self.maxGoodStep = 0.0
        self.minBadStep = 1e100


    def calcObj(self,P,I,J):
        """This is the function which defines the objective function."""

        # TODO: Find a way to properly parameterize this so that the user can
        # set the objective function by setting numerical (or enumerative)
        # parameters in the .its file.

        return J

#        return (I + (self.k)*P)
#        if P > self.tolP:
#            return (I + (self.k)*(P-self.tolP))
#        else:
#            return I


    def getLast(self):
        """ Get the atributes for the last applied step. """

        P = self.histP[self.cont]
        I = self.histI[self.cont]
        Obj = self.histObj[self.cont]

        return P, I, Obj

    def check(self,alfa,Obj,P,pi):
        """ Perform a validity check in this step value, updating the limit
        values for searching step values if it is the case."""

        # Check for violations on the pi conditions.
        # Again, some limit is 'None' when it is not active.
        PiCondVio = False
        for i in range(len(pi)):
            # violated here in lower limit condition
            if self.piLowLim[i] is not None and pi[i] < self.piLowLim[i]:
                PiCondVio = True; break # already violated, no need to continue
            # violated here in upper limit condition
            if self.piHighLim[i] is not None and pi[i] > self.piHighLim[i]:
                PiCondVio = True; break # already violated, no need to continue
        #


        # "Bad point" conditions:
        BadPntCode = False
        BadPntMsg = ''
        if Obj > self.Obj0:
            BadPntCode = True
            BadPntMsg += ("\n-  Obj-Obj0 = {:.4E}".format(Obj-self.Obj0) + \
                         " > 0")
        if Obj < 0.0:
            BadPntCode = True
            BadPntMsg += ("\n-  Obj = {:.4E} < 0".format(Obj))
        if P >= self.limP:
            BadPntCode = True
            BadPntMsg += ("\n-  P = {:.4E}".format(P) + \
                         " > {:.4E} = limP".format(self.limP))
        if PiCondVio:
            BadPntCode = True
            BadPntMsg += ("\n-   piLowLim: " + str(self.piLowLim) + \
                          "\n          pi: " + str(pi) + \
                          "\n   piHighLim: " + str(self.piHighLim))


        if BadPntCode:
            # Bad point!
            self.log.printL("> In check. Got a bad point," + \
                            BadPntMsg + "\nwith alfa = " + str(alfa) + \
                            ", Obj = "+str(Obj))
            # update minimum bad step, if necessary
            if alfa < self.minBadStep:
                self.minBadStep = alfa
            return False, 1.1 * self.Obj0
        else:
            # Good point; update maximum good step, if necessary
            if alfa > self.maxGoodStep:
                self.maxGoodStep = alfa
            return True, Obj

    def calcBase(self,sol,P0,I0,J0):
        """ Calculate the baseline values, that is, the functions P, I, J for
        the solution with no correction applied.

        Please note that there is no need for this P0, I0 and J0 to come from
        outside of this function... this is only here to avoid calculating
        these guys twice. """

        Obj0 = self.calcObj(P0,I0,J0)
        self.log.printL("> I0 = {:.4E}, ".format(I0) + \
                        "Obj0 = {:.4E}".format(Obj0))
        self.P0 = P0
        self.I0 = I0
        self.J0 = J0
        self.Obj0 = Obj0
        self.log.printL("\n> Setting Obj0 to "+str(Obj0))
        self.best['obj'] = Obj0
        return Obj0

    def tryStep(self,sol,alfa,plotSol=False,plotPint=False):
        """ Try this given step value.

        Some additional procedures are performed, though:
            - update "best step" record ...       """

        self.cont += 1

        if self.mustPrnt:
            self.log.printL("\n> Trying alfa = {:.4E}".format(alfa))

        newSol = sol.copy()
        newSol.aplyCorr(alfa,self.corr)
        if plotSol:
            newSol.plotSol()

        P,_,_ = newSol.calcP(mustPlotPint=plotPint)
        J,_,_,I,_,_ = newSol.calcJ()
        Obj = self.calcObj(P,I,J)
        isOk, Obj = self.check(alfa,Obj,P,newSol.pi)#self.check(alfa,Obj,P)

        if isOk:
            # Update best value (if it is the case)
            if self.best['obj'] > Obj:
                self.best = {'step': alfa, 'obj': Obj}
                if self.mustPrnt:
                    self.log.printL("\n> Updating best result: \n" + \
                                    "alfa = {:.4E}, ".format(alfa) + \
                                    "Obj = {:.4E}".format(Obj))
            #

            if self.mustPrnt:
                resStr = "\n- Results (good point):\n" + \
                         "step = {:.4E}".format(alfa) + \
                         " P = {:.4E}".format(P) + " I = {:.4E}".format(I) + \
                         " J = {:.7E}".format(J) + " Obj = {:.7E}".format(Obj)
        else:
            if self.mustPrnt:
                resStr = "\n- Results (bad point):\n" + \
                         "step = {:.4E}".format(alfa) + \
                         " P = {:.4E}".format(P) + " I = {:.4E}".format(I) + \
                         " J = {:.7E}".format(J) + \
                         " CorrObj = {:.7E}".format(Obj)
        #

        if self.mustPrnt:
            resStr += "\n- Evaluation #" + str(self.cont) + ",\n" + \
                      "- BestStep: {:.6E}, ".format(self.best['step']) + \
                      "BestObj: {:.6E}".format(self.best['obj']) + \
                      "\n- MaxGoodStep: {:.6E}".format(self.maxGoodStep) + \
                      "\n- MinBadStep: {:.6E}".format(self.minBadStep)
            self.log.printL(resStr)
        #
        self.histStep.append(alfa)
        self.histP.append(P)
        self.histI.append(I)
        self.histObj.append(Obj)

        return self.getLast()

    def tryStepSens(self,sol,alfa,plotSol=False,plotPint=False):
        """ Try a given step and the sensitivity as well. """

        da = self.findLimStepTol * .1
        NotAlgn = True
        P, I, Obj = self.tryStep(sol, alfa, plotSol=plotSol, plotPint=plotPint)

        while NotAlgn:
            stepArry = numpy.array([alfa*(1.-da), alfa, alfa*(1.+da)])
            Pm, Im, Objm = self.tryStep(sol, stepArry[0])
            PM, IM, ObjM = self.tryStep(sol, stepArry[2])

            dalfa = stepArry[2] - stepArry[0]
            gradP = (PM-Pm) / dalfa
            gradI = (IM-Im) / dalfa
            gradObj = (ObjM-Objm) / dalfa

            outp = {'P': numpy.array([Pm,P,PM]), 'I': numpy.array([Im,I,IM]),
                    'obj': numpy.array([Objm,Obj,ObjM]), 'step': stepArry,
                    'gradP': gradP, 'gradI': gradI, 'gradObj': gradObj}

            algnP = testAlgn(stepArry,outp['P'])
            algnI = testAlgn(stepArry,outp['I'])
            algnObj = testAlgn(stepArry,outp['obj'])
            # Test alignment; if not aligned, keep lowering
            if max([abs(algnP),abs(algnI),abs(algnObj)]) < 1.0e-10:
                NotAlgn = False
            da *= .1
        #

        return outp

    def fitNewStep(self,alfaList,ObjList):
        """ Fit a quadratic curve through the points given; then use this model
        to find the optimum step value (to minimize objective function)."""

        M = numpy.ones((3,3))
        M[:,0] = alfaList ** 2.0
        M[:,1] = alfaList
        # find the quadratic coefficients
        coefList = numpy.linalg.solve(M,ObjList)
        self.log.printL("\n> Objective list: "+str(ObjList))
        self.log.printL("\n> Step value list: "+str(alfaList))
        self.log.printL("\n> Quadratic interpolation coefficients: " + \
                        str(coefList))

        # this corresponds to the vertex of the parabola
        alfaOpt = -coefList[1]/coefList[0]/2.0
        self.log.printL("> According to this quadratic fit, best step " + \
                        "value is {:.4E}.".format(alfaOpt))

        # The quadratic model is obviously wrong if the x^2 coefficient is
        # negative, or if the appointed step is negative, or invalid.
        # These cases must be handled differently.

        if coefList[0] < 0.0:
            # There seems to be a local maximum nearby.
            # Check direction of max decrease in objective
            gradLeft = (ObjList[1]-ObjList[0])/(alfaList[1]-alfaList[0])
            gradRight = (ObjList[2]-ObjList[1])/(alfaList[2]-alfaList[1])
            self.log.printL("\n> Inverted parabola detected.\n" + \
                            "Slopes: left = {:.4E}".format(gradLeft) + \
                            ", right = {:.4E}".format(gradRight))

            if gradLeft * gradRight > 0.0:
                # Both sides have the same tendency; use medium gradient
                grad = (ObjList[2]-ObjList[0])/(alfaList[2]-alfaList[0])
                print("> Using medium slope: {:.4E}".format(grad))
            else:
                if abs(gradLeft) > abs(gradRight):
                    grad = gradLeft
                    print("> Using left slope.")
                else:
                    grad = gradRight
                    print("> Using right slope.")
                #
            #

            alfaOpt = alfaList[1] - 0.5 * ObjList[1]/grad
            self.log.printL("\n> Using the slope in a Newton-Raphson-ish" + \
                            " iteration, next step value is " + \
                            "{:.4E}...".format(alfaOpt))

            if alfaOpt > self.minBadStep:
                alfaOpt = .5 * (alfaList[1] + self.minBadStep)
                self.log.printL("> ...but since that would violate" + \
                                " the max step condition,\n" + \
                                " let's bisect forward to " + \
                                "{:.4E}".format(alfaOpt) + " instead!")
            elif alfaOpt < 0.0:
                alfaOpt = .5 * alfaList[1]
                self.log.printL("> ...but since that would be " + \
                                "negative,\nlet's bisect back to " + \
                                "{:.4E}".format(alfaOpt) + " instead!")

            else:
                self.log.printL("> ... seems legit!")

        elif alfaOpt < 0.0:
            alfaOpt = .5 * min(alfaList)
            self.log.printL("> Quadratic fit suggested a negative step.\n" + \
                           "  Bisecting back into that region with " + \
                            "alfa = {:.4E} instead.".format(alfaOpt))
        elif alfaOpt > self.minBadStep:
            alfaOpt = .5 * (alfaList[1] + self.maxGoodStep)
            self.log.printL("> Quadratic fit suggested a bad step.\n" + \
                            "  Bisecting forward into that region with " + \
                            "alfa = {:.4E} instead.".format(alfaOpt))
        else:
            self.log.printL("  Let's do it!")

        return alfaOpt

    def stopCond(self,alfa,outp):
        """ Decide if the step search can be stopped. """
        gradObj = outp['gradObj']/outp['obj'][1]

        #outp = {'P': numpy.array([Pm,P,PM]), 'I': numpy.array([Im,I,IM]),
        #        'obj': numpy.array([Objm,Obj,ObjM]), 'step': stepArry,
        #        'gradP': gradP, 'gradI': gradI, 'gradObj': gradObj}

        if abs(gradObj) < self.stopObjDerTol:
            if self.mustPrnt:
                self.log.printL("\n> Stopping step search: low sensitivity" + \
                                " of objective function with step.\n" + \
                                "(local minimum, perhaps?)")
            return True
        elif abs(alfa/self.maxGoodStep - 1.0) < self.stopStepLimTol \
                and gradObj < 0.0:
            self.log.printL("\n> Stopping step search: high proximity" + \
                            " to the step limit value.")
            return True

        elif self.cont+1 > self.stopNEvalLim:
            self.log.printL("\n> Stopping step search: too many objective" + \
                            " evaluations.")
            return True

        else:
            return False

    def findStepLim(self,sol):
        """ Find the limit value for the step.
        This means the minimum value so that P = limP, or Obj = Obj(0). """

        if self.mustPrnt:
            self.log.printL("\n> This is findStepLim." + \
                            "\n> I will try to find the step value for" + \
                            " which Obj=Obj0." + \
                            "\n> Trying first alfa = 1 and its neighborhood.")

        IsGoodPnt = True
        # Start with alfaHigh = 1.0, multiply by 10.0 until some condition is
        # violated.
        alfaHigh = .1
        while IsGoodPnt:
            alfaHigh *= 10.0
            P, I, Obj = self.tryStep(sol,alfaHigh)
            if Obj > self.Obj0:
                IsGoodPnt = False

        alfaLow = 0.0
        StepSep, ObjSep = True, True
        if self.mustPrnt:
            self.log.printL("\n> Going for simple bisection.")
        while ObjSep and StepSep:
            alfaMid = .5 * (alfaLow + alfaHigh)
            P, I, Obj = self.tryStep(sol,alfaMid)
            # Based on the condition...
            if Obj >= self.Obj0:
                # Bisect down
                alfaHigh = alfaMid
            else:
                # Bisect up
                alfaLow = alfaMid
            if self.mustPrnt:
                self.log.printL("- alfaLow = {:.4E}, ".format(alfaLow) + \
                                "alfaHigh = {:.4E}".format(alfaHigh))
            #ObjSep = (abs(Obj/self.Obj0 - 1.0) > self.findLimObjTol)
            StepSep = (abs(1.0-alfaLow/alfaHigh) > self.findLimStepTol)
        #

        self.log.printL("\n> Found it! Leaving findStepLim now.")
        return .5*(alfaLow+alfaHigh)
    #

    def endPrntPlot(self,alfa,mustPlot=False):
        """Final plots and prints for this run of calcStepGrad. """
        if mustPlot or self.mustPrnt:
            # Get index for applied step
            for k in range(len(self.histStep)):
                if abs(self.histStep[k]-alfa)<1e-14:
                    break

            # Get final values of P, I and Obj
            P, I, Obj = self.histP[k], self.histI[k], self.histObj[k]
        #

        # plot the history of the tried alfas, and corresponding P/I/Obj's
        if mustPlot:

            # Plot history of P
            a = min(self.histStep)
            cont = 0
            if a == 0:
                newhistStep = sorted(self.histStep)
            while a == 0.0:
                cont += 1
                a = newhistStep[cont]
            linhAlfa = numpy.array([0.9*a,max(self.histStep)])
            plt.loglog(self.histStep,self.histP,'o',label='P(alfa)')
            linP0 = self.P0 + numpy.zeros(len(linhAlfa))
            plt.loglog(linhAlfa,linP0,'--',label='P(0)')
            linTolP = self.tolP + 0.0 * linP0
            plt.loglog(linhAlfa,linTolP,'--',label='tolP')
            linLimP = self.limP + 0.0 * linP0
            plt.loglog(linhAlfa,linLimP,'--',label='limP')
            # Plot final values in squares
            plt.loglog(alfa,P,'s',label='Chosen value')
            plt.xlabel('alpha')
            plt.ylabel('P')
            plt.grid(True)
            plt.legend()
            plt.title("P versus grad step for this grad run")
            plt.show()

            # Plot history of I
            plt.semilogx(self.histStep, 100.0*(self.histI/self.I0-1.0), 'o',\
                         label='I(alfa)')
            plt.semilogx(alfa, 100.0*(I/self.I0-1.0), 's', \
                         label='Chosen value')
            xlim = max([.99*self.maxGoodStep, 1.01*alfa])
            plt.ylabel("I variation (%)")
            plt.xlabel("alpha")
            plt.title("I variation versus grad step for this grad run." + \
                      " I0 = {:.4E}".format(self.I0))
            plt.legend()
            plt.grid(True)
            Imin = min(self.histI)
            if Imin < self.I0:
                ymax = 100.0*(1.0 - Imin/self.I0)
                ymin = - 1.1 * ymax
            else:
                ymax = 50.0 * (Imin/self.I0 - 1.0)
                ymin = - ymax
            plt.xlim(right = xlim)
            plt.ylim(ymax = ymax, ymin = ymin)
            plt.show()

            # Plot history of Obj
            plt.semilogx(self.histStep, 100.0*(self.histObj/self.Obj0-1.0), \
                         'o', label='Obj(alfa)')
            plt.semilogx(alfa, 100.0*(Obj/self.Obj0-1.0), 's', \
                         label='Chosen value')
            plt.ylabel("Obj variation (%)")
            plt.xlabel("alpha")
            plt.title("Obj variation versus grad step for this grad run." + \
                      " Obj0 = {:.4E}".format(self.Obj0))
            plt.legend()
            plt.grid(True)
            Objmin = min(self.histObj)
            if Objmin < self.Obj0:
                ymax = 100.0*(1.0-Objmin/self.Obj0)
                ymin = - 1.1 * ymax
            else:
                ymax = 50.0 * (Objmin/self.Obj0 - 1.0)
                ymin = - ymax
            plt.xlim(right = xlim)
            plt.ylim(ymax = ymax, ymin = ymin)
            plt.show()
        #

        if self.mustPrnt:
            dIp = 100.0 * (I/self.I0 - 1.0)
            dObjp = 100.0 * (Obj/self.Obj0 - 1.0)
            self.log.printL("\n> Chosen alfa = {:.4E}".format(alfa) + \
                            "\n> I0 = {:.4E}".format(self.I0) + \
                            ", I = {:.4E}".format(I) + \
                            ", dI = {:.4E}".format(I-self.I0) + \
                            " ({:.4E})%".format(dIp) + \
                            "\n> Obj0 = {:.4E}".format(self.Obj0) + \
                            ", Obj = {:.4E}".format(Obj) + \
                            ", dObj = {:.4E}".format(Obj-self.Obj0) + \
                            " ({:.4E})%".format(dObjp))

            self.log.printL("> Number of objective evaluations: " + \
                            str(self.cont))
        #
    #
#
###############################################################################

def plotF(self,piIsTime=False):
    """Plot the cost function integrand."""

    self.log.printL("In plotF.")

    argout = self.calcF()

    if isinstance(argout,tuple):
        if len(argout) == 3:
            f, fOrig, fPF = argout
            self.plotCat(f, piIsTime=piIsTime, color='b', labl='Total cost')
            self.plotCat(fOrig, piIsTime=piIsTime, color='k', labl='Orig cost')
            self.plotCat(fPF, piIsTime=piIsTime, color='r', \
                         labl='Penalty function')
        else:
            f = argout[0]
            self.plotCat(f, piIsTime=piIsTime, color='b', labl='Total cost')
    else:
        self.plotCat(f, piIsTime=piIsTime, color='b', labl='Total cost')
    #
    plt.title('Integrand of cost function (grad iter #' + \
              str(self.NIterGrad) + ')')
    plt.ylabel('f [-]')
    plt.grid(True)
    if piIsTime:
        plt.xlabel('Time [s]')
    else:
        plt.xlabel('Adim. time [-]')
    plt.legend()
    self.savefig(keyName='F',fullName='F')

def plotQRes(self,args):
    "Generic plots of the Q residuals"

    iterStr = "\n(grad iter #" + str(self.NIterGrad) + \
                  ", rest iter #"+str(self.NIterRest) + \
                  ", event #" + str(int((self.EvntIndx+1)/2)) + ")"
    # Qx error plot
    plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
    nm1 = self.n+1
    plt.subplot2grid((nm1,1),(0,0))
    self.plotCat(args['normErrQx'],color='b',piIsTime=False)
    plt.grid(True)
    plt.ylabel("Integrand of Qx")
    titlStr = "Qx = int || dlam - f_x + phi_x^T*lam || " + \
              "= {:.4E}".format(args['Qx'])
    titlStr += iterStr
    plt.title(titlStr)
    errQx = args['errQx']
    for i in range(self.n):
        plt.subplot2grid((nm1,1),(i+1,0))
        self.plotCat(errQx[:,i,:],piIsTime=False)
        plt.grid(True)
        plt.ylabel("ErrQx_"+str(i))
    plt.xlabel("t [s]")
    self.savefig(keyName='Qx',fullName='Qx')

    # Qu error plot
    plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
    mm1 = self.m+1
    plt.subplot2grid((mm1,1),(0,0))
    self.plotCat(args['normErrQu'],color='b',piIsTime=False)
    plt.grid(True)
    plt.ylabel("Integrand of Qu")
    titlStr = "Qu = int || f_u - phi_u^T*lam || = {:.4E}".format(args['Qu'])
    titlStr += iterStr
    plt.title(titlStr)
    errQu = args['errQu']
    for i in range(self.m):
        plt.subplot2grid((mm1,1),(i+1,0))
        self.plotCat(errQu[:,i,:],color='k',piIsTime=False)
        plt.grid(True)
        plt.ylabel("Qu_"+str(i))
    plt.xlabel("t")
    self.savefig(keyName='Qu',fullName='Qu')

    # Qp error plot
    errQp = args['errQp']; resVecIntQp = args['resVecIntQp']
    p = self.p
    plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
    plt.subplot2grid((p,1),(0,0))
    self.plotCat(errQp[:,0,:],color='k',piIsTime=False)
    plt.grid(True)
    plt.ylabel("ErrQp, j = 0")
    titlStr = "Qp = f_pi - phi_pi^T*lam\nresVecQp = "
    for j in range(p):
        titlStr += "{:.4E}, ".format(resVecIntQp[j])
    titlStr += iterStr
    plt.title(titlStr)
    for j in range(1,p):
        plt.subplot2grid((p,1),(j,0))
        self.plotCat(errQp[:,j,:],color='k')
        plt.grid(True)
        plt.ylabel("ErrQp, j ="+str(j))
    plt.xlabel("t [s]")

    self.savefig(keyName='Qp',fullName='Qp')

def calcJ(self):
    N, s = self.N, self.s
    #x = self.x

    #phi = self.calcPhi()
    psi = self.calcPsi()
    lam = self.lam
    mu = self.mu
    #dx = ddt(x,N)
    I,Iorig,Ipf = self.calcI()

    func = -self.calcErr() #like this, it's dx-phi
    vetL = numpy.empty((N,s))
    #vetIL = numpy.empty((N,s))

    for arc in range(s):
        for t in range(N):
            vetL[t,arc] = lam[t,:,arc].transpose().dot(func[t,:,arc])

    # Perform the integration of Lint array by Simpson's method
    Lint = 0.0
    for arc in range(self.s):
        Lint += simp(vetL[:,arc],N)

    #Lint = vetIL[N-1,:].sum()
    Lpsi = mu.transpose().dot(psi)
    L = Lint + Lpsi

    J_Lint = Lint
    J_Lpsi = Lpsi
    J_I = I
    J = L + J_I
    strJs = "J = {:.6E}".format(J)+", J_Lint = {:.6E}".format(J_Lint)+\
          ", J_Lpsi = {:.6E}".format(J_Lpsi)+", J_I = {:.6E}".format(J_I)
    self.log.printL(strJs)

    return J, J_Lint, J_Lpsi, I, Iorig, Ipf

def calcQ(self,mustPlotQs=False):
    # Q expression from (15).
    # FYI: Miele (2003) is wrong in oh so many ways...
    self.log.printL("In calcQ.")
    N, n, m, p, s = self.N, self.n, self.m, self.p, self.s
    dt = 1.0/(N-1)
    lam, mu = self.lam, self.mu

    # get gradients
    Grads = self.calcGrads()
    phix = Grads['phix']
    phiu = Grads['phiu']
    phip = Grads['phip']
    fx = Grads['fx']
    fu = Grads['fu']
    fp = Grads['fp']
    psiy = Grads['psiy']
    psip = Grads['psip']

    Qx, Qu, Qp, Qt, Q = 0.0, 0.0, 0.0, 0.0, 0.0
    auxVecIntQp = numpy.zeros((p,s))

    errQx = numpy.empty((N,n,s)); normErrQx = numpy.empty((N,s))
    errQu = numpy.empty((N,m,s)); normErrQu = numpy.empty((N,s))
    errQp = numpy.empty((N,p,s)); #normErrQp = numpy.empty(N)

    coefList = simp([],N,onlyCoef=True)
    z = numpy.empty(2*n*s)

    for arc in range(s):
        z[2*arc*n : (2*arc+1)*n] = -lam[0,:,arc]
        z[(2*arc+1)*n : (2*arc+2)*n] = lam[N-1,:,arc]

        # calculate Qx separately. In this way, the derivative avaliation is
        # adequate with the trapezoidal integration method
        med = (lam[1,:,arc]-lam[0,:,arc])/dt -.5*(fx[0,:,arc]+fx[1,:,arc]) + \
                .5 * phix[0,:,:,arc].transpose().dot(lam[0,:,arc]) + \
                .5 * phix[1,:,:,arc].transpose().dot(lam[1,:,arc])

        errQx[0,:,arc] = med
        errQx[1,:,arc] = med
        for k in range(2,N):
            errQx[k,:,arc] = 2.0 * (lam[k,:,arc]-lam[k-1,:,arc]) / dt + \
                        -fx[k,:,arc] - fx[k-1,:,arc] + \
                        phix[k,:,:,arc].transpose().dot(lam[k,:,arc]) + \
                        phix[k-1,:,:,arc].transpose().dot(lam[k-1,:,arc]) + \
                        -errQx[k-1,:,arc]

        for k in range(N):
            errQu[k,:,arc] = fu[k,:,arc] +  \
                            - phiu[k,:,:,arc].transpose().dot(lam[k,:,arc])
            errQp[k,:,arc] = fp[k,:,arc] + \
                            - phip[k,:,:,arc].transpose().dot(lam[k,:,arc])

            normErrQx[k,arc] = errQx[k,:,arc].transpose().dot(errQx[k,:,arc])
            normErrQu[k,arc] = errQu[k,:,arc].transpose().dot(errQu[k,:,arc])

            Qx += normErrQx[k,arc] * coefList[k]
            Qu += normErrQu[k,arc] * coefList[k]
            auxVecIntQp[:,arc] += errQp[k,:,arc] * coefList[k]
        #
    #

    auxVecIntQp *= dt; Qx *= dt; Qu *= dt

    resVecIntQp = numpy.zeros(p)
    for arc in range(s):
        resVecIntQp += auxVecIntQp[:,arc]

    resVecIntQp += psip.transpose().dot(mu)
    Qp = resVecIntQp.transpose().dot(resVecIntQp)

    errQt = z + psiy.transpose().dot(mu)
    Qt = errQt.transpose().dot(errQt)

    Q = Qx + Qu + Qp + Qt
    self.log.printL("Q = {:.4E}".format(Q)+": Qx = {:.4E}".format(Qx)+\
          ", Qu = {:.4E}".format(Qu)+", Qp = {:.7E}".format(Qp)+\
          ", Qt = {:.4E}".format(Qt))

    #self.Q = Q

###############################################################################
    if mustPlotQs:
        args = {'errQx':errQx, 'errQu':errQu, 'errQp':errQp, 'Qx':Qx, 'Qu':Qu,
                'normErrQx':normErrQx, 'normErrQu':normErrQu,
                'resVecIntQp':resVecIntQp}
        self.plotQRes(args)

###############################################################################

    somePlot = False
    for key in self.dbugOptGrad.keys():
        if ('plotQ' in key) or ('PlotQ' in key):
            if self.dbugOptGrad[key]:
                somePlot = True
                break
    if somePlot:
        self.log.printL("\nDebug plots for this calcQ run:\n")
        self.plotSol()

        indMaxQu = numpy.argmax(normErrQu, axis=0)

        for arc in range(s):
            self.log.printL("\nArc =",arc,"\n")
            ind1 = numpy.array([indMaxQu[arc]-20,0]).max()
            ind2 = numpy.array([indMaxQu[arc]+20,N]).min()

            if self.dbugOptGrad['plotQu']:
                plt.plot(self.t,normErrQu[:,arc])
                plt.grid(True)
                plt.title("Integrand of Qu")
                plt.show()

            #for zoomed version:
            if self.dbugOptGrad['plotQuZoom']:
                plt.plot(self.t[ind1:ind2],normErrQu[ind1:ind2,arc],'o')
                plt.grid(True)
                plt.title("Integrand of Qu (zoom)")
                plt.show()

#            if self.dbugOptGrad['plotCtrl']:
#                if self.m==2:
#                    alfa,beta = self.calcDimCtrl()
#                    plt.plot(self.t,alfa[:,arc]*180.0/numpy.pi)
#                    plt.title("Ang. of attack")
#                    plt.show()
#
#                    plt.plot(self.t,beta[:,arc]*180.0/numpy.pi)
#                    plt.title("Thrust profile")
#                    plt.show()
            if self.dbugOptGrad['plotQuComp']:
                plt.plot(self.t,errQu[:,0,arc])
                plt.grid(True)
                plt.title("Qu: component 1")
                plt.show()

                if m>1:
                    plt.plot(self.t,errQu[:,1,arc])
                    plt.grid(True)
                    plt.title("Qu: component 2")
                    plt.show()

            if self.dbugOptGrad['plotQuCompZoom']:
                plt.plot(self.t[ind1:ind2],errQu[ind1:ind2,0,arc])
                plt.grid(True)
                plt.title("Qu: component 1 (zoom)")
                plt.show()

                if m>1:
                    plt.plot(self.t[ind1:ind2],errQu[ind1:ind2,1,arc])
                    plt.grid(True)
                    plt.title("Qu: component 2 (zoom)")
                    plt.show()

            if self.dbugOptGrad['plotLam']:
                plt.plot(self.t,lam[:,0,arc])
                plt.grid(True)
                plt.title("Lambda_h")
                plt.show()

                if n>1:
                    plt.plot(self.t,lam[:,1,arc])
                    plt.grid(True)
                    plt.title("Lambda_v")
                    plt.show()

                if n>2:
                    plt.plot(self.t,lam[:,2,arc])
                    plt.grid(True)
                    plt.title("Lambda_gama")
                    plt.show()

                if n>3:
                    plt.plot(self.t,lam[:,3,arc])
                    plt.grid(True)
                    plt.title("Lambda_m")
                    plt.show()

    # TODO: break these plots into more conditions

#    if numpy.array(self.dbugOptGrad.values()).any:
#        print("\nDebug plots for this calcQ run:")
#
#        if self.dbugOptGrad['plotQx']:
#            plt.plot(self.t,normErrQx)
#            plt.grid(True)
#            plt.title("Integrand of Qx")
#            plt.show()
#
#        if self.dbugOptGrad['plotQu']:
#            plt.plot(self.t,normErrQu)
#            plt.grid(True)
#            plt.title("Integrand of Qu")
#            plt.show()
#
#        # for zoomed version:
#        indMaxQx = normErrQx.argmax()
#        ind1 = numpy.array([indMaxQx-20,0]).max()
#        ind2 = numpy.array([indMaxQx+20,N-1]).min()
#
#        if self.dbugOptGrad['plotQxZoom']:
#            plt.plot(self.t[ind1:ind2],normErrQx[ind1:ind2],'o')
#            plt.grid(True)
#            plt.title("Integrand of Qx (zoom)")
#            plt.show()
#
#        if self.dbugOptGrad['plotSolQxMax']:
#            print("\nSolution on the region of MaxQx:")
#            self.plotSol(intv=numpy.arange(ind1,ind2,1,dtype='int'))
#
#        # for zoomed version:
#        indMaxQu = normErrQu.argmax()
#        ind1 = numpy.array([indMaxQu-20,0]).max()
#        ind2 = numpy.array([indMaxQu+20,N-1]).min()
#
#        if self.dbugOptGrad['plotQuZoom']:
#            plt.plot(self.t[ind1:ind2],normErrQu[ind1:ind2],'o')
#            plt.grid(True)
#            plt.title("Integrand of Qu (zoom)")
#            plt.show()
#
#        if self.dbugOptGrad['plotSolQuMax']:
#            print("\nSolution on the region of MaxQu:")
#            self.plotSol(intv=numpy.arange(ind1,ind2,1,dtype='int'))
#
#
##        if n==4 and m==2:
##
##            plt.plot(tPlot[ind1:ind2],errQx[ind1:ind2,0])
##            plt.grid(True)
##            plt.ylabel("Qx_h")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],errQx[ind1:ind2,1],'g')
##            plt.grid(True)
##            plt.ylabel("Qx_V")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],errQx[ind1:ind2,2],'r')
##            plt.grid(True)
##            plt.ylabel("Qx_gamma")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],errQx[ind1:ind2,3],'m')
##            plt.grid(True)
##            plt.ylabel("Qx_m")
##            plt.show()
##
##            print("\nStates, controls, lambda on the region of maxQx:")
##
##            plt.plot(tPlot[ind1:ind2],x[ind1:ind2,0])
##            plt.grid(True)
##            plt.ylabel("h [km]")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],x[ind1:ind2,1],'g')
##            plt.grid(True)
##            plt.ylabel("V [km/s]")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],x[ind1:ind2,2]*180/numpy.pi,'r')
##            plt.grid(True)
##            plt.ylabel("gamma [deg]")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],x[ind1:ind2,3],'m')
##            plt.grid(True)
##            plt.ylabel("m [kg]")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],u[ind1:ind2,0],'k')
##            plt.grid(True)
##            plt.ylabel("u1 [-]")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],u[ind1:ind2,1],'c')
##            plt.grid(True)
##            plt.xlabel("t")
##            plt.ylabel("u2 [-]")
##            plt.show()
##
##            print("Lambda:")
##
##            plt.plot(tPlot[ind1:ind2],lam[ind1:ind2,0])
##            plt.grid(True)
##            plt.ylabel("lam_h")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],lam[ind1:ind2,1],'g')
##            plt.grid(True)
##            plt.ylabel("lam_V")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],lam[ind1:ind2,2],'r')
##            plt.grid(True)
##            plt.ylabel("lam_gamma")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],lam[ind1:ind2,3],'m')
##            plt.grid(True)
##            plt.ylabel("lam_m")
##            plt.show()
##
###            print("dLambda/dt:")
###
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,0])
###            plt.grid(True)
###            plt.ylabel("dlam_h")
###            plt.show()
###
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,1])
###            plt.grid(True)
###            plt.ylabel("dlam_V")
###            plt.show()
###
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,2],'r')
###            plt.grid(True)
###            plt.ylabel("dlam_gamma")
###            plt.show()
###
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,3],'m')
###            plt.grid(True)
###            plt.ylabel("dlam_m")
###            plt.show()
###
###            print("-phix*lambda:")
###
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,0]-errQx[ind1:ind2,0])
###            plt.grid(True)
###            plt.ylabel("-phix*lambda_h")
###            plt.show()
###
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,1]-errQx[ind1:ind2,1],'g')
###            plt.grid(True)
###            plt.ylabel("-phix*lambda_V")
###            plt.show()
###
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,2]-errQx[ind1:ind2,2],'r')
###            plt.grid(True)
###            plt.ylabel("-phix*lambda_gamma")
###            plt.show()
###
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,3]-errQx[ind1:ind2,3],'m')
###            plt.grid(True)
###            plt.ylabel("-phix*lambda_m")
###            plt.show()
##
##            print("\nBlue: dLambda/dt; Black: -phix*lam")
##
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,0])
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,0]-errQx[ind1:ind2,0],'k')
##            plt.grid(True)
##            plt.ylabel("z_h")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,1])
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,1]-errQx[ind1:ind2,1],'k')
##            plt.grid(True)
##            plt.ylabel("z_V")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,2])
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,2]-errQx[ind1:ind2,2],'k')
##            plt.grid(True)
##            plt.ylabel("z_gamma")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,3])
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,3]-errQx[ind1:ind2,3],'k')
##            plt.grid(True)
##            plt.ylabel("z_m")
##            plt.show()

    if self.dbugOptGrad['pausCalcQ']:
        input("calcQ in debug mode. Press any key to continue...")

    return Q, Qx, Qu, Qp, Qt

def calcStepGrad(self,corr,alfa_0,retry_grad,stepMan):

    self.log.printL("\nIn calcStepGrad.\n")
    prntCond = self.dbugOptGrad['prntCalcStepGrad']

    if retry_grad:
        if prntCond:
            self.log.printL("\n> Retrying alfa." + \
                            " Base value: {:.4E}".format(alfa_0))
        # LOWER alfa!
        alfa = .1 * alfa_0
        if prntCond:
            self.log.printL("\n> Let's try alfa 90% lower.")
        P, I, Obj = stepMan.tryStep(self,alfa)

        while Obj > stepMan.Obj0:
            alfa *= .1
            if prntCond:
                self.log.printL("\n> Let's try alfa 90% lower.")
            P, I, Obj = stepMan.tryStep(self,alfa)

        if prntCond:
            self.log.printL("\n> Ok, this value should work.")

    else:
        # Get initial status (no correction applied)
        if prntCond:
            self.log.printL("\n> alfa = 0.0")
        P0,_,_ = self.calcP()
        J0,_,_,I0,_,_ = self.calcJ()
        # Get all the constants
        ctes = {'tolP': self.tol['P'], #from sgra already,
                'limP': self.constants['GSS_PLimCte'] * self.tol['P'], #1e5
                'stopStepLimTol': self.constants['GSS_stopStepLimTol'], # been using 1e-4
                'stopObjDerTol': self.constants['GSS_stopObjDerTol'],  # been using 1e-4
                'stopNEvalLim': self.constants['GSS_stopNEvalLim'], # been using 100
                'findLimStepTol': self.constants['GSS_findLimStepTol'],# been using 1e-2
                'piLowLim': self.restrictions['pi_min'],
                'piHighLim': self.restrictions['pi_max']}

        # Create new stepManager object
        stepMan = stepMngr(self.log, ctes, corr, prntCond = prntCond)
        # Set the base values
        stepMan.calcBase(self,P0,I0,J0)

        alfaLim = stepMan.findStepLim(self)

        # Start search by quadratic interpolation
        if prntCond:
            self.log.printL("\n> Starting detailed step search...\n")

        # Quadratic interpolation: for each point candidate for best step
        # value, its neighborhood (+1% and -1%) is also tested. With these 3
        # points, a quadratic interpolant is obtained, resulting in a parabola
        # whose minimum is the new candidate for optimal value.
        # The stop criterion is based in small changes to step value (prof.
        # Azevedo would hate this...), basically because it was easy to
        # implement.

        keepSrch = True
        alfaRef = alfaLim * .5

        while keepSrch:
            outp = stepMan.tryStepSens(self,alfaRef)#,plotSol=True,plotPint=True)
            alfaList, ObjList = outp['step'], outp['obj']
            gradObj = outp['gradObj']
            ObjRef = ObjList[1]
            if prntCond:
                self.log.printL("\n> With alfa = {:.4E}".format(alfaRef) + \
                                ", Obj = {:.4E}".format(ObjRef) + \
                                ", dObj/dAlfa = {:.4E}".format(gradObj))

            alfaRef = stepMan.fitNewStep(alfaList,ObjList)
            outp = stepMan.tryStepSens(self,alfaRef)#,plotSol=True,plotPint=True)
            if self.dbugOptGrad['manuInptStepGrad']:

                gradObj = outp['gradObj']
                ObjPos = outp['obj'][1]
                self.log.printL("\n> Now, with alfa = {:.4E}".format(alfaRef) + \
                                ", Obj = {:.4E}".format(ObjPos) + \
                                ", dObj/dAlfa = {:.4E}".format(gradObj))

                self.log.printL("\n> Type S for entering a step value, or" + \
                                " any other key to quit:")
                inp = input("> ")
                if inp == 's':
                    self.log.printL("\n> Type the step value.")
                    inp = input("> ")
                    alfaRef = float(inp)
                else:
                    keepSrch = False
            else:
                keepSrch = not(stepMan.stopCond(alfaRef,outp))
            #
        #
        alfa = stepMan.best['step']
    #

    # "Assured rejection prevention": if P<tolP, make sure I<I0, otherwise
    # there will be a guaranteed rejection later...

    if self.dbugOptGrad['plotCalcStepGrad']:
        # SCREENING:
        self.log.printL("\n> Going for screening...")
        mf = 10.0**(1.0/10.0)
        alfaUsed = alfa
        for j in range(10):
            alfa *= mf
            P, I, Obj = stepMan.tryStep(self,alfa)
        alfa = alfaUsed
        for j in range(10):
            alfa /= mf
            P, I, Obj = stepMan.tryStep(self,alfa)
        alfa = alfaUsed
#
#    # "Sanity checking": Is P0 = P(0), or I0 = I(0), or Obj0 = Obj(0)?
#    P_de_0, I_de_0, Obj_de_0 = stepMan.tryStep(self,0.0)
#    self.log.printL("\n> Sanity checking: ")
#    self.log.printL("P0 = " + str(stepMan.P0) + ", P(0) = " + \
#                    str(P_de_0) + ", DeltaP0 = " + str(P_de_0-stepMan.P0))
#    self.log.printL("I0 = " + str(stepMan.I0) + ", I(0) = " + \
#                    str(I_de_0) + ", DeltaI0 = " + str(I_de_0-stepMan.I0))
#    self.log.printL("Obj0 = " + str(stepMan.Obj0) + ", Obj(0) = " + \
#                    str(Obj_de_0) + ", DeltaObj0 = " + \
#                    str(Obj_de_0-stepMan.Obj0))

    # Final plots and prints
    stepMan.endPrntPlot(alfa,mustPlot=self.dbugOptGrad['plotCalcStepGrad'])

    if self.dbugOptGrad['pausCalcStepGrad']:
        input("\n> Run of calcStepGrad terminated. Press any key to continue.")

    return alfa, stepMan

def grad(self,corr,alfa_0,retry_grad,stepMan):

    self.log.printL("\nIn grad, Q0 = {:.4E}.".format(self.Q))
    #self.log.printL("NIterGrad = "+str(self.NIterGrad))

#    if self.NIterGrad > 99:
#        newConf = {'left':0.12,'right':0.9,'bottom':0.06,
#                   'top':.88,'wspace':0.2,'hspace':0.55}
#        self.log.printL("\nPlotting current sol in interactive mode...")
#        self.plotSol(mustSaveFig=False, subPlotAdjs=newConf)
#        self.log.printL("\nPlotting variations in interactive mode...")
#        self.plotSol(opt={'mode':'var','x':corr['x'],
#                          'u':corr['u'],'pi':corr['pi']},
#                     mustSaveFig=False, subPlotAdjs=newConf)

    # Calculation of alfa
    alfa, stepMan = self.calcStepGrad(corr,alfa_0,retry_grad,stepMan)
    #alfa = 0.1
    #self.log.printL('\n\nBypass cabuloso: alfa arbitrado em '+str(alfa)+'!\n\n')
    self.updtHistGrad(alfa)

    self.plotSol(opt={'mode':'lambda'})
    A, B, C = corr['x'], corr['u'], corr['pi']

#    if self.NIterGrad > 99:
#        self.log.printL("\nPlotting weighted variations in interactive mode...")
#        newConf = {'left':0.12,'right':0.9,'bottom':0.06,
#                   'top':.88,'wspace':0.2,'hspace':0.55}
#        self.plotSol(opt={'mode':'var','x':alfa*A,'u':alfa*B,'pi':alfa*C},
#                     mustSaveFig=False, subPlotAdjs=newConf)
#    else:
#        self.plotSol(opt={'mode':'var','x':alfa*A,'u':alfa*B,'pi':alfa*C})

    self.plotSol(opt={'mode':'var','x':alfa*A,'u':alfa*B,'pi':alfa*C})
    #input("@Grad: Waiting for lambda/corrections check...")

    # Apply correction, update histories in alternative solution
    newSol = self.copy()
    newSol.aplyCorr(alfa,corr)
    newSol.updtHistP()

    self.log.printL("Leaving grad with alfa = "+str(alfa))
    self.log.printL("Delta pi = "+str(alfa*C))

    if self.dbugOptGrad['pausGrad']:
        input('Grad in debug mode. Press any key to continue...')

    return alfa, newSol, stepMan
