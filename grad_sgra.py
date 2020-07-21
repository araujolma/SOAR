#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:39:08 2017

@author: levi
"""

import numpy #, time
#from utils import testAlgn
from utils import simp, testAlgn, avoidRepCalc
import matplotlib.pyplot as plt
eps = 1e-20 # this is for avoiding == 0.0 (float) comparisons
LARG = 1e100 # this is for a "random big number"
class stepMngr:
    """ Class for the step manager, the object used to calculate the step
    value for the gradient phase. The step is chosen through (rudimentary)
    minimization of an objective function "Obj".

    For now, the objective function looks only into J, the extended cost
    functional."""

    def __init__(self, log, ctes, corr, pi, prevStepInfo, prntCond=False):

        # Lists for the histories
        # TODO: maybe pre-allocating numpy arrays performs better!!
        self.histStep = list()
        self.histI = list()
        self.histP = list()
        self.histObj = list()
        self.histStat = list()
        # contains the indexes of the non-pi and non-P lim violating tried steps,
        # in ascending order
        self.sortIndxHistStep = list()

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
        # New: dJ/dAlfa
        self.dJdStepTheo = corr['dJdStepTheo']
        self.corr = corr
        self.log = log
        self.mustPrnt = prntCond # printing condition
        # Previous step information
        self.prevStepInfo = prevStepInfo

        # These refer to the current state of the search, and change as it progresses:
        self.P0, self.I0, self.J0, self.Obj0 = 1., 1., 1., 1. # placeholder values
        # Begin by putting the "null step" in
        self.histStep.append(0.)
        self.sortIndxHistStep.append(0)
        self.histStat.append(True)
        self.cont = -1      # number of objective function evaluation counter
        self.cont_exp = -1  # number of objective function evaluation estimation
        self.mustRaisObj = False # flag for artificial raising of obj when violating
        self.best = {'step':0.0,     # this is for the best step so far
                     'obj': None,    # this is for the best object so far
                     'limP': -1.}    # this is for the smallest distance to P limit
        self.maxGoodStep = 0.0 # maximum good (non-violating) step
        self.minBadStep = LARG # minimum bad (violating) step
        self.isDecr = True # the obj function is monotonic until proven otherwise
        self.pLimSrchStopMotv = 0 # reason for abandoning the P lim step search.
        # this is for the P regression
        self.PRegrAngCoef, self.PRegrLinCoef = 0., 0.
        # this is for the "trissection" search
        #   The convention is: the first and last elements delimit the region of
        #   optimality. Thus;
        #     step[0] < alfa_minObj < step[2]
        #     minObj  < min(Obj[0],Obj[2])
        #   The second (middle) element must be between the first and the last:
        #     step[0] < step[1] < step[2]; hence Obj[1] < min(Obj[0],Obj[2])
        #   The first and second elements must be valid (non-violating). The last
        #   element may be violating.
        #   This "self.trio" attribute holds the indexes of the points in the
        #   histories (histStep/histObj/histP/etc).
        self.trio = [0, -1, -1]  # placeholder values, don't make sense
        #self.trio = {'obj': [0., 0., 0.],
        #             'step': [0., 0., 0.]}  # placeholder values, don't make sense
        self.isTrioSet = False  # True when the trio is set
        # "Stop motive" codes:  0 - step rejected
        #                       1 - local min found
        #                       2 - step limit hit
        #                       3 - too many evals
        self.stopMotv = -1  # placeholder, this will be updated later.

        # TODO: these parameters should go to the config file...
        self.tolStepObj = 1e-2
        self.tolStepLimP = 1e-2
        self.tolLimP = 1e-2
        self.epsStep = 1e-10#min(self.tolStepObj,self.tolStepLimP) / 100.
        self.ffRate = 1.5#2.#10.#

        # Try to find a proper limit for the step

        # initialize arrays with -1
        alfaLimLowPi = numpy.zeros_like(self.piLowLim) - 1.
        alfaLimHighPi = numpy.zeros_like(self.piHighLim) - 1.
        # Find the limit with respect to the pi Limits (lower and upper)
        # pi + alfa * corr['pi'] = piLim => alfa = (piLim-pi)/corr['pi']
        p = min((len(self.piLowLim), len(corr['pi'])))
        for i in range(p):
            if self.piLowLim[i] is not None and corr['pi'][i] < 0.:
                alfaLimLowPi[i] = (self.piLowLim[i] - pi[i]) / corr['pi'][i]
            if self.piHighLim[i] is not None and self.corr['pi'][i] > 0.:
                alfaLimHighPi[i] = (self.piHighLim[i] - pi[i]) / corr['pi'][i]
        noLimPi = True
        alfaLimPi = alfaLimLowPi[0]
        for alfa1, alfa2 in zip(alfaLimLowPi, alfaLimHighPi):
            if alfa1 > 0.:
                if alfa1 < alfaLimPi or noLimPi:
                    alfaLimPi = alfa1
                    noLimPi = False
            if alfa2 > 0.:
                if alfa2 < alfaLimPi or noLimPi:
                    alfaLimPi = alfa2
                    noLimPi = False
        if self.mustPrnt:
            if noLimPi:
                msg = "\n> No limit in step due to pi conditions. Moving on."
            else:
                msg = "\n> Limit in step due to pi conditions: " + \
                      "{}".format(alfaLimPi)
            self.log.printL(msg)

        # convention: 0: pi limits,
        #             1: P < limP,
        #             2: Obj <= Obj0,
        #             3: Obj >= 0
        if noLimPi:
            self.StepLimActv = [False, True, True, True]
            # lowest step that violates each condition
            self.StepLimUppr = [LARG, LARG, LARG, LARG]
            # highest step that does NOT violate each condition
            self.StepLimLowr = [LARG, 0., 0., 0.]
        else:
            self.StepLimActv = [True, True, True, True]
            # lowest step that violates each condition
            self.StepLimUppr = [alfaLimPi, LARG, LARG, LARG]
            # highest step that does NOT violate each condition
            self.StepLimLowr = [alfaLimPi, 0., 0., 0.]

    @staticmethod
    def calcObj(P, I, J):
        """This is the function which defines the objective function."""

        # TODO: Find a way to properly parametrize this so that the user can
        #  set the objective function by setting numerical (or enumerable)
        #  parameters in the .its file.

        return J

#        return (I + (self.k)*P)
#        if P > self.tolP:
#            return (I + (self.k)*(P-self.tolP))
#        else:
#            return I

    def getLast(self):
        """ Get the attributes for the last applied step. """

        P = self.histP[self.cont]
        I = self.histI[self.cont]
        Obj = self.histObj[self.cont]

        return P, I, Obj

    def check(self,alfa,Obj,P,pi):
        """ Perform a validity check in this step value, updating the limit
        values for searching step values if it is the case.

        Conditions checked:  - limits in each of the pi parameters
                             - P <= limP
                             - Obj <= Obj0
                             - Obj >= 0 [why?]

        If any of these conditions are not met, the actual obtained value of Obj
        may not be meaningful. Hence, the strategy adopted is to, upon command, (i.e.,
        via the .mustRaisObj flag) artificially increase the Obj value if any of the
        conditions (except the third, of course) is not met.
        """

        # the value of the objective to be returned.
        ObjRtrn = Obj

        # Bad point code: originally 1, each violation multiplies by a prime:
        # Pi conditions violation: x 2
        # P limit violation:       x 3
        # Obj >= Obj0              x 5
        # Obj < 0                  x 7
        BadPntCode = 1


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
        BadPntMsg = ''
        # condition 0
        if PiCondVio:
            BadPntCode *= 2
            BadPntMsg += ("\n-   piLowLim: " + str(self.piLowLim) +
                          "\n          pi: " + str(pi) +
                          "\n   piHighLim: " + str(self.piHighLim))
            # no need for checking since the pi limits are known beforehand
            ObjRtrn = 1.1 * self.Obj0 # artificial increase in Obj
        # condition 1
        if P >= self.limP:
            BadPntCode *= 3
            BadPntMsg += ("\n-  P = {:.4E}".format(P) +
                         " > {:.4E} = limP".format(self.limP))
            ObjRtrn = 1.1 * self.Obj0 # artificial increase in Obj
            if alfa < self.StepLimUppr[1]:
                self.StepLimUppr[1] = alfa
        else:
            if alfa > self.StepLimLowr[1]:
                self.StepLimLowr[1] = alfa
        # condition 2
        if Obj > self.Obj0:
            BadPntCode *= 5
            BadPntMsg += ("\n-  Obj-Obj0 = {:.4E} > 0".format(Obj-self.Obj0))
            if alfa < self.StepLimUppr[2]:
                self.StepLimUppr[2] = alfa
        else:
            if alfa > self.StepLimLowr[2]:
                self.StepLimLowr[2] = alfa
        # condition 3
        if Obj < 0.0:
            BadPntCode *= 7
            BadPntMsg += ("\n-  Obj = {:.4E} < 0".format(Obj))
            if alfa < self.StepLimUppr[3]:
                self.StepLimUppr[3] = alfa
        else:
            if alfa > self.StepLimLowr[3]:
                self.StepLimLowr[3] = alfa

        # look for limits that are no longer active, deactivate them
        domMsg = ''
        for i in range(4):
            if self.StepLimActv[i]:
                for j in range(4):
                    if not (j == i):
                        if self.StepLimUppr[j] <= self.StepLimLowr[i]:
                            domMsg += "\nIndex {} was ".format(i) + \
                                      "dominated by index {}".format(j) + \
                                      ". No longer active."
                            self.StepLimActv[i] = False
                            break

        if BadPntCode > 1:
            # Bad point! Return saturated version of Obj (if necessary)
            if self.mustPrnt:
                self.log.printL("> In check. Got a bad point," + \
                                BadPntMsg + "\nwith alfa = " + str(alfa) + \
                                ", Obj = "+str(Obj)+domMsg)
            # update minimum bad step, if necessary
            if alfa < self.minBadStep:
                self.minBadStep = alfa
            stat = False
        else:
            # Good point; update maximum good step, if necessary
            if alfa > self.maxGoodStep:
                self.maxGoodStep = alfa
            stat = True
        if self.mustPrnt:
            msg =  "\n  StepLimLowr = "+str(self.StepLimLowr)
            msg += "\n  StepLimUppr = " + str(self.StepLimUppr)
            msg += "\n  StepLimActv = " + str(self.StepLimActv)
            self.log.printL(msg)

        # If it is not necessary to raise the objective,
        # override it back to the original value
        if not self.mustRaisObj:
            ObjRtrn = Obj

        return BadPntCode, ObjRtrn

    def calcBase(self,sol,P0,I0,J0):
        """ Calculate the baseline values, that is, the functions P, I, J for
        the solution with no correction applied.

        Please note that there is no need for this P0, I0 and J0 to come from
        outside of this function... this is only here to avoid calculating
        these guys twice. """

        Obj0 = self.calcObj(P0,I0,J0)
        if self.mustPrnt:
            self.log.printL("> I0 = {:.4E}, ".format(I0) + \
                            "Obj0 = {:.4E}".format(Obj0))
            self.log.printL("\n> Setting Obj0 to "+str(Obj0))
        self.P0, self.I0, self.J0, self.Obj0 = P0, I0, J0, Obj0
        # store the base values on the other histories as well
        self.histP.append(self.P0)
        self.histI.append(self.I0)
        self.histObj.append(self.Obj0)

        self.best['obj'] = Obj0
        return Obj0

    def testDecr(self,alfa:float,Obj:float):
        """
        Test if the decrescent behavior of the objective as a function of the step
         is still maintained.

        This is implemented by keeping ordered lists of the steps and the
        corresponding objective values.

        This function is called in tryStep(), before actually appending the given
        "alfa" to the history.

        In principle, this function would stop being called once the monotonicity on
        the objective function was lost, but the code sometimes overrides that and
        calls this anyways.

        :return: None
        """

        isMonoLost = False # flag for recent loss of monotonicity
        ins = len(self.histStep)

        if len(self.sortIndxHistStep) > 0:
            # the test is different depending on the position of the current
            # step in respect to the list of previous steps.

            if alfa > self.histStep[self.sortIndxHistStep[-1]]:
                # step bigger than all previous steps
                pos = ins # position on the list: final one
                if self.isDecr:
                    if Obj > self.histObj[self.sortIndxHistStep[-1]]:
                        isMonoLost = True
                        self.isDecr = False
            elif alfa < self.histStep[self.sortIndxHistStep[0]]:
                # step smaller than all previous steps
                pos = 0 # position on the list: first one
                if self.isDecr:
                    if Obj < self.histObj[self.sortIndxHistStep[0]]:
                        isMonoLost = True
                        self.isDecr = False
            else:
                # step is in between. Find proper values of steps for comparison
                for i, indx_alfa_ in enumerate(self.sortIndxHistStep):
                    alfa_ = self.histStep[indx_alfa_]
                    if alfa_ <= alfa <= self.histStep[self.sortIndxHistStep[i + 1]]:
                        pos = i+1 # position on the list: the (i+1)-th one
                        break
                else:
                    # this condition should never be reached!
                    msg = "Something is very weird here. Debugging is recommended."
                    raise (Exception(msg))

                if self.isDecr:
                    if Obj < self.histObj[self.sortIndxHistStep[pos]] or \
                            Obj > self.histObj[self.sortIndxHistStep[pos-1]]:
                        isMonoLost = True
                        self.isDecr = False

            # insert the current step at the proper place in the sorted list
            self.sortIndxHistStep.insert(pos, ins)

            # Display message if and only if the monotonicity was lost right now
            if isMonoLost and self.mustPrnt:
                self.log.printL("\n> Monotonicity was just lost...")
        else:
            self.sortIndxHistStep.append(ins)

    def raisObj(self):
        """ Raise the objectives in all bad steps (unless the obj is already high),
        set the raising flag to True."""

        self.mustRaisObj = True

        ObjRtrn = self.Obj0 * 1.1
        for i in range(len(self.histStep)):
            if not self.histStat[i] and self.histObj[i] < self.Obj0:
                if self.mustPrnt:
                    msg = "  Replacing step #{}: from {} to " \
                          "{}".format(i,self.histStep[i],ObjRtrn)
                    self.log.printL(msg)
                self.histStep[i] = ObjRtrn

    def isNewStep(self, alfa):
        """Check if the step alfa is a new one (not previously tried).

        The (relative) tolerance is self.epsStep.
        """

        for alfa_ in self.histStep:
            if abs(alfa_-alfa) < self.epsStep * alfa:
                return False
        else:
            return True

    def tryNewStep(self, sol, alfa, ovrdSort=False, plotSol=False, plotPint=False):
        """ Similar to tryStep(), but making sure it is a new step

        :param sol: sgra
        :param alfa: the step to be tried
        :param ovrdSort: flag for overriding the sorting (sort it regardless)
        :param plotSol: flag for plotting the solution with the correction
        :param plotPint: flag for plotting the P_int with the correction
        :return: P, I, Obj just like tryStep()
        """

        if self.isNewStep(alfa):
            P, I, Obj = self.tryStep(sol, alfa, ovrdSort=ovrdSort,
                                     plotSol=plotSol, plotPint=plotPint)
            return P, I, Obj
        else:
            alfa0 = alfa.copy()
            keepSrch = True
            while keepSrch:
                # TODO: this scheme is more than too simple...
                alfa *= 1.01
                keepSrch = not self.isNewStep(alfa)
            if self.mustPrnt:
                msg = "\n> Changing step from {} to {}".format(alfa0,alfa) + \
                      " due to conflict."
                self.log.printL(msg)
            P, I, Obj = self.tryStep(sol, alfa, ovrdSort=ovrdSort, plotSol=plotSol,
                                     plotPint=plotPint)
            return P, I, Obj

    def tryStep(self, sol, alfa, ovrdSort=False, plotSol=False, plotPint=False):
        """ Try this given step value.

        It returns the values of P, I and Obj associated* with the tried step (alfa).

        Some additional procedures are performed, though:
            - update "best step" record ...
            - update the lower and upper bounds (via self.check())
            - assemble/update the trios for searching the minimum obj"""

        # Increment evaluation counter
        self.cont += 1
        thisStepIndx = len(self.histStep)

        # Useful for determining if this is the first good step (for setting up the
        # trios)
        noGoodStepBfor = (self.maxGoodStep < eps) # maxGoodStep is essentially zero

        if not self.isNewStep(alfa):
            self.log.printL("\n> Possible conflict over here! Debugging is "
                            "recommended...")

        if self.mustPrnt:
            self.log.printL("\n> Trying alfa = {:.4E}".format(alfa))

        # Apply the correction
        newSol = sol.copy()
        newSol.aplyCorr(alfa,self.corr)
        if plotSol:
            newSol.plotSol()

        # Get corresponding P, I, J, etc
        P,_,_ = newSol.calcP(mustPlotPint=plotPint)
        J,_,_,I,_,_ = newSol.calcJ()
        JrelRed = 100.*((J - self.J0)/alfa/self.dJdStepTheo)
        IrelRed = 100.*((I - self.I0)/alfa/self.dJdStepTheo)
        Obj = self.calcObj(P,I,J)

        # Record on the spreadsheets (if it is the case)
        if self.log.xlsx is not None:
            col, row = self.prevStepInfo['NIterGrad']+1, self.cont+1
            self.log.xlsx['step'].write(row,col,alfa)
            self.log.xlsx['P'].write(row,col,P)
            self.log.xlsx['I'].write(row,col,I)
            self.log.xlsx['J'].write(row,col,J)
            self.log.xlsx['obj'].write(row,col,Obj)

        # check if it is a good step, etc
        BadPntCode, Obj = self.check(alfa, Obj, P, newSol.pi)
        isOk = (BadPntCode == 1)

        # update trios for speeding up the objective "trissection"
        if self.isTrioSet:
            # compare current step with the trio's middle step
            if alfa < self.histStep[self.trio[1]]:
                # new step is less than the middle step. Check left step
                if alfa > self.histStep[self.trio[0]]:
                    # new step is between the left and middle step. Check middle obj
                    if Obj < self.histObj[self.trio[1]]:
                        # shift middle and right steps to the left
                        self.trio[1], self.trio[2] = thisStepIndx, self.trio[1]
                        if self.mustPrnt:
                            self.log.printL("\n  Shifting middle and right steps "
                                            "of the trio to the left.")
                    else:
                        # shift the left step to the right
                        self.trio[0] = thisStepIndx
                        if self.mustPrnt:
                            self.log.printL("\n  Changing left step of the trio.")
            else:
                # new step is greater than the middle step. Check right step
                if alfa < self.histStep[self.trio[2]]:
                    # new step is between the middle and right step. Check middle obj
                    if Obj < self.histObj[self.trio[1]]:
                        # shift left and middle steps to the right
                        self.trio[0], self.trio[1] = self.trio[1], thisStepIndx
                        if self.mustPrnt:
                            self.log.printL("\n  Shifting left and middle steps "
                                            "of the trio to the right.")
                    else:
                        # shift the right step to the left
                        self.trio[2] = thisStepIndx
                        if self.mustPrnt:
                            self.log.printL("\n  Changing right step of the trio.")

        resStr = ''
        if isOk:
            # good step, register it as such
            self.histStat.append(True)

            # update the step closest to P limit, if still applicable.
            # Just the best distance is kept, the best step is necessarily
            # StepLimLowr[1]
            if self.StepLimActv[1]:
                distP = 1. - P / self.limP
                if self.mustPrnt:
                    msg = "\n> Current distP = {}.\n".format(distP)
                else:
                    msg = ''
                if self.best['limP'] < 0.:
                    self.best['limP'] = distP
                    if self.mustPrnt:
                        msg += "  First best steplimP: {}.".format(alfa)
                else:
                    if distP < self.best['limP']:
                        self.best['limP'] = distP
                        if self.mustPrnt:
                            msg += "  Updating best steplimP: {}.".format(alfa)
                if self.mustPrnt:
                    self.log.printL(msg)
            if not self.isTrioSet:
                # setting up the trios
                if noGoodStepBfor:
                    # This is the first valid step ever!
                    # Put it in the middle of the trio
                    self.trio[1] = thisStepIndx
                    if self.mustPrnt:
                        self.log.printL("\n  Setting middle step of the trio.")
                    # entry 1 is set. Check entry 2 before switching the 'ready' flag
                    if self.trio[2] > -1:
                        # entry 2 had already been set. Switch the flag!
                        self.isTrioSet = True
                        if self.mustPrnt:
                            self.log.printL("\n  Middle entry was just set, "
                                            "trios are ready.")
                else:
                    # Not the first valid step. Maybe update middle element of trio?
                    if Obj < self.histObj[self.trio[1]]:
                        self.trio[1] = thisStepIndx
                        if self.mustPrnt:
                            self.log.printL("\n> Updating middle entry of trio.")
                    #elif alfa < self.trio['step'][2]: # ok, maybe the right one then?
                    #    self.trio['step'][2] = alfa
                    #    self.trio['obj']

            # Update best value (if it is the case)
            if self.best['obj'] > Obj:
                self.best['step'], self.best['obj'] = alfa, Obj

                if self.mustPrnt:
                    self.log.printL("\n> Best result updated!")
            #
            if self.mustPrnt:
                resStr = "\n- Results (good point):\n" + \
                         "step = {:.4E} P = {:.4E} I = {:.4E}".format(alfa,P,I) + \
                         " J = {:.7E} Obj = {:.7E}\n ".format(J,Obj) + \
                         " Rel. reduction of J = {:.1F}%,".format(JrelRed) + \
                         " Rel. reduction of I = {:.1F}%".format(IrelRed)
        else:
            # not a valid step, register it as such
            self.histStat.append(False)
            if not self.isTrioSet:
                # setting up the trios
                if self.trio[2] == -1 or alfa < self.histStep[self.trio[2]]:
                    # this means the final position of the trio gets the smallest
                    # non-valid step to be found before the trio is set.
                    self.trio[2] = thisStepIndx
                    if self.mustPrnt:
                        self.log.printL("\n  Setting/updating right step of the "
                                        "trio.")
                    # entry 2 is set. Check entry 1 before switching the 'ready' flag
                    if self.trio[1] > -1:
                        # entry 1 had already been set. Switch the flag!
                        self.isTrioSet = True
                        if self.mustPrnt:
                            self.log.printL("\n> Final entry was just set, trios are "
                                            "ready.")
            if self.mustPrnt:
                resStr = "\n- Results (bad point):\n" + \
                         "step = {:.4E} P = {:.4E} I = {:.4E}".format(alfa,P,I) + \
                         " J = {:.7E} CorrObj = {:.7E}\n ".format(J,Obj) + \
                         " J reduction eff. = {:.1F}%,".format(JrelRed) + \
                         " I reduction eff. = {:.1F}%".format(JrelRed)

        # test the non-ascending monotonicity of Obj = f(step). (or not)
        if self.isDecr or ovrdSort:
            # no need to keep testing if the monotonicity is not kept!
            if isOk:
                self.testDecr(alfa, Obj)
            elif BadPntCode == 5:
                # only violation is Obj>Obj0; keep checking for monotononicity
                self.testDecr(alfa, Obj)

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

        if self.mustPrnt:
            msg = "Steps: {}\nStats: {}\n Objs: {}\n   " \
                  "Ps: {}".format(self.histStep, self.histStat,self.histObj,
                                  self.histP)
            self.log.printL(msg)
            self.log.printL("\nsortIndxList:")
            for i, indx in enumerate(self.sortIndxHistStep):
                self.log.printL("i = {}, indx = {},"
                                " alfa = {}".format(i, indx, self.histStep[indx]))
            self.showTrios()

        return P, I, Obj#self.getLast()

    def tryStepSens(self,sol,alfa,plotSol=False,plotPint=False):
        """ Try a given step and the sensitivity as well. """

        da = self.findLimStepTol * .1
        NotAlgn = True
        P, I, Obj = self.tryStep(sol, alfa, plotSol=plotSol, plotPint=plotPint)
        outp = {}
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

    def showTrios(self):
        msg =  "\n  StepTrio:\n" + \
                 "   Indexes: {}\n".format(self.trio) + \
                 "   Steps:   {}\n".format([self.histStep[self.trio[0]],
                                            self.histStep[self.trio[1]],
                                            self.histStep[self.trio[2]]]) + \
                 "   Objs:    {}".format([self.histObj[self.trio[0]],
                                          self.histObj[self.trio[1]],
                                          self.histObj[self.trio[2]]])
        self.log.printL(msg)

    def adjsTrios(self,sol):
        """ Adjust the trios for min obj search.

        :param sol: sgra solution as usual, for trying steps
        :return: None

        This function is called right before starting the min obj search.
        When it is called, the trios are set, that is, there are 3 ascending values
        of step and their corresponding values of objectives.
        Since the middle step holds the best value so far, the objectives are
        necessarily in the V-configuration, that is,
        left_obj > middle_obj and right_obj > middle_obj.

        However, it is possible that the left and right elements are not as "tight"
        around the minimum as they could be.

        This function adjusts the steps in order to reach that condition.
        """

        if self.mustPrnt:
            self.log.printL("\n> Adjusting trios...")
            self.showTrios()

        # Check if the middle step is the best one. If not, something is very wrong!
        if abs(self.histStep[self.trio[1]]-self.best['step']) > eps:
           raise(Exception("Middle element of the trio was not the best step."
                           "\nDebug now!"))

        trio1 = self.histStep[self.trio[1]]
        for i, alfa in enumerate(self.histStep):
            if abs(alfa - trio1) > eps:
                # this 'alfa' is not the best step. Carry on
                if abs(alfa - self.histStep[self.trio[0]]) > eps:
                    # this 'alfa' is not the current first element of the trio.
                    if self.histStep[self.trio[0]] < alfa < trio1:
                        # it is closer to the center than the left element.
                        # Hence, it replaces the left element!
                        self.trio[0] = i
                        if self.mustPrnt:
                            self.log.printL("\n  Changing left element of trio to "
                                            "{}".format(alfa))
                if abs(alfa- self.histStep[self.trio[2]]) > eps:
                    # this 'alfa' is not the current last element of the trio.
                    if trio1 < alfa < self.histStep[self.trio[2]]:
                        # it is closer to the center than the right element.
                        # Hence, it replaces the right element!
                        self.trio[2] = i
                        if self.mustPrnt:
                            self.log.printL("\n  Changing right element of trio to "
                                            "{}".format(alfa))

    def piSafe(self, alfa):
        """Saturate given step with respect to the pi-condition, if necessary."""

        if alfa >= self.StepLimLowr[0]:
            alfa = self.StepLimLowr[0] * .9
            if self.mustPrnt:
                msg = "\n> Oops! That would violate the pi condition.\n" \
                      "  Let's go with {} instead.".format(alfa)
                self.log.printL(msg)
        return alfa

    def keepSrchPLim(self):
        """ Decide to keep looking for P step limit or not.


        keepLookCode is 0 for keep looking. Other values correspond to different
        exit conditions:

        1 - P limit is not active
        2 - upper and lower bounds for P step limit are within tolerance
        3 - monotonic (descent) behavior was lost, obj min is before P limit
        4 - too many objective function evaluations
        5 - P limit is within tolerance"""

        if not self.StepLimActv[1]:
            keepLoopCode = 1
        elif (self.StepLimUppr[1] - self.StepLimLowr[1]) < \
                self.StepLimLowr[1] * self.tolStepLimP:
            keepLoopCode = 2
        elif not self.isDecr:
            keepLoopCode = 3
        elif self.cont >= self.stopNEvalLim:
            keepLoopCode = 4
        elif self.tolLimP > self.best['limP'] > 0.:
            keepLoopCode = 5
        else:
            keepLoopCode = 0

        self.pLimSrchStopMotv = keepLoopCode

    def srchStep(self,sol):
        """The ultimate step search.

        First of all, the algorithm tries to find a step value for which
        P = limP; called AlfaLimP. This is because of the almost linear
        behavior of P with step in loglog plot.
        This AlfaLimP search is conducted until the value is found (with a
        given tolerance, of course) or if Obj > Obj0 before P = limP.

        The second part concerns the min Obj search itself. Ideally, some
        step values have already been tried in the previous search.
        The search is performed by "trissection"; we have 3 ascending values
        of step with the corresponding values of Obj. If the middle point
        corresponds to the lowest value (V condition), the trissection
        begins. Else, the search point moves left of right until the V
        condition is met.
        """

        if self.mustPrnt:
            self.log.printL("\n> Entering ULTIMATE STEP SEARCH!\n")

        # PART 1: alfa_limP search

        # 1.0: Get the stop motive of the previous grad iteration, if it exists
        prevStopMotv = self.prevStepInfo.get('stopMotv', -1)

        # 1.0.1: Get the best step from the previous grad iteration
        prevBest = self.prevStepInfo.get('best', None)
        if prevBest is not None:
            alfa = prevBest['step']
            if self.mustPrnt:
                msg = "\n> First of all, let's try the best step of " \
                      "the previous run,\n  alfa = {}".format(alfa)
                self.log.printL(msg)
            self.tryStep(sol, alfa)

        # 1.0.2: Decide the mode for P limit search, default or not
        if prevStopMotv == 1 or prevStopMotv == 2 or prevStopMotv == 3:
            isPLimSrchDefault = False
        else:
            isPLimSrchDefault = True

        # 1.0.3: If the previous P limit fit is to be used, check validity first
        if not isPLimSrchDefault:
            a = self.prevStepInfo['PRegrAngCoef']
            #b = self.prevStepInfo['PRegrLinCoef']
            if a < eps:
                # Fit is not good, override and return to default
                if self.mustPrnt:
                    self.log.printL("\n> Overriding P limit search to default"
                                    " mode.")
                isPLimSrchDefault = True

        # 1.0.4: calculate P target (just below P limit, but still within tolerance)
        P_targ = self.limP * (1. - self.tolLimP / 2.)
        # ^ This is so that abs(P_targ / self.limP - 1.) = tolLimP/2
        logP_targ = numpy.log(P_targ)

        if isPLimSrchDefault:
            # 1.1: Start with alfa = 1, unless the pi conditions don't allow it
            if self.StepLimActv[0]:
                alfa1 = min([1., self.StepLimLowr[0]*.9])
            else:
                alfa1 = 1.

            if self.mustPrnt:
                self.log.printL("\n> Going for default P limit search.")
            P1, I1, Obj1 = self.tryNewStep(sol, alfa1)
            alfa1 = self.histStep[-1]

            if self.cont > 0:
                # 1.2: If more than 1 step has been tried, make a line
                y1, y2 = numpy.log(self.histP[-1]), numpy.log(self.histP[-2])
                x1 = numpy.log(self.histStep[-1])
                x2 = numpy.log(self.histStep[-2])
                a = (y2 - y1) / (x2 - x1)  # angular coefficient
                b = y1 - x1 * a            # linear coefficient
                self.PRegrAngCoef, self.PRegrLinCoef = a, b  # store them
                alfaLimP = numpy.exp((logP_targ - b) / a)
                if self.mustPrnt:
                    msg = "\n> Ok, that did not work. " \
                          "Let's try alfa = {}\n  (linear fit from" \
                          " the latest tries).".format(alfaLimP)
                    self.log.printL(msg)
            else:
                # 1.2 (alt): first model for alfa with P(0) and P(alfa1) only
                # was actually very bad. Let's go with +- 10%...
                if P1 > P_targ:
                    alfaLimP = alfa1 * 0.9
                    if self.mustPrnt:
                        msg = "\n> Ok, that did not work. " \
                              "Let's try alfa = {} (10% lower).".format(alfaLimP)
                        self.log.printL(msg)
                else:
                    alfaLimP = alfa1 * 1.1
                    if self.mustPrnt:
                        msg = "\n> Ok, that did not work. " \
                              "Let's try alfa = {} (10% higher).".format(alfaLimP)
                        self.log.printL(msg)
            if self.mustPrnt:
                msg = "\n> AlfaLimP = {} (by first model**)".format(alfaLimP)
                self.log.printL(msg)

            # 1.3: update P limit search status
            self.keepSrchPLim()
        else:
            if self.pLimSrchStopMotv == 0:
                # 1.1 (alt): Start with the fit provided by the last grad step
                a = self.prevStepInfo['PRegrAngCoef']
                b = self.prevStepInfo['PRegrLinCoef']
                alfaLimP = numpy.exp((logP_targ - b) / a)
                # ... unless the pi conditions don't allow it
                if self.StepLimActv[0]:
                    alfaLimP = min([alfaLimP, self.StepLimLowr[0] * .9])
                if self.mustPrnt:
                    msg = "\n> Using the fit from the previous step search,\n" \
                          "  (a = {}, b = {}),\n  let's try alfa".format(a, b) + \
                          " = {} to hit P target = {}.".format(alfaLimP, P_targ)
                    self.log.printL(msg)
                # Try alfaLimP, but there can be a conflict...
                P, I, Obj = self.tryNewStep(sol, alfaLimP)

                # 1.2 (alt) Checking if we must proceed...
                self.keepSrchPLim()
            else:
                alfaLimP, P = 1., 1. # These values will not be used.

            # 1.3 (alt): The fit was not that successful... carry on
            if self.pLimSrchStopMotv == 0:
                if self.cont > 0:
                    # 1.3.1 (alt1): If more than 1 step was tried, make a line!
                    y1, y2 = numpy.log(self.histP[-1]), numpy.log(self.histP[-2])
                    x1 = numpy.log(self.histStep[-1])
                    x2 = numpy.log(self.histStep[-2])
                    a = (y2-y1)/(x2-x1) # angular coefficient
                    b = y1 - x1 * a     # linear coefficient
                    self.PRegrAngCoef, self.PRegrLinCoef = a, b # store them
                    alfaLimP = numpy.exp((logP_targ - b) / a)
                    if self.mustPrnt:
                        msg = "\n> Ok, that did not work. "\
                              "Let's try alfa = {}\n  (linear fit from" \
                              " the latest tries).".format(alfaLimP)
                        self.log.printL(msg)
                else:
                    # 1.3.1 (alt2): Unable to fit, let's go with +/-10%...
                    if P > P_targ:
                        alfaLimP *= 0.9 # 10% lower
                        msg = "\n> Ok, that did not work. Let's try" \
                              " alfa = {} (10% lower).".format(alfaLimP)
                    else:
                        alfaLimP *= 1.1 # 10% higher
                        msg = "\n> Ok, that did not work. Let's try" \
                              " alfa = {} (10% higher).".format(alfaLimP)
                    if self.mustPrnt:
                        self.log.printL(msg)

            # 1.4 (alt): update P limit search status
            self.keepSrchPLim()

        # Proceed with this value, unless the pi conditions don't allow it
        alfaLimP = self.piSafe(alfaLimP)

        # 1.3: Main loop for P limit search
        while self.pLimSrchStopMotv == 0:
            # 1.3.1: Safety in alfaLimP: make sure the guess is in the right
            # interval
            if alfaLimP > self.minBadStep or alfaLimP < self.StepLimLowr[1]:
                alfaLimP = .5 * (self.StepLimLowr[1] + self.minBadStep)
                msg = "\n> Model has retrieved a bad guess for alfaLimP..." \
                      "\n  Let's go with {} instead.".format(alfaLimP)
                self.log.printL(msg)

            # 1.3.2: try newest step value
            alfaLimP = self.piSafe(alfaLimP)
            self.tryStep(sol, alfaLimP)

            # 1.3.3: keep in loop or not
            self.keepSrchPLim()

            # 1.3.4: perform regression (if it is the case)
            if self.pLimSrchStopMotv == 0:
                # 1.3.4.1: assemble log arrays for regression
                logStep = numpy.log(numpy.array(self.histStep[1:]))
                logP = numpy.log(numpy.array(self.histP[1:]))
                n = len(self.histStep[1:])
                #logStep = numpy.log(numpy.array(self.histStep))
                #logP = numpy.log(numpy.array(self.histP))
                #n = len(self.histStep)

                # 1.3.4.2: perform actual regression
                X = numpy.ones((n, 2))
                X[:, 0] = logStep
                # weights for regression: priority to points closer to the P limit
                W = numpy.zeros((n, n))
                logLim = numpy.log(self.limP)
                for i, logP_ in enumerate(logP):
                    W[i, i] = 1. / (logP_ - logLim) ** 2.  # inverse
                mat = numpy.dot(X.transpose(), numpy.dot(W, X))
                a, b = numpy.linalg.solve(mat,
                                          numpy.dot(X.transpose(),
                                                    numpy.dot(W, logP)))
                # store the information to the object
                self.PRegrAngCoef, self.PRegrLinCoef = a, b

                # 1.3.4.3: calculate alfaLimP according to current regression
                pass
                # (P target just below the P limit, in order to try to get a point
                # in the tolerance)
                alfaLimP = numpy.exp((logP_targ - b) / a)
                if self.mustPrnt:
                    msg = "\n> Performed fit with n = {} points:\n" \
                          "    Steps: {}\n      Ps: {}\n Weights: {}\n" \
                          "  a = {}, b = {}\n" \
                          "  alfaLimP = {}".format(n, numpy.exp(logStep),
                                                   numpy.exp(logP), W,
                                                   a, b, alfaLimP)
                    self.log.printL(msg)

                # 1.3.4.4: Saturate this guess with respect to the pi condition if so
                alfaLimP = self.piSafe(alfaLimP)
                # DEBUG PLOT
                # plt.loglog(self.histStep, self.histP, 'o', label='P')
                # model = numpy.exp(b) * numpy.array(self.histStep) ** a
                # plt.loglog(self.histStep, model, '--', label = 'model')
                # plt.loglog(self.histStep,
                #            self.limP * numpy.ones_like(self.histP), label='limP')
                # plt.loglog(alfaLimP, self.limP, 'x', label='Next guess')
                # plt.grid()
                # plt.title("P vs step behavior")
                # plt.xlabel('Step')
                # plt.ylabel('P')
                # plt.legend()
                # plt.show()
        if self.mustPrnt:
            msg = '\n> Stopping alfa_limP search: '
            if self.pLimSrchStopMotv == 1:
                msg += "P limit is no longer active." + \
                       "\n  Local min is likely before alfa_limP."
            elif self.pLimSrchStopMotv == 2:
                msg += "P limit step tolerance is met."
            elif self.pLimSrchStopMotv == 3:
                msg += "Monotonicity in objective was lost." + \
                       "\n  Local min is likely before alfa_limP."
            elif self.pLimSrchStopMotv == 4:
                msg += "Too many Obj evals. Debugging is recommended."
            elif self.pLimSrchStopMotv == 5:
                msg += "P limit tolerance is met."
            self.log.printL(msg)

        # 1.4: If the loop was abandoned because of too many evals, leave now.
        alfa = self.best['step']
        if self.pLimSrchStopMotv == 4:
            msg = "\n> Leaving step search due to excessive evaluations." + \
                  "\n  Debugging is recommended."
            self.log.printL(msg)
            self.stopMotv = 3
            return alfa

        # PART 2: min Obj search

        # 2.0: Bad condition: all tested steps were bad (not valid)
        if alfa < eps: # alfa is essentially 0 in this case
            if self.isTrioSet:
                raise(Exception("grad_sgra: No good step but the trios are set."
                                "\nDebug this now!"))
            if self.mustPrnt:
                msg = "\n\n> Bad condition! None of the tested steps was valid."
                self.log.printL(msg)
            # 2.0.1: Get latest tried step
            #Obj, alfa, isOk  = self.histObj[-1], self.histStep[-1], self.histStat[-1]
            isOk = False
            alfa = self.histStep[self.trio[2]]

            # 2.0.2: Go back a few times. We will need 3 points
            while not isOk or self.cont < 2:
                if self.mustPrnt:
                    self.log.printL("  Going back by {}...".format(self.ffRate))
                alfa /= self.ffRate
                self.tryStep(sol,alfa)
                isOk = self.histStat[-1] # get its status

            # The trios are already automatically assembled!

            if self.mustPrnt:
                self.log.printL("\n> Finished looking for valid objs.")
                self.showTrios()
            self.isTrioSet = True

            # DEBUG PLOTS
            # plt.semilogx(self.histStep, self.histObj, 'o', label='Obj')
            # plt.semilogx(self.histStep,
            #              self.Obj0 * numpy.ones_like(self.histStep),
            #              label='Obj0')
            # plt.grid()
            # plt.title("Obj vs step behavior")
            # plt.xlabel('Step')
            # plt.ylabel('Obj')
            # plt.legend()
            # plt.show()

        # 2.1 shortcut: step limit was found, objective seems to be
        # descending with step. Abandon ship with the best step so far and
        # that's it.
        alfa, Obj = self.best['step'], self.best['obj']
        if self.isDecr and self.StepLimActv[1]:
            # calculate the gradient at this point
            if self.mustPrnt:
                self.log.printL("\n> It seems the minimum obj is after the P "
                                "limit.\n  Testing gradient...")
            # TODO: this hardcoded value can cause issues in super sensitive
            #  problems...
            alfa_ = alfa * .999
            P_,I_,Obj_ = self.tryStep(sol,alfa_)
            grad = (Obj-Obj_)/(alfa-alfa_)

            if grad < 0.:
                if self.mustPrnt:
                    msg = "\n> Objective seems to be descending with step.\n" + \
                          "  Leaving calcStepGrad with alfa = {}\n".format(alfa)
                    self.log.printL(msg)
                self.stopMotv = 2 # step limit hit
                return alfa
            else:
                if self.mustPrnt:
                    self.log.printL("\n> Positive gradient ({})...".format(grad))

        if self.mustPrnt:
            self.log.printL("\n> Starting min obj search...")

        # for now on, it is useful to raise the objective if the step is not valid
        self.raisObj()

        # 2.2: If the shortcut 2.1 was not taken, we must set the trios
        if not self.isTrioSet:
            if self.mustPrnt:
                self.log.printL("\n> Let's finish setting the trios. Here they are"
                                " so far:")
                self.showTrios()

            # 2.2.1: Get index for best step so far
            alfa = self.best['step']
            for ind in self.sortIndxHistStep:
                alfa_ = self.histStep[self.sortIndxHistStep[ind]]
                if abs(alfa_-alfa) < eps:
                    indxAlfa = self.sortIndxHistStep[ind]
                    break
            else:
                raise(Exception("grad_sgra: Latest step was not found. Debug now!"))

            # 2.2.2: Special procedures in case the best step is at the
            # beginning or at the end of the list
            n = len(self.sortIndxHistStep) - 1 # max index

            indxAlfa_m, indxAlfa_p = 0, 0 # just to shut PyCharm up...
            if ind > 0: # there is a lesser step; use it
                indxAlfa_m = self.sortIndxHistStep[ind - 1]
            # TODO: it is not necessary to use a good step for trio[2].
            if ind < n: # there is a greater step; use it
                indxAlfa_p = self.sortIndxHistStep[ind + 1]

            if ind <= 0.: # there is no lesser step, try 1% lower
                alfa_m = alfa * .99
                if self.mustPrnt:
                    self.log.printL("\n> No lesser valid step; going for a new one,"
                                    " 1% below the best so far ({}).".format(alfa))
                # a new step will be tried; this will be its index
                indxAlfa_m = len(self.histStep)
                self.tryStep(sol, alfa_m)
            if ind >= n: # there is no greater step, try 1% higher
                alfa_p = alfa * 1.01
                if self.mustPrnt:
                    self.log.printL("\n> No greater valid step; going for a new one,"
                                    " 1% above the best so far ({}).".format(alfa))
                # a new step will be tried; this will be its index
                indxAlfa_p = len(self.histStep)
                self.tryStep(sol, alfa_p)

            # 2.2.3: In any case, assemble the trios. Adjustments are probably
            # necessary.
            if self.mustPrnt:
                self.log.printL("\n  Trios before:")
                self.showTrios()
            self.trio = [indxAlfa_m, indxAlfa, indxAlfa_p]
            self.isTrioSet = True
            if self.mustPrnt:
                self.log.printL("\n> Trios are set, maybe some adjustments are still"
                                " necessary.")
                self.showTrios()

        # 2.3:  Adjust the trios (if necessary)
        self.adjsTrios(sol)

        # 2.4: Final step search by "trissection"
        pass #just to break the comment
        # up to this point, stepTrio has three ascending values of step,
        # and objTrio has the corresponding values of Obj, with
        #   objTrio[1] < min(objTrio[0], objTrio[2])

        # 2.4.-1: Estimation of the number of evaluations until convergence
        n = int(numpy.ceil(numpy.log(self.tolStepObj /
             (self.histStep[self.trio[2]] - self.histStep[self.trio[0]]) ) /
             numpy.log((numpy.sqrt(5.) - 1.) / 2.)))
        if n < 0: # Apparently we are about to converge!
            n = 0
        if self.mustPrnt:
            self.log.printL("\n> According to the Golden Section approximation, "
                            "n = {} evals until convergence...".format(n))
        self.cont_exp = n + self.cont

        # 2.4.0: Move left or right depending on which side is the largest
        pass #just to break the comment
        # if self.trio['step'][1] / self.trio['step'][0] > \
        #     self.trio['step'][2] / self.trio['step'][1]:
        #     # left side is largest, move left
        #     leftRght = True  # indicator of left (True) or right (False) movement
        # else:
        #     # right side is largest, move right
        #     leftRght = False # indicator of left (True) or right (False) movement
        leftRght = False # indicator of left (True) or right (False) movement
        if self.mustPrnt:
            self.showTrios()
        while self.histStep[self.trio[2]] - self.histStep[self.trio[0]] > \
                self.tolStepObj * self.histStep[self.trio[1]] and \
                self.cont < self.stopNEvalLim:
            # 2.4.1: Narrow the boundaries either left or right
            # (alternatively)
            if leftRght:
                alfa = .5 * (self.histStep[self.trio[0]] +
                             self.histStep[self.trio[1]])
            else:
                alfa = .5 * (self.histStep[self.trio[2]] +
                             self.histStep[self.trio[1]])

            # 2.4.2: Try the new step. Updating the trios is done as well
            self.tryStep(sol,alfa)

            # DEBUG PLOT
            # plt.semilogx(self.histStep, self.histObj, 'o', label='Obj')
            # plt.semilogx(self.histStep,
            #              self.Obj0 * numpy.ones_like(self.histStep),
            #              label='Obj0')
            # plt.grid()
            # plt.title("Obj vs step behavior")
            # plt.xlabel('Step')
            # plt.ylabel('Obj')
            # plt.legend()
            # plt.show()

            # 2.4.4: Decide the search direction again
            # if self.trio['step'][1] / self.trio['step'][0] > \
            #         self.trio['step'][2] / self.trio['step'][1]:
            #     # left side is largest, move left
            #     leftRght = True
            #     # indicator of left (True) or right (False) movement
            # else:
            #     # right side is largest, move right
            #     leftRght = False
            #     # indicator of left (True) or right (False) movement
            leftRght = not leftRght # reverse search direction
            #input(">> ")

        # 2.5: End conditions
        if self.cont >= self.stopNEvalLim:
            # too many evals, complain and return the best so far
            msg = "\n> Leaving step search due to excessive evaluations."
            self.log.printL(msg)
            if self.mustPrnt:
                self.showTrios()
            self.stopMotv = 3 # too many evals
        else:
            # minimum was actually found!
            if self.mustPrnt:
                gr_p = (self.histObj[self.trio[2]] - self.histObj[self.trio[1]]) / \
                       (self.histStep[self.trio[2]] - self.histStep[self.trio[1]])
                gr_n = (self.histObj[self.trio[1]] - self.histObj[self.trio[0]]) / \
                       (self.histStep[self.trio[1]] - self.histStep[self.trio[0]])
                grad = gr_p - gr_n
                msg = "\n> Local obj minimum was found.\n  Gradients: grad_p = "\
                      "{}, grad_n = {}, grad = {}".format(gr_p,gr_n,grad)
                self.log.printL(msg)
                self.showTrios()
            self.stopMotv = 1 # local min found!
        return self.best['step']#self.histStep[self.trio[1]]

    def endPrntPlot(self,alfa,mustPlot=False):
        """Final plots and prints for this run of calcStepGrad. """
        P, I, Obj = 1., 1., 1.
        if mustPlot or self.mustPrnt:
            # Get index for applied step
            k = 0
            for k in range(len(self.histStep)):
                if abs(self.histStep[k]-alfa) < eps:
                    break

            # Get final values of P, I and Obj
            P, I, Obj = self.histP[k], self.histI[k], self.histObj[k]
        #

        # plot the history of the tried alfas, and corresponding P/I/Obj's
        if mustPlot:

            # Plot history of P
            a = min(self.histStep)
            cont = 0
            if a == 0.:
                newhistStep = sorted(self.histStep)
                while a == 0.0:
                    cont += 1
                    a = newhistStep[cont]
                #
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
            # noinspection PyTypeChecker
            plt.semilogx(self.histStep, 100.0*(self.histI/self.I0-1.0), 'o',
                         label='I(alfa)')
            plt.semilogx(alfa, 100.0*(I/self.I0-1.0), 's',
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
            # noinspection PyTypeChecker
            plt.semilogx(self.histStep, 100.0*(self.histObj/self.Obj0-1.0),
                         'o', label='Obj(alfa)')
            plt.semilogx(alfa, 100.0*(Obj/self.Obj0-1.0), 's',
                         label='Chosen value')
            #minStep, maxStep = min(self.histStep), max(self.histStep)
            #steps = numpy.linspace(minStep,maxStep,num=1000)
            #dJTheoPerc = (100.0 * self.dJdStepTheo / self.Obj0) * steps
            #plt.semilogx(steps, dJTheoPerc, label='TheoObj(alfa)')
            # noinspection PyTypeChecker
            dJTheoPerc = (100.0 * self.dJdStepTheo) * (self.histStep/self.Obj0)
            plt.semilogx(self.histStep, dJTheoPerc,'o', label='TheoObj(alfa)')
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

        if self.mustPrnt:
            dIp = 100.0 * (I/self.I0 - 1.0)
            dObjp = 100.0 * (Obj/self.Obj0 - 1.0)

            msg = "\n> Chosen alfa = {:.4E}\n> I0 = {:.4E}, I = {:.4E}" \
                  ", dI = {:.4E} ({:.4E})%".format(alfa,self.I0,I,I-self.I0,dIp) + \
                  "\n> Obj0 = {:.4E}, Obj = {:.4E}".format(self.Obj0,Obj) + \
                  ", dObj = {:.4E} ({:.4E})%".format(Obj-self.Obj0, dObjp)
            self.log.printL(msg)

            #self.log.printL(">  Number of objective evaluations: " + \
            #                str(self.cont+1))

            # NEW STUFF:
            #self.log.printL("  HistStep = " + str(self.histStep))
            #self.log.printL("  HistI = " + str(self.histI))
            #self.log.printL("  HistObj = " + str(self.histObj))
            #self.log.printL("  HistP = " + str(self.histP))
            corr = self.corr # get the correction
            prevStepInfo = self.prevStepInfo # get previous step info
            self.corr = 'OMITTED FOR PRINT PURPOSES' # self-explanatory
            self.prevStepInfo = 'OMITTED FOR PRINT PURPOSES'  # self-explanatory
            self.log.printL("\n> State of stepMan at the end of the run:\n")
            self.log.pprint(self.__dict__)
            self.corr = corr # put it back
            self.prevStepInfo = prevStepInfo
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
            self.plotCat(fPF, piIsTime=piIsTime, color='r',
                         labl='Penalty function')
        else:
            f = argout[0]
            self.plotCat(f, piIsTime=piIsTime, color='b', labl='Total cost')
    else:
        self.plotCat(argout, piIsTime=piIsTime, color='b', labl='Total cost')
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

def plotQRes(self,args,mustSaveFig=True,addName=''):
    """ Generic plots of the Q residuals. """

    iterStr = "\n(grad iter #" + str(self.NIterGrad) + \
                  ", rest iter #"+str(self.NIterRest) + \
                  ", event #" + str(int((self.EvntIndx+1)/2)) + ")"
    # Qx error plot
    plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
    nm2 = self.n+2
    plt.subplot2grid((nm2,1),(0,0))
    self.plotCat(args['accQx'],color='b',piIsTime=False)
    plt.grid(True)
    plt.ylabel("Accumulated int.")
    titlStr = "Qx = int || dlam - f_x + phi_x^T*lam || " + \
              "= {:.4E}".format(args['Qx'])
    titlStr += iterStr
    plt.title(titlStr)
    plt.subplot2grid((nm2,1),(1,0))
    self.plotCat(args['normErrQx'],color='b',piIsTime=False)
    plt.grid(True)
    plt.ylabel("Integrand of Qx")
    errQx = args['errQx']
    for i in range(self.n):
        plt.subplot2grid((nm2,1),(i+1,0))
        self.plotCat(errQx[:,i,:],piIsTime=False)
        plt.grid(True)
        plt.ylabel("ErrQx_"+str(i))
    plt.xlabel("t [-]")

    if mustSaveFig:
        self.savefig(keyName=('Qx'+addName),fullName='Qx')
    else:
        plt.show()
        plt.clf()

    # Qu error plot
    plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
    mm2 = self.m+2
    plt.subplot2grid((mm2,1),(0,0))
    self.plotCat(args['accQu'],color='b',piIsTime=False)
    plt.grid(True)
    plt.ylabel("Accumulated int.")
    titlStr = "Qu = int || f_u - phi_u^T*lam || = {:.4E}".format(args['Qu'])
    titlStr += iterStr
    plt.title(titlStr)
    plt.subplot2grid((mm2,1),(1,0))
    self.plotCat(args['normErrQu'],color='b',piIsTime=False)
    plt.grid(True)
    plt.ylabel("Integrand")
    errQu = args['errQu']
    for i in range(self.m):
        plt.subplot2grid((mm2,1),(i+2,0))
        self.plotCat(errQu[:,i,:],color='k',piIsTime=False)
        plt.grid(True)
        plt.ylabel("Qu_"+str(i))
    plt.xlabel("t [-]")
    if mustSaveFig:
        self.savefig(keyName=('Qu'+addName),fullName='Qu')
    else:
        plt.show()
        plt.clf()


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
        self.plotCat(errQp[:,j,:],color='k',piIsTime=False)
        plt.grid(True)
        plt.ylabel("ErrQp, j ="+str(j))
    plt.xlabel("t [-]")
    if mustSaveFig:
        self.savefig(keyName=('Qp'+addName),fullName='Qp')
    else:
        plt.show()
        plt.clf()

#@avoidRepCalc(fieldsTuple=('J','J_Lint','J_Lpsi','I','Iorig','I_pf')) # not worth it
def calcJ(self):
    #self.log.printL("\nIn calcJ.")
    N, s = self.N, self.s
    #x = self.x

    #phi = self.calcPhi()
    psi = self.calcPsi()
    lam = self.lam
    mu = self.mu
    #dx = ddt(x,N)
    I,Iorig,Ipf = self.calcI()

    func = -self.calcErr() #like this, it's dx-phi
    vetL = numpy.einsum('nis,nis->ns',lam,func)

    # Perform the integration of Lint array by Simpson's method
    vetIL = numpy.empty(s)
    Lint = 0.0
    for arc in range(s):
        vetIL[arc] = simp(vetL[:,arc],N)
        Lint += vetIL[arc]
    self.log.printL("L components, by arc: "+str(vetIL))
    #Lint = vetIL[N-1,:].sum()
    Lpsi = mu.transpose().dot(psi)
    L = Lint + Lpsi

    J_Lint = Lint
    J_Lpsi = Lpsi
    J_I = I
    J = L + J_I
    strJs = "J = {:.6E}, J_Lint = {:.6E}, ".format(J,J_Lint) + \
            "J_Lpsi = {:.6E}, J_I = {:.6E}".format(J_Lpsi,J_I)
    self.log.printL(strJs)

    return J, J_Lint, J_Lpsi, I, Iorig, Ipf

@avoidRepCalc(fieldsTuple=('Q', 'Qx', 'Qu', 'Qp', 'Qt'))
def calcQ(self,mustPlotQs=False,addName=''):
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

    errQx = numpy.empty((N,n,s))
    coefList = simp([],N,onlyCoef=True)
    z = numpy.empty(2*n*s)

    # this computes fu - (phiu^T)*lam, for all times and arcs
    errQu = fu - numpy.einsum('njis,njs->nis', phiu, lam)
    # this computes fp - (phip^T)*lam, for all times and arcs
    errQp = fp - numpy.einsum('njis,njs->nis', phip, lam)
    # this computes |errQu|², for all times and arcs
    normErrQu = numpy.einsum('nis,nis->ns', errQu, errQu)

    # assemble z term
    # TODO: this can probably be rewritten without the for loop...
    for arc in range(s):
        z[2*arc*n : (2*arc+1)*n] = -lam[0,:,arc]
        z[(2*arc+1)*n : (2*arc+2)*n] = lam[N-1,:,arc]

    # calculate Qx separately. In this way, the derivative evaluation is
    # adequate with the trapezoidal integration method

    # this computes (phix ^ T) * lam, for all times and arcs
    phixTlam = numpy.einsum('njis,njs->nis',phix,lam)
    med = (lam[1,:,:]-lam[0,:,:])/dt -.5*(fx[0,:,:]+fx[1,:,:]) + \
            .5 * (phixTlam[0,:,:] + phixTlam[1,:,:])

    errQx[0,:,:] = med
    aux = numpy.empty((self.N-1,self.n,self.s))
    aux[1:,:,:] = 2. * (lam[2:,:,:] - lam[1:(self.N-1),:,:]) / self.dt + \
                  -(fx[2:,:,:] + fx[1:(self.N-1),:,:]) + \
                  phixTlam[2:, :, :] + phixTlam[1:(self.N-1), :, :]
    aux[0,:,:] = med

    # this computes errQx[k+1,:,:] = aux[k,:,:] - errQx[k,:,:], for all k
    errQx[1:,:,:] = numpy.einsum('ij,jks->iks',self.CalcErrMat,aux)
    # this computes |errQx|², for all times and arcs
    normErrQx = numpy.einsum('nis,nis->ns', errQx, errQx)

    if mustPlotQs:
        accQx = numpy.empty((N, s))
        accQu = numpy.empty((N, s))
        auxVecIntQp = numpy.zeros((p, s))
        for arc in range(s):
            accQx[0,arc] = normErrQx[0,arc] * coefList[0]
            accQu[0,arc] = normErrQu[0,arc] * coefList[0]
            auxVecIntQp[:,arc] += errQp[0,:,arc] * coefList[0]
            for k in range(1,N):
                accQx[k,arc] = accQx[k-1,arc] + normErrQx[k,arc] * coefList[k]
                accQu[k,arc] = accQu[k-1,arc] + normErrQu[k,arc] * coefList[k]
                auxVecIntQp[:,arc] += errQp[k,:,arc] * coefList[k]
            #
            Qx += accQx[N-1,arc]; Qu += accQu[N-1,arc]
        #

        # Correct the accumulation
        for arc in range(1,s):
            accQx[:,arc] += accQx[-1,arc-1]
            accQu[:,arc] += accQu[-1,arc-1]

        resVecIntQp = numpy.zeros(p)
        for arc in range(s):
            resVecIntQp += auxVecIntQp[:, arc]
    else:
        # this computes sum(coefList[k]*normErrQx[k,arc])
        Qx = numpy.einsum('n,ns->', coefList, normErrQx)
        # this computes sum(coefList[k]*normErrQu[k,arc])
        Qu = numpy.einsum('n,ns->', coefList, normErrQu)
        # this computes sum(coefList[k]*errQp[k,:,arc])
        resVecIntQp = numpy.einsum('n,nis->i',coefList, errQp)

    resVecIntQp += psip.transpose().dot(mu)
    Qp = resVecIntQp.transpose().dot(resVecIntQp)

    errQt = z + psiy.transpose().dot(mu)
    Qt = errQt.transpose().dot(errQt)

    Q = Qx + Qu + Qp + Qt
    self.log.printL("Q = {:.4E}".format(Q)+": Qx = {:.4E}".format(Qx)+\
          ", Qu = {:.4E}".format(Qu)+", Qp = {:.7E}".format(Qp)+\
          ", Qt = {:.4E}".format(Qt))

    if mustPlotQs:
        args = {'errQx':errQx, 'errQu':errQu, 'errQp':errQp, 'Qx':Qx, 'Qu':Qu,
                'normErrQx':normErrQx, 'normErrQu':normErrQu,
                'resVecIntQp':resVecIntQp, 'accQx':accQx, 'accQu':accQu}
        self.plotQRes(args,addName=addName)


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

    if self.dbugOptGrad['pausCalcQ']:
        self.log.prom("calcQ in debug mode. Press any key to continue...")

    return Q, Qx, Qu, Qp, Qt

def calcStepGrad(self,corr,alfa_0,retry_grad,stepMan):
    """This method calculates the gradient step.
     It is a key function, called in each gradient phase.

     The first major bifurcation is if this Gradient phase is actually a "retry" from
     a previous gradient phase. If it is the case, all this function does is to return
     a lesser value of the step, hoping that after restoration, the I<I_old condition
     will hold. No further checks are performed because the step value is lower than a
     previous "successful" value in terms of all the other conditions.

     Outside of this condition, the main algorithm applies.

     The main idea is to find the step value (alpha value) so that the objective
     function is minimized, subject to the P <= limP condition (P condition).
     Actually, there are other restrictions that apply as well and are not
     automatically satisfied when the P condition is, so they must be checked as well.

     Miele (2003) recommends using the J function as the objective instead of the I
     function, because the J function somehow incorporates the I and P functions.
     Nevertheless, the P value (and the other constraints) still has to be monitored.
     Naturally, the idea is to lower the value of I, so the objective function is
     expected to be lowered as well.

     This search can be stopped for different reasons, either the local
     minimum in Obj is found, or the step limit was hit (with a negative gradient
     on Obj), or there have been too many evaluations of the Objective function
     (this is to prevent infinite loops).

     After creating the stepMngr object, the stepLimit is found, which is the highest
     value of step so that the constraints are still respected. Then, the program
     proceeds to making successive quadratic interpolations until a local minimum is
     found. At that point the program halts and the best feasible value of step is
     returned.

     """

    self.log.printL("\nIn calcStepGrad.\n")
    # flag for printing all the steps tried.
    prntCond = self.dbugOptGrad['prntCalcStepGrad']

    if retry_grad:
        # Retrying a same gradient correction
        stepFact = 0.1#0.9 #
        if prntCond:
            self.log.printL("\n> Retrying alfa." + \
                            " Base value: {:.4E}".format(alfa_0))
        # LOWER alfa!
        alfa = stepFact * alfa_0
        if prntCond:
            self.log.printL("\n> Let's try alfa" + \
                            " {:.1F}% lower.".format(100.*(1.-stepFact)))
        P, I, Obj = stepMan.tryStep(self,alfa)

        while Obj > stepMan.Obj0:
            alfa *= stepFact
            if prntCond:
                msg = "\n> Let's try alfa" + \
                      " {:.1F}% lower.".format(100.*(1.-stepFact))
                self.log.printL(msg)
            P, I, Obj = stepMan.tryStep(self,alfa)

        if prntCond:
            self.log.printL("\n> Ok, this value should work.")
        stepMan.stopMotv = 0
    else:
        # Get initial status (no correction applied)
        if prntCond:
            self.log.printL("\n> alfa = 0.0")
        P0,_,_ = self.calcP()
        J0,_,_,I0,_,_ = self.calcJ()
        # Get all the constants
        ctes = {'tolP': self.tol['P'], #from sgra already,
                'limP': self.constants['GSS_PLimCte'] * self.tol['P'],
                'stopStepLimTol': self.constants['GSS_stopStepLimTol'],
                'stopObjDerTol': self.constants['GSS_stopObjDerTol'],
                'stopNEvalLim': self.constants['GSS_stopNEvalLim'],
                'findLimStepTol': self.constants['GSS_findLimStepTol'],
                'piLowLim': self.restrictions['pi_min'],
                'piHighLim': self.restrictions['pi_max']}

        if stepMan is None:
            prevStepInfo = {}
        else:
            prevStepInfo = stepMan.__dict__
            # remove unuseful entries to shorten the print, and please no recursion!
            for key in ['corr','prevStepInfo']:
                prevStepInfo.pop(key)

        prevStepInfo['NIterGrad'] = self.NIterGrad

        # Create new stepManager object
        stepMan = stepMngr(self.log, ctes, corr, self.pi, prevStepInfo,
                           prntCond = prntCond)
        # Set the base values
        stepMan.calcBase(self,P0,I0,J0)

        # Write base values on the spreadsheet (if it is the case)
        if self.log.xlsx is not None:
            col = self.NIterGrad+1
            self.log.xlsx['step'].write(0, col, 0.)
            self.log.xlsx['P'].write(0, col, P0)
            self.log.xlsx['I'].write(0, col, I0)
            self.log.xlsx['J'].write(0, col, J0)
            self.log.xlsx['obj'].write(0, col, stepMan.Obj0)

        # Proceed to the step search itself
        alfa = stepMan.srchStep(self)

    # TODO: "Assured rejection prevention"
    #  if P<tolP, make sure I<I0, otherwise,
    #  there will be a guaranteed rejection later...

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
        msg = "\n> Run of calcStepGrad terminated. "+\
              "Press any key to continue."
        self.log.prom(msg)

    return alfa, stepMan

def grad(self,corr,alfa_0:float,retry_grad:bool,stepMan:stepMngr):

    self.log.printL("\nIn grad, Q0 = {:.4E}.".format(self.Q))
    #self.log.printL("NIterGrad = "+str(self.NIterGrad))

    # Calculation of alfa
    alfa, stepMan = self.calcStepGrad(corr,alfa_0,retry_grad,stepMan)

    # write on the spreadsheets (if it is the case)
    if self.log.xlsx is not None:
        # Update maximum number of evals
        if self.log.xlsx['maxEvals'] < stepMan.cont:
            self.log.xlsx['maxEvals'] = stepMan.cont

        row, col = 0, self.NIterGrad+1
        for i in range(4): # writing step limits and status
            self.log.xlsx['misc'].write(row, col, stepMan.StepLimActv[i])
            row += 1
            self.log.xlsx['misc'].write(row, col, stepMan.StepLimLowr[i])
            row += 1
            self.log.xlsx['misc'].write(row, col, stepMan.StepLimUppr[i])
            row += 2
        # Writing the trios
        self.log.xlsx['misc'].write(row, col, stepMan.histStep[stepMan.trio[0]])
        row += 1
        self.log.xlsx['misc'].write(row, col, alfa)
        row += 1
        self.log.xlsx['misc'].write(row, col, stepMan.histStep[stepMan.trio[2]])
        row += 2
        # Writing the stop motives (for P Lim search and the step search itself)
        self.log.xlsx['misc'].write(row, col, stepMan.pLimSrchStopMotv)
        row += 1
        self.log.xlsx['misc'].write(row, col, stepMan.stopMotv)
        # Writing the linear and angular coefficients for the P limit regression
        row += 2
        self.log.xlsx['misc'].write(row, col, stepMan.PRegrAngCoef)
        row += 1
        self.log.xlsx['misc'].write(row, col, stepMan.PRegrLinCoef)

    #alfa = 0.1
    #self.log.printL('\n\nBypass: alfa arbitrado em '+str(alfa)+'!\n\n')
    self.updtHistGrad(alfa,stepMan.stopMotv)

    # Apply correction, update histories in alternative solution
    newSol = self.copy()
    newSol.aplyCorr(alfa,corr)
    newSol.updtHistP()

    self.log.printL("Leaving grad with alfa = "+str(alfa))
    self.log.printL("Delta pi = "+str(alfa*corr['pi']))

    if self.dbugOptGrad['pausGrad']:
        self.log.prom('Grad in debug mode. Press any key to continue...')

    return alfa, newSol, stepMan
