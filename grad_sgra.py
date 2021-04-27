#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:39:08 2017

@author: levi
"""

import numpy #, time
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
        # for the P limit, it is useful to keep track of lower and upper bounds,
        # that is, steps for which P < limP and P > limP.
        self.pLimLowrBnd = 0.    # the largest alfa so that P(alfa) < limP
        self.pLimUpprBnd = LARG  # the smallest alfa so that P(alfa) > limP
        self.bestLimP = -1 # this is for the smallest distance to P limit

        # convention: 0: pi limits,
        #             1: P < limP,
        #             2: Obj <= Obj0,
        #             3: Ascending gradient detected on Obj
        # lowest step that violates each condition
        self.minStepVio = [LARG, LARG, LARG, LARG]
        # this is for monitoring the presence of "up slopes" on the objective function
        #   a given step alfa is said to be non-decrescent if there exists another
        #   step alfa' < alfa so that Obj(alfa') < Obj(alfa).
        #   This variable holds the index w.r.t. sortIndxHistStep of the smallest
        #   non-decrescent step.
        self.minNonDecrStepIndx = None # the obj function is monotonic until contrary
        # so yes, it is redundant, self.minStepVio[3] always equals self.histStep[
        #   self.sortIndxHistStep[self.minNonDecrStepIndx]] whenever
        #   self.minNonDecrStepIndx is not None.
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
        self.isTrioSet = False  # True when the trio is set
        self.leftRght = False   # True: left, False: right
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
        self.pLim_ffRate = 10.

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
        #             3: Ascending gradient detected on Obj
        if not noLimPi:
            self.minStepVio[0] = alfaLimPi

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

    def bestStep(self, get_sort_index=False):
        """Returns the best step so far."""
        if self.trio[1] == -1:
            if get_sort_index:
                return 0., 0
            else:
                return 0.
        else:
            if get_sort_index:
                # find the sorted index
                for sort_index, indx in enumerate(self.sortIndxHistStep):
                    if indx == self.trio[1]:
                        break
                else:
                    raise Exception("Could not find the proper index... debug now!")
                return self.histStep[self.trio[1]], sort_index
            else:
                return self.histStep[self.trio[1]]

    def bestObj(self):
        """Returns the best objective so far."""
        if self.trio[1] == -1:
            return self.Obj0
        else:
            return self.histObj[self.trio[1]]

    def maxGoodStep(self):
        """"Returns the maximum good step tried so far."""
        i = 1
        for i, ind in enumerate(self.sortIndxHistStep):
            if not self.histStat[ind]:
               break
        else:
            i += 1
        return self.sortIndxHistStep[i-1]

    def minNonDecrStep(self):
        """Returns the minimum non-descrescent step."""
        if self.minNonDecrStepIndx is not None:
            return self.histStep[self.sortIndxHistStep[self.minNonDecrStepIndx]]

    def minBadStep(self):
        """Returns the minimum step that violates some condition."""
        return min(self.minStepVio)

    def check(self, pos: int, alfa: float, Obj: float, P: float, pi):
        """ Perform a validity check in this step value, updating the limit
        values for searching step values if it is the case.

        Conditions checked:  - limits in each of the pi parameters
                             - P <= limP
                             - Obj <= Obj0
                             - Descending behavior of Obj(alfa)

        If any of these conditions are not met, this step is treated as a bad
        (invalid) step. Otherwise, it is a good step.
        """

        # Bad point code: originally 1, each violation multiplies by a prime:
        # Pi conditions violation:        x 2
        # P limit violation:              x 3
        # Obj >= Obj0                     x 5
        # Obj < Obj(x) for some x > alfa  x 7
        # Step limit violation:           x 11
        BadPntCode = 1

        # Check for violations on the pi conditions.
        # Again, a limit is 'None' when it is not active.
        PiCondVio = False
        for i in range(len(pi)):
            # violated here in lower limit condition
            if self.piLowLim[i] is not None and pi[i] < self.piLowLim[i]:
                PiCondVio = True; break # already violated, no need to continue
            # violated here in upper limit condition
            if self.piHighLim[i] is not None and pi[i] > self.piHighLim[i]:
                PiCondVio = True; break # already violated, no need to continue
        # "Bad point" conditions:
        BadPntMsg = ''
        # condition 0:  pi < piLowLim or pi > piHighLim
        if PiCondVio:
            BadPntCode *= 2
            BadPntMsg += ("\n-   piLowLim: " + str(self.piLowLim) +
                          "\n          pi: " + str(pi) +
                          "\n   piHighLim: " + str(self.piHighLim))
            # no need for checking since the pi limits are known beforehand
        # condition 1:
        if P >= self.limP:
            BadPntCode *= 3
            BadPntMsg += ("\n-  P = {:.4E}".format(P) +
                         " > {:.4E} = limP".format(self.limP))
            # update minStepVio
            if alfa < self.minStepVio[1]:
                self.minStepVio[1] = alfa
        # condition 2:
        if Obj > self.Obj0:
            BadPntCode *= 5
            BadPntMsg += ("\n-  Obj-Obj0 = {:.4E} > 0".format(Obj-self.Obj0))
            # update minStepVio
            if alfa < self.minStepVio[2]:
                self.minStepVio[2] = alfa
        # condition 3:
        isCond3 = False
        # get the step associated with minNonDecrStepIndx
        if self.minNonDecrStepIndx is not None:
            if pos == self.minNonDecrStepIndx:
                # this case has to be treated specially because the current step
                # (alfa) is not on the histStep yet
                alfa_minNonDecr = alfa
            else:
                alfa_minNonDecr = self.minNonDecrStep()
            # check if minStepVio needs to be updated
            if alfa_minNonDecr < self.minStepVio[3]:
                self.minStepVio[3] = alfa_minNonDecr
                # EVERY SUBSEQUENT STEP BECOMES A BAD STEP!!
                isCond3 = True
                for i in range(pos+1, len(self.sortIndxHistStep)):
                    if self.mustPrnt and self.histStat[self.sortIndxHistStep[i]]:
                        msg = "> Changing the status of step #{} ({}) to False!".\
                            format(self.sortIndxHistStep[i], self.histStep[
                            self.sortIndxHistStep[i]])
                        self.log.printL(msg)
                    self.histStat[self.sortIndxHistStep[i]] = False
        # condition 4:
        if BadPntCode == 1:
            for i in range(4):
                if alfa >= self.minStepVio[i]:
                    BadPntCode *= 11
                    BadPntMsg += "\n-  alfa = {:.4E} >= {:.4E} =" \
                                 " minStepVio[{}]".format(alfa,
                                                          self.minStepVio[i], i)
                    break

        if BadPntCode > 1 and self.mustPrnt:
            self.log.printL("> In check. Got a bad point," + BadPntMsg + \
                            f"\nwith alfa = {alfa}, Obj = {Obj}")

        if self.mustPrnt:
            msg = "\n- minStepVio = " + str(self.minStepVio)
            if isCond3:
                msg += "\n- Condition 3 was triggered."\
                       " All subsequent steps have become bad steps!"
            self.log.printL(msg)

        return BadPntCode

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

        return Obj0

    def testDecr(self, alfa:float, Obj:float):
        """Test decrescent behavior.

        Test if the decrescent behavior of the objective as a function of the step
         is still maintained.

        This is implemented by keeping an ordered list of the steps (or, more
        precisely, a list of the indexes of the steps in histStep, but in ascending
        order of the steps), called sortIndxHistStep.

        This function is called in tryStep(), before actually appending the given
        "alfa" to the history.

        :return: the position of the sorted list in which the current step will be
        placed
        """

        decrFrstLostUpdt = False # flag for changing point of loss of monotonicity
        alfa_decr_lost = 0. # placeholder
        prev_alfa_decr_lost = 0. #placeholder

        ins = len(self.histStep)  # index that will be inserted on the list

        if len(self.sortIndxHistStep) == 0:
            # This is the first step. It goes to the first position and that's it.
            self.sortIndxHistStep.append(ins)
            pos = 0
        else:
            # the test is different depending on the position of the current
            # step in respect to the list of previous steps.

            if alfa > self.histStep[self.sortIndxHistStep[-1]]:
                # step bigger than all previous steps
                pos = ins # position on the list: final one
                if self.minNonDecrStepIndx is None:
                    if Obj > self.histObj[self.sortIndxHistStep[-1]]:
                        decrFrstLostUpdt = True
                        alfa_decr_lost = alfa
                        self.minNonDecrStepIndx = pos
            elif alfa < self.histStep[self.sortIndxHistStep[0]]:
                # step smaller than all previous steps. Impossible!
                raise Exception("This should never happen...")
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
                    raise Exception(msg)

                # Of all the tried steps, alfa_m is the one immediately before alfa;
                # and alfa_p is the one immediately above alfa. Obj_m and Obj_p are
                # their objective values, respectively.
                Obj_m = self.histObj[self.sortIndxHistStep[pos-1]]
                alfa_p = self.histStep[self.sortIndxHistStep[pos]]
                Obj_p = self.histObj[self.sortIndxHistStep[pos]]
                # get the previous non-descrescent step
                if self.minNonDecrStepIndx is not None:
                    prev_alfa_decr_lost = self.minNonDecrStep()
                alfa_decr_lost = prev_alfa_decr_lost+0.

                if Obj < Obj_p:
                    # Crescent behavior detected between alfa and alfa_p
                    if self.minNonDecrStepIndx is None or \
                            alfa_p < prev_alfa_decr_lost:
                        # increasing behavior starts before
                        decrFrstLostUpdt = True
                        self.minNonDecrStepIndx = pos + 1
                        alfa_decr_lost = alfa_p
                        if self.mustPrnt:
                            msg = "> Changing minNonDecrStepIndx to {} (alfa={})".\
                                format(self.minNonDecrStepIndx, alfa_p)
                            self.log.printL(msg)
                if Obj > Obj_m:
                    # Crescent behavior detected between alfa_m and alfa
                    if self.minNonDecrStepIndx is None or alfa < prev_alfa_decr_lost:
                        decrFrstLostUpdt = True
                        self.minNonDecrStepIndx = pos
                        alfa_decr_lost = alfa
                        if self.mustPrnt:
                            msg = "> Changing minNonDecrStepIndx to {} (alfa={})".\
                                format(self.minNonDecrStepIndx, alfa)
                            self.log.printL(msg)

            # insert the current step at the proper place in the sorted list
            self.sortIndxHistStep.insert(pos, ins)

            # correct the index to keep the order!
            if self.minNonDecrStepIndx is not None and not decrFrstLostUpdt and \
                    alfa < alfa_decr_lost:
                self.minNonDecrStepIndx += 1

            # Display messages
            if self.mustPrnt and decrFrstLostUpdt:
                msg = "\n> The descending section was shortened... \n"\
                      "  index = {}, alfa = {}".format(self.minNonDecrStepIndx,
                                                       alfa_decr_lost)
                self.log.printL(msg)

        return pos

    def isNewStep(self, alfa):
        """Check if the step alfa is a new one (not previously tried).

        The (relative) tolerance is self.epsStep.
        """

        for alfa_ in self.histStep:
            if abs(alfa_-alfa) < self.epsStep * alfa:
                return False
        else:
            return True

    def tryStep(self, sol, alfa, plotSol=False, plotPint=False):
        """ Try this given step value.

        It returns the values of P, I and Obj associated* with the tried step (alfa).

        Some additional procedures are performed, though:
            - update "best step" record ...
            - update the lower and upper bounds (via self.check())
            - assemble/update the trios for searching the minimum obj"""

        # Increment evaluation counter
        self.cont += 1
        thisStepIndx = len(self.histStep)

        # Useful for determining if this is the first good step
        noGoodStepBfor = (self.trio[1] == -1)

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

        # Put the step in the list of sorted steps
        pos = self.testDecr(alfa, Obj)

        # check if it is a good step, etc
        BadPntCode = self.check(pos, alfa, Obj, P, newSol.pi)
        isOk = (BadPntCode == 1)

        # Update P limit upper and lower bounds (if applicable, of course)
        if P > self.limP and alfa < self.pLimUpprBnd:
            self.pLimUpprBnd = alfa
        elif P < self.limP and alfa > self.pLimLowrBnd:
            self.pLimLowrBnd = alfa
        if isOk: # Good step!
            # update the step closest to P limit
            # Just the best distance is kept, the closest good step is necessarily
            # self.pLimLowrBnd
            distP = 1. - P / self.limP
            if self.mustPrnt:
                msg = f"\n> Current distP = {distP}.\n"
            else:
                msg = ''
            if self.bestLimP < 0.:
                self.bestLimP = distP
                if self.mustPrnt:
                    msg += f"  First best steplimP: {alfa}."
            else:
                if distP < self.bestLimP:
                    self.bestLimP = distP
                    if self.mustPrnt:
                        msg += f"  Updating best steplimP: {alfa}."
            if self.mustPrnt:
                self.log.printL(msg)

        # STORE THE RESULTS IN THE HISTORY ARRAYS
        self.histStat.append(isOk)
        self.histStep.append(alfa)
        self.histP.append(P)
        self.histI.append(I)
        self.histObj.append(Obj)

        # UPDATE TRIO FOR OBJECTIVE "TRISSECTION"
        if self.isTrioSet:  # trio is ready; update it
            alfa0, Obj0 = self.histStep[self.trio[0]], self.histObj[self.trio[0]]
            alfa1, Obj1 = self.histStep[self.trio[1]], self.histObj[self.trio[1]]
            alfa2, Obj2 = self.histStep[self.trio[2]], self.histObj[self.trio[2]]
            if alfa1 < alfa < alfa2:
                if isOk:
                    if Obj < Obj1: # "shift right"
                        self.trio[0] = self.trio[1]
                        self.trio[1] = thisStepIndx
                    else: # not optimal, bring upper bound closer
                        self.trio[2] = thisStepIndx
                else: # not good, bring the upper bound closer
                    self.trio[2] = thisStepIndx
            elif alfa0 < alfa < alfa1:
                if isOk:
                    if Obj < Obj1: # "shift left"
                        self.trio[2] = self.trio[1]
                        self.trio[1] = thisStepIndx
                    elif Obj < Obj0: # bring the lower bound closer
                        self.trio[0] = thisStepIndx
                    else: # Obj > Obj1 and Obj > Obj0... REVOLUTION!
                        self.trioRevo()
                else: # bad step between the optimal and the lower bound? REVOLUTION!
                    self.trioRevo()
            elif alfa < alfa0:
                if Obj < Obj0: # lower bound is not a true lower bound? REVOLUTION!
                    self.trioRevo()
        else:  # trio is not ready yet; either trio[1] == -1 or trio[2] == -1
            if isOk:  # this step is a good one
                if noGoodStepBfor:
                    # This is the first valid step ever!
                    # Put it in the middle of the trio
                    self.trio[1] = thisStepIndx
                    if self.mustPrnt:
                        self.log.printL("\n  Setting middle step of the trio.")
                    # entry 1 is set. Check entry 2...
                    if self.trio[2] > -1:
                        # entry 2 had already been set. Switch the flag!
                        self.isTrioSet = True
                        if self.mustPrnt:
                            self.log.printL("\n  Middle entry was just set, "
                                            "trio is ready.")
                else:
                    # Not the first valid step. Maybe update middle element?
                    if Obj < self.histObj[self.trio[1]]:
                        self.trio[1] = thisStepIndx
                        if self.mustPrnt:
                            self.log.printL("\n> Updating middle entry of trio."
                                            "\n  Best result updated!")
                    elif alfa > self.histStep[self.trio[1]]:
                        if self.trio[2] == -1:
                            self.trio[2] = thisStepIndx
                            # Both trio[1] and trio[2] are set. Switch the flag
                            self.isTrioSet = True
                            if self.mustPrnt:
                                self.log.printL("\n  Setting right step of the "
                                                "trio.")
            else: # not a good step
                # there has been a good step and this step is lower but bad, so REVO!
                if self.trio[1] > -1 and alfa < self.histStep[self.trio[1]]:
                    self.trioRevo()
                else:
                    if self.trio[2] == -1 or alfa < self.histStep[self.trio[2]]:
                        # this means the final position of the trio gets the smallest
                        # non-valid step to be found before the trio is set.
                        self.trio[2] = thisStepIndx
                        if self.mustPrnt:
                            self.log.printL("\n  Setting/updating right step of the "
                                            "trio.")
                        # entry 2 is set. Check entry 1...
                        if self.trio[1] > -1:
                            # entry 1 had already been set. Switch the flag!
                            self.isTrioSet = True
                            if self.mustPrnt:
                                self.log.printL("\n> Final entry was just set, trio "
                                                "is ready.")

        # PRINT RESULTS
        if self.mustPrnt:
            if isOk:
                goodBad = 'good'
            else:
                goodBad = 'bad'

            resStr = f"\n- Results ({goodBad} point):\n" + \
                     "step = {:.4E} P = {:.4E} I = {:.4E}".format(alfa, P, I) + \
                     " J = {:.7E} CorrObj = {:.7E}\n ".format(J, Obj) + \
                     " J reduction eff. = {:.1F}%,".format(JrelRed) + \
                     " I reduction eff. = {:.1F}%".format(IrelRed) + \
                     f"\n- Evaluation #{self.cont},\n" + \
                     "- BestStep: {:.6E}, ".format(self.bestStep()) + \
                     "BestObj: {:.6E}".format(self.bestObj()) + \
                     "\n- MinBadStep: {:.6E}".format(self.minBadStep()) + \
                     "\n- P Limits: [{:.4E}, {:.4E}]".format(self.pLimLowrBnd,
                                                             self.pLimUpprBnd)
            self.log.printL(resStr)
            msg = '\nHere are the steps so far (sorted):\n'\
                  ' i  indx    alfa         d obj        red,%     P/limP   Status'
            for i, indx in enumerate(self.sortIndxHistStep):
                if i == 0:
                    red = 100.
                else:
                    red = 100.*(self.histObj[indx] - self.J0)/\
                                self.histStep[indx]/self.dJdStepTheo
                msg += "\n{:2d}    {:2d} {:.4E} {:+.7E} {:+.3E} {:.4E} {}".format(
                    i, indx, self.histStep[indx],
                    self.histObj[indx]-self.Obj0, red,
                    self.histP[indx]/self.limP, self.histStat[indx])
                if indx == self.trio[0]:
                    msg += "  (< best step)"
                elif indx == self.trio[2]:
                    msg += " (> best step)"
                if indx == self.trio[1]:
                    msg += "  (  best step)"
                if i == self.minNonDecrStepIndx:
                    msg += " (min non decr step)"

            self.log.printL(msg)
            self.showTrio()

        #input("Ponto testado.")
        return P, I, Obj

    def trioRevo(self):
        """Trio revolution.

        This is called when there is some violation of the expected obj vs. step
        behavior. The trio must be rebuilt from scratch."""
        # forget everything
        self.trio = [0, -1, -1]
        self.isTrioSet = False
        # trio[1] has to be the best valid step, so let's find it:
        bestObj = self.Obj0
        i = 1
        for i in range(1, len(self.sortIndxHistStep)):
            sortIndx = self.sortIndxHistStep[i]
            if self.histStat[sortIndx]:
                if self.histObj[sortIndx] < bestObj:
                    bestObj = self.histObj[sortIndx]
                    self.trio[1] = sortIndx
            else:
                i -= 1
                break

        # trio[2] has to be the next step, valid or not
        if self.trio[1] == -1:
            # no good step so far. trio[2] gets the smallest tried step
            self.trio[2] = self.sortIndxHistStep[1]
            # trio[0] stays at the default value (0.0)
        else:
            # there has been a good step. Let's try setting trio[0] and trio[2]
            if i < len(self.sortIndxHistStep) - 1:
                # the best step is not the last. trio[2] is the next one
                self.trio[2] = self.sortIndxHistStep[i+1]
                # this is the only scenario in which the trio is set.
                self.isTrioSet = True
            if i > 1:
                # the best step is not the first. trio[0] is the previous one
                self.trio[0] = self.sortIndxHistStep[i-1]

        if self.mustPrnt:
            self.log.printL("\n > REVOLUTION! This is the trio now:")
            self.showTrio()

    def showTrio(self):
        """Show the trio as it currently is."""
        if self.trio[1] == -1:
            s1, o1 = 'Not set', 'Not set'
        else:
            s1, o1 = self.histStep[self.trio[1]], self.histObj[self.trio[1]]
        if self.trio[2] == -1:
            s2, o2 = 'Not set', 'Not set'
        else:
            s2, o2 = self.histStep[self.trio[2]], self.histObj[self.trio[2]]
        msg =  "\n  StepTrio:\n" + \
               "   Indexes: {}\n".format(self.trio) + \
               "   Steps:   {}\n".format([self.histStep[self.trio[0]],
                                          s1, s2]) + \
               "   Objs:    {}".format([self.histObj[self.trio[0]], o1, o2])
        self.log.printL(msg)

    def saturate(self, alfa):
        """Saturate given step with respect to P and pi conditions, if necessary.

        This method is only called during the P limit search.
        The main idea is that it makes no sense to test a step value outside of the
        viable region. That means "saturating" the given step with respect to the
        pi conditions (minStepVio[0]) or pLimLowrBnd/pLimUpprBnd."""

        if alfa >= self.minStepVio[0]:
            alfa = self.minStepVio[0] * .9
            if self.mustPrnt:
                msg = "\n> Oops! That would violate the pi condition.\n" \
                      f"  Let's go with {alfa} instead."
                self.log.printL(msg)
        if alfa < self.pLimLowrBnd or alfa > self.pLimUpprBnd:
            alfa = .5 * (self.pLimLowrBnd + self.pLimUpprBnd)
            if self.mustPrnt:
                msg = "\n> Oops! That would be outside the bounds for P.\n" \
                      f"  Let's go with {alfa} instead."
                self.log.printL(msg)
        return alfa

    def keepSrchPLim(self):
        """ Decide to keep looking for P step limit or not.

        keepLookCode is 0 for keep looking. Other values correspond to different
        exit conditions:

        1 - P limit is not active
            This means that either:
              - the pi limits are more restrictive than the P limit
              - the objective function crosses its initial value before P=limP
              - the objective function rises before P=limP
        2 - upper and lower bounds for P step limit are within tolerance
        3 - monotonic (descent) behavior was lost, obj min is before P limit
        4 - too many objective function evaluations
        5 - P limit is within tolerance
            This means that an alfa was found so that 0 < limP - P(alfa) < tolLimP."""

        if min(self.minStepVio) <= self.pLimLowrBnd:
            keepLoopCode = 1  # P limit is no longer active
        elif (self.pLimUpprBnd - self.pLimLowrBnd) < \
                 self.pLimLowrBnd * self.tolStepLimP:
            keepLoopCode = 2  # upper&lower bounds for P step limit within tolerance
        elif self.minNonDecrStepIndx is not None and self.minNonDecrStep() < \
                self.pLimLowrBnd:
            keepLoopCode = 3  # obj min is before P limit
        elif self.cont >= self.stopNEvalLim:
            keepLoopCode = 4  # too many objective function evaluations (debug maybe?)
        elif self.tolLimP > self.bestLimP > 0.:
            keepLoopCode = 5  # P limit within tolerance
        else:
            keepLoopCode = 0  # keep looking!

        self.pLimSrchStopMotv = keepLoopCode

    def pLimSrch(self,sol):
        """ First part of the step search: the P-limit search.

        The main idea here is to find a value for the step value whose P value is
        smaller than limP but sufficiently close to it. The way this implementation
        works is to keep fitting straight lines for step vs. P in log-log space.

        The stopping criteria are in keepSrchPLim().

        :param sol:
        :return: None
        """

        # 1: Initial guess: get the best step from the previous grad iteration
        prevBest = self.prevStepInfo.get('best', None)
        if prevBest is None: # no previous best, use 1 (if possible)
            alfa1 = self.saturate(1.)
            self.log.printL("\n> Going for default P limit search.")
        else:
            alfa1 = self.saturate(prevBest['step'])
            if self.mustPrnt:
                msg = "\n> First of all, let's try the best step of " \
                      "the previous run,\n  alfa = {}".format(alfa1)
                self.log.printL(msg)
        P1, _, _ = self.tryStep(sol, alfa1)

        # 2: Find a second point, alfa2, "on the other side" of the P=limP line
        alfa2, P2 = alfa1 + 0., P1 + 0.
        # use alfa2 = alfa1 * mult, and mult > 1 or < 1
        if P1 > self.limP:              # alfa1 made P1 > limP...
            mult = 1./self.pLim_ffRate  # get alfa2 < alfa1 so that P2 < limP
            if self.mustPrnt:
                self.log.printL(f"\n> Oops, too large. Let's go back by {mult}.")
        else:                           # alfa1 made P1 < limP...
            mult = self.pLim_ffRate     # get alfa2 > alfa1 so that P2 > limP
            if self.mustPrnt:
                self.log.printL(f"\n> Oops, too small. Let's go forward by {mult}.")
        # keep increasing/decreasing until P1 and P2 are on opposite sides
        while (P2-self.limP) * (P1-self.limP) > 0 and self.pLimSrchStopMotv == 0:
            alfa2 *= mult
            if self.mustPrnt:
                self.log.printL(f"\n> Setting alfa2 = {alfa2}...")
            # TODO: this is actually wrong. If there are Pi limits applicable,
            #  then this saturation will disturb this while loop...
            alfa2 = self.saturate(alfa2)
            P2, _, _ = self.tryStep(sol, alfa2)
            self.keepSrchPLim()

        # 3: First regression, with only two points
        if self.pLimSrchStopMotv == 0:
            # 3.1: calculate P target: just below P limit, still within tolerance
            P_targ = self.limP * (1. - self.tolLimP / 2.)
            # ^ This is so that abs(P_targ / self.limP - 1.) = tolLimP/2
            logP_targ = numpy.log(P_targ)

            # 3.2: perform the first regression (with 2 points)
            x1, x2, y1, y2 = numpy.log(numpy.array([alfa1, alfa2, P1, P2]))
            self.PRegrAngCoef = a = (y2 - y1) / (x2 - x1)  # angular coefficient
            self.PRegrLinCoef = b = y1 - x1 * a  # linear coefficient
            alfaLimP = numpy.exp((logP_targ - b) / a)
            if self.mustPrnt:
                msg = f"\n> Ok, continuing. Let's try alfa = {alfaLimP}\n  " \
                      "(linear fit from the latest tries)."
                self.log.printL(msg)
        else:
            alfaLimP, logP_targ = 0, 0 # just for shutting PyCharm up

        # 4: Main loop for P limit search
        while self.pLimSrchStopMotv == 0:
            # 4.1: Proceed with this value, unless the pi conditions don't allow it
            alfaLimP = self.saturate(alfaLimP)

            # 4.2: try newest step value
            self.tryStep(sol, alfaLimP)

            # 4.3: keep in loop or not
            self.keepSrchPLim()

            # 4.4: perform regression (if it is the case)
            if self.pLimSrchStopMotv == 0:
                # 4.4.1: assemble log arrays for regression
                logStep = numpy.log(numpy.array(self.histStep[1:]))
                logP = numpy.log(numpy.array(self.histP[1:]))
                n = len(self.histStep[1:])

                # 4.4.2: perform actual regression
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

                # 4.4.3: calculate alfaLimP according to current regression
                alfaLimP = numpy.exp((logP_targ - b) / a)
                if self.mustPrnt:
                    msg = f"\n> Performed fit with n = {n} points:\n" \
                          f"  Steps: {numpy.exp(logStep)}\n" \
                          f"  Ps: {numpy.exp(logP)}\n Weights: {W}\n" \
                          f"  a = {a}, b = {b}\n" \
                          f"  alfaLimP = {alfaLimP}"
                    self.log.printL(msg)

        # 5: Print conditions for end of P limit search
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

    def minObjSrch(self, sol):
        """ Second part of the step search: minimum objective search.

        The program only enters this function after the P limit search, so at least
        two points must have been tried.

        First of all, a few special cases are treated before the "main" case.

        The first special case is when none of the previously tried steps is valid.
        In this case, the smallest bad step is taken and the algorithm sequentially
        tries values of step smaller and smaller until a good step is found.

        The second special case is when the objective function is strictly descending
        with step throughout the valid step interval.

        Finally, the search itself. It is implemented in two main "states", one where
        the trissection is performed (reducing the bounds on the local minimum) and
        one where the trio is recovered. This is necessary because at any given time
        a "revolution" might happen, rendering some previously valid steps as invalid.
        :param sol:
        :return: None
        """
        alfa = self.bestStep() # start with the best step so far
        # 1: Bad condition -- all tested steps were bad (not valid)
        if alfa < eps:  # alfa is 0 in this case
            if self.isTrioSet:
                raise Exception("grad_sgra: No good step but the trios are set."
                                "\nDebug this now!")
            if self.mustPrnt:
                msg = "\n\n> Bad condition! None of the tested steps was valid."
                self.log.printL(msg)
            # 1.1: Get the smallest bad step tried
            isOk = False
            alfa = self.histStep[self.trio[2]]

            # 1.2: Go back until a good point is found
            while not isOk:
                if self.mustPrnt:
                    self.log.printL(f"\n>  Going back by {self.ffRate}...")
                alfa /= self.ffRate
                self.tryStep(sol, alfa)
                isOk = self.histStat[-1]  # get its status

            # The trios are already automatically assembled!
            if self.mustPrnt:
                self.log.printL("\n> Finished looking for valid objs.")
                self.showTrio()

        # 2: Check possibility: Decrescent behavior throughout the valid steps region
        alfa, sort_index = self.bestStep(get_sort_index=True)
        Obj = self.bestObj()
        if self.minNonDecrStepIndx is None or \
                self.minNonDecrStep() > self.pLimUpprBnd:
            # 2.1: Calculate the gradient at this point
            if self.mustPrnt:
                self.log.printL("\n> It seems the minimum obj is after the P "
                                "limit.\n  Testing gradient...")
            # TODO: this hardcoded value can cause issues in super sensitive
            #  problems...
            # This is the first attempt for a step that is very close
            alfa_ = alfa * .999
            if sort_index > 0:
                alfa_prev = self.histStep[self.sortIndxHistStep[sort_index-1]]
                if alfa_ <= alfa_prev:
                    alfa_ += 0.001 * alfa_prev

            P_, I_, Obj_ = self.tryStep(sol, alfa_)
            grad = (Obj - Obj_) / (alfa - alfa_)

            # 2.2: Test the gradient
            if grad < 0.: # negative gradient! Let's leave
                if self.mustPrnt:
                    msg = "\n> Objective seems to be descending with step.\n" + \
                          f"  Leaving calcStepGrad with alfa = {alfa}\n"
                    self.log.printL(msg)
                self.stopMotv = 2  # step limit hit
                return alfa
            else: # positive gradient... back to main algorithm then
                if self.mustPrnt:
                    self.log.printL(f"\n> Positive gradient ({grad})...")

        if self.mustPrnt:
            self.log.printL("\n> Starting min obj search...")

        keepLook = True
        # 3: THE MAIN LOOP
        while keepLook:
            if self.isTrioSet:  # 3.1: Trio is set, perform a trissection move

                # 3.1.1: Decide the search direction
                if self.histStep[self.trio[1]] - self.histStep[self.trio[0]] > \
                    self.histStep[self.trio[2]] - self.histStep[self.trio[1]]:
                    self.leftRght = True
                else:
                    self.leftRght = False

                # 3.1.2: Perform the move
                self.trissection_move(sol)

                # 3.1.3: Check for finishing conditions
                if self.cont >= self.stopNEvalLim: # too many evals
                    keepLook = False
                else:
                    # the trio may have been undone on the "trissection" move...
                    if self.isTrioSet:
                        keepLook = self.histStep[self.trio[2]] - \
                                   self.histStep[self.trio[0]] > \
                                   self.tolStepObj * self.histStep[self.trio[1]]
            else:  # 3.2: Trio is not set: Set it then!
                self.setTrio(sol)

        # 4: End conditions
        if self.cont >= self.stopNEvalLim:
            # 4.1: too many evals, complain and return the best so far
            msg = "\n> Leaving step search due to excessive evaluations."
            self.log.printL(msg)
            if self.mustPrnt:
                self.showTrio()
            self.stopMotv = 3  # too many evals
        else:
            # 4.2: minimum was actually found!
            if self.mustPrnt:
                gr_p = (self.histObj[self.trio[2]] - self.histObj[self.trio[1]]) / \
                       (self.histStep[self.trio[2]] - self.histStep[self.trio[1]])
                gr_n = (self.histObj[self.trio[1]] - self.histObj[self.trio[0]]) / \
                       (self.histStep[self.trio[1]] - self.histStep[self.trio[0]])
                grad = gr_p - gr_n
                msg = "\n> Local obj minimum was found.\n  Gradients: grad_p = " \
                      "{}, grad_n = {}, grad = {}".format(gr_p, gr_n, grad)
                self.log.printL(msg)
                self.showTrio()
            self.stopMotv = 1  # local min found!

    def setTrio(self, sol):
        """Set the trio.

        For some reason the trio was not set, so this function attempts on new values
        of step until the trio is set, not necessarily aiming for optimality yet.

        This actually only sets the middle element of the trio, since the left
        element has a good default value (0.0), and the right element is also
        necessarily set.
        """

        if self.mustPrnt:
            self.log.printL("\n> Let's finish setting the trios. Here they are"
                            " so far:")
            self.showTrio()

        while not self.isTrioSet:
            if self.trio[1] == -1:
                # this means that there haven't been any good steps. Go back!
                alfa = self.histStep[self.sortIndxHistStep[1]] / self.ffRate
                if self.mustPrnt:
                    self.log.printL("\n> All attempted steps were bad. "
                                    f"Going back by {self.ffRate} to alfa = {alfa}")
                self.tryStep(sol, alfa)

            else:
                # there has been at least a good step. Not a bad one, though...
                raise Exception("How can this even happen?")
                # this should be impossible because a bad step is guaranteed to be
                # found during the pSearch.

    def trissection_move(self, sol):
        """Do a single "trissection" move.

        It is assumed that the trio is already set; so this performs a move towards
        reducing the upper step bound (trio[2]) and increasing the lower step bound
        (trio[0]). """

        # Estimation of the number of evaluations until convergence
        n_float = numpy.log(self.tolStepObj * self.histStep[self.trio[1]] / \
                (self.histStep[self.trio[2]] - self.histStep[self.trio[0]])) / \
                numpy.log((numpy.sqrt(5.) - 1.) / 2.)
        n = int(numpy.ceil(n_float))
        if n < 0:  # Apparently we are about to converge!
            n = 0
        if self.mustPrnt:
            self.log.printL("\n> According to the Golden Section approximation, "
                            f"n = {n} evals until convergence...")
        self.cont_exp = n + self.cont

        if self.mustPrnt:
            self.showTrio()

        # Narrow the boundaries either left (trio[0]) or right (trio[2])
        if self.leftRght: # move left
            alfa = .5 * (self.histStep[self.trio[0]] +
                         self.histStep[self.trio[1]])
        else: # move right
            alfa = .5 * (self.histStep[self.trio[2]] +
                         self.histStep[self.trio[1]])

        # Try the new step. Updating the trios is done as well, of course
        self.tryStep(sol, alfa)

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
        self.pLimSrch(sol)

        # If the P limit search was abandoned because of too many evals, leave now.
        if self.pLimSrchStopMotv == 4:
            msg = "\n> Leaving step search due to excessive evaluations." + \
                  "\n  Debugging is recommended."
            self.log.printL(msg)
            self.stopMotv = 3
            return self.bestStep()

        # PART 2: min Obj search
        self.minObjSrch(sol)

        # In any case, return the best (valid) step so far
        return self.bestStep()

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
            xlim = max([.99*self.maxGoodStep(), 1.01*alfa])
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

def calcStepGrad(self, corr: dict, alfa_0: float, retry_grad: bool, stepMan:stepMngr):
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
        # TODO: it would not be that hard to put a test on I reduction efficiency...
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
            prevStepInfo['best'] = {'step': stepMan.bestStep(),
                                    'obj': stepMan.bestObj()}

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
            self.log.xlsx['misc'].write(row, col, stepMan.minStepVio[i])
            row += 1
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
