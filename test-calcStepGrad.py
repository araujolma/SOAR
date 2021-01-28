#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 12:12:07 2020

@author: levi
"""
import numpy
import matplotlib.pyplot as plt
import grad_sgra
from interf import logger
from tabulate import tabulate
from scipy.interpolate import interp1d

parsDefault = {'useObjPars': True,
               'usePPars': True,
               'minVal': 1.,
               'minPt': 2.,
               'sigma': 1.,
               'scale': 2.,
               'P0': 1e-13,
               'limP_step': 5.,#1.5,#
               'tolP': 1e-12,
               'GSS_PLimCte': 1e8,
               'GSS_stopStepLimTol': 1e-2,
               'GSS_stopObjDerTol': 1e-4,
               'GSS_stopNEvalLim': 100,
               'GSS_findLimStepTol': 1e-2,
               'pi_min': [None],
               'pi_max': [None],
               'prntCalcStepGrad': True,
               'plotCalcStepGrad': True,
               'pausCalcStepGrad': False}

def miele_curve(t, pars=None):
    """Make a curve that looks like common Obj vs. step behavior in SGRA problems."""
    if pars is None:
        pars = parsDefault
    #return -10.*numpy.exp(-((t-2.)/0.5)**2) + numpy.exp((t**2)/10.)
    aux = (t-pars['minPt'])/pars['sigma'] - 1.
    return pars['minVal'] + pars['scale'] * \
           (numpy.exp(-1) + numpy.exp(aux) * aux)

class SgraDummy:
    """A dummy version of SGRA, for testing calcStepGrad."""

    def __init__(self, corr, makeDir=False, pars=None):
        self.log = logger('test', isManu=True, makeDir=makeDir, mode='screen')
        self.corr = corr
        if pars is None:
            pars = parsDefault
        #logger('test',makeDir=makeDir)
        self.pars = pars
        self.x = 0.
        self.pi = numpy.array([1.])

        if self.pars['useObjPars']:
            self.obj = None
        else:
            # set up the interpolation model
            obj = interp1d(self.pars['step_J_P_array'][0,:],
                           self.pars['step_J_P_array'][1,:])
            self.obj = obj

        if self.pars['usePPars']:
            self.Pfit = None
        else:
            # set up the interpolation model
            Pfit = interp1d(self.pars['step_J_P_array'][0,:],
                           self.pars['step_J_P_array'][2,:])
            self.Pfit = Pfit

        self.tol = {'P': pars['tolP']}
        self.constants, self.restrictions = {}, {}
        for key in ['GSS_PLimCte', 'GSS_stopStepLimTol', 'GSS_stopObjDerTol',
                    'GSS_stopNEvalLim', 'GSS_findLimStepTol']:
            self.constants[key] = pars[key]
        for key in ['pi_min', 'pi_max']:
            self.restrictions[key] = pars[key]

        self.NIterGrad = 1
        self.dbugOptGrad = {}
        for key in ['prntCalcStepGrad', 'plotCalcStepGrad', 'pausCalcStepGrad']:
            self.dbugOptGrad[key] = pars[key]

    def calcP(self, mustPlotPint=False):
        """Calculate current P value."""
        if self.pars['usePPars']:
            k = (self.constants['GSS_PLimCte'] * self.tol['P'] - self.pars['P0'])/\
                self.pars['limP_step']
            P = self.pars['P0'] + k * self.x
        else:
            P = self.PfitSat(self.x)

        return P, P, 0.

    def PfitSat(self, x):
        """A saturated version of P fit"""
        xmin, xmax = min(self.Pfit.x), max(self.Pfit.x)
        if x > xmax:
            x = xmax
        elif x < xmin:
            x = xmin
        return self.Pfit(x)

    def getPLimStep(self, limPtoo=False):
        """Get the value of alfa so that P(alfa)=P_lim."""
        if self.pars['usePPars']:
            if limPtoo:
                limP = self.pars['tolP'] * self.pars['GSS_PLimCte']
                return self.pars['limP_step'], limP
            else:
                return self.pars['limP_step']
        else:
            limP = self.pars['tolP'] * self.pars['GSS_PLimCte']

            # basic bisection. Prepare upper and lower bounds
            alfa_low = 0.
            alfa_high = 1.
            P = self.PfitSat(alfa_high)
            while P < limP:
                alfa_high *= 2.
                P = self.PfitSat(alfa_high)
            # perform bisection itself
            alfa = .5 * (alfa_low + alfa_high)
            P = self.PfitSat(alfa)
            while abs(P-limP) > 1e-3 * limP:
                if P > limP:
                    alfa_high = alfa
                else:
                    alfa_low = alfa
                alfa = .5 * (alfa_low + alfa_high)
                P = self.PfitSat(alfa)

            if limPtoo:
                return alfa, limP
            else:
                return alfa

    def calcJ(self):
        """Calculate current J value."""
        if self.pars['useObjPars']:
            I = miele_curve(self.x, self.pars)
        else:
            x, xmin, xmax = self.x, min(self.obj.x), max(self.obj.x)

            if x > xmax:
                x = xmax
            elif x < xmin:
                x = xmin

            I = self.obj(x)
        #J, J_Lint, J_Lpsi, I, Iorig, Ipf
        return I, 0., 0., I, I, 0.

    def getMinJStep(self, minJtoo=False):
        """Get the value of alfa that minimizes objective J."""
        if self.pars['useObjPars']:
            if minJtoo:
                return self.pars['minPt'], self.pars['minVal']
            else:
                return self.pars['minPt']
        else:
            arr = self.pars['step_J_P_array']
            argMinPt = arr[1, :].argmin()
            if minJtoo:
                return arr[0, argMinPt], arr[1, argMinPt]
            else:
                return arr[0, argMinPt]

    def copy(self):
        """Make a copy of itself."""
        copia = SgraDummy(self.corr, pars=self.pars.copy())
        copia.x = self.x + 0.
        return copia

    def aplyCorr(self, alfa, correction):
        """Apply the correction."""
        self.x += alfa * correction['x']
        self.pi += alfa * correction['pi']

    def calcStepGrad(self, *args, **kwargs):
        return grad_sgra.calcStepGrad(self, *args, **kwargs)

    def showPJ(self, alfaMin=0., alfaMax=5., autoscale=True):
        """Make plots of the P and J curves as functions of alfa."""

        if autoscale and 'step_J_P_array' in self.pars.keys():
            alfaMin = min(self.pars['step_J_P_array'][0, :])*.95
            alfaMax = max(self.pars['step_J_P_array'][0, :])*1.05
        alfa = numpy.linspace(alfaMin, alfaMax, num=1000)
        P = numpy.empty_like(alfa)
        J = numpy.empty_like(alfa)
        for ind, alfa_ in enumerate(alfa):
            self.aplyCorr(alfa_, self.corr)
            P_, _, _ = self.calcP()
            aux = self.calcJ()
            self.aplyCorr(-alfa_, self.corr)
            J_ = aux[0]
            P[ind], J[ind] = P_, J_

        f, (ax1, ax2) = plt.subplots(2, 1, sharex='col')
        ax1.semilogy(alfa, P, label='P')
        ax1.plot(alfa, P[0] + numpy.zeros_like(P), '--', label='P(0)')
        ax1.plot(alfa, self.tol['P'] + numpy.zeros_like(P), '--', label='tolP')
        limP = self.constants['GSS_PLimCte'] * self.tol['P']
        ax1.plot(alfa, limP + numpy.zeros_like(P), '--', label='limP')
        Ppt = self.getPLimStep()
        ax1.plot(Ppt, limP, 'o', label='limP_pt')
        ax1.legend()
        ax1.grid()
        ax1.set_xlabel('Step')
        ax1.set_ylabel('P')
        ax1.set_title('P and J vs. step')

        ax2.plot(alfa, J, label='J')
        minPt, minVal = self.getMinJStep(minJtoo=True)
        ax2.plot(minPt, minVal, 'o', label='min_pt')
        ax2.grid()
        ax2.set_xlabel('Step')
        ax2.set_ylabel('J')
        ax2.legend()
        plt.show()

class TestSeq:
    """Test sequence."""

    def __init__(self, parsList, showPJ=False):
        alfa_0 = 0.
        corr = {'x': 1.,
                'pi': numpy.array([1e-10]),
                'dJdStepTheo': -1.}
        retry_grad = False
        stepMan = None
        self.stepError, self.stepRelError, self.nAttempts = [], [], []
        self.sgraList, self.stepManList, self.stepList = [], [], []
        self.stopMotv, self.minObjPt, self.limPPt = [], [], []
        for pars in parsList:
            sgra = SgraDummy(corr, pars=pars)
            if showPJ:
                sgra.showPJ()
            self.sgraList.append(sgra)
            alfa, stepMan = sgra.calcStepGrad(sgra.corr, alfa_0, retry_grad, stepMan)
            self.stepManList.append(stepMan)
            self.stepList.append(alfa)
            self.stopMotv.append(stepMan.stopMotv)
            self.minObjPt.append(sgra.getMinJStep())
            self.limPPt.append(sgra.getPLimStep())
            alfa_0 = alfa

    def checkPerf(self):
        """Check performance of calcStepGrad on the runs"""
        nTest = 0
        for alfa, stepMan, sgra in zip(self.stepList, self.stepManList, self.sgraList):
            alfa_J_min = sgra.getMinJStep()
            alfa_P = sgra.getPLimStep()
            alfa_exact = min(alfa_J_min, alfa_P)

            self.stepError.append(alfa - alfa_exact)
            self.stepRelError.append((alfa - alfa_exact)/alfa_exact)
            self.nAttempts.append(stepMan.cont)
            nTest += 1
        print("\nTest concluded! Here are the results:")
        # making the results
        headers = ["N", 'obj evals', "Min obj pt", "P=limP pt", "Error",
                   "Error, %", "Stop motv"]
        # data = {'N': numpy.linspace(0,nTest-1,num=nTest)
        #         'evals': self.nAttempts,
        #         'Min obj pt': self.minObjPt,
        #         'P=limP pt': self.limPPt,
        #         'Step error:'} # not good because loses the order...
        data = []
        for k in range(nTest):
            data.append((k, self.nAttempts[k], self.minObjPt[k],
                         self.limPPt[k], self.stepError[k],
                         self.stepRelError[k]*100., self.stopMotv[k]))

        print(tabulate(data, headers=headers, floatfmt='.3G'))

if __name__ == "__main__":

    thesePars = parsDefault
    thesePars['limP_step'] = 1.5
    thesePars['plotCalcStepGrad'] = False
    parsList = [thesePars]

    newPars = thesePars.copy()
    newPars['limP_step'] = 10.
    parsList.append(newPars)

    newPars = newPars.copy()
    newPars['minPt'] = 1e-4
    parsList.append(newPars)

    parsNonPar = parsDefault.copy()
    parsNonPar['limP_step'] = 4.
    parsNonPar['useObjPars'] = False # this makes it interpolate for calculating J
    parsNonPar['usePPars'] = False # this makes it interpolate for calculating P
    sJP = numpy.empty((3, 11))
    sJP[0,:] = numpy.linspace(0.,10.,num=11)
    sJP[1,:] = numpy.array([1., 1., 0.7, 0.5, -.25, -1., -.6, .5, .5, .7, 1.]) + 2.
    sJP[2,:] = parsNonPar['P0'] + 1.1e-5 * sJP[0,:]
    parsNonPar['step_J_P_array'] = sJP
    parsList.append(parsNonPar)

    TS = TestSeq(parsList, showPJ=True)
    TS.checkPerf()


