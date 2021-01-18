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

parsDefault = {'minVal': 1.,
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
        k = (self.constants['GSS_PLimCte'] * self.tol['P'] - self.pars['P0'])/\
            self.pars['limP_step']
        P = self.pars['P0'] + k * self.x
        return P, P, 0.

    def calcJ(self):
        I = miele_curve(self.x, self.pars)
        #J, J_Lint, J_Lpsi, I, Iorig, Ipf
        return I, 0., 0., I, I, 0.

    def copy(self):
        copia = SgraDummy(self.corr)
        copia.x = self.x.copy()
        copia.pars = self.pars.copy()
        return copia

    def aplyCorr(self, alfa, correc):
        self.x += alfa * correc['x']
        self.pi += alfa * correc['pi']

    def calcStepGrad(self, *args, **kwargs):
        return grad_sgra.calcStepGrad(self, *args, **kwargs)

    def showPJ(self, alfaMin=0., alfaMax=5.):
        """Make plots of the P and J curves as functions of alfa."""
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
        ax1.legend()
        ax1.grid()
        ax1.set_xlabel('Step')
        ax1.set_ylabel('P')
        ax1.set_title('P and J vs. step')

        ax2.plot(alfa, J, label='J')
        ax2.plot(self.pars['minPt'], self.pars['minVal'], 'o', label='min_pt')
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
            self.minObjPt.append(sgra.pars['minPt'])
            self.limPPt.append(sgra.pars['limP_step'])
            alfa_0 = alfa

    def checkPerf(self):
        """Check performance of calcStepGrad on the runs"""
        nTest = 0
        for alfa, stepMan, sgra in zip(self.stepList, self.stepManList, self.sgraList):
            if sgra.pars['minPt'] < sgra.pars['limP_step']:
                alfa_exact = sgra.pars['minPt']
            else:
                alfa_exact = sgra.pars['limP_step']
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
    # corr = {'x': 1.,
    #         'pi': numpy.array([1e-10]),
    #         'dJdStepTheo': -1.}
    # thesePars = parsDefault
    # thesePars['limP_step'] = 1.5
    # sgra = SgraDummy(corr, pars=thesePars)
    #
    # sgra.showPJ()
    #
    # alfa_0 = 0.
    # retry_grad = False
    # stepMan = None
    # alfa, stepMan = sgra.calcStepGrad(sgra.corr, alfa_0, retry_grad, stepMan)

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
    TS = TestSeq(parsList, showPJ=True)
    TS.checkPerf()


