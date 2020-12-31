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

parsDefault = {'minVal': 1.,
               'minPt': 2.,
               'sigma': 1.,
               'scale': 2.,
               'P0': 1e-13,
               'limP_step': 5.}#1.5}

def miele_curve(t, pars=None):
    if pars is None:
        pars = parsDefault
    #return -10.*numpy.exp(-((t-2.)/0.5)**2) + numpy.exp((t**2)/10.)
    aux = (t-pars['minPt'])/pars['sigma'] - 1.
    return pars['minVal'] + pars['scale'] * \
           (numpy.exp(-1) + numpy.exp(aux) * aux)

class SgraDummy():
    """aa"""

    def __init__(self,makeDir=False,pars=None):
        self.log = logger('test',isManu=True,makeDir=False,mode='screen')
        #logger('test',makeDir=makeDir)
        self.x = 0.
        self.pi = numpy.array([1.])

        self.tol = {'P': 1e-12}
        self.constants = {'GSS_PLimCte': 1e8,
                          'GSS_stopStepLimTol': 1e-2,
                          'GSS_stopObjDerTol': 1e-4,
                          'GSS_stopNEvalLim': 100,
                          'GSS_findLimStepTol': 1e-2}
        self.restrictions = {'pi_min': [None],
                             'pi_max': [None]}

        if pars is None:
            pars = parsDefault
        self.pars = pars
        self.NIterGrad = 1
        self.dbugOptGrad = {'prntCalcStepGrad': True,
                            'plotCalcStepGrad': True,
                            'pausCalcStepGrad': False}


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
        copia = SgraDummy()
        copia.x = self.x
        return copia

    def aplyCorr(self, alfa, correc):
        self.x += alfa * correc['x']
        self.pi += alfa * correc['pi']

    def calcStepGrad(self, *args, **kwargs):
        return grad_sgra.calcStepGrad(self,*args, **kwargs)

    def showPJ(self, alfaMin=0., alfaMax=5.):
        alfa = numpy.linspace(alfaMin, alfaMax, num=1000)
        P = numpy.empty_like(alfa)
        J = numpy.empty_like(alfa)
        for ind, alfa_ in enumerate(alfa):
            self.aplyCorr(alfa_, corr)
            P_, _, _ = self.calcP()
            aux = self.calcJ()
            self.aplyCorr(-alfa_, corr)
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

if __name__ == "__main__":
    # x = numpy.linspace(0.,5.,num=1000)
    # y = miele_curve(x)
    # plt.plot(x,y)
    # plt.grid()
    # plt.show()

    sgra = SgraDummy()
    corr = {'x': 1.,
            'pi': numpy.array([1e-10]),
            'dJdStepTheo': -1.}

    sgra.showPJ()

    alfa_0 = 0.
    retry_grad = False
    stepMan = None
    sgra.calcStepGrad(corr,alfa_0,retry_grad,stepMan)


