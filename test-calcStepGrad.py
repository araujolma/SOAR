#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 12:12:07 2020

@author: levi
"""
import os
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

            # in this case, there is no alfa so that P(alfa) = limP ...
            if max(self.pars['step_J_P_array'][2,:]) < limP:
                if limPtoo:
                    return None, None
                else:
                    return None

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

    def __init__(self, givenParsList, showPJ=False):
        alfa_0 = 0.
        corr = {'x': 1.,
                'pi': numpy.array([1e-10]),
                'dJdStepTheo': -1.}
        retry_grad = False
        stepMan = None
        self.stepError, self.stepRelError, self.nAttempts = [], [], []
        self.sgraList, self.stepManList, self.stepList = [], [], []
        self.stopMotv, self.minObjPt, self.limPPt = [], [], []
        for pars in givenParsList:
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

        # making the results
        headers = ["N", 'obj evals', "Step", "Min obj pt", "P=limP pt", "Error",
                   "Error, %", "Stop motv"]
        # data = {'N': numpy.linspace(0,nTest-1,num=nTest)
        #         'evals': self.nAttempts,
        #         'Min obj pt': self.minObjPt,
        #         'P=limP pt': self.limPPt,
        #         'Step error:'} # not good because loses the order...
        data = []
        for k in range(len(self.sgraList)):
            # get theoretical values for comparison
            alfa_J_min = self.minObjPt[k]
            alfa_P = self.limPPt[k]
            if alfa_P is None:
                alfa_exact = alfa_J_min
            else:
                alfa_exact = min(alfa_J_min, alfa_P)
            # get the value returned by the search
            alfa = self.stepList[k]
            # error, absolute and relative
            self.stepError.append(alfa - alfa_exact)
            self.stepRelError.append((alfa - alfa_exact)/alfa_exact)
            # number of attempts
            self.nAttempts.append(self.stepManList[k].cont)
            #print("\nnTest = {}, alfa_J_min = {}, alfa_P = {}".format(k,
            #                                                          alfa_J_min,
            #                                                          alfa_P))
            data.append((k, self.nAttempts[k], alfa, alfa_J_min,
                         alfa_P, self.stepError[k],
                         self.stepRelError[k]*100., self.stopMotv[k]))

        print("\nTest concluded! Here are the results:")
        print(tabulate(data, headers=headers, floatfmt='.3G'))

class LogParser:
    """A class for the log parser.

    The purpose of this is to parse a given log.txt file of a SGRA run so that
    new algorithms can be tested."""

    def __init__(self, foldername):

        # this parses the .its file as well to get the 'pars'
        pars = self.make_pars_default(foldername)

        # read the whole file
        with open(foldername + os.sep + 'log.txt', 'r') as fhand:
            whole = fhand.read()

        # this is the string that gets printed at the end of every calcGradStep session
        str_match = "> State of stepMan at the end of the run:"
        # split the whole file by that string
        chunks = whole.split(str_match)
        N = len(chunks)
        # amount of "chunks" = amount of 'runs' + 1
        if N == 1: # minimum number of outputs of .split() is 1
            raise Exception("Sorry, key string was not found.")

        # iterate over the chunks to get the information
        thisParsList = []
        for i in range(1,N):
            this_chunk = chunks[i]

            # the idea here is that the end of the dictionary (with }) marks the
            # end of the parameters part
            pars_part = this_chunk.split('}')[0]

            # get the 'histories'
            histObj = self.get_field(pars_part, 'histObj')
            histP = self.get_field(pars_part, 'histP')
            histStep = self.get_field(pars_part, 'histStep')

            sJP_array = numpy.empty((3, len(histStep)))
            # sorting the arrays using the step history
            indxSort = numpy.argsort(histStep)
            sJP_array[0, :] = histStep[indxSort]
            sJP_array[1, :] = histObj[indxSort]
            sJP_array[2, :] = histP[indxSort]

            # store the information on the pars dictionary
            pars['step_J_P_array'] = sJP_array
            pars['P0'] = histP[0]

            thisParsList.append(pars.copy())
        self.parsList = thisParsList

    @staticmethod
    def get_field(pars_chunk:str, field_name:str, dtype:str ='float',
                  from_its=False):
        """Get given field from the text.

        This works for two main 'modes' when getting the information from log files
        or from .its configuration files."""
        # the string to be matched
        if from_its:
            # .its file default (from ConfigParser)
            str_match = field_name + " = "
        else:
            # log.txt file default (from pprint)
            str_match = "'" + field_name + "':"
        # its length
        L = len(str_match)
        # start of the string
        ind1 = pars_chunk.index(str_match)
        # this is the first 'enter'
        if from_its:
            ind2 = pars_chunk.index('\n', ind1)
        else:
            ind2 = pars_chunk.index(',\n', ind1)

        param_str = pars_chunk[(ind1 + L):ind2]

        if '[' in param_str:
            # found a '['; this means it's an array (only 1D arrays are supported)
            is_scalar = False
            #  remove the '[' so the string can be converted to a number properly
            param_str = param_str.replace('[','')
        else:
            is_scalar = True

        if dtype == 'float':
            param = float(param_str)
        elif dtype == 'int':
            param = int(param_str)
        else:
            raise Exception("Unsupported data type '{}'.".format(dtype))

        if is_scalar:
            return param
        else:
            # assemble a list
            ret = [param]
            keep_look = True
            while keep_look:  # the search continues until the first ']'
                ind1 = ind2+2
                ind2 = pars_chunk.index(',\n', ind1)
                param_str = pars_chunk[ind1:ind2]
                if ']' in param_str:
                    param_str = param_str.replace(']', '')
                    keep_look = False

                if dtype == 'float':
                    param = float(param_str)
                elif dtype == 'int':
                    param = int(param_str)
                else:
                    raise Exception("Unsupported data type '{}'.".format(dtype))

                ret.append(param)

            return numpy.array(ret)

    @staticmethod
    def make_pars_default(foldername: str):
        """Read from a .its file and make a pars dictionary"""
        # start with parsDefault, make some alterations
        pars = parsDefault
        pars['plotCalcStepGrad'] = False
        pars['usePPars'] = False
        pars['useObjPars'] = False

        # get the names of the .its files in this folder, continue with the first one
        its_files = [f for f in os.listdir(foldername) if f.endswith('.its')]
        if len(its_files) > 1:
            print("\nWARNING: more than 1 .its files. Proceeding with the first "
                  "one...\n\n")

        # read the whole file
        with open(foldername + os.sep + its_files[0], 'r') as fhand:
            whole = fhand.read()

        # use the get_field method to extract these keys from the file
        for key in ['tolP', 'GSS_PLimCte', 'GSS_stopStepLimTol', 'GSS_stopObjDerTol',
                    'GSS_stopNEvalLim', 'GSS_findLimStepTol']:
            pars[key] = LogParser.get_field(whole, key, from_its=True)

        #print("These are the pars that I got:\n{}".format(pars))
        return pars

if __name__ == "__main__":

    LP = LogParser('probLand_2020_11_30_18_48_29_372897')# + os.sep + 'log.txt')
    TS = TestSeq(LP.parsList)
    TS.checkPerf()

    #input("\nIAE?")

    # thesePars = parsDefault
    # thesePars['limP_step'] = 1.5
    # thesePars['plotCalcStepGrad'] = False
    # parsList = [thesePars]
    #
    # newPars = thesePars.copy()
    # newPars['limP_step'] = 10.
    # parsList.append(newPars)
    #
    # newPars = newPars.copy()
    # newPars['minPt'] = 1e-4
    # parsList.append(newPars)
    #
    # parsNonPar = parsDefault.copy()
    # parsNonPar['limP_step'] = 4.
    # parsNonPar['useObjPars'] = False # this makes it interpolate for calculating J
    # parsNonPar['usePPars'] = False # this makes it interpolate for calculating P
    # sJP = numpy.empty((3, 11))
    # sJP[0,:] = numpy.linspace(0.,10.,num=11)
    # sJP[1,:] = numpy.array([1., 1., 0.7, 0.5, -.25, -1., -.6, .5, .5, .7, 1.]) + 2.
    # sJP[2,:] = parsNonPar['P0'] + 1.1e-5 * sJP[0,:]
    # parsNonPar['step_J_P_array'] = sJP
    # parsList.append(parsNonPar)
    #
    # TS = TestSeq(parsList, showPJ=True)
    # TS.checkPerf()
