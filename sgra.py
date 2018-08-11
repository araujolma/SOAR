#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:07:35 2017

@author: levi
"""

import rest_sgra, grad_sgra, hist_sgra, numpy, copy, os
import matplotlib.pyplot as plt
from lmpbvp import LMPBVPhelp
from utils import simp
from multiprocessing import Pool
from itsme import problemConfigurationSGRA
#from utils import ddt


class binFlagDict(dict):
    """Class for binary flag dictionaries.
    Provides good grounding for any settings or options dictionary. """

    def __init__(self,inpDict={},inpName='options'):
        self.name = inpName
        for key in inpDict.keys():
            self[key] = inpDict[key]

    def setAll(self,tf=True,opt={}):
        for key in self.keys():
            self[key] = (tf and opt.get(key,True))

#        self.log.printL("\nSetting '"+self.name+"' as follows:")
#        self.log.pprint(self)

class sgra():
    """Class for a general instance of the SGRA problem.

    Here are all the methods and variables that are independent of a specific
    instance of a problem.

    Each instance of an optimization problem must then inherit these methods
    and properties. """

    probName = 'probSGRA'

    def __init__(self,parallel={}):
        # these numbers should not make any sense;
        # they should change with the problem
        N,n,m,p,q,s = 50000,4,2,1,3,2

        self.N, self.n, self.m, self.p, self.q, self.s = N, n, m, p, q, s

        self.x = numpy.zeros((N,n))
        self.u = numpy.zeros((N,m))
        self.pi = numpy.zeros(p)
        self.lam = numpy.zeros((N,n))
        self.mu = numpy.zeros(q)

        self.boundary, self.constants, self.restrictions = {}, {}, {}
        self.P, self.Q, self.I, self.J = 1.0, 1.0, 1.0, 1.0

        # Histories
        self.declHist()

        self.tol = {'P':1e-7,'Q':1e-7}

        # Debugging options
        tf = False
        self.dbugOptRest = binFlagDict(inpDict={'pausRest':tf,
                            'pausCalcP':tf,
                            'plotP_int':tf,
                            'plotP_intZoom':tf,
                            'plotIntP_int':tf,
                            'plotSolMaxP':tf,
                            'plotRsidMaxP':tf,
                            'plotErr':tf,
                            'plotCorr':tf,
                            'plotCorrFin':tf},\
                                        inpName='Debug options for Rest')
        tf = False#True#
        self.dbugOptGrad = binFlagDict(inpDict={'pausGrad':tf,
                            'pausCalcQ':tf,
                            'prntCalcStepGrad':tf,
                            'plotCalcStepGrad': tf,
                            'pausCalcStepGrad':tf,
                            'manuInptStepGrad': tf,
                            'plotQx':tf,
                            'plotQu':tf,
                            'plotLam':tf,
                            'plotQxZoom':tf,
                            'plotQuZoom':tf,
                            'plotQuComp':tf,
                            'plotQuCompZoom':tf,
                            'plotSolQxMax':tf,
                            'plotSolQuMax':tf,
                            'plotCorr':tf,
                            'plotCorrFin':tf,
                            'plotF':tf,
                            'plotFint':tf},\
                                        inpName='Debug options for Grad')

        # Solution plot saving status:
        self.save = binFlagDict(inpDict={'currSol':True,
                     'histP':True,
                     'histQ':True,
                     'histI':True,
                     'histGradStep':True,
                     'traj':True,
                     'comp':True,
                     'eig':True},\
                                inpName='Plot saving options')

        # Paralellism options
        self.isParallel = dict()
        self.isParallel['gradLMPBVP'] = parallel.get('gradLMPBVP',False)
        self.isParallel['restLMPBVP'] = parallel.get('restLMPBVP',False)


    def copy(self):
        """Copy the solution. It is useful for applying corrections, generating
        baselines for comparison, etc.
        Special care must be given for the logging object, however."""

        # Get logging object (always by reference)
        log = self.log
        # Clear the reference in the solution
        self.log = None
        # Do the copy
        newSol = copy.deepcopy(self)
        # Point the logging object back into the solutions (original and copy)
        newSol.log = log
        self.log = log
        return newSol

    def aplyCorr(self,alfa,corr):
        self.log.printL("\nApplying alfa = "+str(alfa))
        self.x  += alfa * corr['x']
        self.u  += alfa * corr['u']
        self.pi += alfa * corr['pi']

    def initGues(self):
        # Must be implemented by child classes
        pass

    def loadParsFromFile(self,file):
        pConf = problemConfigurationSGRA(fileAdress=file)
        pConf.sgra()

        N = pConf.con['N']
        tolP = pConf.con['tolP']
        tolQ = pConf.con['tolQ']

        for key in ['GSS_PLimCte','GSS_stopStepLimTol','GSS_stopObjDerTol',\
                    'GSS_stopNEvalLim','GSS_findLimStepTol']:
            self.constants[key] = pConf.con[key]

        for key in ['pi_min','pi_max']:
            self.restrictions[key] = pConf.con[key]

        self.tol = {'P': tolP,
                    'Q': tolQ}
        self.N = N

        self.dt = 1.0/(N-1)
        self.t = numpy.linspace(0,1.0,N)

    def printPars(self):
        dPars = self.__dict__
#        keyList = dPars.keys()
        self.log.printL("These are the attributes for the current solution:\n")
        self.log.pprint(dPars)

    def plotCat(self,func,mark='',markSize=1.,color='b',labl='',\
                piIsTime=True,intv=[]):
        """Plot a given function with several subarcs.
            Since this function should serve all the SGRA instances, the pi
            parameters (if exist!) are not necessarily the times for each
            subarc. Hence, the optional parameter "piIsTime".

            However, this function does consider the arcs to be concatenated.
            If this property does not hold for any future problems to be
            considered, then the function must be rewritten.
        """

        s, t, N = self.s, self.t, self.N
        pi = self.pi
        dt = 1.0/(N-1)
        dtd = dt

        # Set upper and lower bounds
        lowrBnd = 0.0
        if piIsTime:
            uperBnd = pi.sum()
        else:
            TimeDur = t[-1]
            uperBnd = TimeDur * s

        if len(intv)==0:
            intv = [lowrBnd,uperBnd]

        # Check consistency of requested time interval, override if necessary
        if intv[0] < lowrBnd or intv[1] > uperBnd:
            self.log.printL("plotCat: inadequate time interval used!" + \
                            " Ignoring...")
            if intv[0] < lowrBnd:
                intv[0] = lowrBnd
            if intv[1] > uperBnd:
                intv[1] = uperBnd

        accTime = 0.0
        mustLabl = True
        isBgin = True

        for arc in range(s):
            if piIsTime:
                TimeDur = pi[arc]
                dtd = dt * TimeDur

            # check if this arc gets plotted at all
            #print("\narc =",arc)
            #print("accTime =",accTime)
            #print("TimeDur =",TimeDur)
            #print("intv =",intv)
            #print("condition1:",accTime <= intv[1])
            #print("condition2:",accTime + TimeDur >= intv[0])
            if (accTime <= intv[1]) and (accTime + TimeDur >= intv[0]):

                # From this point on, the arc will be plotted.
                # Find the index for the first point to be plotted:

                if isBgin:
                    # accTime + ind * dtd = intv[0]
                    indBgin = int((intv[0] - accTime)/dtd)
                    isBgin = False
                    if intv[0] <= accTime:
                        plt.plot(accTime + TimeDur*t[0],func[0,arc],'o'+color,\
                                 ms=markSize)
                else:
                    indBgin = 0
                    # arc beginning with circle
                    plt.plot(accTime + TimeDur*t[0],func[0,arc],'o'+color,\
                             ms=markSize)

                #print("indBgin =",indBgin)
                if accTime + TimeDur > intv[1]:
                    indEnd = int((intv[1] - accTime)/dtd)
                    if indEnd == (N-1):
                        plt.plot(accTime + TimeDur*t[-1], \
                         func[-1,arc],'s'+color,ms=markSize)
                else:
                    indEnd = N-1
                    # arc end with square
                    plt.plot(accTime + TimeDur*t[-1],func[-1,arc],'s'+color,\
                             ms=markSize)

                #print("indEnd =",indEnd)

                # Plot the function at each arc.
                #Label only the first drawed arc
                if mustLabl:
                    plt.plot(accTime + TimeDur * t[indBgin:indEnd], \
                             func[indBgin:indEnd,arc],\
                             mark+color,label=labl)
                    mustLabl = False
                else:
                    plt.plot(accTime + TimeDur * t[indBgin:indEnd], \
                             func[indBgin:indEnd,arc],\
                             mark+color)
                #
            #
            accTime += TimeDur

    def savefig(self,keyName='',fullName=''):
        if self.save.get(keyName,'False'):
#            fileName = self.log.folderName + '/' + self.probName + '_' + \
#                        keyName + '.pdf'
            fileName = self.log.folderName + os.sep + keyName + '.pdf'
            self.log.printL('Saving ' + fullName + ' plot to ' + fileName + \
                            '!')
            try:
                plt.savefig(fileName, bbox_inches='tight', pad_inches=0.1)
            except:
                self.log.printL("Sorry, pdf saving failed... " + \
                                "Are you using Windows?\n" + \
                                "Anyway, you can always load the object " + \
                                "and use some of its plotting methods "+ \
                                "later, I guess.")
        else:
            plt.show()
        #
        plt.clf()
        plt.close('all')

#%% Just for avoiding compatibilization issues with other problems
    # These methods are all properly implemented in probRock class.

    def plotTraj(self,*args,**kwargs):
        self.log.printL("plotTraj: unimplemented method.")
        pass

    def compWith(self,*args,**kwargs):
        self.log.printL("compWith: unimplemented method.")
        pass

    def plotSol(self,*args,**kwargs):
        titlStr = "Current solution"

        plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
        Np = self.n + self.m


        # First state (just because of the "title"...)
        plt.subplot2grid((Np,1),(0,0))
        self.plotCat(self.x[:,0,:],piIsTime=False)
        plt.grid(True)
        plt.ylabel("State #1")
        plt.title(titlStr)

        ind = 1
        for i in range(1,self.n):
            plt.subplot2grid((Np,1),(ind,0))
            ind+=1
            self.plotCat(self.x[:,i,:],piIsTime=False)
            plt.grid(True)
            plt.ylabel("State #"+str(i+1))

        # Controls
        for i in range(self.m):
            plt.subplot2grid((Np,1),(ind,0))
            ind+=1
            self.plotCat(self.u[:,i,:],piIsTime=False)
            plt.grid(True)
            plt.ylabel("Control #"+str(i+1))

        self.savefig(keyName='currSol',fullName='solution')

    def calcI(self,*args,**kwargs):
        pass

    def calcF(self,*args,**kwargs):
        pass

#%% RESTORATION-WISE METHODS

    def rest(self,*args,**kwargs):
        rest_sgra.rest(self,*args,**kwargs)

    def calcStepRest(self,*args,**kwargs):
        return rest_sgra.calcStepRest(self,*args,**kwargs)

    def calcP(self,*args,**kwargs):
        return rest_sgra.calcP(self,*args,**kwargs)

#%% GRADIENT-WISE METHODS

    def grad(self,*args,**kwargs):
        return grad_sgra.grad(self,*args,**kwargs)

    def calcStepGrad(self,*args,**kwargs):
        return grad_sgra.calcStepGrad(self,*args,**kwargs)

    def calcJ(self,*args,**kwargs):
        return grad_sgra.calcJ(self,*args,**kwargs)

    def calcQ(self,*args,**kwargs):
        return grad_sgra.calcQ(self,*args,**kwargs)

    def plotQRes(self,args):
        return grad_sgra.plotQRes(self,args)

    def plotF(self,*args,**kwargs):
        return grad_sgra.plotF(self,*args,**kwargs)

#%% HISTORY-RELATED METHODS (P, Q, step sizes, events)
    def declHist(self,*args, **kwargs):
        return hist_sgra.declHist(self, *args, **kwargs)

    def updtEvntList(self,*args,**kwargs):
        return hist_sgra.updtEvntList(self,*args,**kwargs)

#    def updtHistGRrate(self,*args,**kwargs):
#        return hist_sgra.updtHistGRrate(self,*args,**kwargs)

    def updtHistP(self,*args,**kwargs):
        return hist_sgra.updtHistP(self,*args,**kwargs)

    def updtHistRest(self,*args,**kwargs):
        return hist_sgra.updtHistRest(self,*args,**kwargs)

    def showHistP(self,*args,**kwargs):
        return hist_sgra.showHistP(self,*args,**kwargs)

    def updtGradCont(self,*args,**kwargs):
        return hist_sgra.updtGradCont(self,*args,**kwargs)

    def updtHistGrad(self,*args,**kwargs):
        return hist_sgra.updtHistGrad(self,*args,**kwargs)

    def showHistQ(self,*args,**kwargs):
        return hist_sgra.showHistQ(self,*args,**kwargs)

    def showHistI(self,*args,**kwargs):
        return hist_sgra.showHistI(self,*args,**kwargs)

    def showHistGradStep(self,*args,**kwargs):
        return hist_sgra.showHistGradStep(self,*args,**kwargs)

    def showHistGRrate(self,*args,**kwargs):
        return hist_sgra.showHistGRrate(self,*args,**kwargs)

    def copyHistFrom(self,*args,**kwargs):
        return hist_sgra.copyHistFrom(self,*args,**kwargs)

#%% LMPBVP
    def calcErr(self):#,inRest=False):

        # Old method (which is adequate for Euler + leapfrog, actually...)
#        phi = self.calcPhi()
#        err = phi - ddt(self.x,self.N)

        # New method, adequate for trapezoidal intergration scheme
        phi = self.calcPhi()
        err = numpy.zeros((self.N,self.n,self.s))

        #if inRest:
        m = .5*(phi[0,:,:] + phi[1,:,:]) + \
                -(self.x[1,:,:]-self.x[0,:,:])/self.dt
        err[0,:,:] = m
        err[1,:,:] = m
        for k in range(2,self.N):
            err[k,:,:] = (phi[k,:,:] + phi[k-1,:,:]) + \
            -2.0*(self.x[k,:,:]-self.x[k-1,:,:])/self.dt + \
            -err[k-1,:,:]
        #else:
        #    for k in range(2,self.N-1):
        #        err[k,:,:] = phi[k,:,:] + \
        #                    -(self.x[k+1,:,:]-self.x[k-1,:,:])/self.dt


        return err

    def LMPBVP(self,rho=0.0,isParallel=False):

        helper = LMPBVPhelp(self,rho)

        if isParallel:
            pool = Pool()
            res = pool.map(helper.propagate,range(self.Ns+1))
            pool.close()
            pool.join()
        else:
            if rho>0.5:
                self.log.printL("\nRunning GRAD in sequential " + \
                                "(non-parallel) mode...\n")
            else:
                self.log.printL("\nRunning REST in sequential " + \
                                "(non-parallel) mode...\n")
            res = list()
            for j in range(self.Ns+1):
                outp = helper.propagate(j)
                res.append(outp)
            #
        #

        A,B,C,lam,mu = helper.getCorr(res,self.log)
        corr = {'x':A, 'u':B, 'pi':C}

        if rho > 0.5:
            if self.save.get('eig',False):
                helper.showEig(self.N,self.n,self.s)#,mustShow=True)
                self.savefig(keyName='eig',fullName='eigenvalues')

            # TODO: Use the 'self.save' dictionary here as well...
            if self.NIterGrad % 10 == 0:
                self.plotSol(opt={'mode':'lambda'})
                self.plotSol(opt={'mode':'lambda'},piIsTime=False)

            BBvec = numpy.empty((self.N,self.s))
            BB = 0.0
            for arc in range(self.s):
                for k in range(self.N):
                    BBvec[k,arc] = B[k,:,arc].transpose().dot(B[k,:,arc])
                #
                BB += simp(BBvec[:,arc],self.N)
            #

            CC = C.transpose().dot(C)
            dJdStep = -BB-CC; corr['dJdStepTheo'] = dJdStep

            self.log.printL("\nBB = {:.4E}".format(BB) + \
                            ", CC = {:.4E},".format(CC) + \
                            " dJ/dAlfa = {:.4E}".format(dJdStep))
            # TODO: Use the 'self.save' dictionary here as well...
            if self.NIterGrad % 10 == 0:
                self.plotSol(opt={'mode':'var','x':A,'u':B,'pi':C})
                self.plotSol(opt={'mode':'var','x':A,'u':B,'pi':C},\
                             piIsTime=False)

            #self.log.printL("\nWaiting 5.0 seconds for lambda/corrections check...")
            #time.sleep(5.0)
            #input("\n@Grad: Waiting for lambda/corrections check...")
        #

        return corr,lam,mu
