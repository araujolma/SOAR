#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:07:35 2017

@author: levi
"""

import rest_sgra, grad_sgra, numpy, copy, os
import matplotlib.pyplot as plt
from lmpbvp import LMPBVPhelp
from multiprocessing import Pool
from itsme import problemConfigurationSGRA



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

        self.N = N
        self.n = n
        self.m = m
        self.p = p
        self.q = q
        self.s = s

        self.x = numpy.zeros((N,n))
        self.u = numpy.zeros((N,m))
        self.pi = numpy.zeros(p)
        self.lam = numpy.zeros((N,n))
        self.mu = numpy.zeros(q)

        self.boundary = {}
        self.constants = {}
        self.restrictions = {}

        self.P = 1.0
        self.Q = 1.0
        self.I = 1.0

        # Basic maximum number of iterations for grad/rest.
        # May be overriden in the problem definition
        MaxIterRest = 100000
        self.MaxIterRest = MaxIterRest
        self.NIterRest = 0
        self.histStepRest = numpy.zeros(MaxIterRest)
        self.histP = numpy.zeros(MaxIterRest)
        self.histPint = numpy.zeros(MaxIterRest)
        self.histPpsi = numpy.zeros(MaxIterRest)

        MaxIterGrad = 10000
        self.MaxIterGrad = MaxIterGrad
        self.NIterGrad = 0

        self.histStepGrad = numpy.zeros(MaxIterGrad)
        self.histQ = numpy.zeros(MaxIterGrad)
        self.histQx = numpy.zeros(MaxIterGrad)
        self.histQu = numpy.zeros(MaxIterGrad)
        self.histQp = numpy.zeros(MaxIterGrad)
        self.histQt = numpy.zeros(MaxIterGrad)

        self.histI = numpy.zeros(MaxIterGrad)
        self.histIorig = numpy.zeros(MaxIterGrad)
        self.histIpf = numpy.zeros(MaxIterGrad)

        self.histGRrate = numpy.zeros(MaxIterGrad)

        self.tol = {'P':1e-7,'Q':1e-7}

        # Gradient-Restoration EVent List (1-grad, 0-rest)
        self.GREvList = numpy.ones(MaxIterRest+MaxIterGrad,dtype='bool')
        self.GREvIndx = -1

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
                     'comp':True},\
                                inpName='Plot saving options')

        # Paralellism options
        self.isParallel = dict()
        self.isParallel['gradLMPBVP'] = parallel.get('gradLMPBVP',False)
        self.isParallel['restLMPBVP'] = parallel.get('restLMPBVP',False)


    def updtGRrate(self):

        # find last gradient step index in event list
        LastGradIndx = self.GREvIndx - 1

        while self.GREvList[LastGradIndx] == False and LastGradIndx > 0:
            LastGradIndx -= 1

        # number of restorations after last gradient step
        if LastGradIndx == 0:
            nRest = self.GREvIndx-1
        else:
            nRest = self.GREvIndx - 1 - LastGradIndx

        self.histGRrate[self.NIterGrad] = nRest

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
        k = pConf.con['gradStepSrchCte']

        self.tol = {'P': tolP,
                    'Q': tolQ}
        self.constants['gradStepSrchCte'] = k

        self.N = N

        self.dt = 1.0/(N-1)
        self.t = numpy.linspace(0,1.0,N)

    def printPars(self):
        dPars = self.__dict__
#        keyList = dPars.keys()
        self.log.printL("These are the attributes for the current solution:\n")
        self.log.pprint(dPars)

    def plotCat(self,func,mark='',color='b',labl='',piIsTime=True,intv=[]):
        """Plot a given function with several subarcs.
            Since this function should serve all the SGRA instances, the pi
            parameters (if exist!) are not necessarily the times for each
            subarc. Hence, the optional parameter "piIsTime".

            However, this function does consider the arcs to be concatenated.
            If this property does not hold for any future problems to be
            considered, then the function must be rewritten.
        """

        s = self.s
        t = self.t
        pi = self.pi
        N = self.N
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
            self.log.printL("plotCat: inadequate time interval used! Ignoring...")
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
                        plt.plot(accTime + TimeDur*t[0],func[0,arc],'o'+color)
                else:
                    indBgin = 0
                    # arc beginning with circle
                    plt.plot(accTime + TimeDur*t[0],func[0,arc],'o'+color)

                #print("indBgin =",indBgin)
                if accTime + TimeDur > intv[1]:
                    indEnd = int((intv[1] - accTime)/dtd)
                    if indEnd == (N-1):
                        plt.plot(accTime + TimeDur*t[-1], \
                         func[-1,arc],'s'+color)
                else:
                    indEnd = N-1
                    # arc end with square
                    plt.plot(accTime + TimeDur*t[-1],func[-1,arc],'s'+color)

                #print("indEnd =",indEnd)

                # Plot the function at each arc. Label only the first drawed arc
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
                self.log.printL("Sorry, pdf saving failed... Are you using Windows?")
                self.log.printL("Anyway, you can always load the object and use some "+ \
                      "of its plotting methods later, I guess.")
        else:
            plt.show()
        #
        plt.clf()
        plt.close('all')

#%% Just for avoiding compatibilization issues with other problems
    # These methods are all properly implemented in probRock class.

    def plotTraj(self):
        self.log.printL("plotTraj: unimplemented method.")
        pass

    def compWith(self,*args,**kwargs):
        self.log.printL("compWith: unimplemented method.")
        pass

    def plotSol(self,*args,**kwargs):
        titlStr = "Current solution"

        plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
        Np = self.n + self.m


        # First state
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

    def updtHistP(self,alfa,mustPlotPint=False):

        NIterRest = self.NIterRest+1

        P,Pint,Ppsi = self.calcP(mustPlotPint=mustPlotPint)
        self.P = P
        self.histP[NIterRest] = P
        self.histPint[NIterRest] = Pint
        self.histPpsi[NIterRest] = Ppsi
        self.histStepRest[NIterRest] = alfa
        self.NIterRest = NIterRest

    def showHistP(self):
        IterRest = numpy.arange(0,self.NIterRest+1,1)

        if self.histP[IterRest].any() > 0:
            plt.semilogy(IterRest,self.histP[IterRest],'b',label='P')

        if self.histPint[IterRest].any() > 0:
            plt.semilogy(IterRest,self.histPint[IterRest],'k',label='P_int')

        if self.histPpsi[IterRest].any() > 0:
            plt.semilogy(IterRest,self.histPpsi[IterRest],'r',label='P_psi')

        plt.plot(IterRest,self.tol['P']+0.0*IterRest,'-.b',label='tolP')
        plt.title("Convergence report on P")
        plt.grid(True)
        plt.xlabel("Rest iterations")
        plt.ylabel("P values")
        plt.legend(loc="upper left", bbox_to_anchor=(1,1))

        self.savefig(keyName='histP',fullName='P')

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

    def updtHistQ(self,alfa,mustPlotQs=False):
        """ Updates the history of Qs, Is and gradStep. """

        NIterGrad = self.NIterGrad+1

        Q,Qx,Qu,Qp,Qt = self.calcQ(mustPlotQs=mustPlotQs)
        self.Q = Q
        self.histQ[NIterGrad] = Q
        self.histQx[NIterGrad] = Qx
        self.histQu[NIterGrad] = Qu
        self.histQp[NIterGrad] = Qp
        self.histQt[NIterGrad] = Qt
        self.histStepGrad[NIterGrad] = alfa

        I,Iorig,Ipf = self.calcI()
        self.histI[NIterGrad] = I
        self.histIorig[NIterGrad] = Iorig
        self.histIpf[NIterGrad] = Ipf
        self.I = I

        self.NIterGrad = NIterGrad


    def showHistQ(self):
        IterGrad = numpy.arange(1,self.NIterGrad+1,1)

        if self.histQ[IterGrad].any() > 0:
            plt.semilogy(IterGrad,self.histQ[IterGrad],'b',label='Q')

        if self.histQx[IterGrad].any() > 0:
            plt.semilogy(IterGrad,self.histQx[IterGrad],'k',label='Qx')

        if self.histQu[IterGrad].any() > 0:
            plt.semilogy(IterGrad,self.histQu[IterGrad],'r',label='Qu')

        if self.histQp[IterGrad].any() > 0:
            plt.semilogy(IterGrad,self.histQp[IterGrad],'g',label='Qp')

        if self.histQt[IterGrad].any() > 0:
            plt.semilogy(IterGrad,self.histQt[IterGrad],'y',label='Qt')

        plt.plot(IterGrad,self.tol['Q']+0.0*IterGrad,'-.b',label='tolQ')
        plt.title("Convergence report on Q")
        plt.grid(True)
        plt.xlabel("Grad iterations")
        plt.ylabel("Q values")
        plt.legend(loc="upper left", bbox_to_anchor=(1,1))

        self.savefig(keyName='histQ',fullName='Q convergence history')

    def showHistI(self):
        IterGrad = numpy.arange(1,self.NIterGrad+1,1)

        plt.title("Convergence report on I")
        plt.semilogy(IterGrad,self.histI[IterGrad],label='I')
        plt.semilogy(IterGrad,self.histIorig[IterGrad],label='Iorig')
        plt.semilogy(IterGrad,self.histIpf[IterGrad],label='Ipf')
        plt.grid(True)
        plt.xlabel("Grad iterations")
        plt.ylabel("I values")
        plt.legend()

        self.savefig(keyName='histI',fullName='I convergence history')

    def showHistGradStep(self):
        IterGrad = numpy.arange(1,self.NIterGrad+1,1)

        plt.title("Gradient step history")
        plt.semilogy(IterGrad,self.histStepGrad[IterGrad])
        plt.grid(True)
        plt.xlabel("Grad iterations")
        plt.ylabel("Step values")

        self.savefig(keyName='histGradStep',fullName='GradStep convergence history')

    def showHistGRrate(self):
        IterGrad = numpy.arange(1,self.NIterGrad+1,1)

        if self.histGRrate[IterGrad].any() > 0:
            plt.title("Gradient-restoration rate history")
            plt.semilogy(IterGrad,self.histGRrate[IterGrad])
            plt.grid(True)
            plt.xlabel("Grad iterations")
            plt.ylabel("Step values")

            self.savefig(keyName='histGRrate',fullName='Grad-Rest rate history')
        else:
            self.log.printL("showHistGRrate: No positive values. Skipping...")

#%% LMPBVP

    def LMPBVP(self,rho=0.0,isParallel=False):

        helper = LMPBVPhelp(self,rho)

        if isParallel:
            pool = Pool()
            res = pool.map(helper.propagate,range(self.Ns+1))
            pool.close()
            pool.join()
        else:
            if rho>0.5:
                self.log.printL("\nRunning GRAD in sequential (non-parallel) mode...\n")
            else:
                self.log.printL("\nRunning REST in sequential (non-parallel) mode...\n")
            res = list()
            for j in range(self.Ns+1):
                outp = helper.propagate(j)
                res.append(outp)

        A,B,C,lam,mu = helper.getCorr(res,self.log)

        return A,B,C,lam,mu
