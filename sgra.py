#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:07:35 2017

@author: levi
"""

import rest_sgra, grad_sgra, hist_sgra, numpy, copy, os
import matplotlib.pyplot as plt
from lmpbvp import LMPBVPhelp
from utils import simp, getNowStr
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

class sgra:
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
        # TODO: since this formula actually does not change, this should be a method here...
        self.Ns = 2*n*s + p
        self.dt = 1.0/(N-1)
        self.t = numpy.linspace(0,1.0,N)

        self.x = numpy.zeros((N,n))
        self.u = numpy.zeros((N,m))
        self.pi = numpy.zeros(p)
        self.lam = numpy.zeros((N,n))
        self.mu = numpy.zeros(q)

        self.boundary, self.constants, self.restrictions = {}, {}, {}
        self.P, self.Q, self.I, self.J = 1.0, 1.0, 1.0, 1.0

        # Histories
        self.declHist()
        self.NIterGrad = 0

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
                            'plotCorrFin':tf},
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
                            'plotFint':tf},
                                        inpName='Debug options for Grad')

        # Solution plot saving status:
        self.save = binFlagDict(inpDict={'currSol':True,
                     'histP':True,
                     'histQ':True,
                     'histI':True,
                     'histGradStep':True,
                     'traj':True,
                     'comp':True,
                     'eig':True},
                                inpName='Plot saving options')

        # Parallelism options
        self.isParallel = {'gradLMPBVP': parallel.get('gradLMPBVP',False),
                           'restLMPBVP': parallel.get('restLMPBVP',False)}

    # Basic "utility" methods

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

    def loadParsFromFile(self,file):
        pConf = problemConfigurationSGRA(fileAdress=file)
        pConf.sgra()

        N = pConf.con['N']

        for key in ['GSS_PLimCte','GSS_stopStepLimTol','GSS_stopObjDerTol',
                    'GSS_stopNEvalLim','GSS_findLimStepTol']:
            self.constants[key] = pConf.con[key]

        for key in ['pi_min','pi_max']:
            self.restrictions[key] = pConf.con[key]

        self.tol = {'P': pConf.con['tolP'], 'Q': pConf.con['tolQ']}
        self.N = N
        self.dt = 1.0/(N-1)
        self.t = numpy.linspace(0,1.0,N)

    def printPars(self):
        dPars = self.__dict__
#        keyList = dPars.keys()
        self.log.printL("These are the attributes for the current solution:\n")
        self.log.pprint(dPars)

    def plotCat(self,func,mark='',markSize=1.,color='b',labl='',
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

        # Set upper and lower bounds
        lowrBnd = 0.0
        if piIsTime:
            uperBnd = pi.sum()
        else:
            TimeDur = t[-1]
            uperBnd = TimeDur * s
            dtd = dt #dimensional dt, makes no sense if piIsTime=False

        # if no interval is specified, default to a full plot
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

        # Accumulated time between arcs
        accTime = 0.0
        # Flag for labeling plots
        mustLabl = True
        # Flag that marks if the current arc is the first to be plotted
        isBgin = True

        for arc in range(s):
            # TimeDur: time duration of this arc
            if piIsTime:
                TimeDur = pi[arc]
                dtd = dt * TimeDur #dimensional dt

            # check if this arc gets plotted at all
            # noinspection PyUnboundLocalVariable
            if (accTime <= intv[1]) and (accTime + TimeDur >= intv[0]):

                # From this point on, the arc will be plotted.
                # Find the index for the first point to be plotted:

                # Plot the first point of the arc
                if isBgin:
                    # accTime + ind * dtd = intv[0]
                    # noinspection PyUnboundLocalVariable
                    indBgin = int(numpy.floor((intv[0] - accTime) / dtd))
                    if intv[0] <= accTime:
                        plt.plot(accTime + TimeDur*t[0],func[0,arc],'o'+color,
                                 ms=markSize)
                    isBgin = False
                else:
                    indBgin = 0
                    # arc beginning with circle
                    plt.plot(accTime + TimeDur*t[0],func[0,arc],'o'+color,
                             ms=markSize)

                # Plot the last point of the arc, with square
                if accTime + TimeDur > intv[1]:
                    indEnd = int(numpy.ceil((intv[1] - accTime)/dtd))
                    if indEnd >= (N-1):
                        plt.plot(accTime + TimeDur*t[-1],
                         func[-1,arc],'s'+color,ms=markSize)
                else:
                    indEnd = N
                    plt.plot(accTime + TimeDur*t[-1],func[-1,arc],'s'+color,
                             ms=markSize)

                # Plot the function at each arc.
                # Label only the first drawn arc
                if mustLabl:
                    plt.plot(accTime + TimeDur * t[indBgin:indEnd],
                             func[indBgin:indEnd,arc],
                             mark+color,label=labl)
                    mustLabl = False
                else:
                    plt.plot(accTime + TimeDur * t[indBgin:indEnd],
                             func[indBgin:indEnd,arc],
                             mark+color)
                #
            #
            # Correct accumulated time, for next arc
            accTime += TimeDur

    def savefig(self,keyName='',fullName=''):
        if self.save.get(keyName,'False'):
#            fileName = self.log.folderName + '/' + self.probName + '_' + \
#                        keyName + '.pdf'
            now = ''#'_' + getNowStr()
            fileName = self.log.folderName + os.sep + keyName + now + '.pdf'
            self.log.printL('Saving ' + fullName + ' plot to ' + fileName + \
                            '!')
            try:
                plt.savefig(fileName, bbox_inches='tight', pad_inches=0.1)
            except KeyboardInterrupt:
                raise
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

    def pack(self, f: numpy.array) -> numpy.array:
        """ 'Stacks' an array of size N*s x 1 into a N x s array."""

        Nf = len(f)
        if not (Nf == self.N * self.s):
            raise(Exception("Unable to pack from array with size " +
                            str(Nf) + "."))

        F = numpy.empty((self.N,self.s))

        for arc in range(self.s):
            F[:,arc] = f[arc*self.N : (arc+1)*self.N]

        return F

    def unpack(self,F: numpy.array) -> numpy.array:
        """ 'Unpacks' an array of size N x s into a long N*s array."""

        NF, sF = F.shape
        if not ((NF == self.N) and (sF == self.s)):
            raise(Exception("Unable to unpack array with shape " +
                            str(NF) + " x " +str(sF) + "."))

        f = numpy.empty(self.N*self.s)

        for arc in range(self.s):
            f[arc * self.N : (arc + 1) * self.N] = F[:,arc]

        return f

    def intgEulr(self, df: numpy.array, f0: float):
        """ Integrate a given function, by Euler method.
        Just one initial condition (f0) is required, since the arcs are
        concatenated. """

        f = numpy.empty((self.N,self.s))

        for arc in range(self.s):
            # initial condition
            f[0, arc] = f0
            # dimensional dt
            dtd = self.dt * self.pi[arc]
            for i in range(1, self.N):
                # Integrate by Euler method using derivative, df
                f[i, arc] = f[i - 1, arc] + dtd * df[i - 1,arc]
            # Set initial condition for next arc
            f0 = f[-1, arc]

        return f

    # These methods SHOULD all be properly implemented in each problem class.

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

    # These methods MUST all be properly implemented in each problem class.

    def initGues(self):
        # Must be implemented by child classes
        pass

    def calcI(self,*args,**kwargs):
        pass

    def calcF(self,*args,**kwargs):
        pass

    def calcPhi(self,*args,**kwargs):
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

    def plotQRes(self,*args,**kwargs):
        return grad_sgra.plotQRes(self,*args,**kwargs)

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
    def calcErr(self):

        # Old method (which is adequate for Euler + leapfrog, actually...)
#        phi = self.calcPhi()
#        err = phi - ddt(self.x,self.N)

        # New method, adequate for trapezoidal intergration scheme
        phi = self.calcPhi()
        err = numpy.zeros((self.N,self.n,self.s))

        m = .5*(phi[0,:,:] + phi[1,:,:]) + \
                -(self.x[1,:,:]-self.x[0,:,:])/self.dt
        err[0,:,:] = m
        err[1,:,:] = m
        for k in range(2,self.N):
            err[k,:,:] = (phi[k,:,:] + phi[k-1,:,:]) + \
            -2.0*(self.x[k,:,:]-self.x[k-1,:,:])/self.dt + \
            -err[k-1,:,:]

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
            if self.NIterGrad % 20 == 0:
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

            if self.NIterGrad % 20 == 0:
                self.plotSol(opt={'mode':'var','x':A,'u':B,'pi':C})
                self.plotSol(opt={'mode':'var','x':A,'u':B,'pi':C},
                             piIsTime=False)
            #if self.NIterGrad > 380:
            #    raise Exception("Mandou parar, parei.")

            #self.log.printL("\nWaiting 5.0 seconds for lambda/corrections check...")
            #time.sleep(5.0)
            #input("\n@Grad: Waiting for lambda/corrections check...")
        #

        return corr,lam,mu
