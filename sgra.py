#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:07:35 2017

@author: levi
"""

import rest_sgra, grad_sgra, hist_sgra, numpy, copy, os, functools
import matplotlib.pyplot as plt
from lmpbvp import LMPBVPhelp
from utils import simp, getNowStr
from multiprocessing import Pool
from itsme import problemConfigurationSGRA
#from utils import ddt


class binFlagDict(dict):
    """Class for binary flag dictionaries.
    Provides good grounding for any settings or options dictionary. """

    def __init__(self, inpDict=None, inpName='options'):
        super().__init__()
        if inpDict is None:
            inpDict = {}
        self.name = inpName
        for key in inpDict.keys():
            self[key] = inpDict[key]

    def copy(self):
        """
        :return: a shallow copy of itself.
        """

        return binFlagDict(inpDict=self,inpName=self.name)

    # def setAll(self, tf=True, opt=None):
    #     if opt is None:
    #         opt = {}
    #     for key in self.keys():
    #         self[key] = (tf and opt.get(key,True))
    def setAll(self, tf=True, opt=None):
        if opt is None:
            for key in self.keys():
                self[key] = tf
        else:
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

    probName = 'probSGRA' # this also gets altered in each instance of the problem

    def __init__(self, parallel=None, MaxIterGrad=10000):
        # these numbers should not make any sense;
        # they should change with the problem
        if parallel is None:
            parallel = {}
        N,n,m,p,q,s = 50000,4,2,1,3,2

        self.log = None  # For logging, this will be updated later
        self.N, self.n, self.m, self.p, self.q, self.s = N, n, m, p, q, s
        # TODO: since this formula actually does not change, this should be a method...
        self.Ns = 2*n*s + p
        self.dt = 1.0/(N-1)
        self.t = numpy.linspace(0,1.0,N)

        self.x = numpy.zeros((N,n))
        self.u = numpy.zeros((N,m))
        self.pi = numpy.zeros(p)
        self.lam = numpy.zeros((N,n))
        self.mu = numpy.zeros(q)

        # If the problem has unnecessary variations (typically, boundary conditions
        # containing begin of arc specified values for states), it should override this
        # attribute with a cropped version of the Ns+1 identity matrix, omitting the
        # columns corresponding to the unnecessary variations, according to the order
        # shown in equation (26) in Miele and Wang(2003).
        self.omitEqMat = numpy.eye(self.Ns+1)
        self.omitVarList = list(range(self.Ns+1))
        self.omit = False

        self.boundary, self.constants, self.restrictions = {}, {}, {}
        self.P, self.Q, self.I, self.J = 1.0, 1.0, 1.0, 1.0
        self.Pint, self.Ppsi = 1.0, 1.0
        self.Iorig, self.I_pf = 1., 1
        self.J_Lint, self.J_Lpsi = 1., 1.
        self.Qx, self.Qu, self.Qp, self.Qt = 1., 1., 1., 1.

        # These attributes will be used if the problem makes use of the
        # @utils.avoidRepCalc decorator.
        self.f = numpy.empty((N,s))
        self.fOrig, self.f_pf = numpy.empty((N,s)), numpy.empty((N,s))
        self.phi = numpy.empty((N,n,s))
        #self.psi = numpy.empty(q) # not worth it...
        # self.isUpdated = binFlagDict(inpName='Update status',
        #                              inpDict={'f': False, 'phi': False, 'psi': False,
        #                                       'I': False, 'J': False,
        #                                       'P': False, 'Q': False})
        self.isUpdated = binFlagDict(inpName='Update status',
                                     inpDict={'f': False, 'phi': False,
                                              'I': False,
                                              'P': False, 'Q': False})

        # Existence of exact solutions, default is not having them.
        self.hasExactSol = False
        # Their parameters and errors
        self.I_opt = -1.
        self.relErrI_opt = -1.
        self.x_opt = None
        self.u_opt = None
        self.pi_opt = None
        self.rmsErr_x_opt = -1.
        self.rmsErr_u_opt = -1.
        self.rmsErr_pi_opt = -1.

        # Histories
        self.declHist(MaxIterGrad=MaxIterGrad)
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
                     'eig':True,
                     'hamCheck':True},
                                inpName='Plot saving options')

        # Parallelism options
        self.isParallel = {'gradLMPBVP': parallel.get('gradLMPBVP',False),
                           'restLMPBVP': parallel.get('restLMPBVP',False)}
        self.timer = -1. # timer for the run

        self.CalcErrMat = None#numpy.zeros((self.N))

    # Basic "utility" methods

    def copy(self):
        """Copy the solution. It is useful for applying corrections, generating
        baselines for comparison, etc.

        This used to make deep, recursive copies, but shallow ones are much faster."""

        # Do the copy - shallow copy is much faster, but the elements are passed by
        # reference... special attention must be given to the elements that will be
        # changed
        newSol = copy.copy(self)

        # Assign x, u and pi to proper copies of the former values.
        newSol.x = self.x.copy()
        newSol.u = self.u.copy()
        newSol.pi = self.pi.copy()
        newSol.isUpdated = self.isUpdated.copy()
        newSol.P = self.P + 0. # To the change the reference from self.P ...

        return newSol

    def aplyCorr(self,alfa,corr):
        self.log.printL("\nApplying alfa = "+str(alfa))
        self.x  += alfa * corr['x']
        self.u  += alfa * corr['u']
        self.pi += alfa * corr['pi']
        # Set all the statuses (?) to False, since the states, controls and parameters
        # were very likely changed
        self.isUpdated.setAll(False)

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
                piIsTime=True,intv=None):
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
            arcsDur = pi.copy()
        else:
            uperBnd = t[-1] * s
            arcsDur = [t[-1]] * s

        # First and last arcs to be plotted
        FrstArc, LastArc = 0, s
        # Indexes for begin and and of plot
        BginPlotIndx, EndPlotIndx = [0] * s, [N] * s

        # if no interval is specified, default to a full plot.
        # else, we need to find out what gets plotted!
        if intv is not None:

            # Check consistency of requested time interval,
            # override if necessary
            if intv[0] < lowrBnd or intv[1] > uperBnd:
                msg = "plotCat: bounds: [{},{}], ".format(lowrBnd, uperBnd) + \
                      "given interval: [{},{}]".format(intv[0], intv[1]) + \
                      "\nInadequate time interval used! Ignoring..."
                self.log.printL(msg)
                if intv[0] < lowrBnd:
                    intv[0] = lowrBnd - 1.
                if intv[1] > uperBnd:
                    intv[1] = uperBnd + 1.

            # Find out which arcs get plotted, which don't; and for those who
            # do get plotted, find the proper indexes to start and end plot.

            # This is a partial sum of times (like a CDF)
            PartSumTimeArry = numpy.zeros(s)
            # accumulated time since beginning
            accTime = 0.0
            # Flag for finding the first and last arcs
            MustFindFrstArc, MustFindLastArc = True, True
            for arc in range(s):
                # update accumulated time, partial sums array
                accTime += arcsDur[arc]
                PartSumTimeArry[arc] = accTime

                dtd = dt * arcsDur[arc] # dimensional dt
                if MustFindFrstArc and intv[0] <= PartSumTimeArry[arc]:
                    # Found the first arc!
                    MustFindFrstArc=False
                    FrstArc = arc
                    # Find the index for start of plot
                    indFlt = (intv[0] - PartSumTimeArry[arc]+arcsDur[arc]) /\
                             dtd

                    BginPlotIndx[arc] = max([int(numpy.floor(indFlt)),0])
                if MustFindLastArc and intv[1] <= PartSumTimeArry[arc]:
                    # Found last arc!
                    MustFindLastArc = False
                    LastArc = arc+1 # python indexing
                    # Find the index for end of plot
                    indFlt = (intv[1] - PartSumTimeArry[arc]+arcsDur[arc]) /\
                             dtd

                    EndPlotIndx[arc] = min([int(numpy.ceil(indFlt))+1,N]) # idem
            #
        #

        # Accumulated time between arcs
        accTime = sum(arcsDur[0:FrstArc])
        # Flag for labeling plots
        # (so that only the first plotted arc is labeled)
        mustLabl = True

        for arc in range(FrstArc,LastArc):
            # Plot the function at each arc.
            # Label only the first drawn arc
            indBgin, indEnd = BginPlotIndx[arc], EndPlotIndx[arc]
            if mustLabl:
                plt.plot(accTime + arcsDur[arc] * t[indBgin:indEnd],
                         func[indBgin:indEnd, arc],
                         mark + color, label=labl)
                mustLabl = False
            else:
                plt.plot(accTime + arcsDur[arc] * t[indBgin:indEnd],
                         func[indBgin:indEnd, arc],
                         mark + color)
            #
            # Plot arc beginning with circle
            if indBgin == 0:
                plt.plot(accTime + arcsDur[arc] * t[0], func[0, arc],
                         'o' + color, ms=markSize)
            # Plot the last point of the arc, with square
            if indEnd == N:
                plt.plot(accTime + arcsDur[arc] * t[-1], func[-1, arc],
                         's' + color, ms=markSize)

            #
            # Correct accumulated time, for next arc
            accTime += arcsDur[arc]

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

    def prepCalcErrMat(self):
        """Prepare the error-calculating matrix.

        It is basically a (N-1) x (N-1) matrix with the structure as follows:
        M = [[ 1  0  0  0  0 ...  0  0],
             [-1  1  0  0  0 ...  0  0],
             [ 1 -1  1  0  0 ...  0  0],
             [-1  1 -1  1  0 ...  0  0],
                             ... ,
             [-1  1 -1  1 -1 ...  1  0],
             [ 1 -1  1 -1  1 ... -1  1]] .

        """
        CalcErrMat = numpy.zeros((self.N-1,self.N-1))
        # auxiliary column with 1, -1, 1, -1, ...
        auxCol = numpy.ones(self.N-1)
        auxCol[1:self.N-1:2] = -1.

        for j in range(self.N-1):
            CalcErrMat[j:,j] = auxCol[0:(self.N-1-j)]

        self.CalcErrMat = CalcErrMat

    def calcOptErr(self):
        """ Calculate the error with respect to the optimal solution."""

        # Root mean square errors
        self.rmsErr_x_opt = numpy.sqrt(numpy.einsum('nis,nis->',self.x_opt-self.x,
                                           self.x_opt-self.x)/self.N/self.s)
        self.rmsErr_u_opt = numpy.sqrt(numpy.einsum('mis,mis->', self.u_opt - self.u,
                                        self.u_opt - self.u) / self.N / self.s)
        self.rmsErr_pi_opt = numpy.sqrt(numpy.einsum('s,s->', self.pi_opt - self.pi,
                                        self.pi_opt - self.pi) / self.s)
        # Relative error of I with respect to I optimum
        self.relErrI_opt = (self.I - self.I_opt) / self.I_opt
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
        return 0., 0., 0.

    def calcF(self,*args,**kwargs):
        return numpy.empty((self.N,self.s)), numpy.empty((self.N,self.s)), \
               numpy.empty((self.N,self.s))

    def calcPhi(self,*args,**kwargs):
        return numpy.empty((self.N,self.n,self.s))

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

    def showHistQvsI(self,*args,**kwargs):
        return hist_sgra.showHistQvsI(self,*args,**kwargs)

    def showHistGradStep(self,*args,**kwargs):
        return hist_sgra.showHistGradStep(self,*args,**kwargs)

    def showHistGRrate(self,*args,**kwargs):
        return hist_sgra.showHistGRrate(self,*args,**kwargs)

    def showHistObjEval(self,*args,**kwargs):
        return hist_sgra.showHistObjEval(self,*args,**kwargs)

    def copyHistFrom(self,*args,**kwargs):
        return hist_sgra.copyHistFrom(self,*args,**kwargs)

#%% LMPBVP
    def calcErr(self, plotErr=False):
        """Calculate the "error", that is the residual between phi and dx/dt."""

        # Old method (which is adequate for Euler + leapfrog, actually...)
        # phi = self.calcPhi()
        # err = phi - ddt(self.x,self.N)

        # New method, adequate for trapezoidal integration scheme
        phi = self.calcPhi()
        err = numpy.zeros((self.N,self.n,self.s))

        m = .5*(phi[0,:,:] + phi[1,:,:]) + \
                -(self.x[1,:,:]-self.x[0,:,:])/self.dt
        err[0,:,:] = m

        aux = numpy.empty((self.N-1,self.n,self.s))
        aux[1:,:,:] = phi[2:,:,:] + phi[1:(self.N-1),:,:]  + \
              - 2. * (self.x[2:,:,:]-self.x[1:(self.N-1)]) / self.dt
        aux[0,:,:] = m # err[1,:,:] = m as well.
        # this computes err[k+1,:,:] = aux[k,:,:] - err[k,:,:] for each k
        err[1:,:,:] = numpy.einsum('ij,jks->iks',self.CalcErrMat,aux)

        if plotErr:  # plot the error (for debugging purposes)
            indFig = 0
            for arc in range(self.s):
                for j in range(self.n):
                    plt.figure(indFig)
                    plt.plot(self.t, err[:,j,arc])
                    plt.xlabel('Time, -')
                    plt.ylabel('Residual')
                    plt.title('Residual component, state #{}, arc #{}'.format(j,arc))
                    plt.grid()
                    indFig += 1
            plt.show()
        return err

    def LMPBVP(self,rho=0.0,isParallel=False):
        """Solves the Linear Multi-Point Boundary Value Problem, which is
        either grad (rho=1) or rest (rho=0). """

        # create the LMPBVP helper object
        helper = LMPBVPhelp(self,rho)

        # get proper range according to grad or rest and omit or not
        if rho > .5 and self.omit:
            # Grad and omit: use only the elements from the omitted list
            rang = self.omitVarList
        else:
            # Rest or no omit: use all the elements
            rang = list(range(self.Ns+1))

        if isParallel:
            pool = Pool()
            res = pool.map(helper.propagate, rang)
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
            for j in rang:
                outp = helper.propagate(j)
                res.append(outp)
            #
        #

        A,B,C,lam,mu = helper.getCorr(res,self.log)
        corr = {'x':A, 'u':B, 'pi':C}

        # these are all non-essential for the algorithm itself
        if rho > 0.5:
            if self.save.get('eig',False):
                helper.showEig(self.N,self.n,self.s)#,mustShow=True)
                self.savefig(keyName='eig',fullName='eigenvalues')

            if self.save.get('lambda', False):
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
            dJdStep = -BB-CC; corr['dJdStepTheo'] = -self.Q#dJdStep

            self.log.printL("\nBB = {:.4E}".format(BB) + \
                            ", CC = {:.4E},".format(CC) + \
                            " dJ/dAlfa = {:.4E}".format(dJdStep))

            if self.save.get('var', False):
                self.plotSol(opt={'mode':'var','x':A,'u':B,'pi':C})
                self.plotSol(opt={'mode':'var','x':A,'u':B,'pi':C},
                             piIsTime=False)

            #self.log.printL("\nWaiting 5.0 seconds for lambda/corrections check...")
            #time.sleep(5.0)
            #input("\n@Grad: Waiting for lambda/corrections check...")
        #

        return corr, lam, mu

#%% Validation
    def calcHam(self):
        """Calculate Hamiltonian."""

        H,_,_ = self.calcF()
        phi = self.calcPhi()

        H = H - numpy.einsum('nis,nis->ns',self.lam,phi)

        return H

    def checkHamMin(self,mustPlot=False):
        """Check Hamiltonian minimality conditions."""

        # TODO: something
        self.log.printL("\nChecking Hamiltonian conditions.")
        H0 = self.calcHam()

        warnMsg = "\nHAMILTONIAN CHECK FAILED!\n"
        testFail = False

        if mustPlot:
            colorList = ['k','b','r', 'g','y','m','c']; c = 0
            nc = len(colorList)

        # For each control, check for the effect of small variations on H
        delta = 0.01
        for j in range(self.m):
            # get a dummy copy of solution for applying variations
            dummySol = self.copy()
            dummySol.isUpdated.setAll(False)
            # apply positive increment
            dummySol.u[:,j,:] += delta
            DHp = dummySol.calcHam() - H0
            argMinDHp = numpy.argmin(DHp)
            indMinDHp = numpy.unravel_index(argMinDHp,(self.N,self.s))
            minDHp = DHp[indMinDHp]
            msg = "Positive increment on control #{}:".format(j)
            if minDHp>0.:
                msg += " min H-H0 = {:.1E} > 0.".format(minDHp)
            else:
                msg += " min H-H0 = {:.1E} < 0 !".format(minDHp)
                msg += warnMsg
                testFail = True

            self.log.printL(msg)
            if mustPlot:
                labl = 'DH(u'+str(j)+'+)'

                # noinspection PyUnboundLocalVariable
                self.plotCat(DHp, labl=labl, color=colorList[c%nc],
                             piIsTime=False)
                c+=1

            # apply negative increment
            dummySol.u[:, j, :] += -2. * delta
            dummySol.isUpdated.setAll(False)
            DHm = dummySol.calcHam() - H0
            argMinDHm = numpy.argmin(DHm)
            indMinDHm = numpy.unravel_index(argMinDHm, (self.N, self.s))
            minDHm = DHm[indMinDHm]
            msg = "Negative increment on control #{}:".format(j)
            if minDHm > 0.:
                msg += " min H-H0 = {:.1E} > 0.".format(minDHm)
            else:
                msg += " min H-H0 = {:.1E} < 0 !".format(minDHm)
                msg += warnMsg
                testFail = True

            self.log.printL(msg)
            if mustPlot:
                labl = 'DH(u' + str(j) + '-)'
                # noinspection PyUnboundLocalVariable
                self.plotCat(DHm, labl=labl, color=colorList[c%nc],
                             piIsTime=False)
                c+=1

        if testFail:
           self.log.printL("\nSome test has failed. "
                           "Solution is not truly optimal...")
        else:
            self.log.printL("\nAll tests passed.")

        if mustPlot:
            plt.xlabel("Non-dim. time [-]")
            plt.grid(True)
            plt.legend()
            plt.title("Hamiltonian variations")
            self.savefig(keyName='hamCheck',fullName='Hamiltonian check')

        # TODO: add tests for pi conditions as well!
