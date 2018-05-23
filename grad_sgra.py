#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:39:08 2017

@author: levi
"""

import numpy
#from utils import ddt, testAlgn
import matplotlib.pyplot as plt

class stepMngr():
    """ Class for the step manager, the object used to calculate the step
    value for the gradient phase. The step is chosen through (rudimentary)
    minimization of an objective function "Obj".

    For now, the objective function looks only into J, the extended cost
    functional."""

    def __init__(self,log, k=1e3, tolP=1e-4, corr = None, prntCond=False):
        self.cont = -1
        self.histStep = list()
        self.histI = list()
        self.histP = list()
        self.histObj = list()
        self.k = k
        self.tolP = tolP
        self.limP = 1e-2#numpy.sqrt(tolP)
        self.corr = corr
        self.log = log
        self.mustPrnt = prntCond
        self.best = {'step':0.0,
                     'obj': None}
        self.maxGoodStep = 1e100
        self.minBadStep = 0.0


    def calcObj(self,P,I,J):
        """This is the function which defines the objective function."""

        # TODO: Find a way to properly parameterize this so that the user can
        # set the objective function by setting numerical (or enumerative)
        # parameters in the .its file.

        return J

#        return (I + (self.k)*P)
#        if P > self.tolP:
#            return (I + (self.k)*(P-self.tolP))
#        else:
#            return I


    def getLast(self):
        """ Get the atributes for the last applied step. """

        P = self.histP[self.cont]
        I = self.histI[self.cont]
        Obj = self.histObj[self.cont]

        return P, I, Obj

    def check(self,alfa,Obj,P):
        """ Perform a check in this step value, updating the limit values for
        searching step values if it is the case."""

        if Obj >= self.Obj0 or Obj < 0.0 or P >= self.limP:
            # Bad point!
            if alfa < self.minBadStep:
                self.minBadStep = alfa
            return False, 1.1 * self.Obj0
        else:
            if alfa > self.maxGoodStep:
                self.maxGoodStep = alfa
            # Good point
            return True, Obj

    def calcBase(self,sol,P0,I0,J0):
        """ Calculate the baseline values, that is, the functions P, I, J for
        the solution with no correction applied.

        Please note that there is no need for this P0, I0 and J0 to come from
        outside of this function... this is only here to avoid calculating
        these guys twice. """

        Obj0 = self.calcObj(P0,I0,J0)
        self.log.printL("> I0 = {:.4E}, ".format(I0) + \
                        "Obj0 = {:.4E}".format(Obj0))
        self.P0 = P0
        self.I0 = I0
        self.J0 = J0
        self.Obj0 = Obj0

        return Obj0

    def tryStep(self,sol,alfa,plotSol=False,plotPint=False):
        """ Try this given step value.

        Some additional procedures are performed, though:
            - update "best step" record ...       """

        self.cont += 1

        if self.mustPrnt:
            self.log.printL("\n> Trying alfa = {:.4E}".format(alfa))

        newSol = sol.copy()
        newSol.aplyCorr(alfa,self.corr)
        if plotSol:
            newSol.plotSol()

        P,_,_ = newSol.calcP(mustPlotPint=plotPint)
        J,_,_,I,_,_ = newSol.calcJ()
        Obj = self.calcObj(P,I,J)
        isOk, Obj = self.check(alfa,Obj,P)


        if isOk:
            if self.mustPrnt:
                self.log.printL("\n> Results:\nalfa = {:.4E}".format(alfa) + \
                      " P = {:.4E}".format(P) + " I = {:.4E}".format(I) + \
                      " J = {:.7E}".format(J) + " Obj = {:.7E}".format(Obj))

            # Update best value (if it is the case)
            if self.best['obj'] is not None:
                if self.best['obj'] > Obj:
                    self.best = {'step': alfa,
                                 'obj': Obj}
                    self.log.printL("\n> Updating best result: \n" + \
                                    "alfa = {:.4E}, ".format(alfa) + \
                                    "Obj = {:.4E}".format(Obj))
            else:
                self.best = {'step': alfa,
                             'obj': Obj}
            #

        else:
            if self.mustPrnt:
                self.log.printL("\nBad point:\nalfa = {:.4E}".format(alfa) + \
                      " P = {:.4E}".format(P) + " I = {:.4E}".format(I) + \
                      " J = {:.7E}".format(J) + " CorrObj = {:.7E}".format(Obj))

        self.histStep.append(alfa)
        self.histP.append(P)
        self.histI.append(I)
        self.histObj.append(Obj)

        return self.getLast()

    def fitQuad(self,alfaList,ObjList):
        """ Fit a quadratic curve through the points given; then use this model
        to find the optimum step value (to minimize objective function)."""

        M = numpy.ones((3,3))
        M[:,0] = alfaList ** 2.0
        M[:,1] = alfaList
        # find the quadratic coefficients
        coefList = numpy.linalg.solve(M,ObjList)
        self.log.printL("\n> Objective list: "+str(ObjList))
        self.log.printL("\n> Step value list: "+str(alfaList))
        self.log.printL("\n> Quadratic interpolation coefficients: " + \
                        str(coefList))

        # this corresponds to the vertex of the parabola
        alfaOpt = -coefList[1]/coefList[0]/2.0
        self.log.printL("> According to this fit, best step value is " + \
                        str(alfaOpt))

        # The quadratic model is obviously wrong if the x^2 coefficient is
        # negative, or if the appointed step is negative.
        # These cases must be handled differently.

        if coefList[0] < 0.0:
            # There seems to be a local maximum nearby.
            # Check direction of max decrease in objective
            gradLeft = (ObjList[1]-ObjList[0])/(alfaList[1]-alfaList[0])
            gradRight = (ObjList[2]-ObjList[1])/(alfaList[2]-alfaList[1])
            self.log.printL("\n> Inverted parabola detected.\n" + \
                            "Slopes: left = {:.4E}".format(gradLeft) + \
                            ", right = {:.4E}".format(gradRight))

            if gradLeft * gradRight > 0.0:
                # Both sides have the same tendency; use medium gradient
                grad = (ObjList[2]-ObjList[0])/(alfaList[2]-alfaList[0])
                print("> Using medium slope: {:.4E}".format(grad))
            else:
                if abs(gradLeft) > abs(gradRight):
                    grad = gradLeft
                    print("> Using left slope.")
                else:
                    grad = gradRight
                    print("> Using right slope.")
                #
            #

            alfaOpt = alfaList[1] - 0.5 * ObjList[1]/grad
            self.log.printL("\n> Using the slope in a Newton-Raphson-ish" + \
                            " iteration, next step value is " + \
                            "{:.4E}...".format(alfaOpt))

            if alfaOpt > self.minBadStep:
                while alfaOpt > self.minBadStep:
                    alfaOpt = .5 * (alfaList[1] + self.minBadStep)
                    self.log.printL("> ...but since that would violate" + \
                                    " the max step condition,\n" + \
                                    " let's bisect to " + \
                                    "{:.4E}".format(alfaOpt) + " instead!")
            else:
                self.log.printL("> ... seems legit!")

        elif alfaOpt < 0.0:
            alfaOpt = 0.9 * alfaList[1] #.5 * min(alfaList)
            self.log.printL("  What to do here? I don't know. Let's stay with " + \
                            "{:.4E} instead.".format(alfaOpt))
            input("> ???")
        else:
            self.log.printL("  Let's do it!")

        return alfaOpt

    def stopCond(self,alfa,driv):
        """ Decide if the step search can be stopped. """

        # TODO: these would be excellent parameters for external
        # configuration file...

        tolDer = 1e-4
        tolAlfaLim = 1e-4
        Ntry = 100

        if alfa > self.minBadStep:
            return False

        elif abs(driv) < tolDer:
            self.log.printL("\n> Stopping step search: low sensitivity" + \
                            " of objective function with step.\n" + \
                            "(local minimum, perhaps?)")
            return True

        elif (self.minBadStep/self.maxGoodStep - 1.0) < tolAlfaLim and \
                driv < 0.0:
            self.log.printL("\n> Stopping step search: high proximity" + \
                            " to the step limit value.")
            return True

        elif self.cont+1 > Ntry:
            self.log.printL("\n> Stopping step search: too many attempts.")
            return True

        else:
            return False

def plotF(self,piIsTime=False):
    """Plot the cost function integrand."""

    self.log.printL("In plotF.")

    argout = self.calcF()

    if isinstance(argout,tuple):
        if len(argout) == 3:
            f, fOrig, fPF = argout
            self.plotCat(f, piIsTime=piIsTime, color='b', labl='Total cost')
            self.plotCat(fOrig, piIsTime=piIsTime, color='k', labl='Orig cost')
            self.plotCat(fPF, piIsTime=piIsTime, color='r', \
                         labl='Penalty function')
        else:
            f = argout[0]
            self.plotCat(f, piIsTime=piIsTime, color='b', labl='Total cost')
    else:
        self.plotCat(f, piIsTime=piIsTime, color='b', labl='Total cost')
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

def plotQRes(self,args):
    "Generic plots of the Q residuals"

    iterStr = "\n(grad iter #" + str(self.NIterGrad) + \
                  ", rest iter #"+str(self.NIterRest) + \
                  ", event #" + str(int((self.EvntIndx+1)/2)) + ")"
    # Qx error plot
    plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
    nm1 = self.n+1
    plt.subplot2grid((nm1,1),(0,0))
    self.plotCat(args['normErrQx'],color='b',piIsTime=False)
    plt.grid(True)
    plt.ylabel("Integrand of Qx")
    titlStr = "Qx = int || dlam - f_x + phi_x^T*lam || " + \
              "= {:.4E}".format(args['Qx'])
    titlStr += iterStr
    plt.title(titlStr)
    errQx = args['errQx']
    for i in range(self.n):
        plt.subplot2grid((nm1,1),(i+1,0))
        self.plotCat(errQx[:,i,:],piIsTime=False)
        plt.grid(True)
        plt.ylabel("ErrQx_"+str(i))
    plt.xlabel("t [s]")
    self.savefig(keyName='Qx',fullName='Qx')

    # Qu error plot
    plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
    mm1 = self.m+1
    plt.subplot2grid((mm1,1),(0,0))
    self.plotCat(args['normErrQu'],color='b',piIsTime=False)
    plt.grid(True)
    plt.ylabel("Integrand of Qu")
    titlStr = "Qu = int || f_u - phi_u^T*lam || = {:.4E}".format(args['Qu'])
    titlStr += iterStr
    plt.title(titlStr)
    errQu = args['errQu']
    for i in range(self.m):
        plt.subplot2grid((mm1,1),(i+1,0))
        self.plotCat(errQu[:,i,:],color='k',piIsTime=False)
        plt.grid(True)
        plt.ylabel("Qu_"+str(i))
    plt.xlabel("t")
    self.savefig(keyName='Qu',fullName='Qu')

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
        self.plotCat(errQp[:,j,:],color='k')
        plt.grid(True)
        plt.ylabel("ErrQp, j ="+str(j))
    plt.xlabel("t [s]")

    self.savefig(keyName='Qp',fullName='Qp')

def calcJ(self):
    N,s,dt = self.N,self.s,self.dt
    #x = self.x

    #phi = self.calcPhi()
    psi = self.calcPsi()
    lam = self.lam
    mu = self.mu
    #dx = ddt(x,N)
    I,Iorig,Ipf = self.calcI()

    func = self.calcErr()#dx-phi
    vetL = numpy.empty((N,s))
    vetIL = numpy.empty((N,s))


    for arc in range(s):
        for t in range(N):
            vetL[t,arc] = lam[t,:,arc].transpose().dot(func[t,:,arc])

    for arc in range(s):
        vetIL[0,arc] = (17.0/48.0) * vetL[0,arc]
        vetIL[1,arc] = vetIL[0,arc] + (59.0/48.0) * vetL[1,arc]
        vetIL[2,arc] = vetIL[1,arc] + (43.0/48.0) * vetL[2,arc]
        vetIL[3,arc] = vetIL[2,arc] + (49.0/48.0) * vetL[3,arc]
        for t in range(4,N-4):
            vetIL[t] = vetIL[t-1,arc] + vetL[t,arc]
        vetIL[N-4,arc] = vetIL[N-5,arc] + (49.0/48.0) * vetL[N-4,arc]
        vetIL[N-3,arc] = vetIL[N-4,arc] + (43.0/48.0) * vetL[N-3,arc]
        vetIL[N-2,arc] = vetIL[N-3,arc] + (59.0/48.0) * vetL[N-2,arc]
        vetIL[N-1,arc] = vetIL[N-2,arc] + (17.0/48.0) * vetL[N-1,arc]

    vetIL *= dt

    Lint = vetIL[N-1,:].sum()
    Lpsi = mu.transpose().dot(psi)
    L = Lint + Lpsi

    J_Lint = Lint
    J_Lpsi = Lpsi
    J_I = I
    J = L + J_I
    strJs = "J = {:.6E}".format(J)+", J_Lint = {:.6E}".format(J_Lint)+\
          ", J_Lpsi = {:.6E}".format(J_Lpsi)+", J_I = {:.6E}".format(J_I)
    self.log.printL(strJs)

    return J, J_Lint, J_Lpsi, I, Iorig, Ipf

def calcQ(self,mustPlotQs=False):
    # Q expression from (15).
    # FYI: Miele (2003) is wrong in oh so many ways...
    self.log.printL("In calcQ.")
    N,n,m,p,s = self.N,self.n,self.m,self.p,self.s
    dt = 1.0/(N-1)

#    x = self.x
#    u = self.u
    lam = self.lam
    mu = self.mu


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
    #dlam = ddt(lam,N)

    Qx = 0.0
    Qu = 0.0
    Qp = 0.0
    Qt = 0.0
    Q = 0.0
    auxVecIntQp = numpy.zeros((p,s))

    errQx = numpy.empty((N,n,s)); normErrQx = numpy.empty((N,s))
    errQu = numpy.empty((N,m,s)); normErrQu = numpy.empty((N,s))
    errQp = numpy.empty((N,p,s)); #normErrQp = numpy.empty(N)

    coefList = numpy.ones(N)
    coefList[0] = 17.0/48.0; coefList[N-1] = coefList[0]
    coefList[1] = 59.0/48.0; coefList[N-2] = coefList[1]
    coefList[2] = 43.0/48.0; coefList[N-3] = coefList[2]
    coefList[3] = 49.0/48.0; coefList[N-4] = coefList[3]
    z = numpy.empty(2*n*s)


    for arc in range(s):
        z[2*arc*n : (2*arc+1)*n] = -lam[0,:,arc]
        z[(2*arc+1)*n : (2*arc+2)*n] = lam[N-1,:,arc]

        # calculate Qx separately. In this way, the derivative avaliation is
        # adequate with the trapezoidal integration method
        med = (lam[1,:,arc]-lam[0,:,arc])/dt -.5*(fx[0,:,arc]+fx[1,:,arc]) + \
                .5 * phix[0,:,:,arc].transpose().dot(lam[0,:,arc]) + \
                .5 * phix[1,:,:,arc].transpose().dot(lam[1,:,arc])

        errQx[0,:,arc] = med
        errQx[1,:,arc] = med
        for k in range(2,N):
            errQx[k,:,arc] = 2.0 * (lam[k,:,arc]-lam[k-1,:,arc]) / dt + \
                        -fx[k,:,arc] - fx[k-1,:,arc] + \
                        phix[k,:,:,arc].transpose().dot(lam[k,:,arc]) + \
                        phix[k-1,:,:,arc].transpose().dot(lam[k-1,:,arc]) + \
                        -errQx[k-1,:,arc]

        for k in range(N):
#            errQx[k,:,arc] = dlam[k,:,arc] - fx[k,:,arc] + \
#                             phix[k,:,:,arc].transpose().dot(lam[k,:,arc])

            errQu[k,:,arc] = fu[k,:,arc] +  \
                            - phiu[k,:,:,arc].transpose().dot(lam[k,:,arc])
            errQp[k,:,arc] = fp[k,:,arc] + \
                            - phip[k,:,:,arc].transpose().dot(lam[k,:,arc])

            normErrQx[k,arc] = errQx[k,:,arc].transpose().dot(errQx[k,:,arc])
            normErrQu[k,arc] = errQu[k,:,arc].transpose().dot(errQu[k,:,arc])

            Qx += normErrQx[k,arc] * coefList[k]
            Qu += normErrQu[k,arc] * coefList[k]
            auxVecIntQp[:,arc] += errQp[k,:,arc] * coefList[k]
        #
    #

    auxVecIntQp *= dt
    Qx *= dt
    Qu *= dt

    resVecIntQp = numpy.zeros(p)
    for arc in range(s):
        resVecIntQp += auxVecIntQp[:,arc]

    resVecIntQp += psip.transpose().dot(mu)
    Qp = resVecIntQp.transpose().dot(resVecIntQp)

    errQt = z + psiy.transpose().dot(mu)
    Qt = errQt.transpose().dot(errQt)

    Q = Qx + Qu + Qp + Qt
    self.log.printL("Q = {:.4E}".format(Q)+": Qx = {:.4E}".format(Qx)+\
          ", Qu = {:.4E}".format(Qu)+", Qp = {:.7E}".format(Qp)+\
          ", Qt = {:.4E}".format(Qt))

    #self.Q = Q

###############################################################################
    if mustPlotQs:
        args = {'errQx':errQx,
                'errQu':errQu,
                'errQp':errQp,
                'Qx':Qx,
                'Qu':Qu,
                'normErrQx':normErrQx,
                'normErrQu':normErrQu,
                'resVecIntQp':resVecIntQp}
        self.plotQRes(args)

###############################################################################

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

    # TODO: break these plots into more conditions

#    if numpy.array(self.dbugOptGrad.values()).any:
#        print("\nDebug plots for this calcQ run:")
#
#        if self.dbugOptGrad['plotQx']:
#            plt.plot(self.t,normErrQx)
#            plt.grid(True)
#            plt.title("Integrand of Qx")
#            plt.show()
#
#        if self.dbugOptGrad['plotQu']:
#            plt.plot(self.t,normErrQu)
#            plt.grid(True)
#            plt.title("Integrand of Qu")
#            plt.show()
#
#        # for zoomed version:
#        indMaxQx = normErrQx.argmax()
#        ind1 = numpy.array([indMaxQx-20,0]).max()
#        ind2 = numpy.array([indMaxQx+20,N-1]).min()
#
#        if self.dbugOptGrad['plotQxZoom']:
#            plt.plot(self.t[ind1:ind2],normErrQx[ind1:ind2],'o')
#            plt.grid(True)
#            plt.title("Integrand of Qx (zoom)")
#            plt.show()
#
#        if self.dbugOptGrad['plotSolQxMax']:
#            print("\nSolution on the region of MaxQx:")
#            self.plotSol(intv=numpy.arange(ind1,ind2,1,dtype='int'))
#
#        # for zoomed version:
#        indMaxQu = normErrQu.argmax()
#        ind1 = numpy.array([indMaxQu-20,0]).max()
#        ind2 = numpy.array([indMaxQu+20,N-1]).min()
#
#        if self.dbugOptGrad['plotQuZoom']:
#            plt.plot(self.t[ind1:ind2],normErrQu[ind1:ind2],'o')
#            plt.grid(True)
#            plt.title("Integrand of Qu (zoom)")
#            plt.show()
#
#        if self.dbugOptGrad['plotSolQuMax']:
#            print("\nSolution on the region of MaxQu:")
#            self.plotSol(intv=numpy.arange(ind1,ind2,1,dtype='int'))
#
#
##        if n==4 and m==2:
##
##            plt.plot(tPlot[ind1:ind2],errQx[ind1:ind2,0])
##            plt.grid(True)
##            plt.ylabel("Qx_h")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],errQx[ind1:ind2,1],'g')
##            plt.grid(True)
##            plt.ylabel("Qx_V")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],errQx[ind1:ind2,2],'r')
##            plt.grid(True)
##            plt.ylabel("Qx_gamma")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],errQx[ind1:ind2,3],'m')
##            plt.grid(True)
##            plt.ylabel("Qx_m")
##            plt.show()
##
##            print("\nStates, controls, lambda on the region of maxQx:")
##
##            plt.plot(tPlot[ind1:ind2],x[ind1:ind2,0])
##            plt.grid(True)
##            plt.ylabel("h [km]")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],x[ind1:ind2,1],'g')
##            plt.grid(True)
##            plt.ylabel("V [km/s]")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],x[ind1:ind2,2]*180/numpy.pi,'r')
##            plt.grid(True)
##            plt.ylabel("gamma [deg]")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],x[ind1:ind2,3],'m')
##            plt.grid(True)
##            plt.ylabel("m [kg]")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],u[ind1:ind2,0],'k')
##            plt.grid(True)
##            plt.ylabel("u1 [-]")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],u[ind1:ind2,1],'c')
##            plt.grid(True)
##            plt.xlabel("t")
##            plt.ylabel("u2 [-]")
##            plt.show()
##
##            print("Lambda:")
##
##            plt.plot(tPlot[ind1:ind2],lam[ind1:ind2,0])
##            plt.grid(True)
##            plt.ylabel("lam_h")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],lam[ind1:ind2,1],'g')
##            plt.grid(True)
##            plt.ylabel("lam_V")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],lam[ind1:ind2,2],'r')
##            plt.grid(True)
##            plt.ylabel("lam_gamma")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],lam[ind1:ind2,3],'m')
##            plt.grid(True)
##            plt.ylabel("lam_m")
##            plt.show()
##
###            print("dLambda/dt:")
###
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,0])
###            plt.grid(True)
###            plt.ylabel("dlam_h")
###            plt.show()
###
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,1])
###            plt.grid(True)
###            plt.ylabel("dlam_V")
###            plt.show()
###
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,2],'r')
###            plt.grid(True)
###            plt.ylabel("dlam_gamma")
###            plt.show()
###
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,3],'m')
###            plt.grid(True)
###            plt.ylabel("dlam_m")
###            plt.show()
###
###            print("-phix*lambda:")
###
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,0]-errQx[ind1:ind2,0])
###            plt.grid(True)
###            plt.ylabel("-phix*lambda_h")
###            plt.show()
###
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,1]-errQx[ind1:ind2,1],'g')
###            plt.grid(True)
###            plt.ylabel("-phix*lambda_V")
###            plt.show()
###
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,2]-errQx[ind1:ind2,2],'r')
###            plt.grid(True)
###            plt.ylabel("-phix*lambda_gamma")
###            plt.show()
###
###            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,3]-errQx[ind1:ind2,3],'m')
###            plt.grid(True)
###            plt.ylabel("-phix*lambda_m")
###            plt.show()
##
##            print("\nBlue: dLambda/dt; Black: -phix*lam")
##
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,0])
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,0]-errQx[ind1:ind2,0],'k')
##            plt.grid(True)
##            plt.ylabel("z_h")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,1])
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,1]-errQx[ind1:ind2,1],'k')
##            plt.grid(True)
##            plt.ylabel("z_V")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,2])
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,2]-errQx[ind1:ind2,2],'k')
##            plt.grid(True)
##            plt.ylabel("z_gamma")
##            plt.show()
##
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,3])
##            plt.plot(tPlot[ind1:ind2],dlam[ind1:ind2,3]-errQx[ind1:ind2,3],'k')
##            plt.grid(True)
##            plt.ylabel("z_m")
##            plt.show()

    if self.dbugOptGrad['pausCalcQ']:
        input("calcQ in debug mode. Press any key to continue...")

    #input("Done calculating Q.\n")
    return Q,Qx,Qu,Qp,Qt

def calcStepGrad(self,corr,alfa_0,retry_grad):

    self.log.printL("\nIn calcStepGrad.\n")
    prntCond = self.dbugOptGrad['prntCalcStepGrad']

    # Get initial status (no correction applied)
    self.log.printL("\nalfa = 0.0")
    #Q0,_,_,_,_ = self.calcQ()
    P0,_,_ = self.calcP()
    J0,_,_,I0,_,_ = self.calcJ()
    k = self.constants['gradStepSrchCte'] * I0/self.tol['P']
    stepMan = stepMngr(self.log, k = k, tolP = self.tol['P'], \
                       corr = corr, prntCond = prntCond)

    Obj0 = stepMan.calcBase(self,P0,J0,I0)

    if retry_grad:
        if prntCond:
            self.log.printL("\n> Retrying alfa." + \
                            " Base value: {:.4E}".format(alfa_0))
        Pbase, Ibase, Objbase = stepMan.tryStep(self,alfa_0)

        # LOWER alfa!
        alfa = .9 * alfa_0
        if prntCond:
            self.log.printL("\n> Let's try alfa 10% lower.")
        P, I, Obj = stepMan.tryStep(self,alfa)

        while Obj > Obj0:
            alfa *= .9
            if prntCond:
                self.log.printL("\n> Let's try alfa 10% lower.")
            P, I, Obj = stepMan.tryStep(self,alfa)

        if prntCond:
            self.log.printL("\n> Ok, this value should work.")

    else:
        if self.NIterGrad == 0:

            # Get status associated with integral correction (alfa=1.0)
            if prntCond:
                self.log.printL("\n> Trying alfa = 1, fingers crossed...")
            alfa = 1.0
            P1, I1, Obj1 = stepMan.tryStep(self,alfa)


            # Search for a better starting point for alfa
            Obj = Obj1; keepLook = False; dAlfa = 1.0
            if Obj>1.1*Obj0:#Obj>Obj0:#I/I0 >= 10.0:#Q>Q0:#
                if prntCond:
                    self.log.printL("\n> Whoa! Going back to safe " + \
                                    "region of alphas...\n")
                keepLook = True
                dAlfa = 0.1
                cond = lambda nObj,Obj: nObj>Obj0 #nQ/Q0>1.1
            # Or increase alfa, if the conditions seem Ok for it
            elif Obj<Obj0:
                if prntCond:
                    self.log.printL("\n> This seems boring. Going forward!\n")
                keepLook = True
                dAlfa = 10.0
                cond = lambda nObj,Obj: nObj<Obj

            nObj = Obj.copy()
            while keepLook:
                Obj = nObj.copy()
                alfa *= dAlfa
                nP,nI,nObj = stepMan.tryStep(self,alfa)
                keepLook = cond(nObj,Obj)
            #
            if dAlfa > 1.0:
                alfa /= dAlfa
            elif dAlfa < 1.0:
                Obj = nObj.copy()

            alfa0 = alfa

        else:
            alfa0 = alfa_0
            P, I, Obj = stepMan.tryStep(self,alfa0)
        #

        # Start search by quadratic interpolation
        if prntCond:
            self.log.printL("\n> Starting detailed step search...\n")

        # Quadratic interpolation: for each point candidate for best step
        # value, its neighborhood (+1% and -1%) is also tested. With these 3
        # points, a quadratic interpolant is obtained, resulting in a parabola
        # whose minimum is the new candidate for optimal value.
        # The stop criterion is based in small changes to step value (prof.
        # Azevedo would hate this...), basically because it was easy to
        # implement.

        keepSrch = True
        alfaRef = alfa0

        while keepSrch:
            PRef, IRef, ObjRef = stepMan.tryStep(self, alfaRef, \
                                                 plotSol=True, plotPint=True)

            #if alfaRef >= stepMan.minBadStep:
            #    alfaRef = .5 * (alfaRef + stepMan.maxGoodStep)
            #    self.log.printL("> Reached a bad point in alfaRef. " + \
            #                    "Bisecting back to {:.4E},".format(alfaRef) + \
            #                    " based in {:.4E}".format(stepMan.histStep[stepMan.cont-1]))
            #    continue

            alfaM = 1.01 * alfaRef
            PM, IM, ObjM = stepMan.tryStep(self,alfaM)
            #if alfaM >= stepMan.maxStep:
            #    alfaRef = .5 * (alfaM + stepMan.maxGoodStep)
            #    self.log.printL("> Reached a bad point in alfaM. " + \
            #                    "Bisecting back to {:.4E},".format(alfaRef) + \
            #                    " based in {:.4E}".format(stepMan.histStep[stepMan.cont-1]))
            #    continue

            alfam = .99 * alfaRef
            Pm, Im, Objm = stepMan.tryStep(self,alfam)

            driv = (ObjM-Objm) / (alfaM-alfam)
            self.log.printL("\n> With alfa = {:.4E}".format(alfaRef) + \
                            ", Obj = {:.4E}".format(ObjRef) + \
                            ", dObj/dAlfa = {:.4E}".format(driv))
            #input("> Oia!")

            alfaList = numpy.array([alfam, alfaRef, alfaM])
            ObjList = numpy.array([Objm, ObjRef, ObjM])
            alfaRef = stepMan.fitQuad(alfaList,ObjList)
            Ppos, Ipos, ObjPos = stepMan.tryStep(self,alfaRef)
            self.log.printL("\n> Now, With alfa = {:.4E}".format(alfaRef) + \
                            ", Obj = {:.4E}".format(ObjPos))


            self.log.printL("\n> Type S for entering a step value, or Q to quit:")
            inp = input("> ")
            if inp == 's':
                self.log.printL("\n> Type the step value.")
                inp = input("> ")
                alfaRef = float(inp)
            elif inp == 'q':
                keepSrch = False
            else:
                self.log.printL("\n> Did not understand input. Going to next cycle.")


            #isOk = stepMan.stopCond(alfaRef,driv)
            #if isOk:
            #    keepSrch = False
            #else:
            #    self.log.printL("\n> Stop condition was not met. " + \
            #                    "Keep looking!")
            #    alfaList = numpy.array([alfam, alfaRef, alfaM])
            #    ObjList = numpy.array([Objm, ObjRef, ObjM])
            #    alfaRef = stepMan.fitQuad(alfaList,ObjList)
            #



#            err = abs(driv)#(ObjOpt/ObjRef-1.0)
#            if err < tol:#0.01:
#                keepSrch = False
#            else:
#                self.log.printL("\nErr = {:.4E}".format(err) + \
#                      " > {:.4E}, vamos dar mais uma volta!".format(tol))
#                input("> ")
#                alfaList = numpy.array([alfam, alfaRef, alfaM])
#                ObjList = numpy.array([Objm, ObjRef, ObjM])
#                alfaRef = stepMan.fitQuad(alfaList,ObjList)
#            #
        #
        alfa = stepMan.best['step']
    #

    # SCREENING:
    self.log.printL("\n> Going for screening...")
    mf = 10.0**(1.0/10.0)
    alfaBase = alfa
    for j in range(10):
        alfa *= mf
        P, I, Obj = stepMan.tryStep(self,alfa)
    alfa = alfaBase
    for j in range(10):
        alfa /= mf
        P, I, Obj = stepMan.tryStep(self,alfa)
    alfa = alfaBase


    # Get histories from step manager object
    histAlfa = stepMan.histStep
    histP = stepMan.histP
    histI = stepMan.histI
    # Get index for applied alfa
    for k in range(len(histAlfa)):
        if abs(histAlfa[k]-alfa)<1e-14:
            break

    # Get final values of Q and P
    P, I = histP[k], histI[k]

    # after all this analysis, plot the history of the tried alfas, and
    # corresponding Q's

    if self.dbugOptGrad['plotCalcStepGrad']:
        histObj = stepMan.histObj

        linhAlfa = numpy.array([0.9*min(histAlfa),max(histAlfa)])
        plt.loglog(histAlfa,histP,'o',label='P(alfa)')
        linP0 = P0 + numpy.zeros(len(linhAlfa))
        plt.loglog(linhAlfa,linP0,'--',label='P(0)')
        linTolP = stepMan.tolP + 0.0 * linP0
        plt.loglog(linhAlfa,linTolP,'--',label='tolP')
        linLimP = stepMan.limP + 0.0 * linP0
        plt.loglog(linhAlfa,linLimP,'--',label='limP')
        plt.xlabel('alpha')
        plt.ylabel('P')
        plt.grid(True)
        # Plot final values of Q and P in squares
        plt.loglog(alfa,P,'s',label='Chosen value')

        plt.legend()
        plt.title("P versus Grad Step for this grad run")
        plt.show()

        # Plot history of I
        plt.loglog(histAlfa,histI,'o',label='I(alfa)')
        linI = I0 + numpy.zeros(len(linhAlfa))
        plt.loglog(linhAlfa,linI,'--',label='I(0)')
        plt.plot(alfa,histI[k],'s',label='Chosen value')
        plt.ylabel("I")
        plt.xlabel("alpha")
        plt.title("I versus grad step for this grad run")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot history of Obj
        plt.loglog(histAlfa,histObj,'o',label='Obj(alfa)')
        linObj = Obj0 + numpy.zeros(len(linhAlfa))
        plt.loglog(linhAlfa,linObj,'--',label='Obj(0)')
        plt.plot(alfa,histObj[k],'s',label='Chosen value')
        plt.ylabel("Obj")
        plt.xlabel("alpha")
        plt.title("Obj versus grad step for this grad run")
        plt.grid(True)
        plt.show()

    if prntCond:
        dIp = 100.0 * (I/I0 - 1.0)
        self.log.printL("\n> Chosen alfa = {:.4E}".format(alfa))
        self.log.printL("> I0 = {:.4E}".format(I0) + \
                        ", I = {:.4E}".format(I) + \
                        ", dI = {:.4E}".format(I-I0) + \
                        " ({:.4E})%".format(dIp))
        self.log.printL("> Number of objective evaluations: " + \
                        str(stepMan.cont))

    if self.dbugOptGrad['pausCalcStepGrad']:
        input("\n> Run of calcStepGrad terminated. Press any key to continue.")

    return alfa

def grad(self,corr,alfa_0,retry_grad):

    self.log.printL("\nIn grad, Q0 = {:.4E}.".format(self.Q))
    #self.log.printL("NIterGrad = "+str(self.NIterGrad))

    #self.plotSol(opt={'mode':'var','x':A,'u':B,'pi':C})
    #input("Olha lá a correção...")

    # Calculation of alfa
    alfa = self.calcStepGrad(corr,alfa_0,retry_grad)
    #alfa = 0.1
    #self.log.printL('\n\nBypass cabuloso: alfa arbitrado em '+str(alfa)+'!\n\n')

    self.updtHistGrad(alfa)

    self.plotSol(opt={'mode':'lambda'})
    A, B, C = corr['x'], corr['u'], corr['pi']
    self.plotSol(opt={'mode':'var','x':alfa*A,'u':alfa*B,'pi':alfa*C})
    #input("@Grad: Waiting for lambda/corrections check...")


    # Apply correction, update histories in alternative solution
    newSol = self.copy()
    newSol.aplyCorr(alfa,corr)
    newSol.updtHistP()

#    newSol.updtGradCont(alfa)
#    newSol.updtHistP()
#
#    newSol.plotSol(opt={'mode':'lambda'})
#    newSol.plotSol(opt={'mode':'var','x':alfa*A,'u':alfa*B,'pi':alfa*C})
#    input("@Grad: Waiting for lambda/corrections check...")

    self.log.printL("Leaving grad with alfa = "+str(alfa))
    self.log.printL("Delta pi = "+str(alfa*C))

    if self.dbugOptGrad['pausGrad']:
        input('Grad in debug mode. Press any key to continue...')

    return alfa, newSol
