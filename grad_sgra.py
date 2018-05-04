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

    For now, the objective function only takes into account the values of I
    and P. The I value is most relevant, and the P value is secondary. The
    k parameter defines the importance of P with respect to I."""

    def __init__(self,log, k=1e3, tolP=1e-4, prntCond=False):
        self.cont = -1
        self.histStep = list()
        self.histI = list()
        self.histP = list()
        self.histQ = list()
        self.histObj = list()
        self.k = k
        self.tolP = tolP
        self.log = log
        self.mustPrnt = prntCond


    def calcObj(self,P,Q,I,J):
        return J
#        return (I + (self.k)*P)
#        if P > self.tolP:
#            return (I + (self.k)*(P-self.tolP))
#        else:
#            return I


    def getLast(self):
        P = self.histP[self.cont]
        Q = self.histQ[self.cont]
        I = self.histI[self.cont]
        Obj = self.histObj[self.cont]

        return P,Q,I,Obj

    def tryStep(self,sol,corr,alfa):
        self.cont += 1

        if self.mustPrnt:
            self.log.printL("\nTrying alfa = {:.4E}".format(alfa))

        newSol = sol.copy()
        newSol.aplyCorr(alfa,corr)

        P,_,_ = newSol.calcP()
        #Q,_,_,_,_ = newSol.calcQ()
        Q = 1.0
        J,_,_,I,_,_ = newSol.calcJ()

        Obj = self.calcObj(P,Q,I,J)

        if self.mustPrnt:
            self.log.printL("\nResults:\nalfa = {:.4E}".format(alfa)+\
                  " P = {:.4E}".format(P)+" Q = {:.4E}".format(Q)+\
                  " I = {:.4E}".format(I)+" J = {:.7E}".format(J)+\
                  " Obj = {:.7E}".format(Obj))

        self.histStep.append(alfa)
        self.histP.append(P)
        self.histQ.append(Q)
        self.histI.append(I)
        self.histObj.append(Obj)

        return self.getLast()

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
          ", J_Lpsi = {:.6E}.".format(J_Lpsi)+", J_I = {:.6E}".format(J_I)
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
    # Get initial status (Q0, no correction applied)

    self.log.printL("\nalfa = 0.0")
    #Q0,_,_,_,_ = self.calcQ()
    Q0 = 1.0
    P0,_,_ = self.calcP()
    J0,_,_,I0,_,_ = self.calcJ()
    k = self.constants['gradStepSrchCte'] * I0/self.tol['P']
    stepMan = stepMngr(self.log, k = k, tolP = self.tol['P'], \
                       prntCond = prntCond)
    #stepMan = stepMngr(k = 1e-5*I0/P0)#stepMngr(k = 1e-5*I0/P0)#
    # TODO: ideias
    # usar tolP ao inves de P0
    # usar P-tolP ao inves de P
    Obj0 = stepMan.calcObj(P0,Q0,I0,J0)
    self.log.printL("I0 = {:.4E}".format(I0)+" Obj0 = {:.4E}".format(Obj0))

    if retry_grad:
        if prntCond:
            self.log.printL("\n> Retrying alfa. Base value: {:.4E}".format(alfa_0))
        Pbase,Qbase,Ibase,Objbase = stepMan.tryStep(self,corr,alfa_0)

        # LOWER alfa!
        alfa = .9 * alfa_0
        if prntCond:
            self.log.printL("\n> Let's try alfa 10% lower.")
        P,Q,I,Obj = stepMan.tryStep(self,corr,alfa)

        while Obj > Obj0:
            alfa *= .9
            if prntCond:
                self.log.printL("\n> Let's try alfa 10% lower.")
            P,Q,I,Obj = stepMan.tryStep(self,corr,alfa)

        if prntCond:
            self.log.printL("\n> Ok, this value should work.")

    else:
        if self.NIterGrad == 0:

            # Get status associated with integral correction (alfa=1.0)
            if prntCond:
                self.log.printL("\n> Trying alfa = 1.0 first, fingers crossed...")
            alfa = 1.0
            P1,Q1,I1,Obj1 = stepMan.tryStep(self,corr,alfa)


            # Search for a better starting point for alfa
            Obj = Obj1; keepLook = False; dAlfa = 1.0
            if Obj>1.1*Obj0:#Obj>Obj0:#I/I0 >= 10.0:#Q>Q0:#
                if prntCond:
                    self.log.printL("\n> Whoa! Going back to safe region of " + \
                                    "alphas...\n")
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
                nP,nQ,nI,nObj = stepMan.tryStep(self,corr,alfa)
                keepLook = cond(nObj,Obj)
            #
            if dAlfa > 1.0:
                alfa /= dAlfa
            elif dAlfa < 1.0:
                Obj = nObj.copy()

            alfa0 = alfa

        else:
            alfa0 = alfa_0
            #alfa0 = self.histStepGrad[self.NIterGrad]
            P,Q,I,Obj = stepMan.tryStep(self,corr,alfa0)


        # Now Obj is not so much bigger than Obj0. Start "bilateral analysis"
        if prntCond:
            self.log.printL("\n> Starting bilateral analysis...\n")

        alfa = 1.1*alfa0
        PM,QM,IM,ObjM = stepMan.tryStep(self,corr,alfa)

        alfa = .9*alfa0
        Pm,Qm,Im,Objm = stepMan.tryStep(self,corr,alfa)

        # Start refined search
        self.log.printL("\nObj = "+str(Obj))
        if Objm < Obj:
            if self.dbugOptGrad['prntCalcStepGrad']:
                self.log.printL("\n> Beginning search for decreasing alfa...")
            # if the tendency is to still decrease alfa, do it...
            nObj = Obj.copy(); keepSearch = True
            while keepSearch and alfa > 1.0e-15:
                Obj = nObj.copy()
                alfa *= .9
                nP,nQ,nI,nObj = stepMan.tryStep(self,corr,alfa)

                # TODO: use testAlgn to test alignment of points,
                # and use it to improve calcStepGrad. E.G.:

    #            isAlgn = (abs(testAlgn(histAlfa[(cont-4):(cont-1)],\
    #                           histQ[(cont-4):(cont-1)])) < 1e-3)
    #
    #            if isAlgn:
    #                m = (histQ[cont-2]-histQ[cont-3]) / \
    #                    (histAlfa[cont-2]-histAlfa[cont-3])
    #                b = histQ[cont-2] - m * histAlfa[cont-2]
    #                self.log.printL("m =",m,"b =",b)

                if nObj < Obj0:
                    keepSearch = ((nObj-Obj)/Obj < -.1)#-.001)#nQ<Q#
            alfa /= 0.9
        else:

#            # no overdrive!
#            if prntCond:
#                self.log.printL("\n> No overdrive! Let's stay with " + \
#                                "alfa = {:.4E}.".format(alfa0))
#            alfa = alfa0 # BRASIL

            if Obj <= ObjM:
                alfa = alfa0 # BRASIL

                if prntCond:
                    self.log.printL("\n> Apparently alfa = " + str(alfa0) + \
                                    " is the best.")

            else:
                # ObjM is the lowest. But increasing alfa is riskier...

                alfa = 1.1 * alfa0
                if prntCond:
                    self.log.printL("\n> Let's try alfa = " + str(alfa) + \
                                    " (very carefully!).")
                #
            #
        #
    #

    # LOUCURA
    self.log.printL("\n> Going for screening...")
    mf = 10.0**(1.0/10.0)
    alfaBase = alfa
    for j in range(10):
        alfa *= mf
        P,Q,I,Obj = stepMan.tryStep(self,corr,alfa)
    alfa = alfaBase
    for j in range(10):
        alfa /= mf
        P,Q,I,Obj = stepMan.tryStep(self,corr,alfa)
    alfa = alfaBase
    #

    # Get histories from step manager object
    histAlfa = stepMan.histStep
    histP = stepMan.histP
    histQ = stepMan.histQ
    histI = stepMan.histI
    # Get index for applied alfa
    for k in range(len(histAlfa)):
        if abs(histAlfa[k]-alfa)<1e-14:
            break

    # Get final values of Q and P
    Q = histQ[k]; P = histP[k]; I = histI[k]

    # after all this analysis, plot the history of the tried alfas, and
    # corresponding Q's

    if self.dbugOptGrad['plotCalcStepGrad']:
        histObj = stepMan.histObj

        # Ax1: convergence history of Q
        fig, ax1 = plt.subplots()
        ax1.loglog(histAlfa, histQ, 'ob')
        linhAlfa = numpy.array([0.9*min(histAlfa),max(histAlfa)])
        linQ0 = Q0 + 0.0*numpy.empty_like(linhAlfa)
        ax1.loglog(linhAlfa,linQ0,'--b')
        ax1.set_xlabel('alpha')
        ax1.set_ylabel('Q', color='b')
        ax1.tick_params('y', colors='b')
        ax1.grid(True)

        # Ax2: convergence history of P
        ax2 = ax1.twinx()
        ax2.loglog(histAlfa,histP,'or')
        linP0 = P0 + numpy.zeros(len(linhAlfa))
        ax2.loglog(linhAlfa,linP0,'--r')
        ax2.set_ylabel('P', color='r')
        ax2.tick_params('y', colors='r')
        ax2.grid(True)

        # Plot final values of Q and P in squares
        ax1.loglog(alfa,Q,'sb'); ax2.loglog(alfa,P,'sr')

        plt.title("Q and P versus Grad Step for this grad run")
        plt.show()

        # Plot history of I
        plt.loglog(histAlfa,histI,'o')
        linI = I0 + numpy.zeros(len(linhAlfa))
        plt.loglog(linhAlfa,linI,'--')
        plt.plot(alfa,histI[k],'s')
        plt.ylabel("I")
        plt.xlabel("alpha")
        plt.title("I versus grad step for this grad run")
        plt.grid(True)
        plt.show()

        # Plot history of Obj
        plt.loglog(histAlfa,histObj,'o')
        linObj = Obj0 + numpy.zeros(len(linhAlfa))
        plt.loglog(linhAlfa,linObj,'--')
        plt.plot(alfa,histObj[k],'s')
        plt.ylabel("Obj")
        plt.xlabel("alpha")
        plt.title("Obj versus grad step for this grad run")
        plt.grid(True)
        plt.show()

    if prntCond:
        dIp = 100.0 * (I/I0 - 1.0)
        self.log.printL("\n> Chosen alfa = {:.4E}".format(alfa) + \
                        ", Q = {:.4E}".format(Q))
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
