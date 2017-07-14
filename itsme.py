#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:14:40 2017

@author: Carlos

Initial Trajectory Setup ModulE

Version 2.0
Including:
    Variable multi-stage model
    New parametric model

"""

import numpy
import configparser
import matplotlib.pyplot as plt
from scipy.integrate import ode
from atmosphere import rho

totalTrajectorySimulationCounter = 0

def its(*arg):

    # arguments analisys
    if len(arg) == 0:
        fname = 'default.its'
    elif len(arg) == 1:
        fname = arg[0]
    else:
        raise Exception('itsme saying: too many arguments on its')

    problem = itsYourProblem(fname)

    solution1 = problem.initialGuess()

    solution1.displayResults()

    solution2 = problem.fineTunning()

    solution2.trajectory.plotResultsAed(problem)

    solution2.displayResults()

    return solution2

def mdlDer(t,x,arg):

    # initialization
    h,v,gama,M = x[0],x[1],x[2],x[3]
    alfaProg,betaProg,con = arg
    T = con['T']
    Isp = con['Isp']
    g0 = con['g0']
    R = con['R']

    if numpy.isnan(h):
       raise Exception('itsme saying: h is not a number')

    # controls calculation
    betat = betaProg.value(t)
    alfat = alfaProg.value(t)

    # Other calculations
    btm = betat*T/M
    sinGama = numpy.sin(gama)
    g = g0*(R/(R+h))**2 - (con['we']**2)*(R + h)

    # Aerodynamics
    CL = con['CL0'] + con['CL1']*alfat
    CD = con['CD0'] + con['CD2']*(alfat**2)

    qdin = 0.5 * rho(h) * (v**2)
    L = qdin * con['s_ref'] * CL
    D = qdin * con['s_ref'] * CD

    return numpy.array([v*sinGama,\
    btm*numpy.cos(alfat) - g*sinGama - (D/M),\
    btm*numpy.sin(alfat)/v + (v/(h+R)-g/v)*numpy.cos(gama) + (L/(v*M)) + 2*con['we'],\
    -btm*M/g0/Isp])

class simpleStep():

    def __init__(self,v1,v2,tref):
        self.v1 = v1
        self.v2 = v2
        self.tref = tref

    def value(self,t)    :
        if t < self.tref:
            ans = self.v1
        else:
            ans = self.v2

        return ans

    def multValue(self,t):
        N = len(t)
        ans = numpy.full((N,1),0.0)
        for jj in range(0,N):
            ans[jj] = self.value(t[jj])

        return ans

class propulsiveModel():

    def __init__(self,p1,p2,tf,v1,v2,softness):
        self.t1 = p1.tf[-1]
        t2 = tf - p2.tb[-1]
        self.t3 = tf

        self.v1 = v1
        self.v2 = v2

        f = softness/2

        d1  = self.t1 # width of retangular and 0.5 soft part
        self.c1  = d1*f # width of the 0.5 soft part
        self.fr1  = d1 - self.c1 # final of the retangular part
        self.fs1  = d1 + self.c1 # final of the retangular part

        d2  = self.t3 - t2 # width of retangular and 0.5 soft part
        self.c2  = d2*f # width of the 0.5 soft part
        self.r2  = d2 - self.c2 # width of the retangular part
        self.ir2 = t2 + self.c2 # start of the retangular part
        self.is2 = t2 - self.c2 # start of the soft part

        self.dv21 = v2 - v1

        # List of time events and jetsoned masses
        self.tflist = p1.tf[0:-1].tolist()+[self.fr1,  self.fs1]+[self.is2,self.ir2,        tf]
        self.melist = p1.me[0:-1].tolist()+[     0.0, p1.me[-1]]+[    0.0,      0.0, p2.me[-1]]

        self.fail = False
        if len(p1.tf) > 2:
            if p1.tf[-2] >= self.fr1:
                self.fail = True

        self.tlist1 = p1.tf[0:-1].tolist()+[self.fs1]
        self.Tlist1 = p1.Tlist
        self.Tlist2 = p2.Tlist

        self.Isp1 = p1.Isp
        self.Isp2 = p2.Isp

    def value(self,t):
        if (t <= self.fr1):
            ans = self.v1
        elif (t <= self.fs1):
            cos = numpy.cos( numpy.pi*(t - self.fr1)/(2*self.c1) )
            ans = self.dv21*(1 - cos)/2 + self.v1
        elif (t <= self.is2):
            ans = self.v2
        elif (t <= self.ir2):
            cos = numpy.cos( numpy.pi*(t - self.is2)/(2*self.c2) )
            ans = -self.dv21*(1 - cos)/2 + self.v2
        elif (t <= self.t3):
            ans = self.v1
        else:
            ans = 0.0

        return ans

    def multValue(self,t):
        N = len(t)
        ans = numpy.full((N,1),0.0)
        for jj in range(0,N):
            ans[jj] = self.value(t[jj])

        return ans

    def thrust(self,t):

        T = 0.0
        tlist = self.tlist1
        Tlist = self.Tlist1
        for ii in range(0,len(tlist)):
            if t <= tlist[ii]:
                T = Tlist[ii]

        if t > self.is2:
            T = self.Tlist2[0]

        return T

    def Isp(self,t):

        if t <= self.fs1:
            Isp = self.Isp1
        else:
            Isp = self.Isp2

        return Isp

class cosSoftPulse():

    def __init__(self,t1,t2,v1,v2):
        self.t1 = t1
        self.t2 = t2

        self.v1 = v1
        self.v2 = v2

    def value(self,t):
        if (t >= self.t1) and (t < self.t2):
            ans = (self.v2 - self.v1) * (1 - numpy.cos(2*numpy.pi * (t - self.t1)  / (self.t2 - self.t1))) / 2 + self.v1
            return ans
        else:
            return self.v1

    def multValue(self,t):
        N = len(t)
        ans = numpy.full((N,1),0.0)
        for jj in range(0,N):
            ans[jj] = self.value(t[jj])

        return ans

class optimalStaging():
    # optimalStaging() returns a object with information of optimal staging factor
    # The maximal staging reason is defined as reducing the total mass for a defined
    # delta V.
    # Structural eficience and thrust shall variate for diferent stages, but
    # specific impulse must be the same for all stages
    # Based in Cornelisse (1979)

    def __init__(self,effList,dV,Tlist,con,Isp,g0,mu):
        self.Tlist = Tlist
        self.e = numpy.array(effList)
        self.T = numpy.array(self.Tlist)
        self.dV = dV
        self.mu = mu
        self.Isp = Isp
        self.c = Isp*g0
        self.mflux = self.T/self.c
        self.mE = self.calcGeoMeanE()
        self.m1E = self.calcGeoMean1E()
        self.fail = False
        self.lamb = self.calc_lamb()
        self.phi = (1 - self.e)*(1 - self.lamb)

        self.mtot = self.calc_mtot()             # Total sub-rocket mass
        self.mp = self.mtot*self.phi             # Propelant mass on each stage
        self.me = self.mp*(self.e/(1 - self.e))  # Strutural mass of each stage
        self.tb = self.mp/self.mflux             # Duration of each stage burning
        self.tf = self.calc_tf()                 # Final burning time of each stage

    def calcGeoMeanE(self):

        a = self.e
        m = 1.0
        for v in a:
            m = m*v
        m = m ** (1/a.size)
        return m

    def calcGeoMean1E(self):

        a = 1 - self.e
        m = 1.0
        for v in a:
            m = m*v
        m = m ** (1/a.size)
        return m

    def calc_lamb(self):

        LtN = ( numpy.exp(-self.dV/(self.c*self.e.size)) - self.mE)/self.m1E
        self.LtN = LtN

        if LtN <= 0:
            self.fail = True

        lamb = (self.e/(1 - self.e))*(LtN*self.m1E/self.mE)
        return lamb

    def calc_mtot(self):

        mtot = self.e*0.0
        N = self.e.size-1
        for ii in range(0,N+1):
            if ii == 0:
                mtot[N - ii] = self.mu/self.lamb[N - ii]
            else:
                mtot[N - ii] = mtot[N - ii + 1]/self.lamb[N - ii]
        return mtot

    def calc_tf(self):

        tf = self.e*0.0
        N = self.tb.size-1
        for ii in range(0,N+1):
            if ii == 0:
                tf[ii] = self.tb[ii]
            else:
                tf[ii] = self.tb[ii] + tf[ii-1]
        return tf

    def printInfo(self):

        print("dV =",self.dV)
        print("mu =",self.mu)
        print("mp =",self.mp)
        print("me =",self.me)
        print("mtot =",self.mtot)
        print("mflux =",self.mflux)
        print("tb =",self.tb)
        print("tf =",self.tf,"\n\r")

class initialEstimate():

    def __init__(self,con):

        efflist = con['efflist']
        Tlist = con['Tlist']

        lamb = numpy.exp( 0.5*con['V_final']/(con['Isp2']*con['g0']) )
        self.Mu = con['Mu']*lamb
        self.hf = con['h_final']
        self.M0max = Tlist[0]/con['g0']
        self.c = con['Isp1']*con['g0']
        self.mflux = numpy.mean(Tlist[0:-1])/self.c
        self.GM = con['GM']
        self.R = con['R']
        self.e = numpy.exp( numpy.mean( numpy.log(efflist[0:-1]) ) )
        self.g0 = con['g0'] - con['R']*(con['we'] ** 2)
        self.t1max = self.M0max/self.mflux
        self.fail = False

        self.calculate()

        self.vx = 0.5*self.V1*(con['R'] + self.h1)/(con['R']+self.hf)

    def newRap(self):

        N = 100
        t1max = self.t1max

        dt = t1max/N
        t1 = 0.1*t1max

        cont = 0
        erro = 1.0

        while (abs(erro) > 1e-6) and not self.fail:

            erro = self.hEstimate(t1) - self.hf
            dedt = (self.hEstimate(t1+dt) - self.hEstimate(t1-dt))/(2*dt)
            t1 = t1 - erro/dedt
            cont += 1
            if cont == 100:
                raise Exception('itsme saying: initialEstimate failed')

        self.cont = cont
        self.t1 = t1
        return None

    def hEstimate(self,t1):

        mflux = self.mflux
        Mu = self.Mu
        g0 = self.g0
        c = self.c

        Mp = mflux*t1
        Me = Mp*self.e/(1 - self.e)
        M0 = Mu + Me + Mp
        x = (Mu + Me)/M0
        V1 = c*numpy.log(1/x) - g0*t1
        h1 = (c*M0/mflux) * ( (x*numpy.log(x) -x ) + 1) - g0*(t1**2)/2
        h = h1 + self.GM/(self.GM/(self.R + h1) - (V1**2)/2) - self.R

        return h

    def dvt(self):

        t1 = self.t1
        mflux = self.mflux
        Mu = self.Mu
        g0 = self.g0
        c = self.c

        M0 = Mu + mflux*t1
        V1 = c*numpy.log(M0/Mu) - g0*t1
        dv = V1 + g0*t1
        x = Mu/M0
        h1 = (c*M0/mflux) * ( (x*numpy.log(x) -x ) + 1) - g0*(t1**2)/2

        h = self.hEstimate(self.t1)

        rr = (self.R + h1)/(self.R + h)
        if rr > 1:
            raise Exception('itsme saying: initialEstimate failed (h1 > h)')
        theta = numpy.arccos( numpy.sqrt( rr ) )
        ve = numpy.sqrt(2*self.GM/(self.R + h))

        t = t1 + ( (self.R + h)/ve ) * ( numpy.sin(2*theta)/2 + theta )

        self.h1 = h1
        self.h = h
        self.t = t
        self.dv = dv
        self.V1 = V1

        return None

    def calculate(self):

        self.newRap()
        self.dvt()

        return None

class itsYourProblem():

    def __init__(self,fileAdress):

        global totalTrajectorySimulationCounter
        totalTrajectorySimulationCounter = 0

        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(fileAdress)
        self.con = dict()

        #######################################################################
        # Enviromental constants
        section = 'enviroment'
        items = config.items(section)
        for para in items:
            self.con[para[0]] = config.getfloat(section,para[0])
        self.con['g0'] = self.con['GM']/(self.con['R']**2)            # [km/s2] gravity acceleration on earth surface

        #######################################################################
        # General constants
        self.con['pi'] = numpy.pi
        self.con['d2r'] = numpy.pi/180.0

        #######################################################################
        #Initial state constants
        self.con['h_initial'] = config.getfloat('initial','h')
        self.con['V_initial'] = config.getfloat('initial','V')
        self.con['gamma_initial'] = config.getfloat('initial','gamma')*self.con['d2r']

        #######################################################################
        #Final state constants
        self.con['h_final'] = config.getfloat('final','h')
        self.con['V_final'] = numpy.sqrt(self.con['GM']/(self.con['R']+self.con['h_final'])) - self.con['we']*(self.con['R']+self.con['h_final'])   # km/s Circular velocity
        self.con['gamma_final'] = config.getfloat('final','gamma')*self.con['d2r']

        #######################################################################
        #Vehicle parameters
        section = 'vehicle'
        items = config.items(section)

        if not config.has_option(section,'homogeneous'):
            self.con['homogeneous'] = True
        else:
            self.con['homogeneous'] = config.getboolean(section,'homogeneous')

        # This flag show indicates if the vehicle shall be considered as having the same
        # values of structural mass and thrust for all stages
        if self.con['homogeneous']:
            self.getVehicleHomogeneous(config)
        else:
            self.getVehicleHeterogeneous(config)

        #######################################################################
        # Trajectory parameters
        section = 'trajectory'
        items = config.items(section)
        for para in items:
            self.con[para[0]] = config.getfloat(section,para[0])
        self.con['torb'] = 2*self.con['pi']*(self.con['R'] + self.con['h_final'])/self.con['V_final'] # Time of one orbit using the final velocity

        #######################################################################
        # Solver parameters
        self.con['tol'] = config.getfloat('solver','tol')

        # Superior and inferior limits
        auxstr = config.get('solver','guess')
        auxstr = auxstr.split(',')
        auxnum = []
        for n in auxstr:
            auxnum = auxnum+[float(n)]
        guess = numpy.array(auxnum)

        auxstr = config.get('solver','limit')
        auxstr = auxstr.split(',')
        auxnum = []
        for n in auxstr:
            auxnum = auxnum+[float(n)]
        limit = numpy.array(auxnum)

        self.con['guess'] = guess
        self.con['fsup'] = guess + limit
        self.con['finf'] = guess - limit

        #######################################################################
        # Reference values
        iniEst = initialEstimate(self.con)
        self.con['Dv1ref'] = iniEst.dv
        self.con['tref'] = iniEst.t
        self.con['vxref'] = iniEst.vx

        return None

    def getVehicleHomogeneous(self,config):

        section = 'vehicle'
        items = config.items(section)

        for para in items:
            self.con[para[0]] = config.getfloat(section,para[0])

        self.con['NStag'] = config.getint('vehicle','NStag')# Number of stages
        self.con['Isp1'] = self.con['Isp']
        self.con['Isp2'] = self.con['Isp']

        # This flag show indicates if the vehicle shall be considered as having the same
        # values of structural mass and thrust for all stages
        efflist = []
        Tlist = []
        if self.con['NStag'] > 1:
            for jj in range(0,self.con['NStag']):
                efflist = efflist+[self.con['efes']]
                Tlist = Tlist+[self.con['T']]
        else:
            # This cases are similar to NStag == 2, the differences are:
            # for NStag == 0 no mass is jetsoned
            # for NStag == 1 all structural mass is jetsoned at the end of all burning
            for jj in range(0,2):
                efflist = efflist+[self.con['efes']]
                Tlist = Tlist+[self.con['T']]

        self.con['efflist'] = efflist
        self.con['Tlist'] = Tlist

    def getVehicleHeterogeneous(self,config):

        section = 'vehicle'
        items = config.items(section)

        for para in items:
            if (para[0] != 'efes') and (para[0] != 'T'):
                self.con[para[0]] = config.getfloat(section,para[0])

        self.con['NStag'] = config.getint(section,'NStag')# Number of stages

        print(self.con['Isp1'])

        auxstr = config.get(section,'efes')
        auxstr = auxstr.split(',')
        auxnum = []
        for n in auxstr:
            auxnum = auxnum+[float(n)]
        self.con['efflist'] = auxnum

        auxstr = config.get(section,'T')
        auxstr = auxstr.split(',')
        auxnum = []
        for n in auxstr:
            auxnum = auxnum+[float(n)]
        self.con['Tlist'] = auxnum


    def initialGuess(self):
        #######################################################################
        # First guess trajectory

        traj = itsTrajectory()
        errors, _, _, tabAlpha, tabBeta = traj.trajectorySimulate(self.con['guess'],self.con,"design")

        self.tabAlpha = tabAlpha
        self.tabBeta = tabBeta
        self.errors = errors
        self.factors = self.con['guess']

        solution = itsSolution(self.con,tabAlpha,tabBeta,self.con['guess'])

        return solution

    def fineTunning(self):
        ##########################################################################
        # Bisection altitude loop
        con = self.con
        tol = con['tol']
        fsup = con['fsup']
        finf = con['finf']

        # Parameters initialization
        # Loop initialization
        stop = False
        count = 0
        Nmax = 50

        # Fators initilization
        df = abs( (fsup[2] - finf[2])/5 )
        factors = (fsup + finf)/2
        step = df.copy()
        f1 = (fsup[2] + finf[2])/2
        e1,factors = self.bisecSpeedAndAng(factors,f1)
        f2 = f1 + step

        # Loop
        while (not stop) and (count <= Nmax):

            # bisecSpeedAndAng: Error update from speed and gamma loop
            e2,factors = self.bisecSpeedAndAng(factors,f2)

            # Loop checks
            if  (abs(e2) < tol):
                stop = True
                # Display final information
                num = "8.6e"
                print("\n\####################################################")
                print("bisecAltitude final iteration: ",count)
                print(("Error     : %"+num) % e2)
                print(("Sup limits: %"+num+", %"+num+", %"+num) % (   fsup[0],   fsup[1],   fsup[2]))
                print(("Factors   : %"+num+", %"+num+", %"+num) % (factors[0],factors[1],        f2))
                print(("Inf limits: %"+num+", %"+num+", %"+num) % (   finf[0],   finf[1],   finf[2]))

            else:
                # Calculation of the new factor
                # Checkings
                # Division and step check
                de = (e2 - e1)
                if abs(de) < tol*1e-2:
                    step = df

                else:
                    der = (f2 - f1)/de
                    step = e2*der
                    if step > df:
                        step = 0.0 + df
                    elif step < -df:
                        step = 0.0 - df

                # Factor definition and check
                f3 = f2 - step

                if f3 > fsup[2]:
                    f3 = 0.0 + fsup[2]
                elif f3 < finf[2]:
                    f3 = 0.0 + finf[2]

                # Parameters update
                f1 = f2.copy()
                f2 = f3.copy()
                e1 = e2.copy()
                count += 1

                # Display information
                num = "8.6e"
                print("\n\####################################################")
                print("bisecAltitude iteration: ",count)
                print(("Error     : %"+num) % e2)
                print(("Sup limits: %"+num+", %"+num+", %"+num) % (   fsup[0],   fsup[1],   fsup[2]))
                print(("Factors   : %"+num+", %"+num+", %"+num) % (factors[0],factors[1],        f2))
                print(("Inf limits: %"+num+", %"+num+", %"+num) % (   finf[0],   finf[1],   finf[2]))
            # if end

        traj = itsTrajectory()
        errors, _, _, tabAlpha, tabBeta = traj.trajectorySimulate(factors,con,"design")
        num = "8.6e"
        print("\n\####################################################")
        print("ITS the end (lol)")
        print(("Error     : %"+num+", %"+num+", %"+num) % ( errors[0], errors[1], errors[2]))
        print(("Sup limits: %"+num+", %"+num+", %"+num) % (   fsup[0],   fsup[1],   fsup[2]))
        print(("Factors   : %"+num+", %"+num+", %"+num) % (factors[0],factors[1],factors[2]))
        print(("Inf limits: %"+num+", %"+num+", %"+num) % (   finf[0],   finf[1],   finf[2]))

        traj = itsTrajectory()
        errors, _, _, tabAlpha, tabBeta = traj.trajectorySimulate(factors,self.con,"design")

        self.tabAlpha = tabAlpha
        self.tabBeta = tabBeta
        self.errors = errors
        self.factors = factors

        solution = itsSolution(self.con,tabAlpha,tabBeta,factors)

        return solution

    def bisecSpeedAndAng(self,factors1,f3):
        ####################################################################
        # Bissection speed and gamma loop
        tol = self.con['tol']
        fsup = self.con['fsup']
        finf = self.con['finf']
        con = self.con

        # Initializing parameters
        # Loop initialization
        stop = False
        count = 0
        Nmax = 50

        # Fators initilization
        df = abs( (fsup - finf)/5 )
        # Making the 3 factor variarions null
        factors1[2] = f3 + 0.0
        df[2] = 0.0
        factors2 = factors1 + df
        traj = itsTrajectory()
        errors1, tt, xx, _, _ = traj.trajectorySimulate(factors1,con,"design")
        step = df + 0.0
        factors3 = factors2 + 0.0

        # Loop
        while (not stop) and (count <= Nmax):

            # Error update
            traj = itsTrajectory()
            errors2, _, _, _, _ = traj.trajectorySimulate(factors2,con,"design")

            converged = abs(errors2) < tol
            if converged[0] and converged[1]:
                stop = True
                # Display information
                print("\n\####################################################")
                if count == Nmax:
                    print("bisecSpeedAndAng total iterations: ", count," (max)")
                else:
                    print("bisecSpeedAndAng total iterations: ", count)
                num = "8.6e"
                print(("Errors    : %"+num+", %"+num) % ( errors2[0],  errors2[1]))
                print(("Sup limits: %"+num+", %"+num+", %"+num) % (    fsup[0],     fsup[1],     fsup[2]))
                print(("Factors   : %"+num+", %"+num+", %"+num) % (factors2[0], factors2[1], factors2[2]))
                print(("Inf limits: %"+num+", %"+num+", %"+num) % (    finf[0],     finf[1],     finf[2]))
            else:
                de = (errors2 - errors1)
                for ii in range(0,2):
                    # Division and step check
                    if de[ii] == 0:
                        step[ii] = 0
                    else:
                        step[ii] = errors2[ii]*(factors2[ii] - factors1[ii])/de[ii]

                    if step[ii] > df[ii]:
                        step[ii] = df[ii] + 0.0
                    elif step[ii] < -df[ii]:
                        step[ii] = -df[ii] + 0.0
                    # if end

                    # factor check
                    factors3[ii] = factors2[ii] - step[ii]

                    if factors3[ii] > fsup[ii]:
                        factors3[ii] = fsup[ii] + 0.0
                    elif factors3[ii] < finf[ii]:
                        factors3[ii] = finf[ii] + 0.0
                    #if end

                # for end
                errors1 = errors2 + 0.0
                factors1 = factors2 + 0.0
                factors2 = factors3 + 0.0
                count += 1
            #if end
        #while end
        # Define output
        errorh = errors2[2]
        return errorh,factors2

class itsTrajectory():

    def __init__(self):

        self.tt = []
        self.xx = []
        self.tp = []
        self.xp = []
        self.flagAppend = False
        self.simulCounter = 0

    def append(self,tt,xx):

        self.tt.append(tt)
        self.xx.append(xx)

    def appendP(self,tp,xp):

        self.tp.append(tp)
        self.xp.append(xp)

    def numpyArray(self):

        self.tt = numpy.array(self.tt)
        self.xx = numpy.array(self.xx)
        self.tp = numpy.array(self.tp)
        self.xp = numpy.array(self.xp)

    def cntrCalculate(self,tabAlpha,tabBeta):

        self.uu = numpy.concatenate([tabAlpha.multValue(self.tt),tabBeta.multValue(self.tt)], axis=1)
        self.up = numpy.concatenate([tabAlpha.multValue(self.tp),tabBeta.multValue(self.tp)], axis=1)

    def trajectorySimulate(self,factors,con,typeResult):
        ##########################################################################
        self.simulCounter += 1

        ##########################################################################
        # Delta V estimates
        fdv2,ftf,fdv1 = factors
        Dv1 = fdv1*con['Dv1ref']
        tf = ftf*con['tref']
        Dv2 =  con['V_final'] - fdv2*con['vxref']

        ##########################################################################
        # Staging calculation
        p1,p2 = self.stagingCalculate(Dv1,Dv2,con)

        ##########################################################################
        # Thrust program
        tabBeta = propulsiveModel(p1,p2,tf,1.0,0.0,con['softness'])
        if tabBeta.fail:
            raise Exception('itsme saying: Softness too high!')

        ##########################################################################
        # Attitude program definition
        tAoA2 = con['tAoA1'] + con['tAoA']
        tabAlpha = cosSoftPulse(con['tAoA1'],tAoA2,0,-con['AoAmax']*con['d2r'])

        ##########################################################################
        #Integration

        # Initial conditions
        t0 = 0.0
        x0 = numpy.array([con['h_initial'],con['V_initial'],con['gamma_initial'],p1.mtot[0]])

        # Integrator setting
        # ode set:
        #         atol: absolute tolerance
        #         rtol: relative tolerance
        ode45 = ode(mdlDer).set_integrator('dopri5',nsteps = 1,atol = con['tol']/1,rtol = con['tol']/10)
        ode45.set_initial_value(x0, t0).set_f_params((tabAlpha,tabBeta,con))

        # Phase times, incluiding the initial time in the begining
        tphases   = [ t0,con['tAoA1'],tAoA2]+tabBeta.tflist
        mjetsoned = [0.0,         0.0,  0.0]+tabBeta.melist
        if (typeResult == "orbital"):
            tphases   =   tphases+[con['torb']]
            mjetsoned = mjetsoned+[        0.0]

        # Integration using rk45 separated by phases
        # Automatic multiphase integration
        if (typeResult == "design"):
            # Fast running
            self.totalIntegration(tphases,mjetsoned,ode45,t0,x0)
            errors = self.errorCalculate(con)
            ans = errors, self.tp[-1], self.xp[-1], tabAlpha, tabBeta

        elif (typeResult == "plot") or (typeResult == "orbital"):
            # Slow running
            if (typeResult == "plot"):
                # Vehicle properties display
                self.displayInfo(p1,p2,con,tphases,mjetsoned)

            for ii in range(1,len(tphases)):
                if tphases[ii - 1] >= tphases[ii]:
                    raise Exception('itsme saying: tphases does not increase monotonically!')

            self.flagAppend = True
            self.totalIntegration(tphases,mjetsoned,ode45,t0,x0)
            self.cntrCalculate(tabAlpha,tabBeta)
            ans = None
    #        for ii in range(1,len(sol.tt)):
    #            if sol.tt[ii - 1] >= sol.tt[ii]:
    #                print('ii = ',ii,' sol.tt[ii-1] = ',sol.tt[ii-1],' sol.tt[ii] = ',sol.tt[ii])
                    #raise Exception('itsme saying: sol.tt does not increase monotonically!')

        return ans

    def totalIntegration(self,tphases,mjetsoned,ode45,t0,x0):

        global totalTrajectorySimulationCounter
        totalTrajectorySimulationCounter += 1
        self.simulCounter += 0

        # Output variables
        self.append(t0,x0)
        self.appendP(t0,x0)

        for ii in range(1,len(tphases)):
            self.phaseIntegration(tphases[ii - 1],tphases[ii],mjetsoned[ii-1],ode45)
            if self.flagAppend:
                if ii == 1:
                    print("Phase integration iteration: 1",end=', '),
                elif ii == (len(tphases) - 1):
                    print('')
                else:
                    print(ii,end=', ')

        self.numpyArray()

    def phaseIntegration(self,t_initial,t_final,mj,ode45):

        flagAppend = self.flagAppend
        contraction = 1e-10
        t_initial = t_initial + contraction
        t_final   =   t_final - contraction

        y = ode45.y
        y[3] = y[3] - mj
        self.append(t_initial,y)

        ode45.set_initial_value(y,t_initial)
        ode45.first_step = (t_final - t_initial)*0.01

        while ode45.t < t_final:
            ode45.integrate(t_final)
            if flagAppend:
                self.append(ode45.t,ode45.y)

        ode45.integrate(t_final)
        self.append(ode45.t,ode45.y)
        self.appendP(ode45.t+contraction,ode45.y)

    def errorCalculate(self,con):

        h,v,gamma,M = self.xp[-1]
        errors = ((v - con['V_final'])/0.01, (gamma - con['gamma_final'])/0.1, (h - con['h_final'])/10)
        errors = numpy.array(errors)

        return errors

    def stagingCalculate(self,Dv1,Dv2,con):

        efflist = con['efflist']
        Tlist = con['Tlist']
        p2 = optimalStaging([efflist[  -1]],Dv2,[Tlist[  -1]],con,con['Isp2'],con['g0'], con['Mu'])
        p1 = optimalStaging( efflist[0:-1] ,Dv1, Tlist[0:-1] ,con,con['Isp1'],con['g0'],p2.mtot[0])

        if con['NStag'] == 0:
            p2.mu = p1.me[0] + p2.me[0] + p2.mu
            p2.me[0] = 0.0
            p1.me[0] = 0.0
        if con['NStag'] == 1:
            p2.me[0] = p1.me[0] + p2.me[0]
            p1.me[0] = 0.0

        if p1.fail or p2.fail:
            raise Exception('itsme saying: Too few stages!')

        if p1.mtot[0]*con['g0'] > Tlist[0]:
            raise Exception('itsme saying: weight greater than thrust!')

        return p1,p2

    def displayInfo(self,p1,p2,con,tphases,mjetsoned):

        if con['NStag'] == 0:
            print("\n\rSpacecraft properties (NStag == 0):")
            mp = p1.mp[0] + p2.mp[0]
            print("m_empty =",p1.mtot[0] - mp)
            print("mp =",mp)
            print("mtot =",p1.mtot[0])
            print("tf =",p2.tf[-1],"\n\r")

        elif con['NStag'] == 1:
            print("\n\rSSTO properties (NStag == 1):")
            print("mu =",p2.mu)
            print("mp =",p1.mp[0] + p2.mp[0])
            print("me =",p2.me[0])
            print("mtot =",p1.mtot[0])
            print("tf =",p2.tf[-1],"\n\r")

        elif con['NStag'] == 2:
            print("\n\rTSTO first stage properties (NStag == 2):")
            p1.printInfo()
            print("TSTO final stage properties (NStag == 2):")
            p2.printInfo()

        else:
            print("\n\rInitial stages properties:")
            p1.printInfo()
            print("Final stage properties:")
            p2.printInfo()

        print('\n\rtphases: ',tphases)
        print('\n\rmjetsoned: ',mjetsoned)
        print('\n\r')

    def plotResults(self):

        tt,xx,uu,tp,xp,up = self.tt,self.xx,self.uu,self.tp,self.xp,self.up

        ii = 0
        plt.subplot2grid((6,4),(0,0),rowspan=2,colspan=2)
        plt.hold(True)
        plt.plot(tt,xx[:,ii],'.-b')
        plt.plot(tp,xp[:,ii],'.r')
        plt.hold(False)
        plt.grid(True)
        plt.ylabel("h [km]")

        ii = 1
        plt.subplot2grid((6,4),(0,2),rowspan=2,colspan=2)
        plt.hold(True)
        plt.plot(tt,xx[:,ii],'.-b')
        plt.plot(tp,xp[:,ii],'.r')
        plt.hold(False)
        plt.grid(True)
        plt.ylabel("V [km/s]")

        ii = 2
        plt.subplot2grid((6,4),(2,0),rowspan=2,colspan=2)
        plt.hold(True)
        plt.plot(tt,xx[:,ii]*180.0/numpy.pi,'.-b')
        plt.plot(tp,xp[:,ii]*180.0/numpy.pi,'.r')
        plt.hold(False)
        plt.grid(True)
        plt.ylabel("gamma [deg]")

        ii = 3
        plt.subplot2grid((6,4),(2,2),rowspan=2,colspan=2)
        plt.hold(True)
        plt.plot(tt,xx[:,ii],'.-b')
        plt.plot(tp,xp[:,ii],'.r')
        plt.hold(False)
        plt.grid(True)
        plt.ylabel("m [kg]")

        ii = 0
        plt.subplot2grid((6,4),(4,0),rowspan=2,colspan=2)
        plt.hold(True)
        plt.plot(tt,uu[:,ii]*180/numpy.pi,'.-b')
        plt.plot(tp,up[:,ii]*180/numpy.pi,'.r')
        plt.hold(False)
        plt.grid(True)
        plt.ylabel("alfa [deg]")

        ii = 1
        plt.subplot2grid((6,4),(4,2),rowspan=2,colspan=2)
        plt.hold(True)
        plt.plot(tt,uu[:,ii],'.-b')
        plt.plot(tp,up[:,ii],'.r')
        plt.hold(False)
        plt.grid(True)
        plt.xlabel("t")
        plt.ylabel("beta [adim]")

        plt.show()

        return None

    def plotResultsAed(self,problem):

        tt = self.tt
        tp = self.tp
        # Aed plots
        LL, DD, CCL, CCD, QQ = self.calcAedTab(self.tt,self.xx,self.uu,problem.con)
        Lp, Dp, CLp, CDp, Qp = self.calcAedTab(self.tp,self.xp,self.up,problem.con)

        plt.subplot2grid((6,2),(0,0),rowspan=2,colspan=2)
        plt.hold(True)
        plt.plot(tt,LL,'.-b',tp,Lp,'.r')
        plt.plot(tt,DD,'.-g',tp,Dp,'.r')
        plt.hold(False)
        plt.grid(True)
        plt.ylabel("L and D [kN]")

        plt.subplot2grid((6,2),(2,0),rowspan=2,colspan=2)
        plt.hold(True)
        plt.plot(tt,CCL,'.-b',tp,CLp,'.r')
        plt.plot(tt,CCD,'.-g',tp,CDp,'.r')
        plt.hold(False)
        plt.grid(True)
        plt.ylabel("CL and CD [-]")

        plt.subplot2grid((6,2),(4,0),rowspan=2,colspan=2)
        plt.hold(True)
        plt.plot(tt,QQ,'.-b',tp,Qp,'.r')
        plt.hold(False)
        plt.grid(True)
        plt.ylabel("qdin [kPa]")

        plt.show()

        return None

    def calcAedTab(self,tt,xx,uu,con):

        LL = tt.copy()
        DD = tt.copy()
        CCL = tt.copy()
        CCD = tt.copy()
        QQ = tt.copy()
        for ii in range( 0, len(tt) ):

            h = xx[ii,0]
            v = xx[ii,1]
            alfat = uu[ii,0]
            # Aerodynamics
            CL = con['CL0'] + con['CL1']*alfat
            CD = con['CD0'] + con['CD2']*(alfat**2)

            qdin = 0.5 * rho(h) * (v**2)
            L = qdin * con['s_ref'] * CL
            D = qdin * con['s_ref'] * CD

            LL[ii] = L
            DD[ii] = D
            CCL[ii] = CL
            CCD[ii] = CD
            QQ[ii] = qdin

        return LL, DD, CCL, CCD, QQ


    def orbitResults(self,GM,R):

        h,v,gama,M = numpy.transpose(self.xx[-1,:])

        r = R + h
        cosGama = numpy.cos(gama)
        sinGama = numpy.sin(gama)
        momAng = r * v * cosGama
        print("Ang mom:",momAng)
        en = .5 * v * v - GM/r
        print("Energy:",en)
        a = - .5*GM/en
        print("Semi-major axis:",a)
        aux = v * momAng / GM
        e = numpy.sqrt((aux * cosGama - 1)**2 + (aux * sinGama)**2)
        print("Eccentricity:",e)

        print("Final altitude:",h)
        ph = a * (1.0 - e) - R
        print("Perigee altitude:",ph)
        ah = 2*(a - R) - ph
        print("Apogee altitude:",ah)

        self.e = e

class itsSolution():

    def __init__(self,con,tabAlpha,tabBeta,factors):

        self.tabAlpha = tabAlpha
        self.tabBeta = tabBeta
        self.factors = factors

        self.GM = con['GM']
        self.R = con['R']

        traj = itsTrajectory()
        traj.trajectorySimulate(factors,con,"plot")
        self.trajectory = traj

        traj = itsTrajectory()
        traj.trajectorySimulate(factors,con,"orbital")
        self.trajOrbital = traj

    def displayResults(self):

        # Results without orbital phase
        self.trajectory.orbitResults(self.GM,self.R)
        self.trajectory.plotResults()

        # Results with orbital phase
        if abs(self.trajectory.e - 1) > 0.1:
            # The eccentricity test avoids simulations too close of the singularity
            self.trajOrbital.orbitResults(self.GM,self.R)
            self.trajOrbital.plotResults()
        print('Initial states:',self.trajectory.xx[ 0])
        print('Final   states:',self.trajOrbital.xx[-1])

        return None

    def sgra(self):

        ans = self.trajectory.tt,self.trajectory.xx,self.trajectory.uu,self.tabAlpha,self.tabBeta
        return ans

if __name__ == "__main__":

    print("itsme: Inital Trajectory Setup Module")
    ################
    #    Errors:

    #    (v - V_final)/0.01
    #    (gamma - gamma_final)/0.1
    #    (h - h_final)/10

    ################
    trajectory = its()
    print('Total number of trajectory simulations: ',totalTrajectorySimulationCounter)
    #input("Press any key to finish...")
