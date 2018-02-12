#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:57:52 2018

@author: carlos
"""

import numpy


class modelInitialEstimate():

    def __init__(self, con: dict):

        if con['NStag'] < 2:
            iniEst = modelInitialEstimateHomogeneous(con)
        else:
            iniEst = modelInitialEstimateSimple2(con)

        self.dv = iniEst.dv
        self.t = iniEst.t
        self.vx = iniEst.vx
        print([self.dv, self.t, self.vx])


class modelInitialEstimateHomogeneous():

    def __init__(self, con: dict):

        efflist = con['efflist']
        Tlist = con['Tlist']

        lamb = numpy.exp(0.5*con['V_final']/(con['Isp2']*con['g0']))
        self.Mu = con['Mu']*lamb
        self.hf = con['h_final']
        self.M0max = Tlist[0]/con['g0']
        self.c = con['Isp1']*con['g0']
        self.mflux = numpy.mean(Tlist[0:-1])/self.c
        self.GM = con['GM']
        self.R = con['R']
        self.e = numpy.exp(numpy.mean(numpy.log(efflist[0:-1])))
        self.g0 = con['g0'] - con['R']*(con['we'] ** 2)
        self.t1max = self.M0max/self.mflux
        self.fail = False

        self.__newtonRaphson()
        self.__dvt()

        self.vx = 0.5*self.V1*(con['R'] + self.h1)/(con['R']+self.hf)

    def __newtonRaphson(self):

        N = 500
        t1max = self.t1max

        dt = t1max/N
        t1 = 0.1*t1max

        cont = 0
        erro = 1.0

        while (abs(erro) > 1e-6) and not self.fail:

            erro = self.__hEstimate(t1) - self.hf
            dedt = (self.__hEstimate(t1+dt) - self.__hEstimate(t1-dt))/(2*dt)
            t1 = t1 - erro/dedt
            cont += 1
            if cont == N:
                raise Exception('itsme saying: initialEstimate failed')

        self.cont = cont
        self.t1 = t1
        return None

    def __hEstimate(self, t1: float)-> float:

        mflux = self.mflux
        Mu = self.Mu
        g0 = self.g0
        c = self.c

        Mp = mflux*t1
        Me = Mp*self.e/(1 - self.e)
        M0 = Mu + Me + Mp
        x = (Mu + Me)/M0
        V1 = c*numpy.log(1/x) - g0*t1
        h1 = (c*M0/mflux) * ((x*numpy.log(x) - x) + 1) - g0*(t1**2)/2
        h = h1 + self.GM/(self.GM/(self.R + h1) - (V1**2)/2) - self.R

        return h

    def __dvt(self)-> None:

        t1 = self.t1
        mflux = self.mflux
        Mu = self.Mu
        g0 = self.g0
        c = self.c

        M0 = Mu + mflux*t1
        V1 = c*numpy.log(M0/Mu) - g0*t1
        dv = V1 + g0*t1
        x = Mu/M0
        h1 = (c*M0/mflux)*((x*numpy.log(x) - x) + 1) - g0*(t1**2)/2

        h = self.__hEstimate(self.t1)

        rr = (self.R + h1)/(self.R + h)
        if rr > 1:
            raise Exception('itsme saying: initialEstimate failed (h1 > h)')
        theta = numpy.arccos(numpy.sqrt(rr))
        ve = numpy.sqrt(2*self.GM/(self.R + h))

        t = t1 + ((self.R + h)/ve) * (numpy.sin(2*theta)/2 + theta)

        self.h1 = h1
        self.h = h
        self.t = t
        self.dv = dv
        self.V1 = V1

        return None


class modelInitialEstimateSimple():

    #  This class is a try of creating a simplified initial estimate method
    #  that could be used in a heterogeneous rocket configuration
    def __init__(self, con: dict):

        self.fail = False
        self.GM = con['GM']
        self.R = con['R']
        self.g0 = con['g0']
        self.hf = con['h_final']
        self.a = 0.4*self.g0
        self.vc = numpy.sqrt(self.GM/(self.R + self.hf))
        self.vx = self.vc*0.18
        self.__newtonRaphson()
        self.__tCalculate()

    def __newtonRaphson(self)-> None:

        N = 100
        ddv = self.vc/N
        dv = self.vc

        cont = 0
        erro = 1.0

        while (abs(erro) > 1e-6) and not self.fail:

            erro = self.__dvEstimate(dv)
            dedt = (self.__dvEstimate(dv+ddv) -
                    self.__dvEstimate(dv - ddv))/(2*ddv)
            dv = dv - erro/dedt
            cont += 1
            if cont == N:
                raise Exception('itsme saying: initialEstimate failed')
                self.fail = True

        self.cont = cont
        self.dv = dv
        return None

    def __dvEstimate(self, dv: float)-> float:

        self.h1 = dv**2/(2*self.a)
        aux = ((dv**2)/2 + self.GM/(self.R + self.hf) -
               self.GM/(self.R + self.h1))
        if aux < 0:
            aux = 0
        perda_gravitacional = dv - numpy.sqrt(2*aux) - dv*0.05
        perda_arrasto = dv*0.1

        erro = self.vc - (self.vx + dv - perda_gravitacional - perda_arrasto)
        return erro

    def __tCalculate(self)-> None:

        vx0 = self.vx*(self.R + self.hf)/(self.R + self.h1)
        vy = numpy.sqrt(self.dv**2 - vx0**2)
        self.t = vy/self.g0


class modelInitialEstimateSimple2():

    # Its is recomendable use all initial guess values equal one
    # TODO: Understand why the conventional itsme dv estimate is so small
    def __init__(self, con: dict):

        self.fail = False
        self.GM = con['GM']
        self.R = con['R']
        self.g0 = con['g0']
        self.hf = con['h_final']
        self.vc = numpy.sqrt(self.GM/(self.R + self.hf))

        # Losses:
        # Gravitational
        self.lg = 0.1
        # Drag
        self.ld = 0.1

        # Fraction of final dv
        self.dv2 = self.vc*con['fracVel']

        # Calculations
        self.__newtonRaphson()
        self.__tCalculate()
        self.vx = self.v2

    def __newtonRaphson(self)-> None:

        N = 100
        ddv = self.vc/N
        dv = self.vc

        cont = 0
        erro = 1.0

        while (abs(erro) > 1e-6) and not self.fail:

            erro = self.__dvEstimate(dv)
            dedt = (self.__dvEstimate(dv + ddv) -
                    self.__dvEstimate(dv - ddv))/(2*ddv)
            dv = dv - erro/dedt
            cont += 1
            if cont == N:
                raise Exception('itsme saying: initialEstimate failed')
                self.fail = True

        self.cont = cont
        self.dv = dv
        return None

    def __dvEstimate(self, dv: float)-> float:

        # First propulsive phases
        self.v1 = (1 - self.lg - self.ld)*dv
        at1 = self.g0/self.lg
        a_res1 = at1*(1 - self.lg - self.ld)
        self.h1 = (self.v1**2)/(2*a_res1)

        # Coast phase
        aux = ((self.v1**2) + 2*self.GM/(self.R + self.hf) -
               2*self.GM/(self.R + self.h1))
        if aux < 0:
            aux = 0

        self.v2 = numpy.sqrt(aux)

        erro = self.vc - self.v2 - self.dv2
        return erro

    def __tCalculate(self)-> None:

        # Coast phase duration estimate
        vx1 = self.v2*(self.R + self.hf)/(self.R + self.h1)
        vy1 = numpy.sqrt(self.v1**2 - vx1**2)
        self.t = 2*vy1/(self.GM/(self.R + self.hf)**2 +
                        self.GM/(self.R + self.h1)**2)
