#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:25:28 2017

@author: levi
"""
import numpy, itsme
from sgra import sgra
from atmosphere import rho, rhoSGRA
import matplotlib.pyplot as plt
from utils import simp#, getNowStr
from naivRock import naivGues

d2r = numpy.pi / 180.0

class prob(sgra):
    probName = 'probRock'

    def storePars(self,inputDict,inputFile):
        """Get parameters from dictionary, load them into self properly."""
        # For lazy referencing here
        n = self.n
        # Payload mass
        self.mPayl = inputDict['Mu']

        # Earth constants
        r_e, GM = inputDict['R'], inputDict['GM']
        grav_e = GM / r_e / r_e

        # Target heights for separation.
        # This is only here because the Isp, Thrust and s_f arrays must be
        # padded to put the extra arcs there.
        TargHeig = numpy.array(inputDict['TargHeig'])
        # Number of additional arcs:
        addArcs = len(TargHeig); self.addArcs = addArcs
        # Number of arcs:
        s = inputDict['NStag'] + addArcs; self.s = s

        # Extra arcs go in the beginning only.
        isStagSep = numpy.ones(s, dtype='bool')
        for i in range(addArcs):
            isStagSep[i] = False
        self.isStagSep = isStagSep

        # There are always as many pi's as there are arcs
        p = s; self.p = p

        # n + n-1: beginning and end conditions (free final mass);
        # n*(s-addArcs-1) are for "normal" arc  interfaces
        #  [hf=hi,vf=vi,gf=gi,mf=mi or mf=mi+jettisoned mass];
        # (n+1)* addArcs are for additional arc interfaces
        #  [hf=hTarg,hi=hTarg,vf=vi,gf=gi,mf=mi]
        q = (n + n - 1) + n * (s - addArcs - 1) + (n + 1) * addArcs
        self.q = q
        # This is the 'N' variable in Miele (2003)
        self.Ns = 2 * n * s + p

        # rocket constants
        ones = numpy.ones(s)
        if 'Isplist' in inputDict.keys():
            Isp = numpy.array(inputDict['Isplist'])
            Isp = numpy.pad(Isp, (addArcs, 0),
                            'constant', constant_values=Isp[0])
        else:
            Isp = inputDict['Isp'] * numpy.ones(s)

        Thrust = numpy.array(inputDict['Tlist'])
        Thrust = numpy.pad(Thrust, (addArcs, 0),
                           'constant', constant_values=Thrust[0])
        s_f = numpy.array(inputDict['efflist'])
        s_f = numpy.pad(s_f, (addArcs, 0),
                        'constant', constant_values=s_f[0])

        CL0, CL1 = inputDict['CL0'] * ones, inputDict['CL1'] * ones
        CD0, CD2 = inputDict['CD0'] * ones, inputDict['CD2'] * ones
        s_ref = inputDict['s_ref'] * ones
        DampCent, DampSlop = inputDict['DampCent'],inputDict['DampSlop']
        acc_max = inputDict['acc_max'] * grav_e  # in km/s²

        # Penalty function settings

        # This approach to Kpf is that if, in any point during flight, the
        # acceleration exceeds the limit by acc_max_tol, then the penalty
        # function at that point is PFtol times greater than the maximum
        # value of the original cost functional.

        PFmode = inputDict['PFmode']
        costFuncVals = Thrust / grav_e / Isp / (1.0 - s_f)
        PFtol = inputDict['PFtol']
        acc_max_relTol = inputDict['acc_max_relTol']
        acc_max_tol = acc_max_relTol * acc_max
        if PFmode == 'lin':
            Kpf = PFtol * max(costFuncVals) / acc_max_tol
        elif PFmode == 'quad':
            Kpf = PFtol * max(costFuncVals) / (acc_max_tol ** 2)
        elif PFmode == 'tanh':
            Kpf = PFtol * max(costFuncVals) / numpy.tanh(acc_max_relTol)
        else:
            self.log.printL('Error: unknown PF mode "' + str(PFmode) + '"')
            raise KeyError

        constants = {'grav_e': grav_e, 'Thrust': Thrust,
                     'Isp': Isp, 's_f': s_f,
                     'r_e': r_e, 'GM': GM,
                     'CL0': CL0, 'CL1': CL1, 'CD0': CD0, 'CD2': CD2,
                     's_ref': s_ref,
                     'DampCent': DampCent, 'DampSlop': DampSlop,
                     'PFmode': PFmode, 'Kpf': Kpf}
        self.constants = constants

        # boundary conditions
        h_final = inputDict['h_final']
        V_final = numpy.sqrt(GM / (r_e + h_final))  # km/s
        missDv = numpy.sqrt((GM / r_e) * (2.0 - r_e / (r_e + h_final)))

        self.boundary = {'h_initial': inputDict['h_initial'],
                         'V_initial': inputDict['V_initial'],
                         'gamma_initial': inputDict['gamma_initial'],
                         'm_initial': 0., # Just a place holder!
                         'h_final': h_final,
                         'V_final': V_final,
                         'gamma_final': inputDict['gamma_final'],
                         'mission_dv': missDv,
                         'TargHeig': TargHeig}

        # restrictions
        alpha_max = inputDict['AoAmax'] * d2r * ones  # in rads
        alpha_min = -alpha_max  # in rads
        self.restrictions = {'alpha_min': alpha_min,
                             'alpha_max': alpha_max,
                             'beta_min': 0.0 * ones,
                             'beta_max': ones,
                             'acc_max': acc_max}

        # Load remaining parameters:
        # - Time discretization,
        # - P and Q tolerances,
        # - Gradient step search options [constants]
        # - Time (pi) limitations [restrictions]
        self.loadParsFromFile(file=inputFile)

    def initGues(self,opt={}):

        # matrix sizes
        n, m = 4, 2
        self.n, self.m = n, m

        # Get initialization mode
        initMode = opt.get('initMode','default'); self.initMode = initMode

        if initMode == 'default' or initMode == 'naive':
            N = 500+1
            self.N = N

            dt = 1.0/(N-1)
            t = numpy.arange(0,1.0+dt,dt)
            self.dt = dt
            self.t = t

            s = 2
            addArcs = 0
            p = s
            self.s = s
            self.addArcs = addArcs
            self.p = p
            self.Ns = 2*n*s + p

            q = 2*n - 1 + n * (s-1)
            self.q = q

            x = numpy.zeros((N,n,s))
            u = numpy.zeros((N,m,s))

            #prepare tolerances
            tolP = 1.0e-5
            tolQ = 1.0e-7
            tol = dict()
            tol['P'] = tolP
            tol['Q'] = tolQ
            self.tol = tol

            # Earth constants
            r_e = 6371.0           # km
            GM = 398600.4415       # km^3 s^-2
            grav_e = GM/r_e/r_e    #9.8e-3       km/s^2

            # Payload mass
            self.mPayl = 100

            # rocket constants
            Thrust = 300*numpy.ones(s)                 # kg km/s² [= kN]

            Kpf = 100.0  # First guess........
            DampCent = 0
            DampSlop = 0
            PFmode = 'quad'
            gradStepSrchCte = 1.0e-4

            Isp = 300.0*numpy.ones(s)                     # s
            s_f = 0.05*numpy.ones(s)
            CL0 = 0.0*numpy.ones(s)                       # (B0 Miele 1998)
            CL1 = 0.8*numpy.ones(s)                       # (B1 Miele 1998)
            CD0 = 0.05*numpy.ones(s)                      # (A0 Miele 1998)
            CD2 = 0.5*numpy.ones(s)                       # (A2 Miele 1998)
            s_ref = (numpy.pi*(0.0005)**2)*numpy.ones(s)  # km^2

            # boundary conditions
            h_initial = 0.0            # km
            V_initial = 1e-6           # km/s
            gamma_initial = numpy.pi/2 # rad
            m_initial = 20000          # kg
            h_final = 463.0            # km
            V_final = numpy.sqrt(GM/(r_e+h_final))#7.633   # km/s
            gamma_final = 0.0 # rad

            boundary = dict()
            boundary['h_initial'] = h_initial
            boundary['V_initial'] = V_initial
            boundary['gamma_initial'] = gamma_initial
            boundary['m_initial'] = m_initial
            boundary['h_final'] = h_final
            boundary['V_final'] = V_final
            boundary['gamma_final'] = gamma_final
            boundary['mission_dv'] = numpy.sqrt((GM/r_e)*\
                    (2.0-r_e/(r_e+h_final)))
            self.boundary = boundary

            constants = dict()
            constants['grav_e'] = grav_e
            constants['Thrust'] = Thrust
            constants['Isp'] = Isp
            constants['r_e'] = r_e
            constants['GM'] = GM
            constants['s_f'] = s_f
            constants['CL0'] = CL0
            constants['CL1'] = CL1
            constants['CD0'] = CD0
            constants['CD2'] = CD2
            constants['s_ref'] = s_ref
            constants['DampCent'] = DampCent
            constants['DampSlop'] = DampSlop
            constants['PFmode'] = PFmode
            constants['Kpf'] = Kpf
            constants['gradStepSrchCte'] = gradStepSrchCte

            self.constants = constants

            # restrictions
            alpha_min = -2 * d2r  # in rads
            alpha_max = 2 * d2r   # in rads
            beta_min = 0.0
            beta_max = 1.0
            acc_max = 3.0 * grav_e
            pi_min = numpy.zeros(s)
            pi_max = numpy.empty_like(pi_min)
            for k in range(s):
                pi_max[k] = None
            #pi_max = numpy.array([None])
            restrictions = dict()
            restrictions['alpha_min'] = alpha_min
            restrictions['alpha_max'] = alpha_max
            restrictions['beta_min'] = beta_min
            restrictions['beta_max'] = beta_max
            restrictions['acc_max'] = acc_max
            restrictions['pi_min'] = pi_min
            restrictions['pi_max'] = pi_max
            self.restrictions = restrictions
            PFtol = 1.0e-2
            acc_max_relTol = 0.1

            costFuncVals = Thrust/grav_e/Isp/(1.0-s_f)
            acc_max_tol = acc_max_relTol * acc_max
            if PFmode == 'lin':
                Kpf = PFtol * max(costFuncVals) / (acc_max_tol)
            elif PFmode == 'quad':
                Kpf = PFtol * max(costFuncVals) / (acc_max_tol**2)
            elif PFmode == 'tanh':
                Kpf = PFtol * max(costFuncVals) / numpy.tanh(acc_max_relTol)
            else:
                self.log.printL('Error: unknown PF mode "' + str(PFmode) + '"')
                raise KeyError
            if initMode == 'default':
############### Artesanal handicraft with L and D (Miele 2003)
                arclen = numpy.floor(len(t)/s).astype(int)
                remainder = len(t) % arclen

                tarc = numpy.zeros((arclen,s+1))
                for k in range(s):
                    tarc[:,k] = t[k*arclen:(k+1)*arclen]
                for r in range(remainder):
                    tarc[r,s] = t[s*arclen+r]
                t_complete = numpy.zeros(((s+1)*arclen,s+1))
                for k in range(s+1):
                    t_complete[:,k] = numpy.pad(tarc[:,k],
                              (k*arclen,(s-k)*arclen),'constant')

                for arc in range(s):
                    for line in range(N):
                        x[line,0,arc] = h_final * \
                                numpy.sin(0.5*numpy.pi*t_complete[line,arc])
                        x[line,1,arc] = V_final * \
                                numpy.sin(numpy.pi*t_complete[line,arc]/2)
                        x[line,2,arc] = (numpy.pi/2) * \
                                (numpy.exp(-(t_complete[line,arc]**2)/0.017))+\
                                0.0#+0.06419
                        expt = numpy.exp(-(t_complete[line,arc]**2)/0.02)
                        x[line,3,arc] = m_initial*((0.7979* expt) +
                                        0.1901*numpy.cos(t_complete[line,arc]))
                        #x[:,1,arc] = 1.0e3*(-0.4523*tarc[:,arc]**5 + \
                                    #1.2353*tarc[:,arc]**4 + \
                                    #-1.1884*tarc[:,arc]**3 + \
                                    #0.4527*tarc[:,arc]**2 + \
                                    #-0.0397*tarc[:,arc])
                        #x[:,1,arc] = 3.793*numpy.exp(0.7256*tarc[:,arc]) + \
                                     #-1.585 + \
                                     #-3.661*numpy.cos(3.785*tarc[:,arc]+ \
                                     #0.9552)
                        #x[:,3,arc] = m_initial*(1.0-0.89*tarc[:,arc])
                        #x[:,3,arc] = m_initial*(-2.9*tarc[:,arc]**3 + \
                                        #6.2*tarc[:,arc]**2 + \
                                        #- 4.2*tarc[:,arc] + 1)

                pi_time = 200
                total_time = pi_time*s
                for k in range(N):
                    if total_time*t[k]<200:
                        u[k,1,:] = (numpy.pi/2)
                    elif total_time*t[k]>600:
                        u[k,1,:] = (numpy.pi/2)*0.27

                pi = pi_time*numpy.ones(p)
            else:
############### Naive
                t_shaped = numpy.reshape(t,(N,1))
                t_matrix = t_shaped*numpy.ones((N,s))
                t_flip_shaped = numpy.flipud(numpy.reshape(t,(N,1)))
                #t_flip_matrix = t_flip_shaped*numpy.ones((N,s))
                x[:,0,:] = h_final*t_matrix
                x[:,1,:] = V_final*t_matrix
                x[:,2,:] = gamma_initial*t_flip_shaped
                x[:,3,:] = (m_initial)*t_flip_shaped
#                x[:,0,:] = (numpy.pi/4)*numpy.ones((N,s))
#                x[:,1,:] = V_final*numpy.ones((N,s))
#                x[:,2,:] = (numpy.pi/4)*(numpy.ones((N,s)))
#                x[:,3,:] = (m_initial)*numpy.ones((N,s))
                u[:,0,:] = (0.005)*numpy.ones((N,s))
                u[:,1,:] = (0.5)*numpy.ones((N,s))
                pi = 500*numpy.ones(s)
#
        elif initMode == 'naive2':
            inpFile = opt.get('confFile','defaults/probRock.its')
            sol = naivGues(inpFile,extLog=self.log)

            # Get the parameters from the sol object
            self.constants = sol.constants
            self.restrictions = sol.restrictions
            self.boundary = sol.boundary
            self.mPayl = sol.mPayl
            self.isStagSep = sol.isStagSep
            self.addArcs = sol.addArcs
            self.N, self.p, self.q, self.s = sol.N, sol.p, sol.q, sol.s
            self.dt, self.t = sol.dt, sol.t
            self.Ns = sol.Ns
            self.tol = sol.tol
            # These go outside because of the bypass that comes later
            x, pi = sol.x, sol.pi
            # Doing here just to undo later. Counter-productive, I know.
            u = numpy.empty((self.N,self.m,self.s))
            u[:,0,:], u[:,1,:] = sol.calcDimCtrl()
            # all info retrieved; the sol object can now be safely deleted
            del sol

        elif initMode == 'extSol':

            ###################################################################
            # STEP 1: run itsme
            inpFile = opt.get('confFile','defaults/probRock.its')
            self.log.printL("Starting ITSME with input = " + inpFile)

            t_its, x_its, u_its, tabAlpha, tabBeta, inputDict, tphases, \
            mass0, massJet = itsme.sgra(inpFile)
            # 'inputDict' corresponds to the 'con' dictionary from itsme.

            self.log.printL("\nITSME was run sucessfully, " + \
                            "proceding adaptations...")

            ###################################################################
            # STEP 2: Load constants from file (actually, via inputDict)

            self.storePars(inputDict,inpFile)
            # This is specific to this initMode (itsme)
            self.boundary['m_initial'] = x_its[0, 3]
            # For lazy referencing below:
            N, addArcs, s = self.N, self.addArcs, self.s
            TargHeig = self.boundary['TargHeig']
            constants = self.constants

            ###################################################################
            # STEP 3: Set up the arcs, handle stage separation points

            # Find indices for beginning of arc
            arcBginIndx = numpy.empty(s+1,dtype='int')
            arc = 0; arcBginIndx[arc] = 0
            nt = len(t_its)

            # The goal here is to find the indexes in the x_its array that
            # correspond to where the arcs begin, hence, assembling the
            # 'arcBginIndx' array.

            # First, find the times for the targeted heights:
            j = 0
            for hTarg in TargHeig:
                # search for target height
                keepLook = True
                while keepLook and (j < nt):
                    if x_its[j,0] > hTarg:
                        # Found the index!
                        arc += 1; arcBginIndx[arc] = j; keepLook = False
                    #
                    j+=1
                #
            #
            # Second, search for the times with jettisoned masses:
            nmj = len(massJet)
            for i in range(nmj):
                if massJet[i] > 0.0:
                    # Jettisoned mass found!
                    arc += 1; tTarg = tphases[i]; keepLook = True
                    while keepLook and (j < nt):
                        if abs(t_its[j]-tTarg) < 1e-10:
                            # TODO: this hardcoded 1e-10 may cause problems...
                            # Found the proper time!
                            # get the next time for proper initial conditions
                            j += 1; arcBginIndx[arc] = j; keepLook = False
                        #
                        j += 1
                    #
                #
            #
            # Finally, set the array of interval lengths
            pi = numpy.empty(s)
            for arc in range(s):
                pi[arc] = t_its[arcBginIndx[arc+1]] - t_its[arcBginIndx[arc]]

            # Set up the variation omission configuration:

            # list of variations after omission
            mat = numpy.eye(self.q)
            # matrix for omitting equations
            # Assemble the list of equations to be omitted (start by omitting none)
            omitEqMatList = list(range(self.q))
            # Omit one equation for each assigned state (height)
            Ns = 2 * self.n * self.s + self.p
            for arc in range(addArcs-1,-1,-1):
                i = 5 + 5 * arc
                #psi[i + 1] = x[0, 0, arc + 1] - TargHeig[arc]
                omitEqMatList.pop(i)
            # Removing the first n elements, corresponding to the initial states
            for i in range(self.n):
                omitEqMatList.pop(0)
            self.omitEqMat = mat[omitEqMatList, :]
            # list of variations after omission
            omitVarList = list(range(Ns + 1))
            # this is how it works with 1 added arc
            #self.omitVarList = [  # states for 1st arc (all omitted)
            #    4, 5, 6, 7,  # Lambdas for 1st arc
            #    9, 10, 11,  # states for 2nd arc (height omitted)
            #    12, 13, 14, 15,  # Lambdas for 2nd arc
            #    16, 17,  # pi's, 1st and 2nd arc
            #    18]  # final variation
            for arc in range(addArcs - 1, -1, -1):
                i = 2 * self.n * (arc+1)
                # states in order: height (2x), speed, flight angle and mass
                # psi[i + 1] = x[0, 0, arc + 1] - TargHeig[arc]
                omitVarList.pop(i)
            # Removing the first n elements, corresponding to the initial states
            for i in range(self.n):
                omitVarList.pop(0)
            self.omitVarList = omitVarList
            self.omit = True

            ###################################################################
            # STEP 4: Re-integrate the differential equation with a fixed step

            self.log.printL("Re-integrating ITSME solution with fixed " + \
                            "step scheme...")
            # Re-integration of proposed solution (RK4)
            # Only the controls are used, not the integrated state itself
            x = numpy.zeros((N,n,s)); u = numpy.zeros((N,m,s))
            for arc in range(s):
                # dtd: dimensional time step
                dtd = pi[arc]/(N-1); dtd6 = dtd/6.0
                x[0,:,arc] = x_its[arcBginIndx[arc],:]
                t0arc = t_its[arcBginIndx[arc]]
                uip1 = numpy.array([tabAlpha.value(t0arc),
                                    tabBeta.value(t0arc)])
                # td: dimensional time (for integration)
                for i in range(N-1):
                    td = t0arc + i * dtd
                    ui = uip1
                    u[i,:,arc] = ui

                    uipm = numpy.array([tabAlpha.value(td+.5*dtd),
                                        tabBeta.value(td+.5*dtd)])
                    uip1 = numpy.array([tabAlpha.value(td+dtd),
                                        tabBeta.value(td+dtd)])
                    # this bypass just ensures consistency for control
                    if i == N-2 and arc == s-1:
                        uip1 = ui
                    x1 = x[i,:,arc]
                    f1 = calcXdot(td,x1,ui,constants,arc)
                    tdm = td+.5*dtd # time at half the integration interval
                    x2 = x1 + .5 * dtd * f1 # x at half step, with f1
                    f2 = calcXdot(tdm,x2,uipm,constants,arc)
                    x3 = x1 + .5 * dtd * f2 # x at half step, with f2
                    f3 = calcXdot(tdm,x3,uipm,constants,arc)
                    x4 = x1 + dtd * f3 # x at next step, with f3
                    f4 = calcXdot(td+dtd,x4,uip1,constants,arc)
                    x[i+1,:,arc] = x1 + dtd6 * (f1+f2+f2+f3+f3+f4)
                #
                u[N-1,:,arc] = u[N-2,:,arc]
            #

        lam = numpy.zeros((self.N,n,self.s))
        mu = numpy.zeros(self.q)

        # Bypass
        ones = numpy.ones(self.s)
        self.restrictions['alpha_min'] = -3.0*numpy.pi/180.0 * ones
        self.restrictions['alpha_max'] = 3.0*numpy.pi/180.0 * ones
#        ThrustFactor = 2.0#500.0/40.0
#        self.constants['Thrust'] *= ThrustFactor
#        # Re-calculate the Kpf, since it scales with Thrust...
#        #self.constants['Kpf'] *= ThrustFactor
#        u[:,1,:] *= 1.0/ThrustFactor

        u = self.calcAdimCtrl(u[:,0,:],u[:,1,:])

        self.x, self.u, self.pi = x, u, pi
        self.lam, self.mu = lam, mu

        solInit = self.copy()

        # This segment does the alterations on itsme solution to yield the
        # Miele ratios (TWR=1.3, WL = 1890.)

        # self.log.printL("\nRestoring initial solution.")
        # self.calcP()
        # while self.P > self.tol['P']:
        #     self.rest(parallelOpt={'restLMPBVP':True})
        # self.log.printL("\nSolution was restored. ")
        #
        # TWR = self.constants['Thrust'][0] / self.boundary['m_initial'] / self.constants['grav_e']
        # TWR_targ = 1.3
        # dm = (TWR/TWR_targ) * self.boundary['m_initial']/30.
        # sign = 1.
        # while abs(TWR-TWR_targ)>0.0001:
        #     self.log.printL("TWR = {:.4G}. Time to change that mass!".format(TWR))
        #     #input("\nI am about to mess things up. Be careful. ")
        #     #dm = 10.
        #     #self.x[:,3,:] += dm
        #     if sign * (TWR - TWR_targ) < 0:
        #         dm = -dm/2.
        #         sign *= -1.
        #     self.boundary['m_initial'] += dm
        #     self.log.printL("\nDone. Let's restore it again.")
        #     self.calcP()
        #     while self.P > self.tol['P']:
        #         self.rest(parallelOpt={'restLMPBVP':True})
        #     TWR = self.constants['Thrust'][0] / self.boundary['m_initial'] / self.constants['grav_e']
        #
        #     #self.compWith(solInit,altSolLabl='itsme',piIsTime=False)
        #
        # S = self.constants['s_ref'][0]  # km²
        # WL_targ = 1890.  # kgf/m²
        # S_targ = (self.boundary['m_initial'] / WL_targ) * 1e-6  # km²
        # dS = (S_targ - S) / 10.  # km²
        # print("S = {:.4G} km², S_targ = {:.4G} km²".format(S, S_targ))
        # sign = 1.
        # while abs(S - S_targ) > S_targ * 1e-8:
        #     self.log.printL(
        #         "WL = {:.4G} kgf/m². Time to change that area!".format(self.boundary['m_initial'] / (S * 1e6)))
        #     # input("\nI am about to mess things up. Be careful. ")
        #     # dm = 10.
        #     # self.x[:,3,:] += dm
        #     if sign * (S_targ - S) < 0:
        #         dS = -dS / 2.
        #         sign *= -1.
        #     self.constants['s_ref'] += dS
        #     S = self.constants['s_ref'][0]
        #     self.log.printL("\nDone. Let's restore it again.")
        #     self.calcP()
        #     while self.P > self.tol['P']:
        #         self.rest(parallelOpt={'restLMPBVP': True})
        #     # self.compWith(solInit,altSolLabl='itsme',piIsTime=False)
        #
        # plt.plot(self.t * self.pi[0], self.x[:, 2, 0] / d2r)
        # plt.xlabel('t [s]')
        # plt.ylabel('gamma [deg]')
        # plt.grid()
        # plt.show()
        #
        # self.plotTraj(mustSaveFig=False)
# =============================================================================
#        # Basic desaturation:
#

#        # TODO: This hardcoded bypass MUST be corrected in later versions.

        if initMode == 'extSol':
            msg = "\n!!!\nHeavy hardcoded bypass here:\n" + \
                  "Control limits are being switched;\n" + \
                  "Controls themselves are being 'desaturated'.\n" + \
                  "Check code for the values, and be careful!\n"

            self.log.printL(msg)

            bat = 1.5
            for arc in range(self.s):
                for k in range(self.N):
                    if u[k,1,arc] < -bat:
                        u[k,1,arc] = -bat
                    if u[k,1,arc] > bat:
                        u[k,1,arc] = bat

        # This was a test for a "super honest" desaturation, so that the
        # program would not know when to do the coasting a priori.
        # It turns out this does not actually happen because the "coasting
        # information" is already "encoded" in the states (h,v,gama,m); so
        # probably the restoration simply puts the coasting back. Hence,
        # there is this...
        # TODO: Try to put this before the RK4 re-integration. If this works,
        #  it would be a great candidate to a less-naive method.
# =============================================================================
        self.u = u

        self.compWith(solInit,'Initial guess')
        self.plotSol(piIsTime=False)
        self.plotSol()#opt={'mode':'orbt'})
        self.plotF()

        self.log.printL("\nInitialization complete.\n")
        return solInit
#%%
    def calcDimCtrl(self,ext_u = None):
        """Calculate variables alpha (angle of attack) and beta (thrust), from
        either the object's own control (self.u) or external control
        (additional parameter needed)."""

        restrictions = self.restrictions
        alpha_min = restrictions['alpha_min']
        alpha_max = restrictions['alpha_max']
        beta_min = restrictions['beta_min']
        beta_max = restrictions['beta_max']
        alfa = numpy.empty((self.N,self.s))
        beta = numpy.empty((self.N,self.s))

        if ext_u is None:
            for arc in range(self.s):
                alfa[:,arc] = .5*((alpha_max[arc] + alpha_min[arc]) +
                                  (alpha_max[arc] - alpha_min[arc]) *
                                  numpy.tanh(self.u[:,0,arc]))
                beta[:,arc] = .5*((beta_max[arc] + beta_min[arc]) +
                           (beta_max[arc] - beta_min[arc]) *
                           numpy.tanh(self.u[:,1,arc]))
        else:
            for arc in range(self.s):
                alfa[:,arc] = .5*((alpha_max[arc] + alpha_min[arc]) +
                                  (alpha_max[arc] - alpha_min[arc]) *
                                  numpy.tanh(ext_u[:,0,arc]))
                beta[:,arc] = .5*((beta_max[arc] + beta_min[arc]) +
                                  (beta_max[arc] - beta_min[arc]) *
                                  numpy.tanh(ext_u[:,1,arc]))

        return alfa, beta

    def calcAdimCtrl(self,alfa,beta):
        """Calculate adimensional control 'u' based on external arrays for
        alpha (ang. of attack) and beta (thrust). """

        Nu = len(alfa)
        s = self.s
        u = numpy.empty((Nu,2,s))

        alpha_min = self.restrictions['alpha_min']
        alpha_max = self.restrictions['alpha_max']
        beta_min = self.restrictions['beta_min']
        beta_max = self.restrictions['beta_max']

        a1 = .5*(alpha_max + alpha_min)
        a2 = .5*(alpha_max - alpha_min)
        b1 = .5*(beta_max + beta_min)
        b2 = .5*(beta_max - beta_min)

        for arc in range(self.s):
            alfa[:,arc] -= a1[arc]
            alfa[:,arc] *= 1.0/a2[arc]

            beta[:,arc] -= b1[arc]
            beta[:,arc] *= 1.0/b2[arc]

        u[:,0,:] = alfa.copy()
        u[:,1,:] = beta.copy()

        sat = 0.99999
        # Basic saturation
        for arc in range(s):
            for j in range(2):
                for k in range(Nu):
                    if u[k,j,arc] > sat:
                        u[k,j,arc] = sat
                    elif u[k,j,arc] < -sat:
                        u[k,j,arc] = -sat

        u = numpy.arctanh(u)
        return u

    def calcPhi(self):
        N,n,s = self.N,self.n,self.s

        constants = self.constants
        grav_e = constants['grav_e']
        Thrust = constants['Thrust']
        Isp = constants['Isp']
        r_e = constants['r_e']
        GM = constants['GM']
        CL0 = constants['CL0']
        CL1 = constants['CL1']
        CD0 = constants['CD0']
        CD2 = constants['CD2']
        s_ref = constants['s_ref']
        DampCent = constants['DampCent']
        DampSlop = constants['DampSlop']

        sin, cos = numpy.sin, numpy.cos

        alpha,beta = self.calcDimCtrl()
        x, pi = self.x, self.pi

        # calculate variables CL and CD
        CL = numpy.empty_like(alpha)
        CD = numpy.empty_like(alpha)
        for arc in range(s):
            CL[:,arc] = CL0[arc] + CL1[arc]*alpha[:,arc]
            CD[:,arc] = CD0[arc] + CD2[arc]*(alpha[:,arc]**2)

        # calculate L and D
        # TODO: making atmosphere.rho vectorized (array compatible) would
        # increase performance significantly!

        #dens = numpy.empty((N, s))
        #for arc in range(s):
        #    dens[:,arc] = rhoSGRA(x[:,0,arc])
        dens = rhoSGRA(x[:,0,:].reshape(N*s)).reshape((N,s))

        pDynTimesSref = numpy.empty_like(CL)
        for arc in range(s):
            pDynTimesSref[:,arc] = .5 * dens[:,arc] * \
                                   (x[:,1,arc]**2) * s_ref[arc]
        L = CL * pDynTimesSref
        D = CD * pDynTimesSref

        # calculate r
        r = r_e + x[:,0,:]

        # calculate grav
        grav = GM/r/r

        # calculate phi:
        phi = numpy.empty((N,n,s))

        sinGama = sin(x[:,2,:]); cosGama = cos(x[:,2,:])
        sinAlfa = sin(alpha);    cosAlfa = cos(alpha)
        accDimTime = 0.0
        for arc in range(s):
            td = accDimTime + pi[arc] * self.t # Dimensional time

            phi[:,0,arc] = x[:,1,arc] * sinGama[:,arc]
            phi[:,1,arc] = (beta[:,arc] * Thrust[arc] * cosAlfa[:,arc] +
                            - D[:,arc])/x[:,3,arc] \
                           - grav[:,arc] * sinGama[:,arc]
            phi[:,2,arc] = ((beta[:,arc] * Thrust[arc] * sinAlfa[:,arc] +
                            + L[:,arc])/(x[:,3,arc] * x[:,1,arc]) +
                            cosGama[:,arc] * ( x[:,1,arc]/r[:,arc] +
                            -  grav[:,arc]/x[:,1,arc] )) * \
                            .5*(1.0+numpy.tanh(DampSlop*(td-DampCent)))
            phi[:,3,arc] = - (beta[:,arc] * Thrust[arc])/(grav_e * Isp[arc])
            phi[:,:,arc] *= pi[arc]
            accDimTime += pi[arc]
        return phi

    def calcAcc(self):
        """ Calculate tangential acceleration."""
        acc = numpy.empty((self.N,self.s))
        phi = self.calcPhi()

        for arc in range(self.s):
            acc[:,arc] = phi[:,1,arc] / self.pi[arc]

        return acc

#%%

    def calcGrads(self,calcCostTerm=True):
        # Pre-assign functions
        sin = numpy.sin; cos = numpy.cos; tanh = numpy.tanh

        # Load constants
        N, n, m, p, q, s = self.N, self.n, self.m, self.p, self.q, self.s
        addArcs = self.addArcs
        constants = self.constants

        grav_e = constants['grav_e']; MaxThrs = constants['Thrust']
        Isp = constants['Isp']; g0Isp = Isp * grav_e
        r_e = constants['r_e']; GM = constants['GM']
        CL0 = constants['CL0']; CL1 = constants['CL1']
        CD0 = constants['CD0']; CD2 = constants['CD2']
        s_ref = constants['s_ref']
        DampCent = constants['DampCent']; DampSlop = constants['DampSlop']
        Kpf = constants['Kpf']; PFmode = constants['PFmode']

        restrictions = self.restrictions
        alpha_min = restrictions['alpha_min']
        alpha_max = restrictions['alpha_max']
        beta_min = restrictions['beta_min']
        beta_max = restrictions['beta_max']
        acc_max = restrictions['acc_max']

        # Load states, controls
        u1 = self.u[:,0,:]; u2 = self.u[:,1,:]
        tanhU1 = tanh(u1); tanhU2 = tanh(u2)

        pi = self.pi
        acc = self.calcAcc()

        phix = numpy.zeros((N,n,n,s)); phiu = numpy.zeros((N,n,m,s))

        if p>0:
            phip = numpy.zeros((N,n,p,s))
        else:
            phip = numpy.zeros((N,n,1,s))

        fx = numpy.zeros((N,n,s)); fu = numpy.zeros((N,m,s))
        fp = numpy.zeros((N,p,s))
        fOrig_u = numpy.zeros((N,m,s)); fPF_u = numpy.empty_like(fu)

        ## Psi derivatives

        # For reference:
        # y = [x[0,:,0],\
        #      x[N-1,:,0],\
        #      x[0,:,1],\
        #      x[N-1,:,0],\
        #       ...,\
        #      x[0,:,s-1],
        #      x[N-1,:,s-1]]

        # first arc - second arc
        #psi[4] = x[N-1,0,0] - 50.0e-3
        #psi[5] = x[0,0,1] - 50.0e-3
        #psi[6] = x[0,1,1] - x[N-1,1,0]
        #psi[7] = x[0,2,1] - x[N-1,2,0]
        #psi[8] = x[0,3,1] - x[N-1,3,0]

        # second arc - third arc
        #psi[9] = x[N-1,0,1] - 2.
        #psi[10] = x[0,0,2] - 2.
        #psi[11] = x[0,1,2] - x[N-1,1,1]
        #psi[12] = x[0,2,2] - x[N-1,2,1]
        #psi[13] = x[0,3,2] - x[N-1,3,1]

        # y = [h(t=0,s=0), 0
        #      v(t=0,s=0), 1
        #      g(t=0,s=0), 2
        #      m(t=0,s=0), 3
        #      h(t=1,s=0), 4
        #      v(t=1,s=0), 5
        #      g(t=1,s=0), 6
        #      m(t=1,s=0), 7
        #      h(t=0,s=1), 8
        #      v(t=0,s=1), 9
        #      g(t=0,s=1), 10
        #      m(t=0,s=1), 11
        #      h(t=1,s=1), 12
        #      v(t=1,s=1), 13
        #      g(t=1,s=1), 14
        #      m(t=1,s=1), 15
        #      h(t=0,s=2), 16
        #      v(t=0,s=2), 17
        #      g(t=0,s=2), 18
        #      m(t=0,s=2), 19
        #      h(t=1,s=2), 20
        #      v(t=1,s=2), 21
        #      g(t=1,s=2), 22
        #      m(t=1,s=2)] 23

        psiy = numpy.zeros((q,2*n*s))
        s_f = self.constants['s_f']

        # First n rows: all states have assigned values
        for ind in range(n):
            psiy[ind,ind] = 1.0

        # Intermediate conditions: extra arcs
        i0 = n; j0 = n
        for arc in range(addArcs):
            # This loop sets the interfacing conditions between all states
            # in 'arc' and 'arc+1' (that's why it only goes up to s-1)
            #self.log.printL("arc = "+str(arc))
            # For height:
            psiy[i0,j0] = 1.0 # height, this arc
            #self.log.printL("Writing on ["+str(i0)+","+str(j0)+"] : 1.0")
            psiy[i0+1,j0+n] = 1.0 # height, next arc
            #self.log.printL("Writing on ["+str(i0+1)+","+str(j0+n)+"] : 1.0")
            # For speed, angle and mass:
            for stt in range(1,n):
                psiy[i0+stt+1,j0+stt] = -1.0 #this state, this arc  (end cond)
                #self.log.printL("Writing on ["+str(i0+stt+1)+","+\
                #str(j0+stt)+"] : -1.0")
                psiy[i0+stt+1,j0+stt+n] = 1.0 #this state, next arc (init cond)
                #self.log.printL("Writing on ["+str(i0+stt+1)+","+\
                #str(j0+stt+n)+"] : 1.0")
            i0 += n + 1
            j0 += 2*n
        #
        # Intermediate conditions
        #self.log.printL("End of first for, i0 = "+str(i0)+", j0 = "+str(j0))
        for arc in range(addArcs,s-1):
            # This loop sets the interfacing conditions between all states
            # in 'arc' and 'arc+1' (that's why it only goes up to s-1)

            # For height, speed and angle:
            for stt in range(n-1):
                psiy[i0+stt,j0+stt] = -1.0  # this state, this arc  (end cond)
                #self.log.printL("Writing on ["+str(i0+stt)+","+\
                #str(j0+stt)+"] : -1.0")
                psiy[i0+stt,j0+stt+n] = 1.0 # this state, next arc (init cond)
                #self.log.printL("Writing on ["+str(i0+stt)+","+\
                #str(j0+stt+n)+"] : 1.0")
            # For mass:
            stt = n-1
            #initMode = self.initMode
            #if initMode == 'extSol':
            if self.isStagSep[arc]:
                # mass, next arc (init cond)
                psiy[i0+stt,j0+stt+n] = 1.0
                #self.log.printL("Writing on ["+str(i0+stt)+","+\
                #str(j0+stt+n)+"] : 1.0")
                # mass, this arc (end cond):
                psiy[i0+stt,j0+stt] = -1.0/(1.0-s_f[arc])
                #self.log.printL("Writing on ["+str(i0+stt)+","+\
                #str(j0+stt)+"] : -1/(1-e)...")
                # mass, this arc (init cond):
                psiy[i0+stt,j0+stt-n] = s_f[arc]/(1.0-s_f[arc])
                #self.log.printL("Writing on ["+str(i0+stt)+","+\
                #str(j0+stt-n)+"] : +e/(1-e)...")
            else:
                psiy[i0+stt,j0+stt] = -1.0  # mass, this arc  (end cond)
                #self.log.printL("Writing on ["+str(i0+stt)+","+\
                #str(j0+stt)+"] : -1.0")
                psiy[i0+stt,j0+stt+n] = 1.0 # mass, next arc (init cond)
                #self.log.printL("Writing on ["+str(i0+stt)+","+\
                #str(j0+stt+n)+"] : 1.0")

            i0 += n
            j0 += 2*n
        #
        # Last n-1 rows (no mass eq for end condition of final arc):
        for ind in range(n-1):
            psiy[q-1-ind,2*n*s-2-ind] = 1.0

        psip = numpy.zeros((q,p))

        # calculate r, V, etc
        r = r_e + self.x[:,0,:]; r2 = r * r; r3 = r2 * r
        V = self.x[:,1,:]; V2 = V * V
        # GAMMA FACTOR (fdg)
        fdg = numpy.empty_like(r)
        td0 = 0.0
        for arc in range(s):
            td = td0 + self.t * pi[arc] # dimensional time
            fdg[:,arc] = .5*(1.0+tanh(DampSlop*(td-DampCent)))
            td0 += pi[arc]

        m = self.x[:,3,:]
        m2 = m * m
        sinGama, cosGama = sin(self.x[:,2,:]), cos(self.x[:,2,:])

        # Calculate variables (arrays) alpha and beta

        # TODO: change calcDimCtrl so that the derivatives
        #  DAlfaDu1 and DBetaDu2 are also calculated there...
        #  (with an optional command so that these are not calculated all the
        #  time, of course!)
        alpha,beta = self.calcDimCtrl()
        sinAlpha, cosAlpha = sin(alpha), cos(alpha)

        # Derivatives
        DAlfaDu1 = .5 * (alpha_max - alpha_min) * (1.0 - tanhU1**2)
        DBetaDu2 = .5 * (beta_max - beta_min) * (1.0 - tanhU2**2)

        # calculate variables CL and CD, already multiplied by ref. area
        CLsref = numpy.empty_like(alpha)
        CDsref = numpy.empty_like(alpha)
        for arc in range(s):
            CLsref[:,arc] = (CL0[arc] + CL1[arc]*alpha[:,arc])*s_ref[arc]
            #CL[:,arc] = CL0[arc] + CL1[arc] * alpha[:,arc]
            CDsref[:,arc] = (CD0[arc] + CD2[arc]*(alpha[:,arc]**2))*s_ref[arc]
            #CD[:,arc] = CD0[arc] + CD2[arc] * (alpha[:,arc]**2)

        # atmosphere: numerical gradient
        dens = numpy.empty((N,s))
        del_rho = numpy.empty((N,s))
        for arc in range(s):
            dens[:, arc] = rhoSGRA(self.x[:, 0, arc])
            del_rho[:, arc] = (rhoSGRA(self.x[:, 0, arc] + .05) - dens[:,arc]) / .05


        # calculate gravity (at each time/arc!)
        g = GM/r2

        ## "common" expressions

        # TODO: also compute d thrust d u2
        # Actual (dimensional) thrust:
        thrust = numpy.empty_like(beta)
        for arc in range(s):
            thrust[:,arc] = beta[:,arc] * MaxThrs[arc]

        ## Normal and axial forces
        # forces on velocity direction
        fVel = thrust * cosAlpha - .5 * dens * V2 * CDsref
        # forces normal to velocity (90 deg +)
        fNor = thrust * sinAlpha + .5 * dens * V2 * CLsref

        # acceleration solely due to thrust
        bTm = thrust/m

#==============================================================================

        if calcCostTerm:
            PenaltyIsTrue = (acc > acc_max)
            # Common term
            if PFmode == 'lin':
                K2dAPen = Kpf * PenaltyIsTrue
            elif PFmode == 'quad':
                K2dAPen = 2.0 * Kpf * (acc-acc_max) * PenaltyIsTrue
            elif PFmode == 'tanh':
                K2dAPen = Kpf * PenaltyIsTrue * \
                          (1.0 - tanh(acc/acc_max-1.0)**2) * (1.0/acc_max)

            ## fx derivatives
            # d f d h (incomplete):
            # noinspection PyUnboundLocalVariable
            fx[:,0,:] = K2dAPen * (2.0 * GM * sinGama / r3 -
              (0.5 * del_rho * V2 * CDsref)/m)
            # d f d V (incomplete):
            fx[:,1,:] = - K2dAPen * CDsref * dens * V / m
            # d f d gamma (incomplete):
            fx[:,2,:] = - K2dAPen * g * cosGama
            # d f d m (incomplete):
            fx[:,3,:] = - K2dAPen * fVel / m2
            # fx still lacks pi!

            ## fu derivatives
            # d f d u1 (incomplete):
            fPF_u[:,0,:] = K2dAPen * (-bTm * sinAlpha * DAlfaDu1)
            # d f d u2 (incomplete):
            fOrig_u[:,1,:] = DBetaDu2
            fPF_u[:,1,:] = K2dAPen * cosAlpha * DBetaDu2 / m
            # fPF_u still lacks pi terms!

            for arc in range(s):
                ## fp derivatives
                if PFmode == 'lin':
                    fp[:,arc,arc] = thrust[:,arc]/g0Isp[arc]/(1.0-s_f[arc]) + \
                                    Kpf * PenaltyIsTrue[:,arc] * \
                                    (acc[:,arc]-acc_max)
                elif PFmode == 'quad':
                    fp[:,arc,arc] = thrust[:,arc]/g0Isp[arc]/(1.0-s_f[arc]) + \
                                    Kpf * PenaltyIsTrue[:,arc] * \
                                    (acc[:,arc]-acc_max)**2
                elif PFmode == 'tanh':
                    fp[:,arc,arc] = thrust[:,arc]/g0Isp[arc]/(1.0-s_f[arc]) + \
                                    Kpf * PenaltyIsTrue[:,arc] * \
                                    tanh(acc[:,arc]/acc_max-1.0)

                # fx is ready!
                fx[:,:,arc] *= pi[arc]

                # fOrig_u is ready!
                fOrig_u[:,:,arc] *= pi[arc] * MaxThrs[arc] / g0Isp[arc] /    \
                       (1.0-s_f[arc])
                # fPF_u is ready!
                fPF_u[:,:,arc] *= pi[arc] * MaxThrs[arc]
            #
            fu = fOrig_u + fPF_u
        #

        ## phip derivatives
        for arc in range(s):
        # d hdot d pi
            phip[:,0,arc,arc] = V[:,arc] * sinGama[:,arc]
        # d vdot d pi
            phip[:,1,arc,arc] = fVel[:,arc]/m[:,arc] - g[:,arc]*sinGama[:,arc]
        # d gamadot d pi
            phip[:,2,arc,arc] = fNor[:,arc]/(m[:,arc] * V[:,arc]) + \
            cosGama[:,arc] * (V[:,arc]/r[:,arc] - g[:,arc]/V[:,arc])
        # d mdot d pi
            phip[:,3,arc,arc] = -thrust[:,arc] / g0Isp[arc]

        ## phix derivatives

        # d hdot d h: 0.0
        # d hdot d V:
        phix[:,0,1,:] = sinGama
        # d hdot d gama:
        phix[:,0,2,:] = V * cosGama
        # d hdot d m: 0.0

        # d Vdot d h:
        phix[:,1,0,:] = 2.0 * GM * sinGama/r3 - (0.5 * del_rho * V2 * CDsref)/m
        # d Vdot d v:
        phix[:,1,1,:] = - CDsref * dens * V/m
        # d Vdot d gama:
        phix[:,1,2,:] = - g * cosGama
        # d Vdot d m:
        phix[:,1,3,:] = - fVel / m2

        # d gamadot d h:
        phix[:,2,0,:] = cosGama * (-V/r2 + 2.0*GM/(V*r3)) + \
            (0.5 * CLsref * del_rho * V)/m
        # d gamadot d v:
        phix[:,2,1,:] = - bTm * sinAlpha /V2 + \
        cosGama * ( 1.0/r + g/V2 ) +  0.5 * CLsref * dens/m
        # d gamadot d gama:
        phix[:,2,2,:] = - sinGama * ( V/r - g/V )
        # d gamadot d m:
        phix[:,2,3,:] = -fNor / ( m2 * V )

        # d mdot d h: 0.0
        # d mdot d v: 0.0
        # d mdot d gama: 0.0
        # d mdot d m: 0.0

        ## phiu derivatives
        for arc in range(s):
        # d hdot d u1: 0.0
        # d hdot d u2: 0.0

        # d vdot d u1:
            phiu[:,1,0,arc] = -(bTm[:,arc] * sinAlpha[:,arc] +
            dens[:,arc] * V2[:,arc] * s_ref[arc] * CD2[arc] * alpha[:,arc]) * \
            DAlfaDu1[:,arc] / m[:,arc]
        # d vdot d u2:
            phiu[:,1,1,arc] = MaxThrs[arc] * cosAlpha[:,arc] * \
            DBetaDu2[:,arc] / m[:,arc]

        # d gamadot d u1:
            phiu[:,2,0,arc] = ( thrust[:,arc] * cosAlpha[:,arc] / V[:,arc] +
            0.5 * dens[:,arc] * V[:,arc] * s_ref[arc] * CL1[arc] ) * \
            DAlfaDu1[:,arc] / m[:,arc]
        # d gamadot d u2:
            phiu[:,2,1,arc] = MaxThrs[arc] * DBetaDu2[:,arc] * \
            sinAlpha[:,arc] / (m[:,arc] * V[:,arc])

        # d mdot d u1: 0.0
        # d mdot d u2:
            phiu[:,3,1,arc] = - MaxThrs[arc] * DBetaDu2[:,arc] / g0Isp[arc]

        ## include fdg
        for i in range(4):
            phix[:,2,i,:] *= fdg
        for i in range(2):
            phiu[:,2,i,:] *= fdg
        for i in range(p):
            phip[:,2,i,:] *= fdg

        ## multiplication by "pi"
        for arc in range(s):
            phix[:,:,:,arc] *= pi[arc]
            phiu[:,:,:,arc] *= pi[arc]
#==============================================================================

        Grads = {'phix': phix, 'phiu': phiu, 'phip': phip,
                 'fx': fx, 'fu': fu,  'fp': fp,
                 'psiy': psiy, 'psip': psip}
    #    Grads['gx'] = gx; Grads['gp'] = gp

        return Grads

#%%
    def calcPsi(self):
        #self.log.printL("In calcPsi.")
        boundary = self.boundary
        s_f = self.constants['s_f']
        x = self.x
        N, q, s, addArcs = self.N, self.q, self.s, self.addArcs
        TargHeig = self.boundary['TargHeig']
        psi = numpy.empty(q)

        # Beginning of first subarc
        #strPrnt = "0,1,2,3,"
        psi[0] = x[0,0,0] - boundary['h_initial']
        psi[1] = x[0,1,0] - boundary['V_initial']
        psi[2] = x[0,2,0] - boundary['gamma_initial']
        psi[3] = x[0,3,0] - boundary['m_initial']

        # inter-arc conditions for the extra arcs (if any)
        for arc in range(addArcs):
            i = 4 + 5 * arc
            # states in order: height (2x), speed, flight angle and mass
            psi[i]   = x[N-1,0,arc] - TargHeig[arc]
            psi[i+1] = x[0,0,arc+1] - TargHeig[arc]
            psi[i+2] = x[0,1,arc+1] - x[N-1,1,arc]
            psi[i+3] = x[0,2,arc+1] - x[N-1,2,arc]
            psi[i+4] = x[0,3,arc+1] - x[N-1,3,arc]
            #strPrnt += str(i) + "," + str(i+1) + "," + str(i+2) + \
            #            "," + str(i+3) + "," + str(i+4) + ","
        #
        # inter-arc conditions for between arc and arc+1 for "natural" arcs
        # (that's why the loop only goes up to s-1)
        # noinspection PyUnboundLocalVariable
        i0 = i + 5
        for arc in range(addArcs,s-1):
            #self.log.printL("arc = "+str(arc))
            i = i0 + 4 * (arc-addArcs)
            # four states in order: height, speed, flight angle and mass
            psi[i]   = x[0,0,arc+1] - x[N-1,0,arc]
            psi[i+1] = x[0,1,arc+1] - x[N-1,1,arc]
            psi[i+2] = x[0,2,arc+1] - x[N-1,2,arc]
            # initMode = self.initMode
            # if initMode == 'extSol':
            #     # This if...else is basically a safety net for the case of an
            #     # undefined self.isStagSep.

            if self.isStagSep[arc]:
                psi[i+3] = x[0,3,arc+1] - (1.0/(1.0 - s_f[arc])) * \
                            (x[N-1,3,arc] - s_f[arc] * x[0,3,arc])
            else:
                psi[i+3] = x[0,3,arc+1] - x[N-1,3,arc]

            # else:
            #     self.log.printL("Sorry, this part of calcPsi "+\
            #                     "is not implemented yet.")
            #     raise Exception("Broken compatibility with init methods " +
            #                     "other than 'extSol'.")
            #strPrnt += str(i)+","+str(i+1)+","+str(i+2)+","+str(i+3)+","
        #

        # End of final subarc
        psi[q-3] = x[N-1,0,s-1] - boundary['h_final']
        psi[q-2] = x[N-1,1,s-1] - boundary['V_final']
        psi[q-1] = x[N-1,2,s-1] - boundary['gamma_final']
        #strPrnt += str(q-3)+","+str(q-2)+","+str(q-1)
        #self.log.printL("Psi eval check:\nq = "+str(q))
        #self.log.printL(strPrnt)
        #self.log.printL("Psi = "+str(psi))

        return psi

    def calcF(self):
        constants = self.constants
        restrictions = self.restrictions

        grav_e = constants['grav_e']
        Thrust = constants['Thrust']
        Isp = constants['Isp']
        s_f = constants['s_f']
        Kpf = constants['Kpf']
        PFmode = constants['PFmode']
        acc_max = restrictions['acc_max']
        acc = self.calcAcc()
        # calculate variable beta
        _,beta = self.calcDimCtrl()

        f = numpy.empty((self.N,self.s))
        fOrig = numpy.empty_like(f)
        fPF = numpy.empty_like(f)

        for arc in range(self.s):
            fOrig[:,arc] = self.pi[arc] * beta[:,arc] * Thrust[arc] /  \
                            (grav_e * (1.0-s_f[arc]) * Isp[arc])
            if PFmode == 'lin':
                fPF[:,arc] = self.pi[arc] * Kpf * (acc[:,arc]-acc_max)
            elif PFmode == 'quad':
                fPF[:,arc] = self.pi[arc] * Kpf * (acc[:,arc]-acc_max)**2
            elif PFmode == 'tanh':
                fPF[:,arc] = self.pi[arc] * Kpf * \
                             numpy.tanh(acc[:,arc]/acc_max-1.0)
            #
        #

        # Apply penalty only when acc > acc_max
        fPF *= (acc > acc_max)
        f = fOrig + fPF

        return f,fOrig,fPF

    def calcI(self):
        N, s = self.N, self.s
        _, fOrig, fPF = self.calcF()

#        # METHOD 1: simple simpson integration.
#        Iorig, Ipf = 0.0, 0.0
#        for arc in range(s):
#            Iorig += simp(fOrig[:,arc],N)
#            Ipf += simp(fPF[:,arc],N)

        # METHOD 2: simpson integration, with prints
#        IorigVec, IpfVec = numpy.empty(s), numpy.empty(s)
#        for arc in range(s):
#            IorigVec[arc] = simp(fOrig[:,arc],N)
#            IpfVec[arc] = simp(fPF[:,arc],N)
#
#        Iorig = IorigVec.sum()
#        Ipf = IpfVec.sum()
#
#        self.log.printL("\nIorigVec: "+str(IorigVec))
#        self.log.printL("Iorig: "+str(Iorig))
#        self.log.printL("IpfVec: "+str(IpfVec))
#        self.log.printL("Ipf: "+str(Ipf))

#        # METHOD 3: trapezoidal integration, comparing with simpson
#        IvecOrig = numpy.empty(s)
#        IvecPF = numpy.empty(s)
#
#        for arc in range(s):
#            IvecOrig[arc] = .5 * ( fOrig[0,arc] + fOrig[N-1,arc] )
#            IvecOrig[arc] += fOrig[1:(N-1),arc].sum()
#            IvecPF[arc] = .5 * ( fPF[0,arc] + fPF[N-1,arc] )
#            IvecPF[arc] += fPF[1:(N-1),arc].sum()
#
#        IvecOrig *= 1.0/(N-1)
#        IvecPF *= 1.0/(N-1)
#        Iorig = IvecOrig.sum()
#        Ipf = IvecPF.sum()
#
#        IorigSimp, IpfSimp = 0.0, 0.0
#        for arc in range(s):
#            IorigSimp += simp(fOrig[:,arc],N)
#            IpfSimp += simp(fPF[:,arc],N)
#
#        self.log.printL("\nIorig: {:.4E}".format(Iorig))
#        self.log.printL("IorigSimp: {:.4E}".format(IorigSimp))
#        self.log.printL("Difference in Iorig: {:.4E}".format(Iorig-IorigSimp))
#        self.log.printL("\nIpf: {:.4E}".format(Ipf))
#        self.log.printL("IpfSimp: {:.4E}".format(IpfSimp))
#        self.log.printL("Difference in Ipf: {:.4E}".format(Ipf-IpfSimp))
#        input("\n>> ")

        # METHOD 4: simpson on Ipf, cheating on Iorig
        IorigVec, IpfVec = numpy.empty(s), numpy.empty(s)
        s_f = self.constants['s_f']
        for arc in range(s):
            IorigVec[arc] = (self.x[0,3,arc]-self.x[-1,3,arc])/(1.-s_f[arc])
            IpfVec[arc] = simp(fPF[:,arc],N)
        self.log.printL("I components, by arcs: "+str(IorigVec))
        Iorig = IorigVec.sum()
        Ipf = IpfVec.sum()

        #self.log.printL("\nIorigVec: "+str(IorigVec))
        #self.log.printL("Iorig: "+str(Iorig))
        #self.log.printL("IpfVec: "+str(IpfVec))
        #self.log.printL("Ipf: "+str(Ipf))


        return Iorig+Ipf, Iorig, Ipf

#%% Plotting commands and related functions

    def printPars(self):
        dPars = self.__dict__
#        keyList = dPars.keys()
        self.log.printL("These are the attributes for the current solution:\n")
        self.log.pprint(dPars)

        msg = "\n" + '-'*88
        msg += "\nMiele-comparing parameters:\n"
        msg += '-'*88
        w0 = self.boundary['m_initial'] * self.constants['grav_e'] #kN
        twr = self.constants['Thrust'][0] / w0
        msg += "\nThrust to weight ratio (at take-off)[-]: {:.2F}".format(twr)
        wld = w0 * 1e3 / (self.constants['s_ref'][0] * 1e6) / 9.80665 #kgf/m²
        msg += "\nWing loading (at take-off)[kgf/m²]: {:.1F}".format(wld)
        msg += "\nEngine specific impulse [s]: "+str(self.constants['Isp'])
        msg += "\nStructural factors [-]: "+str(self.constants['s_f'])
        r2d = 180./numpy.pi
        msg += "\nAngle of attack bounds: alpha_min = ["
        for arc in range(self.s):
            msg += " {:.1F}, ".format(r2d*self.restrictions['alpha_min'][arc])
        msg += "] deg,\n" + (' '*24) + "alpha_max = ["
        for arc in range(self.s):
            msg += " {:.1F},".format(r2d*self.restrictions['alpha_max'][arc])
        msg += "] deg,\nTangential acc. bound [m/s²]: "
        msg += "{:.1F}\n".format(self.restrictions['acc_max']*1000.)
        msg += '-'*88
        self.log.printL(msg)

    def calcMassDist(self):
        """Calculate the mass distributions for a given configuration.

        Calculates, for each stage (not arc) and for the complete rocket, the
        initial mass, propellant mass, structural mass and payload mass."""
        NStag = self.s - self.addArcs
        MInit = numpy.zeros(NStag+1)
        MProp = numpy.zeros(NStag+1)
        MStru = numpy.zeros(NStag+1)
        MPayl = numpy.zeros(NStag+1)

        stg = 0; MRem = self.x[0,3,0]; MInit[0] = MRem
        for arc in range(self.s):
            # Propellant mass for this arc:
            MProp[stg] += self.x[0,3,arc] - self.x[-1,3,arc]

            if self.isStagSep[arc]:
                sf = self.constants['s_f'][arc]
                # In case of stage separation, end of the spent propellant
                # accumulation.
                # The corresponding structural mass for the propellant mass is:
                MStru[stg] = MProp[stg] * sf/(1.-sf)
                # The payload for the stage is the rest of the "rocket" above:
                MPayl[stg] = MRem - MProp[stg] - MStru[stg]
                MRem = MPayl[stg]

                stg += 1
                # Initial mass for the next stage.
                # Calculate only if there is a next stage, of course...
                if arc < self.s-1:
                    MInit[stg] = self.x[0,3,arc+1]
            #
        #
        # Last entry corresponds to the "sum" of the rocket
        MInit[NStag] = MInit[0]
        MProp[NStag] = MProp[0:NStag].sum()
        MStru[NStag] = MStru[0:NStag].sum()
        MPayl[NStag] = MPayl[NStag-1]

        massDist = {'i':MInit,'p':MProp,'s':MStru,'u':MPayl}
        return massDist

    def calcIdDv(self):
        """Calculate ideal Delta v provided by all the stages of the rocket as
        it is, using Tsiolkovsky equation."""

        DvAcc = 0.0
        g0Isp = self.constants['Isp']*self.constants['grav_e']
        for ind in range(self.s):
            arc = self.s - ind - 1
            M0 = self.x[0,3,arc]
            Mf = self.x[-1,3,arc]
            DvAcc += g0Isp[arc]*numpy.log(M0/Mf)
        return DvAcc

    def plotSol(self,opt={},intv=None,piIsTime=True,mustSaveFig=True,
                subPlotAdjs={'left':0.0,'right':1.0,'bottom':0.0,
                             'top':10.0,'wspace':0.2,'hspace':0.35}):
        self.log.printL("\nIn plotSol, opt = "+str(opt))
        x = self.x
        u = self.u
        pi = self.pi
        r2d = 180.0/numpy.pi

        if piIsTime:
            timeLabl = 't [s]'
            tVec = [0.0, self.pi.sum()]
        else:
            timeLabl = 'adim. t [-]'
            tVec = [0.0, self.s]

        if opt.get('mode','sol') == 'sol':
            I,Iorig,Ipf = self.calcI()

            self.log.printL("Initial mass: " + str(x[0,3,0]))
            self.log.printL("I: "+ str(I))
            self.log.printL("Design payload mass: " + str(self.mPayl))
            #mFinl = self.calcPsblPayl()
            massDist = self.calcMassDist()
            mFinl = massDist['u'][-1]
            self.log.printL('"Possible" payload mass: ' + str(mFinl))
            paylPercMassGain = 100.0*(mFinl-self.mPayl)/self.mPayl
            DvId = self.calcIdDv()
            self.log.printL("Ideal Delta v (Tsiolkovsky) with used " + \
                            "propellants: "+str(DvId))
            missDv = self.boundary['mission_dv']
            self.log.printL("Mission Delta v (orbital height + speed): " + \
                            str(missDv))
            dvLossPerc = 100.0*(DvId-missDv)/DvId
            self.log.printL("Losses (%): " + str(dvLossPerc))

            titlStr = "Current solution "
            titlStr += "(grad iter #" + str(self.NIterGrad) + "):\n"
            titlStr += "I = {:.4E}".format(I) + \
                       ", P = {:.4E} ".format(self.P) + \
                       ", Q = {:.4E}\n".format(self.Q)
            titlStr += "Payload mass: {:.1F} kg".format(mFinl) + \
                       ", gain = {:.4G}%\n".format(paylPercMassGain)
            titlStr += "Losses (w.r.t. ideal Delta v): "+ \
                       "{:.4G}%\n".format(dvLossPerc)
            titlStr += "       Init. mass   Prop. mass   " + \
                       "Stru. mass   Payl. mass"
            M0 = self.x[0,3,0]
            for i in range(self.s-self.addArcs):
                titlStr += "\nStg. " + str(i+1) + "   "\
                            "{:.4F}       ".format(massDist['i'][i]/M0) + \
                            "{:.4F}       ".format(massDist['p'][i]/M0) + \
                            "{:.4F}       ".format(massDist['s'][i]/M0) + \
                            "{:.4F}".format(massDist['u'][i]/M0)
            titlStr += "\nTotal    1.0000       "+ \
                       "{:.4F}       ".format(massDist['p'][-1]/M0) + \
                       "{:.4F}       ".format(massDist['s'][-1]/M0) + \
                       "{:.4F}".format(massDist['u'][-1]/M0)
            #self.log.printL("\ndebug:\n"+titlStr)

            plt.subplots_adjust(**subPlotAdjs)

            plt.subplot2grid((11,1),(0,0))
            self.plotCat(x[:,0,:],piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("h [km]")
            plt.title(titlStr)
            plt.xlabel(timeLabl)

            plt.subplot2grid((11,1),(1,0))
            self.plotCat(x[:,1,:],color='g',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("V [km/s]")
            plt.xlabel(timeLabl)

            plt.subplot2grid((11,1),(2,0))
            self.plotCat(x[:,2,:]*r2d,color='r',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("gamma [deg]")
            plt.xlabel(timeLabl)

            plt.subplot2grid((11,1),(3,0))
            self.plotCat(x[:,3,:],color='m',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("m [kg]")
            plt.xlabel(timeLabl)

            plt.subplot2grid((11,1),(4,0))
            self.plotCat(u[:,0,:],color='c',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("u1 [-]")
            plt.xlabel(timeLabl)

            plt.subplot2grid((11,1),(5,0))
            self.plotCat(u[:,1,:],color='c',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("u2 [-]")
            plt.xlabel(timeLabl)

            ######################################
            alpha,beta = self.calcDimCtrl()
            plt.subplot2grid((11,1),(6,0))
            self.plotCat(alpha/d2r,piIsTime=piIsTime,intv=intv,color='k')
            plt.grid(True)
            plt.ylabel("alpha [deg]")
            plt.xlabel(timeLabl)

            plt.subplot2grid((11,1),(7,0))
            self.plotCat(beta,piIsTime=piIsTime,intv=intv,color='k')
            plt.grid(True)
            plt.ylabel("beta [-]")
            plt.xlabel(timeLabl)
            ######################################

            ######################################
            plt.subplot2grid((11,1),(8,0))
            s = self.s; thrust = numpy.empty_like(beta)
            L = numpy.empty_like(beta); D = numpy.empty_like(beta)
            dens = numpy.empty_like(beta); pDyn = numpy.empty_like(beta)
            MaxThrs = self.constants['Thrust']
            CL0,CL1 = self.constants['CL0'],self.constants['CL1']
            CD0,CD2 = self.constants['CD0'],self.constants['CD2']
            s_ref = self.constants['s_ref']
            for arc in range(s):
                thrust[:,arc] = beta[:,arc] * MaxThrs[arc]
                dens[:,arc] = rhoSGRA(self.x[:,0,arc])
                pDyn[:,arc] = .5 * dens[:,arc] * (self.x[:,1,arc])**2
                L[:,arc] = pDyn[:, arc] * s_ref[arc] * \
                           (CL0[arc] + alpha[:,arc] * CL1[arc])
                D[:,arc] = pDyn[:, arc] * s_ref[arc] * \
                           (CD0[arc] + ( alpha[:,arc]**2 ) * CD2[arc])
            self.plotCat(thrust,color='k',intv=intv,piIsTime=piIsTime,
                         labl='Thrust')
            self.plotCat(L,color='b',intv=intv,piIsTime=piIsTime,
                         labl='Lift')
            self.plotCat(D,color='r',intv=intv,piIsTime=piIsTime,
                         labl='Drag')
            plt.grid(True)
            plt.xlabel(timeLabl)
            plt.ylabel("Forces [kN]")
            plt.legend()

            ######################################
            plt.subplot2grid((11,1),(9,0))
            acc =  self.calcAcc()
            self.plotCat(acc*1e3,color='y',intv=intv,piIsTime=piIsTime,
                         labl='Accel.')
            plt.plot(tVec,1e3 * self.restrictions['acc_max'] * \
                          numpy.array([1.0,1.0]),'--')
            plt.grid(True)
            plt.xlabel(timeLabl)
            plt.ylabel("Tang. accel. [m/s²]")

            ######################################
            ax = plt.subplot2grid((11,1),(10,0))
            position = numpy.arange(s)
            stages = numpy.arange(1,s+1)
            width = 0.4
            ax.bar(position,pi,width,color='b')
            # Put the values of the arc lenghts on the bars...
            for arc in range(s):
                coord = (float(arc)-.25*width,pi[arc]+10.)
                ax.annotate("{:.1F}".format(pi[arc]),xy=coord,xytext=coord)
            ax.set_xticks(position)
            ax.set_xticklabels(stages)
            plt.grid(True,axis='y')
            plt.xlabel("Arcs")
            plt.ylabel("Duration [s]")

            if mustSaveFig:
                if piIsTime:
                    self.savefig(keyName='currSol',fullName='solution')
                else:
                    self.savefig(keyName='currSol-adimTime',
                                 fullName='solution (non-dim. time)')
            else:
                plt.show()
                plt.clf()

            self.log.printL("Final (injected into orbit) rocket mass: " + \
                  "{:.4E}\n".format(x[-1,3,self.s-1]))
            # get ejected masses:

            #EjctMass = list()
            #initMode = self.initMode
            #if initMode == 'extSol':

            #for arc in range(self.s-1):
            #    if self.isStagSep[arc]:
            #        EjctMass.append(x[-1,3,arc]-x[0,3,arc+1])
            EjctMass = [x[-1,3,arc]-x[0,3,arc+1] for arc in range(self.s-1)\
                        if self.isStagSep[arc]]
            self.log.printL("Ejected masses: " + str(EjctMass))

            #self.plotSol(opt={'mode': 'orbt'},piIsTime=piIsTime)

        elif opt['mode'] == 'lambda':
            titlStr = "Lambdas (grad iter #" + str(self.NIterGrad+1) + ")"

            plt.subplots_adjust(**subPlotAdjs)

            plt.subplot2grid((8,1),(0,0))
            self.plotCat(self.lam[:,0,:],piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("lam - h")
            plt.xlabel(timeLabl)
            plt.title(titlStr)

            plt.subplot2grid((8,1),(1,0))
            self.plotCat(self.lam[:,1,:],color='g',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("lam - V")
            plt.xlabel(timeLabl)

            plt.subplot2grid((8,1),(2,0))
            self.plotCat(self.lam[:,2,:],color='r',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("lam - gamma")
            plt.xlabel(timeLabl)

            plt.subplot2grid((8,1),(3,0))
            self.plotCat(self.lam[:,3,:],color='m',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("lam - m")
            plt.xlabel(timeLabl)

            plt.subplot2grid((8,1),(4,0))
            self.plotCat(u[:,0,:],color='c',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("u1 [-]")
            plt.xlabel(timeLabl)

            plt.subplot2grid((8,1),(5,0))
            self.plotCat(u[:,1,:],color='c',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            #plt.xlabel("t")
            plt.ylabel("u2 [-]")
            plt.xlabel(timeLabl)

            ######################################
            alpha,beta = self.calcDimCtrl()
            alpha *= r2d
            plt.subplot2grid((8,1),(6,0))
            self.plotCat(alpha,piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.xlabel(timeLabl)
            plt.ylabel("alpha [deg]")

            plt.subplot2grid((8,1),(7,0))
            self.plotCat(beta,piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.xlabel(timeLabl)
            plt.ylabel("beta [-]")
            ######################################

            if mustSaveFig:
#                nowStr = getNowStr()
                if piIsTime:
                    self.savefig(keyName='currLamb',fullName='lambdas')
#                    self.savefig(keyName='currLamb-'+nowStr,\
#                                 fullName='lambdas')
                else:
                    self.savefig(keyName='currLamb-adimTime',
                                 fullName='lambdas (adim. Time)')
#                    self.savefig(keyName='currLamb-adimTime-'+nowStr,\
#                                 fullName='lambdas (adim. Time)')
            else:
                plt.show()
                plt.clf()

            self.log.printL("mu = " + str(self.mu))

        elif opt['mode'] == 'var':
            dx = opt['x']
            du = opt['u']
            dp = opt['pi']

            titlStr = "Proposed variations (grad iter #" + \
                      str(self.NIterGrad+1) + ")\n"+"Delta pi: "
            for i in range(self.p):
                titlStr += "{:.4E}, ".format(dp[i])
                #titlStr += str(dp[i])+", "
            titlStr += "\nDelta pi (%): "
            for i in range(self.p):
                titlStr += "{:.1F}, ".format(100.0*dp[i]/self.pi[i])

            plt.subplots_adjust(**subPlotAdjs)

            plt.subplot2grid((8,1),(0,0))
            self.plotCat(dx[:,0,:],piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("h [km]")
            plt.title(titlStr)
            plt.xlabel(timeLabl)

            plt.subplot2grid((8,1),(1,0))
            self.plotCat(dx[:,1,:],color='g',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("V [km/s]")
            plt.xlabel(timeLabl)

            plt.subplot2grid((8,1),(2,0))
            self.plotCat(dx[:,2,:]*r2d,color='r',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("gamma [deg]")
            plt.xlabel(timeLabl)

            plt.subplot2grid((8,1),(3,0))
            self.plotCat(dx[:,3,:],color='m',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("m [kg]")
            plt.xlabel(timeLabl)

            plt.subplot2grid((8,1),(4,0))
            self.plotCat(du[:,0,:],color='c',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("u1 [-]")
            plt.xlabel(timeLabl)

            plt.subplot2grid((8,1),(5,0))
            self.plotCat(du[:,1,:],color='c',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("u2 [-]")
            plt.xlabel(timeLabl)

            ######################################
            new_u = self.u + du
            alpha,beta = self.calcDimCtrl()
            alpha *= r2d
            new_alpha,new_beta = self.calcDimCtrl(ext_u=new_u)
            new_alpha *= r2d
            plt.subplot2grid((8,1),(6,0))
            self.plotCat(new_alpha-alpha,color='k',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("alpha [deg]")
            plt.xlabel(timeLabl)

            plt.subplot2grid((8,1),(7,0))
            self.plotCat(new_beta-beta,color='k',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("beta [-]")
            plt.xlabel(timeLabl)
            ######################################

            if mustSaveFig:
                if piIsTime:
                    self.savefig(keyName='corr',fullName='corrections')
                else:
                    self.savefig(keyName='corr-adimTime',
                                 fullName='corrections (adim. time)')
            else:
                plt.show()
                plt.clf()
        elif opt['mode'] == 'orbt':
            titlStr = "Current solution (orbital parameters)\n"
            titlStr += "(grad iter #" + str(self.NIterGrad) + ")"

            h, V  = self.x[:,0,:], self.x[:,1,:]
            gama = self.x[:,2,:]
            r = h + self.constants['r_e']
            GM = self.constants['GM']

            # Specific mechanical energy
            en = .5 * V * V - GM/r
            # Specific angular momentum
            am = r * V * numpy.cos(gama)
            # Semi-major axis
            a = -.5 * GM / en
            # Eccentricity (module)
            e = numpy.sqrt(1. + 2. * en * (am/GM)**2)

            # range angle (with respect to zenith at launch)
            sigma = self.rockin(opt={'sigma':True})['sigma']
            # horizontal and vertical components of r and V vectors
            rx, rz = r * numpy.sin(sigma),      r * numpy.cos(sigma)
            vx, vz = V * numpy.cos(gama-sigma), V * numpy.sin(gama-sigma)
            # radial "component" for eccentricity vector
            er = V**2 / GM - 1./r
            # velocity "component" for eccentricity vector
            ev = -(r * V * numpy.sin(gama))/GM
            # horizontal and vertical components for eccentricity
            ex, ez = er * rx + ev * vx, er * rz + ev * vz
            # Argument of perigee (with respect to zenith at launch)
            omega = numpy.arctan2(ex,ez)
            # True anomaly
            f = sigma-omega

            # Reference values for orbital parameters
            aRef = self.constants['r_e'] + self.boundary['h_final']
            enRef = - .5 * self.constants['GM'] / aRef
            amRef = aRef * self.boundary['V_final'] * \
                    numpy.cos(self.boundary['gamma_final'])
            eRef = numpy.sqrt(1. + 2. * enRef * (amRef/GM)**2)

            plt.subplots_adjust(**subPlotAdjs)
            nPlot = 11; np = 0
            plt.subplot2grid((nPlot, 1), (np, 0))
            self.plotCat(en, piIsTime=piIsTime, intv=intv)
            plt.plot(tVec,enRef * numpy.array([1.0, 1.0]), '--')
            plt.grid(True)
            plt.ylabel("Spec. mec. energy [MJ/kg]")
            plt.title(titlStr)
            plt.xlabel(timeLabl)

            np += 1
            plt.subplot2grid((nPlot, 1), (np, 0))
            self.plotCat(am, color='g', piIsTime=piIsTime, intv=intv)
            plt.plot(tVec, amRef * numpy.array([1.0, 1.0]), '--')
            plt.grid(True)
            plt.ylabel("Spec. ang. momentum [km²/s]")
            plt.xlabel(timeLabl)

            np += 1
            plt.subplot2grid((nPlot, 1), (np, 0))
            self.plotCat(a, color='r', piIsTime=piIsTime, intv=intv)
            plt.plot(tVec, aRef * numpy.array([1.0, 1.0]), '--')
            plt.grid(True)
            plt.ylabel("Semi-major axis [km]")
            plt.xlabel(timeLabl)

            np += 1
            plt.subplot2grid((nPlot, 1), (np, 0))
            self.plotCat(e, color='m', piIsTime=piIsTime, intv=intv)
            plt.plot(tVec, eRef * numpy.array([1.0, 1.0]), '--')
            plt.grid(True)
            plt.ylabel("Eccentricity [-]")
            plt.xlabel(timeLabl)

            np += 1
            plt.subplot2grid((nPlot, 1), (np, 0))
            self.plotCat(ex, piIsTime=piIsTime, intv=intv, labl='ex')
            self.plotCat(ez, color='k', piIsTime=piIsTime, intv=intv, labl='ez')
            plt.grid(True)
            plt.ylabel("Eccentricity components [-]")
            plt.legend()
            plt.xlabel(timeLabl)

            np += 1
            plt.subplot2grid((nPlot, 1), (np, 0))
            self.plotCat(sigma * r2d, piIsTime=piIsTime, intv=intv)
            plt.grid(True)
            plt.ylabel("Range angle [deg]")
            plt.xlabel(timeLabl)

            np += 1
            plt.subplot2grid((nPlot, 1), (np, 0))
            self.plotCat(omega * r2d, piIsTime=piIsTime, intv=intv)
            plt.grid(True)
            plt.ylabel("Argument of periapsis [deg]")
            plt.xlabel(timeLabl)

            np += 1
            plt.subplot2grid((nPlot, 1), (np, 0))
            self.plotCat(f * r2d, piIsTime=piIsTime, intv=intv)
            plt.grid(True)
            plt.ylabel("True anomaly [deg]")
            plt.xlabel(timeLabl)


            ######################################
            alpha, beta = self.calcDimCtrl()
            alpha *= r2d
            np += 1
            plt.subplot2grid((nPlot, 1), (np, 0))
            self.plotCat(alpha, piIsTime=piIsTime, intv=intv, color='k')
            plt.grid(True)
            plt.ylabel("alpha [deg]")
            plt.xlabel(timeLabl)

            np += 1
            plt.subplot2grid((nPlot, 1), (np, 0))
            self.plotCat(beta, piIsTime=piIsTime, intv=intv, color='k')
            plt.grid(True)
            plt.ylabel("beta [-]")
            plt.xlabel(timeLabl)
            ######################################

            np += 1
            plt.subplot2grid((nPlot, 1), (np, 0))
            acc = self.calcAcc()
            self.plotCat(acc * 1e3, color='y', piIsTime=piIsTime, labl='Accel.')

            plt.plot(tVec,1e3 * self.restrictions['acc_max'] * \
                          numpy.array([1.0, 1.0]), '--')
            plt.grid(True)
            plt.xlabel(timeLabl)
            plt.ylabel("Tang. accel. [m/s²]")

            if mustSaveFig:
                if piIsTime:
                    self.savefig(keyName='currOrbt', fullName='orbital solution')
                else:
                    self.savefig(keyName='currOrbt-adimTime',
                                 fullName='orbital solution (adim. Time)')
            else:
                plt.show()
                plt.clf()

        else:
            raise Exception('plotSol: Unknown mode "' + str(opt['mode']))
        #

    # noinspection PyTypeChecker
    def plotQRes(self,args,mustSaveFig=True,addName=''):
        """Plots of the Q residuals, specifically for the probRock case."""

        # Qx error plot
        plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
        plt.subplot2grid((6,1),(0,0))
        self.plotCat(args['accQx'],color='b',piIsTime=False)
        plt.grid(True)
        plt.ylabel("Accumulated int.")
        titlStr = "Qx = int || dlam - f_x + phi_x^T*lam || " + \
                  "= {:.4E}".format(args['Qx'])
        titlStr += "\n(grad iter #" + str(self.NIterGrad+1) + ")"
        plt.title(titlStr)

        plt.subplot2grid((6,1),(1,0))
        self.plotCat(args['normErrQx'],color='b',piIsTime=False)
        plt.grid(True)
        plt.ylabel("Integrand of Qx")
        errQx = args['errQx']

        plt.subplot2grid((6,1),(2,0))
        self.plotCat(errQx[:,0,:],piIsTime=False)
        plt.grid(True)
        plt.ylabel("ErrQx_h")

        plt.subplot2grid((6,1),(3,0))
        self.plotCat(errQx[:,1,:],color='g',piIsTime=False)
        plt.grid(True)
        plt.ylabel("ErrQx_v")

        plt.subplot2grid((6,1),(4,0))
        self.plotCat(errQx[:,2,:],color='r',piIsTime=False)
        plt.grid(True)
        plt.ylabel("ErrQx_gama")

        plt.subplot2grid((6,1),(5,0))
        self.plotCat(errQx[:,3,:],color='m',piIsTime=False)
        plt.grid(True)
        plt.ylabel("ErrQx_m")

        plt.xlabel("t [-]")
        if mustSaveFig:
            self.savefig(keyName=('Qx'+addName),fullName='Qx')
        else:
            plt.show()
            plt.clf()

        # Qu error plot
        plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
        plt.subplot2grid((4,1),(0,0))
        self.plotCat(args['accQu'],color='b',piIsTime=False)
        plt.grid(True)
        plt.ylabel("Accumulated int.")
        titlStr = "Qu = int || f_u - phi_u^T*lam || = " + \
                  "{:.4E}".format(args['Qu']) + \
                  "\n(grad iter #" + str(self.NIterGrad+1) + ")"
        plt.title(titlStr)
        plt.subplot2grid((4,1),(1,0))
        self.plotCat(args['normErrQu'],color='b',piIsTime=False)
        plt.grid(True)
        plt.ylabel("Integrand of Qu")

        errQu = args['errQu']
        plt.subplot2grid((4,1),(2,0))
        self.plotCat(errQu[:,0,:],color='k',piIsTime=False)
        plt.grid(True)
        plt.ylabel("Qu_alpha")
        plt.subplot2grid((4,1),(3,0))
        self.plotCat(errQu[:,1,:],color='r',piIsTime=False)
        plt.grid(True)
        plt.ylabel("Qu_beta")

        plt.xlabel("t [-]")
        if mustSaveFig:
            self.savefig(keyName=('Qu'+addName),fullName='Qu')
        else:
            plt.show()
            plt.clf()

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
        titlStr += "\n(grad iter #" + str(self.NIterGrad+1) + ")"
        plt.title(titlStr)

        for j in range(1,p):
            plt.subplot2grid((p,1),(j,0))
            self.plotCat(errQp[:,j,:],color='k',piIsTime=False)
            plt.grid(True)
            plt.ylabel("ErrQp, j ="+str(j))
        plt.xlabel("t [-]")
        if mustSaveFig:
            self.savefig(keyName=('Qp'+addName),fullName='Qp')
        else:
            plt.show()
            plt.clf()

    def compWith(self, altSol, altSolLabl='altSol',
                 mustSaveFig=True, piIsTime=True,
                 subPlotAdjs={'left':0.0,'right':1.0,'bottom':0.0,
                             'top':10.0,'wspace':0.2,'hspace':0.35}):
        self.log.printL("\nComparing solutions...\n")
        r2d = 180.0/numpy.pi
        currSolLabl = 'Current solution'
        if piIsTime:
            timeLabl = 't [s]'
        else:
            timeLabl = 'non-dim. t [-]'

        # Comparing final mass:
        mPaySol = self.calcMassDist()['u'][-1]
        mPayAlt = altSol.calcMassDist()['u'][-1]

        if mPayAlt <= 0.:
            paylMassGain = numpy.inf
            paylPercMassGain = numpy.inf
        else:
            paylMassGain = mPaySol - mPayAlt
            paylPercMassGain = 100.0*paylMassGain/mPayAlt

        # Plotting the curves
        plt.subplots_adjust(**subPlotAdjs)

        # Curve 1: height
        plt.subplot2grid((12,1),(0,0))
        self.plotCat(self.x[:,0,:],color='y',labl=currSolLabl,
                     piIsTime=piIsTime)
        altSol.plotCat(altSol.x[:,0,:],mark='--',labl=altSolLabl,
                       piIsTime=piIsTime)
        plt.grid(True)
        plt.ylabel("h [km]")
        #plt.legend(loc="upper left", bbox_to_anchor=(1,1))
        titlStr = "Comparing solutions: " + currSolLabl + " and " + \
                  altSolLabl + \
                  "\nPayload mass gain: {:.4G}%".format(paylPercMassGain)
        titlStr += "\n(grad iter #" + str(self.NIterGrad) + ")"
        titlStr += "\n\n"
        plt.title(titlStr)
        plt.xlabel(timeLabl)
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)

        # Curve 2: speed
        plt.subplot2grid((12,1),(1,0))
        self.plotCat(self.x[:,1,:],color='g',labl=currSolLabl,
                     piIsTime=piIsTime)
        altSol.plotCat(altSol.x[:,1,:],mark='--',labl=altSolLabl,
                       piIsTime=piIsTime)
        plt.grid(True)
        plt.ylabel("V [km/s]")
        plt.xlabel(timeLabl)
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)

        # Curve 3: flight path angle
        plt.subplot2grid((12,1),(2,0))
        self.plotCat(self.x[:,2,:]*180/numpy.pi,color='r',
                     labl=currSolLabl,piIsTime=piIsTime)
        altSol.plotCat(altSol.x[:,2,:]*r2d,mark='--',labl=altSolLabl,
                       piIsTime=piIsTime)
        plt.grid(True)
        plt.ylabel("gamma [deg]")
        plt.xlabel(timeLabl)
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)

        # Curve 4: Mass
        plt.subplot2grid((12,1),(3,0))
        self.plotCat(self.x[:,3,:],color='m',labl=currSolLabl,
                       piIsTime=piIsTime)
        altSol.plotCat(altSol.x[:,3,:],mark='--',labl=altSolLabl,
                       piIsTime=piIsTime)
        plt.grid(True)
        plt.ylabel("m [kg]")
        plt.xlabel(timeLabl)
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)

        # Curve 5: Control #1 (angle of attack)
        plt.subplot2grid((12,1),(4,0))
        self.plotCat(self.u[:,0,:],color='c',labl=currSolLabl,
                       piIsTime=piIsTime)
        altSol.plotCat(altSol.u[:,0,:],mark='--',labl=altSolLabl,
                       piIsTime=piIsTime)
        plt.grid(True)
        plt.ylabel("u1 [-]")
        plt.xlabel(timeLabl)
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)

        # Curve 6: Control #2 (thrust)
        plt.subplot2grid((12,1),(5,0))
        self.plotCat(self.u[:,1,:],color='c',labl=currSolLabl,
                       piIsTime=piIsTime)
        altSol.plotCat(altSol.u[:,1,:],mark='--',labl=altSolLabl,
                       piIsTime=piIsTime)
        plt.grid(True)
        plt.ylabel("u2 [-]")
        plt.xlabel(timeLabl)
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)

        # Curve 7: angle of attack
        alpha,beta = self.calcDimCtrl()
        alpha_alt,beta_alt = altSol.calcDimCtrl()
        plt.subplot2grid((12,1),(6,0))
        self.plotCat(alpha*r2d,color='k',labl=currSolLabl,
                       piIsTime=piIsTime)
        altSol.plotCat(alpha_alt*r2d,mark='--',labl=altSolLabl,
                       piIsTime=piIsTime)
        plt.grid(True)
        plt.xlabel("t")
        plt.ylabel("alpha [deg]")
        plt.xlabel(timeLabl)
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)

        # Curve 8: thrust level
        plt.subplot2grid((12,1),(7,0))
        self.plotCat(beta,color='k',labl=currSolLabl,
                       piIsTime=piIsTime)
        altSol.plotCat(beta_alt,mark='--',labl=altSolLabl,
                       piIsTime=piIsTime)
        plt.grid(True)
        plt.xlabel(timeLabl)
        plt.ylabel("beta [-]")
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)

        # Curve 9: thrust
        plt.subplot2grid((12,1),(8,0))
        thrust = numpy.empty_like(beta)
        thrust_alt = numpy.empty_like(beta_alt)
        L = numpy.empty_like(beta); L_alt = numpy.empty_like(beta)
        D = numpy.empty_like(beta); D_alt = numpy.empty_like(beta)
        dens = numpy.empty_like(beta); dens_alt = numpy.empty_like(beta)
        pDyn = numpy.empty_like(beta); pDyn_alt = numpy.empty_like(beta)
        s = self.s; s_alt = altSol.s
        MaxThrs = self.constants['Thrust']
        CL0, CL1 = self.constants['CL0'], self.constants['CL1']
        CD0, CD2 = self.constants['CD0'], self.constants['CD2']
        s_ref = self.constants['s_ref']
        for arc in range(s):
            thrust[:, arc] = beta[:, arc] * MaxThrs[arc]
            thrust_alt[:, arc] = beta_alt[:, arc] * MaxThrs[arc]
            dens[:, arc] = rhoSGRA(self.x[:, 0, arc])
            dens_alt[:, arc] = rhoSGRA(altSol.x[:, 0, arc])
            pDyn[:,arc] = .5 * dens[:,arc] * (self.x[:, 1, arc]) ** 2
            pDyn_alt[:,arc] = .5 * dens_alt[:,arc] * \
                              (altSol.x[:, 1, arc]) ** 2
            L[:, arc] = pDyn[:, arc] * s_ref[arc] * \
                        (CL0[arc] + alpha[:, arc] * CL1[arc])
            L_alt[:, arc] = pDyn_alt[:, arc] * s_ref[arc] * \
                        (CL0[arc] + alpha_alt[:, arc] * CL1[arc])
            D[:, arc] = pDyn[:, arc] * s_ref[arc] * \
                        (CD0[arc] + (alpha[:, arc] ** 2) * CD2[arc])
            D_alt[:, arc] = pDyn_alt[:, arc] * s_ref[arc] * \
                        (CD0[arc] + (alpha_alt[:, arc] ** 2) * CD2[arc])
        self.plotCat(thrust, color='k', piIsTime=piIsTime, labl='Thrust')
        altSol.plotCat(thrust_alt, mark='--', color='k',
                       labl='Thrust - '+altSolLabl, piIsTime=piIsTime)
        self.plotCat(L, color='b', piIsTime=piIsTime, labl='Lift')
        self.plotCat(L_alt, mark='--', color='b', piIsTime=piIsTime,
                     labl='Lift - '+altSolLabl)
        self.plotCat(D, color='r', piIsTime=piIsTime, labl='Drag')
        self.plotCat(D_alt, mark='--', color='r', piIsTime=piIsTime,
                     labl='Drag - '+altSolLabl)
        plt.grid(True)
        plt.xlabel(timeLabl)
        plt.ylabel("Forces [kN]")
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=2)

        # Curve 10: acceleration
        plt.subplot2grid((12,1),(9,0))
        solAcc =  self.calcAcc()
        altSolAcc = altSol.calcAcc()
        if piIsTime:
            plt.plot([0.0,max(self.pi.sum(),altSol.pi.sum())],
                  1e3*self.restrictions['acc_max']*numpy.array([1.0,1.0]),
                  '--',label='Acceleration limit')
        else:
            plt.plot([0.0,float(self.s)],
                  1e3*self.restrictions['acc_max']*numpy.array([1.0,1.0]),
                  '--',label='Acceleration limit')

        self.plotCat(1e3*solAcc,color='y',labl=currSolLabl,
                       piIsTime=piIsTime)
        altSol.plotCat(1e3*altSolAcc,mark='--',labl=altSolLabl,
                       piIsTime=piIsTime)
        plt.grid(True)
        plt.xlabel(timeLabl)
        plt.ylabel("Tang. accel. [m/s²]")
        plt.legend(loc="lower center",bbox_to_anchor=(0.5,1),ncol=3)

        # 'Curve' 11: pi's (arcs)
        ax = plt.subplot2grid((12,1),(10,0))
        position = numpy.arange(s)
        position_alt = numpy.arange(s_alt)
        arcs = numpy.arange(1,s+1)
        width = 0.4
        current = ax.bar(position,self.pi,width,color='y')
        initial = ax.bar(position_alt + width,altSol.pi,width)
        # Put the values of the arc lengths on the bars...
        for arc in range(s):
            coord = (float(position[arc])-.25*width,self.pi[arc]+10.)
            ax.annotate("{:.1F}".format(self.pi[arc]),xy=coord,xytext=coord)
        for arc in range(s):
            coord = (float(position_alt[arc])+.75*width,altSol.pi[arc]+10.)
            ax.annotate("{:.1F}".format(altSol.pi[arc]),xy=coord,xytext=coord)
        ax.set_xticks(position + width/2)
        ax.set_xticklabels(arcs)
        plt.grid(True,axis='y')
        plt.xlabel("Arcs")
        plt.ylabel("Duration [s]")
        ax.legend((current[0], initial[0]), (currSolLabl,altSolLabl),
                  loc="lower center", bbox_to_anchor=(0.5,1),ncol=2)

        # 'Curve' 12: pi's (stages)
        addArcs = self.addArcs
        if s > addArcs:
            ax = plt.subplot2grid((12,1),(11,0))
            position = numpy.arange(s-addArcs)
            position_alt = numpy.arange(s_alt-addArcs)
            stages = numpy.arange(1,s+1-addArcs)
            width = 0.4
            pi_stages = numpy.zeros(s-addArcs)
            altSol_pi_stages = numpy.zeros(s-addArcs)
            for k in range(len(pi_stages)):
                if k==0:
                    for arc in range(addArcs+1):
                        pi_stages[k] += self.pi[arc]
                        altSol_pi_stages[k] += altSol.pi[arc]
                else:
                    pi_stages[k] = self.pi[k+addArcs]
                    altSol_pi_stages[k] = altSol.pi[k+addArcs]
            current = ax.bar(position,pi_stages,width,color='y')
            initial = ax.bar(position_alt + width,altSol_pi_stages,width)
            # Put the values of the arc lenghts on the bars...
            for st in range(s-addArcs):
                coord = (float(position[st])-.25*width,pi_stages[st]+10.)
                ax.annotate("{:.1F}".format(pi_stages[st]),xy=coord,
                            xytext=coord)
            for st in range(s-addArcs):
                coord = (float(position_alt[st])+.75*width,
                         altSol_pi_stages[st]+10.)
                ax.annotate("{:.1F}".format(altSol_pi_stages[st]),xy=coord,
                            xytext=coord)
            ax.set_xticks(position + width/2)
            ax.set_xticklabels(stages)
            plt.grid(True,axis='y')
            plt.xlabel("Stages")
            plt.ylabel("Duration [s]")
            ax.legend((current[0], initial[0]), (currSolLabl,altSolLabl),
                      loc="lower center", bbox_to_anchor=(0.5,1),ncol=2)
        if mustSaveFig:
            if piIsTime:
                self.savefig(keyName='comp',fullName='comparisons')
            else:
                self.savefig(keyName='comp-adimTime',
                             fullName='comparisons (non-dim. time)')
        else:
            plt.show()
            plt.clf()

        msg = 'Final rocket "payload":\n' + \
              '{}: {:.4E} kg.\n'.format(currSolLabl,mPaySol) + \
              '{}: {:.4E} kg.\n'.format(altSolLabl, mPayAlt) + \
              'Difference: {:.4E} kg, '.format(paylMassGain) + \
              '{:.4G}% more payload!\n'.format(paylPercMassGain)
        self.log.printL(msg)

    def rockin(self,opt=None):
        """ Integrate rocket kinematics.

        Options for range angle, horizontal position and vertical
        position for a topocentric system. """

        if opt is None:
            opt = {}

        ret = {}

        # Range angle
        if opt.get('range',False) or opt.get('sigma',False) or \
                opt.get('X',False) or opt.get('Z',False):
            # range angle derivative
            dsig = self.x[:,1,:] * \
                   numpy.cos(self.x[:,2,:]) / \
                   (self.constants['r_e'] + self.x[:, 0, :])
            # perform integration itself
            sigma = self.intgEulr(dsig,0.)
            ret['sigma'] = sigma

        # Horizontal position 'X'
        if opt.get('X', False):
            # X position derivative
            # noinspection PyUnboundLocalVariable
            dX = self.x[:, 1, :] * \
                   numpy.cos(self.x[:, 2, :] - sigma)
            # perform the integration itself
            ret['X'] = self.intgEulr(dX, 0.)

        # Vertical position 'Z'
        if opt.get('Z', False):
            # Z position derivative
            # noinspection PyUnboundLocalVariable
            dZ = self.x[:, 1, :] * \
                 numpy.sin(self.x[:, 2, :] - sigma)
            # perform the integration itself
            ret['Z'] = self.intgEulr(dZ, self.boundary['h_initial'])

        return ret

    def plotTraj(self, compare=False, altSol: sgra = None,
                 altSolLabl='altSol', mustSaveFig=True, fullOrbt=False,
                 markSize=2.5):
        """ Plot the rocket trajectory in the flight plane.

        Also enables:
         - comparing trajectories
         - calculating maximum dynamic pressure."""

        self.log.printL("\nIn plotTraj!")

        # Pre-assigning functions and variables
        cos = numpy.cos; sin = numpy.sin
        R = self.constants['r_e']
        N, s = self.N, self.s
        d2r = numpy.pi / 180.

        # Density for all times/arcs
        dens = numpy.zeros((N,s))
        # Stage separation points (actually, just the arcs)
        StgSepPnts = numpy.zeros((s,2))
#        StgInitAcc = numpy.zeros(s)
#        StgFinlAcc = numpy.zeros(s)

        # Re-integrate the kinematics for topocentric system (X,Z),
        # range angle is also necessary for orbit plot
        # optRockin = {'X':True, 'Z':True, 'range':True}
        # ret = self.rockin(opt=optRockin)
        # X, Z = self.unpack(ret['X']), self.unpack(ret['Z'])
        # sigma = ret['sigma'][-1, -1]
        ret = self.rockin(opt={'sigma':True})
        sigArray = ret['sigma']
        # Having the height, just the range angle is necessary, actually...
        X = self.unpack((self.x[:,0,:]+R) * sin(sigArray))
        Z = self.unpack((self.x[:,0,:]+R) * cos(sigArray) - R)
        sigma = sigArray[-1,-1]

        if compare:
            if altSol is None:
                self.log.printL("plotTraj: comparing mode is set to True," + \
                                " but no solution was given to which " + \
                                "compare. Ignoring...")
                compare=False
            else:
                # Also do the kinematic integrations for alt. solution
                # noinspection PyUnresolvedReferences
                # ret_alt = altSol.rockin(opt=optRockin)
                # X_alt = altSol.unpack(ret_alt['X'])
                # Z_alt = altSol.unpack(ret_alt['Z'])

                ret_alt = altSol.rockin(opt={'sigma': True})
                sigArray_alt = ret_alt['sigma']
                # Having the height, just the range angle is necessary,
                # actually...
                X_alt = altSol.unpack((altSol.x[:, 0, :] + R) * \
                                  sin(sigArray_alt))
                Z_alt = altSol.unpack((altSol.x[:, 0, :] + R) * \
                                  cos(sigArray_alt) - R)

        # Propulsive phases' starting and ending times, and maxQ calculations.
        # This is implemented with two lists, one for each arc.
        # Each list stores the indices of the points in which the burning
        # begins or ends, respectively.
        isBurn = True
        indBurn = list(); indShut = list()

        # The rocket definitely begins burning at liftoff!
        indBurn.append(0)
        iCont = -1 # continuous counter (all arcs concatenated)
        for arc in range(s):
            dens[:, arc] = rhoSGRA(self.x[:, 0, arc])
            for i in range(N):#range(strtInd,N):
                iCont += 1

                if isBurn:
                    if self.u[i,1,arc] < -3.:# <0.3% thrust
                        isBurn = False
                        indShut.append(iCont)
                else: #not burning
                    if self.u[i,1,arc] > -3.:# >0.3% thrust
                        isBurn = True
                        indBurn.append(iCont)

            #
            # TODO: this must be adjusted to new formulation, where arcs
            #  might not be the stage separation points necessarily
            StgSepPnts[arc,:] = X[iCont],Z[iCont]
        # The rocket definitely ends the trajectory with engine shutdown
        indShut.append(iCont)


        # Calculate dynamic pressure, get point of max pdyn
        pDyn = self.unpack(.5 * dens * (self.x[:,1,:]**2))
        indPdynMax = numpy.argmax(pDyn)
        pDynMax = pDyn[indPdynMax]

        #pairIndPdynMax = numpy.unravel_index(indPdynMax,(N,s))
        #self.log.printL(indPdynMax)
        #self.log.printL("t @ max q (relative to arc):",\
        #self.t[pairIndPdynMax[0]])
        #self.log.printL("State @ max q:")
        #self.log.printL(self.x[pairIndPdynMax[0],:,pairIndPdynMax[1]])

#        self.plotCat(dens*1e-9)
#        plt.grid(True)
#        plt.title("Density vs. time")
#        plt.xlabel('t [s]')
#        plt.ylabel('Dens [kg/m³]')
#        plt.show()
#
#        self.plotCat(pDyn*1e-3)
#        plt.grid(True)
#        plt.title("Dynamic pressure vs. time")
#        plt.xlabel('t [s]')
#        plt.ylabel('P [Pa]')
#        plt.show()

#        indAccMax = numpy.argmax(acc,0)
#        self.plotCat(acc)
#        plt.grid(True)
#        plt.title("Acceleration vs. time")
#        plt.xlabel('t [s]')
#        plt.ylabel('a [g]')
#        plt.show()


        # Draw Earth segment corresponding to flight range
        sigVec = numpy.arange(0,1.01,.01) * sigma
        # If in "full orbit mode", draw Earth itself,
        # else, plot just the segment corresponding to flight range
        if fullOrbt:
            intv = numpy.arange(0,2.*numpy.pi,0.001)
            x = R * cos(intv)
            z = R * (sin(intv) - 1.)
        else:
            x = R * cos(.5 * numpy.pi - sigVec)
            z = R * (sin(.5 * numpy.pi - sigVec) - 1.0)

        plt.plot(x,z,'k',label='Earth surface')

        # Get final orbit parameters
        h,v,gama,M = self.x[-1,:,-1]

        self.log.printL("State @burnout time:\n" + \
                        "h = {:.3F} km".format(h) + \
                        ", v = {:.3F} km/s".format(v) + \
                        ", gama = {:.4F} deg".format(gama*180./numpy.pi) + \
                        ", m = {:.4F} kg".format(M))
        self.log.printL("Not a state, but should be:\n" + \
                        "range angle = {:.1F} deg".format(sigma/d2r))
        GM = self.constants['GM']
        r = R + h
        cosGama, sinGama = cos(gama), sin(gama)
        # specific angular momentum
        momAng = r * v * cosGama
        # specific mechanic energy
        en = .5 * v * v - GM / r
        # "Semi-major axis"
        a = - .5 * GM / en
        # Eccentricity
        aux = v * momAng / GM
        er = aux * cosGama - 1.0 # radial component of ecc. vector
        erOrt = aux * sinGama # orthogonal component of ecc. vector
        e = numpy.sqrt(er**2 + erOrt**2)
        # True anomaly (w.r.t. perigee)
        f = numpy.arccos(er/e)
        ph = a * (1.0 - e) - R
        self.log.printL("Perigee altitude: {:.1F} km".format(ph))
        ah = a * (1. + e) - R
        self.log.printL("Apogee altitude: {:.1F} km".format(ah))
        # semi-latus rectum
        p = momAng**2 / GM #a * (1.0-e)**2

        # shifting angle
        sh = sigma - f - .5 * numpy.pi
        if fullOrbt:
            # If in full orbit mode, plot the complete orbit
            sigVec = numpy.arange(f, f+2.*numpy.pi, .01)
            rOrb = p / (1.0 + e * cos(sigVec))
            xOrb = rOrb * cos(-sigVec-sh)
            zOrb = rOrb * sin(-sigVec-sh) - R
        else:
            # Plot orbit in green over the same range as the Earth shown
            # (and a little bit further)
            sigVec = numpy.arange(f-1.2*sigma,f+.2*sigma,.01*sigma)
            rOrb = p/(1.0+e*cos(sigVec))
            xOrb = rOrb * cos(-sigVec-sh)
            zOrb = rOrb * sin(-sigVec-sh) - R
        plt.plot(xOrb,zOrb,'g--',label='Target orbit')

        # # DEBUG
        # strP = "\nDebugging is fun!\n" + \
        #       "Final states for all arcs: \n" + str(self.x[-1, :, :]) + \
        #       "\nFinal values after kinematic reintegration:" + \
        #       "\nX: " + str(ret['X'][-1, :]) + \
        #       "\nZ: " + str(ret['Z'][-1, :])
        # self.log.printL(strP)
        # trueF = numpy.arccos((p/r - 1.)/e) / d2r
        # strP = "\nTrue anomaly at injection point: {:.2F}".format(trueF)
        # strP += "\nf = {:.2F} deg".format(f/d2r)
        # strP += "\nsigma = {:.2F} deg".format(sigma/d2r)
        # #strP += "\nsigma = {.2F}".format(sigma / d2r)
        # rp = numpy.sqrt(X[-1]**2 + (Z[-1]+R)**2)
        # strP += "\n X at inj point: {:.2F} km".format(X[-1])
        # strP += "\n Z at inj point: {:.2F} km".format(Z[-1])
        # strP += "\nRadial distance at inj point (Pyth): {:.2F} km".format(rp)
        # strP += "\nR+h at inj point: {:.2F} km".format(R+h)
        # ofX, ofZ = (R + h) * sin(sigma), (R+h)*cos(sigma)-R
        # strP += "\n(R+h)*sin(sigma) = {:.2F} km".format(ofX)
        # strP += "\n(R+h)*cos(sigma)-R = {:.2F} km".format(ofZ)
        # strP += "\nh*cos(sigma) = {:.2F} km".format(h * cos(sigma))
        # strP += "\nX error = {:.3F} km".format(X[-1] - ofX)
        # strP += "\nZ error = {:.3F} km".format(Z[-1] - ofZ)
        # strP += "\n"
        # self.log.printL(strP)
        #input("\nIAE? ")

        # Draw orbit injection point (green)
        r0 = p / (1.0 + e * cos(f))
        x0 = r0 * cos(-f-sh)
        z0 = r0 * sin(-f-sh) - R
        plt.plot(x0,z0,'og',ms=markSize)

        # Plot trajectory in default color (blue)
        plt.plot(X,Z,label='Ballistic flight (coasting)')

        # Plot launching point (black)
        plt.plot(X[0],Z[0],'ok',ms=markSize)

        # Plot burning segments in red,
        # label only the first to avoid multiple labels
        mustLabl = True
        for i in range(len(indBurn)):
            ib = indBurn[i]
            # this +1 compensates for Python's lovely slicing standard
            ish = indShut[i]+1
            if mustLabl:
                plt.plot(X[ib:ish],Z[ib:ish],'r',label='Propulsed flight')
                mustLabl = False
            else:
                plt.plot(X[ib:ish],Z[ib:ish],'r')

        # Plot Max Pdyn point in orange
        plt.plot(X[indPdynMax],Z[indPdynMax],marker='o',color='orange',
                 label='Max dynamic pressure',ms=markSize)

        # Plot stage separation points in blue
        mustLabl = True
        #if self.initMode == 'extSol':1
        for arc in range(s-1):
            if self.isStagSep[arc]:
                # this trick only labels the first segment,
                # to avoid multiple labels afterwards
                if mustLabl:
                    plt.plot(StgSepPnts[arc,0],StgSepPnts[arc,1],
                             marker='o', color='blue',ms=markSize,
                             label='Stage separation point')
                    mustLabl = False
                else:
                    plt.plot(StgSepPnts[arc,0],StgSepPnts[arc,1],
                             marker='o', color='blue',ms=markSize)

        # Plot altSol
        if compare:
            # noinspection PyUnboundLocalVariable
            plt.plot(X_alt,Z_alt,'k--',label='Initial guess')

        # Final plotting commands
        plt.grid(True)
        plt.xlabel("X [km]")
        plt.ylabel("Z [km]")
        plt.axis('equal')
        titlStr = "Rocket trajectory over Earth "
        titlStr += "(grad iter #" + str(self.NIterGrad) + ")\n"
        titlStr += "MaxDynPres = {:.3F} MPa".format(pDynMax*1e-9)
        plt.title(titlStr)
        plt.legend(loc="upper left", bbox_to_anchor=(1,1))

        if mustSaveFig:
            self.savefig(keyName='traj',fullName='trajectory')
        else:
            plt.show()
            plt.clf()

#%%
def calcXdot(td,x,u,constants,arc):
    grav_e = constants['grav_e']; Thrust = constants['Thrust']
    Isp = constants['Isp']; s_ref = constants['s_ref']
    r_e = constants['r_e']; GM = constants['GM']
    CL0 = constants['CL0']; CL1 = constants['CL1']
    CD0 = constants['CD0']; CD2 = constants['CD2']
    DampCent = constants['DampCent']; DampSlop = constants['DampSlop']

    sin = numpy.sin; cos = numpy.cos

    # calculate variables alpha and beta
    alpha = u[0]; beta = u[1]

    # calculate variables CL and CD
    CL = CL0[arc] + CL1[arc] * alpha
    CD = CD0[arc] + CD2[arc] * alpha ** 2

    # calculate L and D

    dens = rho(x[0])
    pDynTimesSref = .5 * dens * (x[1]**2) * s_ref[arc]
    L = CL * pDynTimesSref; D = CD * pDynTimesSref

    # calculate r
    r = r_e + x[0]

    # calculate grav
    grav = GM/r/r

    # prepare return array:
    dx = numpy.empty(4)

    # Rocket dynamics
    sinGama = sin(x[2])
    Thr = beta * Thrust[arc]
    dx[0] = x[1] * sinGama
    dx[1] = (Thr * cos(alpha) - D)/x[3] - grav * sinGama
    dx[2] = (Thr * sin(alpha) + L)/(x[3] * x[1]) + \
            cos(x[2]) * ( x[1]/r  -  grav/x[1] )
    # "Gamma factor" in the gamma equation. td is dimensional time
    dx[2] *= .5*(1.0+numpy.tanh(DampSlop*(td-DampCent)))
    dx[3] = -Thr/(grav_e * Isp[arc])
    return dx
