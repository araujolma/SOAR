#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:25:28 2017

@author: levi
"""
import numpy, itsme
from sgra import sgra
from atmosphere import rho
from itsme import problemConfigurationSGRA
import matplotlib.pyplot as plt


class prob(sgra):
    probName = 'probRock'

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

        dt = 1.0/(N-1)
        t = numpy.arange(0,1.0+dt,dt)
        self.dt = dt
        self.t = t

    def initGues(self,opt={}):

        # matrix sizes
        n = 4
        m = 2
        self.n = n
        self.m = m

        d2r = numpy.pi / 180.0
        # Get initialization mode
        initMode = opt.get('initMode','default')

        if initMode == 'default':
            # artesanal handicraft with L and D (Miele 2003)
#            x[:,0] = h_final*numpy.sin(numpy.pi*t/2)
#            x[:,1] = 3.793*numpy.exp(0.7256*t) -1.585 -3.661*numpy.cos(3.785*t+0.9552)
#            #x[:,1] = V_final*numpy.sin(numpy.pi*t.copy()/2)
#            #x[:,1] = 1.0e3*(-0.4523*t.copy()**5 + 1.2353*t.copy()**4-1.1884*t.copy()**3+0.4527*t.copy()**2-0.0397*t.copy())
#            x[:,2] = (numpy.pi/2)*(numpy.exp(-(t.copy()**2)/0.017))+0.06419
#            x[:,3] = m_initial*((0.7979*numpy.exp(-(t.copy()**2)/0.02))+0.1901*numpy.cos(t.copy()))
#            #x[:,3] = m_initial*(1.0-0.89*t.copy())
#            #x[:,3] = m_initial*(-2.9*t.copy()**3 + 6.2*t.copy()**2 - 4.2*t.copy() + 1)
#            for k in range(N):
#                if k<910:
#                    u[k,1] = (numpy.pi/2)
#                else:
#                    if k>4999:
#                        u[k,1] = (numpy.pi/2)*0.27
#            pi = 1100*numpy.ones((p,1))

            N = 10000+1#7500+1#10000 + 1#40000+1#20000+1#5000000 + 1 #
            self.N = N

            dt = 1.0/(N-1)
            t = numpy.arange(0,1.0+dt,dt)
            self.dt = dt
            self.t = t

            s = 1
            p = 1
            self.s = s
            self.p = p
            self.Ns = 2*n*s + p

            q = 2*n - 1 + n * (s-1)
            self.q = q
            # Payload mass
            self.mPayl = 100.0

            x = numpy.zeros((N,n,s))
            u = numpy.zeros((N,m,s))

            #prepare tolerances
            tolP = 1.0e-5
            tolQ = 1.0e-7#5
            tol = dict()
            tol['P'] = tolP
            tol['Q'] = tolQ
            self.tol = tol

            # Earth constants
            r_e = 6371.0           # km
            GM = 398600.4415       # km^3 s^-2
            grav_e = GM/r_e/r_e    #9.8e-3       km/s^2

            # rocket constants
            Thrust = numpy.array([40.0])                 # kg km/s² [= kN] 1.3*m_initial # N

            Kpf = 100.0  # First guess........

            Isp = 450.0*numpy.ones(s)                     # s
            s_f = 0.05*numpy.ones(s)
            CL0 = 0.0*numpy.ones(s)#-0.03                 # (B0 Miele 1998)
            CL1 = 0.8*numpy.ones(s)                       # (B1 Miele 1998)
            CD0 = 0.05*numpy.ones(s)                      # (A0 Miele 1998)
            CD2 = 0.5*numpy.ones(s)                       # (A2 Miele 1998)
            s_ref = (numpy.pi*(0.0005)**2)*numpy.ones(s)  # km^2
            DampCent = 3.0#2.0#
            DampSlop = 3.0

            # boundary conditions
            h_initial = 0.0            # km
            V_initial = 1e-6#0.0       # km/s
            gamma_initial = numpy.pi/2 # rad
            #m_initial = 50000          # kg
            h_final = 463.0   # km
            V_final = numpy.sqrt(GM/(r_e+h_final))#7.633   # km/s
            gamma_final = 0.0 # rad
            #m_final = free   # kg

            boundary = dict()
            boundary['h_initial'] = h_initial
            boundary['V_initial'] = V_initial
            boundary['gamma_initial'] = gamma_initial
            #boundary['m_initial'] = m_initial
            boundary['h_final'] = h_final
            boundary['V_final'] = V_final
            boundary['gamma_final'] = gamma_final

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
            constants['Kpf'] = Kpf
            self.constants = constants

            # restrictions
            alpha_min = -2 * d2r  # in rads
            alpha_max = 2 * d2r   # in rads
            beta_min = 0.0
            beta_max = 1.0
            acc_max = 3.0 * grav_e
            restrictions = dict()
            restrictions['alpha_min'] = alpha_min
            restrictions['alpha_max'] = alpha_max
            restrictions['beta_min'] = beta_min
            restrictions['beta_max'] = beta_max
            restrictions['acc_max'] = acc_max
            self.restrictions = restrictions
            solInit = None

        elif initMode == 'naive':
            N = 10000+1#7500+1#10000 + 1#40000+1#20000+1#5000000 + 1 #
            self.N = N

            dt = 1.0/(N-1)
            t = numpy.arange(0,1.0+dt,dt)
            self.dt = dt
            self.t = t

            #prepare tolerances
            tolP = 1.0e-5
            tolQ = 1.0e-7#5
            tol = dict()
            tol['P'] = tolP
            tol['Q'] = tolQ
            self.tol = tol

#            pis2 = numpy.pi*0.5
#            pi = numpy.array([300.0])
#            dt = pi[0]/(N-1); dt6 = dt/6
#            x[0,:] = numpy.array([0.0,1.0e-6,pis2,2000.0])
#            for i in range(N-1):
#                tt = i * dt
#                k1 = calcXdot(sizes,tt,x[i,:],u[i,:],constants,restrictions)
#                k2 = calcXdot(sizes,tt+.5*dt,x[i,:]+.5*dt*k1,.5*(u[i,:]+u[i+1,:]),constants,restrictions)
#                k3 = calcXdot(sizes,tt+.5*dt,x[i,:]+.5*dt*k2,.5*(u[i,:]+u[i+1,:]),constants,restrictions)
#                k4 = calcXdot(sizes,tt+dt,x[i,:]+dt*k3,u[i+1,:],constants,restrictions)
#                x[i+1,:] = x[i,:] + dt6 * (k1+k2+k2+k3+k3+k4)
            solInit = None

        elif initMode == 'extSol':
            inpFile = opt.get('confFile','')
            self.log.printL("Starting ITSME with input = " + inpFile)

            t_its, x_its, u_its, tabAlpha, tabBeta, inputDict, tphases, \
            mass0, massJet = itsme.sgra(inpFile)
            # The inputDict corresponds to the con dictionary from itsme.

            self.log.printL("\nITSME was run sucessfully, " + \
                            "proceding adaptations...")

            #prepare tolerances
            tol = {'P': inputDict['tolP'],
                   'Q': inputDict['tolP']}
            self.tol = tol

            # Time array
            N = inputDict['N']
            self.N = N
            dt = 1.0/(N-1)
            t = numpy.arange(0,1.0+dt,dt)
            self.dt = dt
            self.t = t

            TargHeig = numpy.array(inputDict['TargHeig'])
            # Additional arcs:
            addArcs = len(TargHeig)
            # Number of arcs:
            s = inputDict['NStag'] + addArcs
            self.s = s

            # TODO: increase flexibility in these conditions

            isStagSep = numpy.ones(s,dtype='bool')
            for i in range(addArcs):
                isStagSep[i] = False

            self.isStagSep = isStagSep

            p = s
            self.p = p
            q = n * (s+1) - 1  #s=1,q=7; s=2,q=11; s=3,q=15
            self.q = q

            x = numpy.zeros((N,n,s))
            u = numpy.zeros((N,m,s))

            self.Ns = 2*n*s + p
            # Payload mass
            self.mPayl = inputDict['Mu']

            # Earth constants
            r_e = inputDict['R']
            GM = inputDict['GM']
            grav_e = GM/r_e/r_e

            # rocket constants
            Thrust = inputDict['T']*numpy.ones(s)
            Isp = inputDict['Isp']*numpy.ones(s)
            s_f = inputDict['efes']*numpy.ones(s)
            CL0 = inputDict['CL0']*numpy.ones(s)
            CL1 = inputDict['CL1']*numpy.ones(s)
            CD0 = inputDict['CD0']*numpy.ones(s)
            CD2 = inputDict['CD2']*numpy.ones(s)
            s_ref = inputDict['s_ref']*numpy.ones(s)
            DampCent = inputDict['DampCent']
            DampSlop = inputDict['DampSlop']
            acc_max = inputDict['acc_max'] * grav_e

            # Penalty function settings

            # This approach to Kpf is that if, in any point during flight, the
            # acceleration exceeds the limit by acc_max_tol, then the penalty
            # function at that point is PFtol times greater than the maximum
            # value of the original cost functional.

            PFmode = inputDict['PFmode']
            costFuncVals = Thrust/grav_e/Isp/(1.0-s_f)
            PFtol = inputDict['PFtol']
            acc_max_relTol = inputDict['acc_max_relTol']
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

            # Gradient step search options
            gradStepSrchCte = inputDict['gradStepSrchCte']

            # boundary conditions
            h_initial = inputDict['h_initial']
            V_initial = inputDict['V_initial']
            gamma_initial = inputDict['gamma_initial']
            h_final = inputDict['h_final']
            V_final = numpy.sqrt(GM/(r_e+h_final))#7.633   # km/s
            gamma_final = inputDict['gamma_final']

            boundary = dict()
            boundary['h_initial'] = h_initial
            boundary['V_initial'] = V_initial
            boundary['gamma_initial'] = gamma_initial
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
            alpha_min = -inputDict['AoAmax'] * d2r  # in rads
            alpha_max = -alpha_min                  # in rads
            beta_min = 0.0
            beta_max = 1.0
            restrictions = dict()
            restrictions['alpha_min'] = alpha_min
            restrictions['alpha_max'] = alpha_max
            restrictions['beta_min'] = beta_min
            restrictions['beta_max'] = beta_max
            restrictions['acc_max'] = acc_max
            self.restrictions = restrictions

            # Find indices for beginning of arc
            arcBginIndx = numpy.empty(s+1,dtype='int')
            arc = 0; arcBginIndx[arc] = 0
            nt = len(t_its)

            # target heights for separation


            j = 0
            for hTarg in TargHeig:
                # search for target height
                keepLook = True
                while (keepLook and (j < nt)):
                    if x_its[j,0] > hTarg:
                        # Found the index!
                        arc += 1
                        arcBginIndx[arc] = j
                        keepLook = False
                    j+=1

            # now, search for jettisoned masses
            nmj = len(massJet)
            for i in range(nmj):

                if massJet[i] > 0.0:
                    # Jettisoned mass found!
                    arc += 1
                    tTarg = tphases[i]

                    keepLook = True
                    while (keepLook and (j < nt)):
                        if abs(t_its[j]-tTarg) < 1e-10:
                            #TODO: this hardcoded 1e-10 may cause problems...

                            # Found the proper time!
                            keepLook = False

                            # get the next time for proper initial conditions
                            j += 1
                            arcBginIndx[arc] = j
                        j += 1
            #
            #self.log.printL(arcBginIndx)

            # Set the array of interval lengths
            pi = numpy.empty(s)
            for arc in range(s):
                pi[arc] = t_its[arcBginIndx[arc+1]] - t_its[arcBginIndx[arc]]

            # get proper initial mass condition
            self.boundary['m_initial'] = x_its[0,3]

            self.log.printL("Re-integrating ITSME solution with fixed " + \
                            "step scheme...")
            # Re-integration of proposed solution (RK4)
            # Only the controls are used, not the integrated state itself
            for arc in range(s):
                # dtd: dimensional time step
                dtd = pi[arc]/(N-1); dtd6 = dtd/6.0
                x[0,:,arc] = x_its[arcBginIndx[arc],:]
                t0arc = t_its[arcBginIndx[arc]]
                uip1 = numpy.array([tabAlpha.value(t0arc),\
                                    tabBeta.value(t0arc)])
                # td: dimensional time (for integration)
                for i in range(N-1):
                    td = t0arc + i * dtd
                    ui = uip1
                    u[i,:,arc] = ui

                    uipm = numpy.array([tabAlpha.value(td+.5*dtd),\
                                        tabBeta.value(td+.5*dtd)])
                    uip1 = numpy.array([tabAlpha.value(td+dtd),\
                                        tabBeta.value(td+dtd)])
                    # this bypass just ensures consistency for control
                    if (i == N-2 and arc == s-1):
                        uip1 = ui

                    f1 = calcXdot(td,x[i,:,arc],ui,constants,arc)
                    tdm = td+.5*dtd # time at half the integration interval
                    x2 = x[i,:,arc] + .5*dtd*f1 # x at half step, with f1
                    f2 = calcXdot(tdm,x2,uipm,constants,arc)
                    x3 = x[i,:,arc] + .5*dtd*f2 # x at half step, with f2
                    f3 = calcXdot(tdm,x3,uipm,constants,arc)
                    x4 = x[i,:,arc] + dtd*f3 # x at next step, with f3
                    f4 = calcXdot(td+dtd,x4,uip1,constants,arc)
                    x[i+1,:,arc] = x[i,:,arc] + dtd6 * (f1+f2+f2+f3+f3+f4)
                #
                u[N-1,:,arc] = u[N-2,:,arc]#numpy.array([tabAlpha.value(pi[0]),tabBeta.value(pi[0])])
            #

        lam = numpy.zeros((N,n,s))
        mu = numpy.zeros(q)

#        # TODO: This hardcoded bypass MUST be corrected in later versions.
#        self.log.printL("\n!!!\nHeavy hardcoded bypass here:")
#        self.log.printL("Control limits are being switched;")
#        self.log.printL("Controls themselves are being 'desaturated'.")
#        self.log.printL("Check code for the values, and be careful!\n")
#
#        self.restrictions['alpha_min'] = -3.0*numpy.pi/180.0
#        self.restrictions['alpha_max'] = 3.0*numpy.pi/180.0
#        ThrustFactor = 2.0#500.0/40.0
#        self.constants['Thrust'] *= ThrustFactor
#        # Re-calculate the Kpf, since it scales with the Thrust...
#        #self.constants['Kpf'] *= ThrustFactor
#        u[:,1,:] *= 1.0/ThrustFactor

        u = self.calcAdimCtrl(u[:,0,:],u[:,1,:])

        self.x = x
        self.u = u
        self.pi = pi
        self.lam = lam
        self.mu = mu

        solInit = self.copy()

# =============================================================================
#        # Desaturation of the controls
#        for arc in range(s):
#            for k in range(N):
#                if u[k,1,arc] < -2.5:
#                    u[k,1,arc] = -2.5
# =============================================================================
        self.u = u

        self.compWith(solInit,'Initial Guess')
        self.plotSol()
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
        if ext_u is None:
            alfa = .5*((alpha_max + alpha_min) + \
                       (alpha_max - alpha_min)*numpy.tanh(self.u[:,0,:]))
            beta = .5*((beta_max + beta_min) +  \
                       (beta_max - beta_min)*numpy.tanh(self.u[:,1,:]))
        else:
            alfa = .5*((alpha_max + alpha_min) + \
                       (alpha_max - alpha_min)*numpy.tanh(ext_u[:,0,:]))
            beta = .5*((beta_max + beta_min) +  \
                       (beta_max - beta_min)*numpy.tanh(ext_u[:,1,:]))

        return alfa,beta

    def calcAdimCtrl(self,alfa,beta):
        """Calculate adimensional control 'u' based on external arrays for
        alpha (ang. of attack) and beta (thrust). """

        Nu = len(alfa)
        s = self.s
        u = numpy.empty((Nu,2,s))

        restrictions = self.restrictions
        alpha_min = restrictions['alpha_min']
        alpha_max = restrictions['alpha_max']
        beta_min = restrictions['beta_min']
        beta_max = restrictions['beta_max']

        a1 = .5*(alpha_max + alpha_min)
        a2 = .5*(alpha_max - alpha_min)
        b1 = .5*(beta_max + beta_min)
        b2 = .5*(beta_max - beta_min)

        alfa -= a1
        alfa *= 1.0/a2

        beta -= b1
        beta *= 1.0/b2

        u[:,0,:] = alfa.copy()
        u[:,1,:] = beta.copy()

        # Basic saturation
        for arc in range(s):
            for j in range(2):
                for k in range(Nu):
                    if u[k,j,arc] > 0.99999:
                        u[k,j,arc] = 0.99999
                    if u[k,j,arc] < -0.99999:
                        u[k,j,arc] = -0.99999

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

        sin = numpy.sin
        cos = numpy.cos

        alpha,beta = self.calcDimCtrl()
        x = self.x
        pi = self.pi

        # calculate variables CL and CD
        CL = numpy.empty_like(alpha)
        CD = numpy.empty_like(alpha)
        for arc in range(s):
            CL[:,arc] = CL0[arc] + CL1[arc]*alpha[:,arc]
            CD[:,arc] = CD0[arc] + CD2[arc]*(alpha[:,arc]**2)

        # calculate L and D
        # TODO: making atmosphere.rho vectorized (array compatible) would
        # increase performance significantly!

        dens = numpy.empty((N,s))
        for arc in range(s):
            for k in range(N):
                dens[k,arc] = rho(x[k,0,arc])

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
            phi[:,1,arc] = (beta[:,arc] * Thrust[arc] * cosAlfa[:,arc] \
                            - D[:,arc])/x[:,3,arc] \
                           - grav[:,arc] * sinGama[:,arc]
            phi[:,2,arc] = ((beta[:,arc] * Thrust[arc] * sinAlfa[:,arc] \
                            + L[:,arc])/(x[:,3,arc] * x[:,1,arc]) + \
                            cosGama[:,arc] * ( x[:,1,arc]/r[:,arc] \
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
        Grads = dict()

        # Pre-assign functions
        sin = numpy.sin
        cos = numpy.cos
        tanh = numpy.tanh

        # Load constants
        N,n,m,p,q,s = self.N,self.n,self.m,self.p,self.q,self.s
        constants = self.constants

        grav_e = constants['grav_e']
        MaxThrs = constants['Thrust']
        Isp = constants['Isp']
        g0Isp = Isp * grav_e
        r_e = constants['r_e']
        GM = constants['GM']
        s_f = constants['s_f']
        CL0 = constants['CL0']
        CL1 = constants['CL1']
        CD0 = constants['CD0']
        CD2 = constants['CD2']
        s_ref = constants['s_ref']
        DampCent = constants['DampCent']
        DampSlop = constants['DampSlop']
        Kpf = constants['Kpf']
        PFmode = constants['PFmode']

        restrictions = self.restrictions
        alpha_min = restrictions['alpha_min']
        alpha_max = restrictions['alpha_max']
        beta_min = restrictions['beta_min']
        beta_max = restrictions['beta_max']
        acc_max = restrictions['acc_max']

        # Load states, controls
        u1 = self.u[:,0,:]
        u2 = self.u[:,1,:]
        tanhU1 = tanh(u1)
        tanhU2 = tanh(u2)

        pi = self.pi
        acc = self.calcAcc()

        phix = numpy.zeros((N,n,n,s))
        phiu = numpy.zeros((N,n,m,s))

        if p>0:
            phip = numpy.zeros((N,n,p,s))
        else:
            phip = numpy.zeros((N,n,1,s))

        fx = numpy.zeros((N,n,s))
        fu = numpy.zeros((N,m,s))
        fOrig_u = numpy.zeros((N,m,s))
        fPF_u = numpy.empty_like(fu)
        fp = numpy.zeros((N,p,s))

        ## Psi derivatives

        # For reference:
        # y = [x[0,:,0],\
        #      x[N-1,:,0],\
        #      x[0,:,1],\
        #      x[N-1,:,0],\
        #       ...,\
        #      x[0,:,s-1],
        #      x[N-1,:,s-1]]
        psiy = numpy.zeros((q,2*n*s))
        s_f = self.constants['s_f']

        # First n rows: all states have assigned values
        for ind in range(n):
            psiy[ind,ind] = 1.0

        # Last n-1 rows (no mass eq for end condition of final arc):
        for ind in range(n-1):
            psiy[q-1-ind,2*n*s-2-ind] = 1.0

        # Intermediate conditions
        i0 = n; j0 = 0
        for arc in range(s-1):
            # This loop sets the interfacing conditions between all states
            # in 'arc' and 'arc+1' (that's why it only goes up to s-1)

            # For height, speed and angle:
            for stt in range(n-1):
                ind = i0 + stt
                psiy[ind,j0+ind] = -1.0  # this state, this arc  (end cond)
                psiy[ind,j0+ind+n] = 1.0 # this state, next arc (init cond)

            # For mass:
            stt = n-1; ind = i0 + stt
            if self.isStagSep[arc]:
                # mass, next arc (init cond)
                psiy[ind,j0+ind+n] = 1.0
                # mass, this arc (end cond):
                psiy[ind,j0+ind] = -1.0/(1.0-s_f[arc])
                # mass, this arc (init cond):
                psiy[ind,j0+ind-n] = s_f[arc]/(1.0-s_f[arc])
            else:
                psiy[ind,j0+ind] = -1.0  # mass, this arc  (end cond)
                psiy[ind,j0+ind+n] = 1.0 # mass, next arc (init cond)

            i0 = ind + 1
            j0 += n
        #

        psip = numpy.zeros((q,p))

        # calculate r, V, etc
        r = r_e + self.x[:,0,:]
        r2 = r * r
        r3 = r2 * r
        V = self.x[:,1,:]
        V2 = V * V
        # GAMMA FACTOR (fdg)
        fdg = numpy.empty_like(r)
        t0 = 0.0
        for arc in range(s):
            t = t0 + self.t * pi[arc]
            fdg[:,arc] = .5*(1.0+tanh(DampSlop*(t-DampCent)))
            t0 += pi[arc]

        m = self.x[:,3,:]
        m2 = m * m
        sinGama, cosGama = sin(self.x[:,2,:]), cos(self.x[:,2,:])

        # Calculate variables (arrays) alpha and beta

        # TODO: change calcDimCtrl so that the derivatives
        # DAlfaDu1 and DBetaDu2 are also calculated there...
        # (with an optional command so that these are not calculated all the
        # time, of course!)
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
            for k in range(N):
                dens[k,arc] = rho(self.x[k,0,arc])
                del_rho[k,arc] = (rho(self.x[k,0,arc]+.05) - dens[k,arc])/.05

#        rhoS = numpy.empty((N,s)) # density times ref. area!
#        del_rhoS = numpy.empty((N,s))
#        for arc in range(s):
#            for k in range(N):
#                rhoS[k,arc] = rho(self.x[k,0,arc])
#                del_rhoS[k,arc] = (rho(self.x[k,0,arc]+.1) - dens[k,arc])/.1


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
            fx[:,0,:] = K2dAPen * (2.0 * GM * sinGama / r3 - \
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
            phiu[:,1,0,arc] = -(bTm[:,arc] * sinAlpha[:,arc] + \
            dens[:,arc] * V2[:,arc] * s_ref[arc] * CD2[arc] * alpha[:,arc]) * \
            DAlfaDu1[:,arc] / m[:,arc]
        # d vdot d u2:
            phiu[:,1,1,arc] = MaxThrs[arc] * cosAlpha[:,arc] * \
            DBetaDu2[:,arc] / m[:,arc]

        # d gamadot d u1:
            phiu[:,2,0,arc] = ( thrust[:,arc] * cosAlpha[:,arc] / V[:,arc] + \
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

        Grads['phix'] = phix
        Grads['phiu'] = phiu
        Grads['phip'] = phip
        Grads['fx'] = fx
        Grads['fu'] = fu
        Grads['fp'] = fp
    #    Grads['gx'] = gx
    #    Grads['gp'] = gp
        Grads['psiy'] = psiy
        Grads['psip'] = psip
        return Grads

#%%
    def calcPsi(self):
        self.log.printL("In calcPsi.")
        boundary = self.boundary
        s_f = self.constants['s_f']
        x = self.x
        N,q,s = self.N,self.q,self.s
        psi = numpy.empty(q)

        # Beginning of first subarc
        psi[0] = x[0,0,0] - boundary['h_initial']
        psi[1] = x[0,1,0] - boundary['V_initial']
        psi[2] = x[0,2,0] - boundary['gamma_initial']
        psi[3] = x[0,3,0] - boundary['m_initial']

        # interstage conditions between arc and arc+1
        # (that's why the loop only goes up to s-1)
        #strPrnt = "0,1,2,3,"
        for arc in range(s-1):
            i0 = 4 * (arc+1)

            # four states in order: position, speed, flight angle and mass
            psi[i0]   = x[0,0,arc+1] - x[N-1,0,arc]
            psi[i0+1] = x[0,1,arc+1] - x[N-1,1,arc]
            psi[i0+2] = x[0,2,arc+1] - x[N-1,2,arc]
            if self.isStagSep[arc]:
                psi[i0+3] = x[0,3,arc+1] - (1.0/(1.0 - s_f[arc-1])) * \
                            (x[N-1,3,arc] - s_f[arc-1] * x[0,3,arc])
            else:
                psi[i0+3] = x[0,3,arc+1] - x[N-1,3,arc]
            #strPrnt += str(i0)+","+str(i0+1)+","+str(i0+2)+","+str(i0+3)+","
        # End of final subarc
        psi[q-3] = x[N-1,0,s-1] - boundary['h_final']
        psi[q-2] = x[N-1,1,s-1] - boundary['V_final']
        psi[q-1] = x[N-1,2,s-1] - boundary['gamma_final']
        #strPrnt += str(q-3)+","+str(q-2)+","+str(q-1)

        self.log.printL("Psi = "+str(psi))

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
#            fPF[:,arc] = self.pi[arc] * Kpf * \
#                .5 * (1.0 + numpy.tanh(500.0*(acc[:,arc]/acc_max-1.001)) )
#0.5*(1.0+numpy.tanh((a-1- 0.001)*500.0 ))

        # Apply penalty only when acc > acc_max
        fPF *= (acc > acc_max)
        f = fOrig + fPF

        return f,fOrig,fPF

    def calcI(self):
        N,s = self.N,self.s
        _,fOrig,fPF = self.calcF()

        IvecOrig = numpy.empty(s)
        IvecPF = numpy.empty(s)

        for arc in range(s):
            IvecOrig[arc] = .5 * ( fOrig[0,arc] + fOrig[N-1,arc] )
            IvecOrig[arc] += fOrig[1:(N-1),arc].sum()
            IvecPF[arc] = .5 * ( fPF[0,arc] + fPF[N-1,arc] )
            IvecPF[arc] += fPF[1:(N-1),arc].sum()

        IvecOrig *= 1.0/(N-1)
        IvecPF *= 1.0/(N-1)
        Iorig = IvecOrig.sum()
        Ipf = IvecPF.sum()
        return Iorig+Ipf, Iorig, Ipf
#%% Plotting commands and related functions

    def calcPsblPayl(self):
        """Calculate the possible ammount of payload for a given configuration.


        This ammount equals the total rocket mass at liftoff, minus all the
        used propellant mass and the associated structural mass. All the
        remaining mass could be replaced by payload at liftoff!"""

        MTot = self.x[0,3,0] # rocket mass at liftoff
        s_f = self.constants['s_f']
        for arc in range(self.s):
            M0 = self.x[0,3,arc]
            Mf = self.x[-1,3,arc]
            MTot -= (M0-Mf) / (1.0 - s_f[arc])

        return MTot

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


    def plotSol(self,opt={},intv=[],piIsTime=True,mustSaveFig=True,\
                subPlotAdjs={'left':0.0,'right':1.0,'bottom':0.0,
                             'top':5.0,'wspace':0.2,'hspace':0.5}):
        # plt.subplots_adjust(left=0.0,right=1,bottom=0.0,top=5,\
        #                        wspace=0.2,hspace=0.5)
        self.log.printL("\nIn plotSol.")
        x = self.x
        u = self.u
        pi = self.pi
        r2d = 180.0/numpy.pi


        if opt.get('mode','sol') == 'sol':
            I,Iorig,Ipf = self.calcI()

            self.log.printL("Initial mass: " + str(x[0,3,0]))
            self.log.printL("I: "+ str(I))
            self.log.printL("Design payload mass: " + str(self.mPayl))
            #mFinl = x[0,3,0] - I/self.constants['costScalingFactor']
            mFinl = self.calcPsblPayl()
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
            titlStr += "Payload mass gain: {:.4G}%\n".format(paylPercMassGain)
            titlStr += "Losses (w.r.t. ideal Delta v): "+ \
                       "{:.4G}%".format(dvLossPerc)

            plt.subplots_adjust(**subPlotAdjs)

            plt.subplot2grid((11,1),(0,0))
            self.plotCat(x[:,0,:],piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("h [km]")
            plt.title(titlStr)
            if piIsTime:
                plt.xlabel("t [s]")

            plt.subplot2grid((11,1),(1,0))
            self.plotCat(x[:,1,:],color='g',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("V [km/s]")
            if piIsTime:
                plt.xlabel("t [s]")

            plt.subplot2grid((11,1),(2,0))
            self.plotCat(x[:,2,:]*r2d,color='r',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("gamma [deg]")
            if piIsTime:
                plt.xlabel("t [s]")

            plt.subplot2grid((11,1),(3,0))
            self.plotCat(x[:,3,:],color='m',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("m [kg]")
            if piIsTime:
                plt.xlabel("t [s]")

            plt.subplot2grid((11,1),(4,0))
            self.plotCat(u[:,0,:],color='c',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("u1 [-]")
            if piIsTime:
                plt.xlabel("t [s]")

            plt.subplot2grid((11,1),(5,0))
            self.plotCat(u[:,1,:],color='c',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("u2 [-]")
            if piIsTime:
                plt.xlabel("t [s]")

            ######################################
            alpha,beta = self.calcDimCtrl()
            alpha *= r2d
            plt.subplot2grid((11,1),(6,0))
            self.plotCat(alpha,piIsTime=piIsTime,intv=intv,color='k')
            #plt.hold(True)
            #plt.plot(t,alpha*0+alpha_max*180/numpy.pi,'-.k')
            #plt.plot(t,alpha*0+alpha_min*180/numpy.pi,'-.k')
            plt.grid(True)
            #plt.xlabel("t")
            plt.ylabel("alpha [deg]")
            if piIsTime:
                plt.xlabel("t [s]")

            plt.subplot2grid((11,1),(7,0))
            self.plotCat(beta,piIsTime=piIsTime,intv=intv,color='k')
            #plt.hold(True)
            #plt.plot(t,beta*0+beta_max,'-.k')
            #plt.plot(t,beta*0+beta_min,'-.k')
            plt.grid(True)
            plt.ylabel("beta [-]")
            if piIsTime:
                plt.xlabel("t [s]")
            ######################################

            ######################################
            plt.subplot2grid((11,1),(8,0))
            thrust = numpy.empty_like(beta)
            s = self.s
            constants = self.constants
            MaxThrs = constants['Thrust']
            for arc in range(s):
                thrust[:,arc] = beta[:,arc] * MaxThrs[arc]
            self.plotCat(thrust,color='y',piIsTime=piIsTime)
            plt.grid(True)
            if piIsTime:
                plt.xlabel("t [s]")
            plt.ylabel("Thrust [kN]")

            ######################################
            plt.subplot2grid((11,1),(9,0))
            acc =  self.calcAcc()
            self.plotCat(acc*1e3,color='y',piIsTime=piIsTime,labl='Accel.')
            if piIsTime:
                plt.plot([0.0,self.pi.sum()],\
                          1e3 * self.restrictions['acc_max'] * \
                          numpy.array([1.0,1.0]),'--')
            else:
                plt.plot([0.0,self.s+1],\
                          1e3 * self.restrictions['acc_max'] * \
                          numpy.array([1.0,1.0]),'--')

            plt.grid(True)
            if piIsTime:
                plt.xlabel("t [s]")
            plt.ylabel("Tang. accel. [m/s²]")

            ######################################
            ax = plt.subplot2grid((11,1),(10,0))
            position = numpy.arange(s)
            stages = numpy.arange(1,s+1)
            width = 0.4
            ax.bar(position,pi,width,color='b')
            ax.set_xticks(position + width/2)
            ax.set_xticklabels(stages)
            plt.grid(True)
            plt.xlabel("Arcs")
            plt.ylabel("Duration [s]")

#            xlabl = "t [s]\n"+"pi = ["
#            for i in range(self.p):
#                xlabl += "{:.4E}, ".format(pi[i])
#            xlabl += "]"
#            plt.xlabel(xlabl)

            if mustSaveFig:
                self.savefig(keyName='currSol',fullName='solution')
            else:
                plt.show()
                plt.clf()

            self.log.printL("Final (injected into orbit) rocket mass: " + \
                  "{:.4E}\n".format(x[-1,3,self.s-1]))
            EjctMass = list()
            # get ejected masses:
            for arc in range(self.s-1):
                if self.isStagSep[arc]:
                    EjctMass.append(x[-1,3,arc]-x[0,3,arc+1])
            self.log.printL("Ejected masses: " + str(EjctMass))

        elif opt['mode'] == 'lambda':
            titlStr = "Lambdas (grad iter #" + str(self.NIterGrad+1) + ")"

            plt.subplots_adjust(**subPlotAdjs)

            plt.subplot2grid((8,1),(0,0))
            self.plotCat(self.lam[:,0,:],piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("lam - h")
            if piIsTime:
                plt.xlabel("t [s]")
            plt.title(titlStr)

            plt.subplot2grid((8,1),(1,0))
            self.plotCat(self.lam[:,1,:],color='g',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("lam - V")
            if piIsTime:
                plt.xlabel("t [s]")

            plt.subplot2grid((8,1),(2,0))
            self.plotCat(self.lam[:,2,:],color='r',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("lam - gamma")
            if piIsTime:
                plt.xlabel("t [s]")

            plt.subplot2grid((8,1),(3,0))
            self.plotCat(self.lam[:,3,:],color='m',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("lam - m")
            if piIsTime:
                plt.xlabel("t [s]")

            plt.subplot2grid((8,1),(4,0))
            self.plotCat(u[:,0,:],color='c',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("u1 [-]")
            if piIsTime:
                plt.xlabel("t [s]")

            plt.subplot2grid((8,1),(5,0))
            self.plotCat(u[:,1,:],color='c',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            #plt.xlabel("t")
            plt.ylabel("u2 [-]")
            if piIsTime:
                plt.xlabel("t [s]")

            ######################################
            alpha,beta = self.calcDimCtrl()
            alpha *= r2d
            plt.subplot2grid((8,1),(6,0))
            self.plotCat(alpha,piIsTime=piIsTime,intv=intv)
            #plt.hold(True)
            #plt.plot(t,alpha*0+alpha_max*180/numpy.pi,'-.k')
            #plt.plot(t,alpha*0+alpha_min*180/numpy.pi,'-.k')
            plt.grid(True)
            if piIsTime:
                plt.xlabel("t [s]")
            plt.ylabel("alpha [deg]")

            plt.subplot2grid((8,1),(7,0))
            self.plotCat(beta,piIsTime=piIsTime,intv=intv)
            #plt.hold(True)
            #plt.plot(t,beta*0+beta_max,'-.k')
            #plt.plot(t,beta*0+beta_min,'-.k')
            plt.grid(True)
            if piIsTime:
                plt.xlabel("t [s]")
            plt.ylabel("beta [-]")
            ######################################

            if mustSaveFig:
                self.savefig(keyName='currLamb',fullName='lambdas')
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

            plt.subplots_adjust(**subPlotAdjs)

            plt.subplot2grid((8,1),(0,0))
            self.plotCat(dx[:,0,:],piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("h [km]")
            plt.title(titlStr)
            if piIsTime:
                plt.xlabel("t [s]")

            plt.subplot2grid((8,1),(1,0))
            self.plotCat(dx[:,1,:],color='g',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("V [km/s]")
            if piIsTime:
                plt.xlabel("t [s]")

            plt.subplot2grid((8,1),(2,0))
            self.plotCat(dx[:,2,:]*r2d,color='r',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("gamma [deg]")
            if piIsTime:
                plt.xlabel("t [s]")

            plt.subplot2grid((8,1),(3,0))
            self.plotCat(dx[:,3,:],color='m',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("m [kg]")
            if piIsTime:
                plt.xlabel("t [s]")

            plt.subplot2grid((8,1),(4,0))
            self.plotCat(du[:,0,:],color='c',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            plt.ylabel("u1 [-]")
            if piIsTime:
                plt.xlabel("t [s]")

            plt.subplot2grid((8,1),(5,0))
            self.plotCat(du[:,1,:],color='c',piIsTime=piIsTime,intv=intv)
            plt.grid(True)
            #plt.xlabel("t")
            plt.ylabel("u2 [-]")
            if piIsTime:
                plt.xlabel("t [s]")

            ######################################
            new_u = self.u + du
            alpha,beta = self.calcDimCtrl()
            alpha *= r2d
            new_alpha,new_beta = self.calcDimCtrl(ext_u=new_u)
            new_alpha *= r2d
            plt.subplot2grid((8,1),(6,0))
            self.plotCat(new_alpha-alpha,color='k',piIsTime=piIsTime,intv=intv)
            #plt.hold(True)
            #plt.plot(t,alpha*0+alpha_max*180/numpy.pi,'-.k')
            #plt.plot(t,alpha*0+alpha_min*180/numpy.pi,'-.k')
            plt.grid(True)
            #plt.xlabel("t")
            plt.ylabel("alpha [deg]")
            if piIsTime:
                plt.xlabel("t [s]")

            plt.subplot2grid((8,1),(7,0))
            self.plotCat(new_beta-beta,color='k',piIsTime=piIsTime,intv=intv)
            #plt.hold(True)
            #plt.plot(t,beta*0+beta_max,'-.k')
            #plt.plot(t,beta*0+beta_min,'-.k')
            plt.grid(True)
            plt.ylabel("beta [-]")
            if piIsTime:
                plt.xlabel("t [s]")

            ######################################

            # TODO: include a plot for visualization of pi!
            if mustSaveFig:
                self.savefig(keyName='corr',fullName='corrections')
            else:
                plt.show()
                plt.clf()
        else:
            titlStr = opt['mode']
        #

    #

    def plotQRes(self,args,mustSaveFig=True):
        "Plots of the Q residuals, specifically for the probRock case."

        # Qx error plot
        plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
        plt.subplot2grid((5,1),(0,0))
        self.plotCat(args['normErrQx'],color='b',piIsTime=False)
        plt.grid(True)
        plt.ylabel("Integrand of Qx")
        titlStr = "Qx = int || dlam - f_x + phi_x^T*lam || " + \
                  "= {:.4E}".format(args['Qx'])
        titlStr += "\n(grad iter #" + str(self.NIterGrad+1) + ")"
        plt.title(titlStr)
        errQx = args['errQx']

        plt.subplot2grid((5,1),(1,0))
        self.plotCat(errQx[:,0,:],piIsTime=False)
        plt.grid(True)
        plt.ylabel("ErrQx_h")

        plt.subplot2grid((5,1),(2,0))
        self.plotCat(errQx[:,1,:],color='g',piIsTime=False)
        plt.grid(True)
        plt.ylabel("ErrQx_v")

        plt.subplot2grid((5,1),(3,0))
        self.plotCat(errQx[:,2,:],color='r',piIsTime=False)
        plt.grid(True)
        plt.ylabel("ErrQx_gama")

        plt.subplot2grid((5,1),(4,0))
        self.plotCat(errQx[:,3,:],color='m',piIsTime=False)
        plt.grid(True)
        plt.ylabel("ErrQx_m")

        plt.xlabel("t [s]")
        if mustSaveFig:
            self.savefig(keyName='Qx',fullName='Qx')
        else:
            plt.show()
            plt.clf()

        # Qu error plot
        plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
        plt.subplot2grid((3,1),(0,0))
        self.plotCat(args['normErrQu'],color='b',piIsTime=False)
        plt.grid(True)
        plt.ylabel("Integrand of Qu")
        titlStr = "Qu = int || f_u - phi_u^T*lam || = " + \
                  "{:.4E}".format(args['Qu']) + \
                  "\n(grad iter #" + str(self.NIterGrad+1) + ")"
        plt.title(titlStr)

        errQu = args['errQu']
        plt.subplot2grid((3,1),(1,0))
        self.plotCat(errQu[:,0,:],color='k',piIsTime=False)
        plt.grid(True)
        plt.ylabel("Qu_alpha")
        plt.subplot2grid((3,1),(2,0))
        self.plotCat(errQu[:,1,:],color='r',piIsTime=False)
        plt.grid(True)
        plt.ylabel("Qu_beta")

        plt.xlabel("t")
        if mustSaveFig:
            self.savefig(keyName='Qu',fullName='Qu')
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
        plt.xlabel("t [s]")
        if mustSaveFig:
            self.savefig(keyName='Qp',fullName='Qp')
        else:
            plt.show()
            plt.clf()


    def compWith(self,altSol,altSolLabl='altSol',mustSaveFig=True,\
                 subPlotAdjs={'left':0.0,'right':1.0,'bottom':0.0,
                             'top':5.0,'wspace':0.2,'hspace':0.5}):
        self.log.printL("\nComparing solutions...\n")
        r2d = 180.0/numpy.pi
        currSolLabl = 'currentSol'

        # Comparing final mass:
        mPaySol = self.calcPsblPayl()
        mPayAlt = altSol.calcPsblPayl()

        paylMassGain = mPaySol - mPayAlt
        paylPercMassGain = 100.0*paylMassGain/mPayAlt

        # Plotting the curves
        plt.subplots_adjust(**subPlotAdjs)

        plt.subplot2grid((11,1),(0,0))
        altSol.plotCat(altSol.x[:,0,:],labl=altSolLabl)
        self.plotCat(self.x[:,0,:],mark='--',color='y',labl=currSolLabl)
        plt.grid(True)
        plt.ylabel("h [km]")
        plt.legend(loc="upper left", bbox_to_anchor=(1,1))
        titlStr = "Comparing solutions: " + currSolLabl + " and " + \
                  altSolLabl+\
                  "\nPayload mass gain: {:.4G}%".format(paylPercMassGain)
        titlStr += "\n(grad iter #" + str(self.NIterGrad) + ")"
        plt.title(titlStr)
        plt.xlabel("t [s]")

        plt.subplot2grid((11,1),(1,0))
        altSol.plotCat(altSol.x[:,1,:],labl=altSolLabl)
        self.plotCat(self.x[:,1,:],mark='--',color='g',labl=currSolLabl)
        plt.grid(True)
        plt.ylabel("V [km/s]")
        plt.xlabel("t [s]")
        plt.legend(loc="upper left", bbox_to_anchor=(1,1))

        plt.subplot2grid((11,1),(2,0))
        altSol.plotCat(altSol.x[:,2,:]*r2d,labl=altSolLabl)
        self.plotCat(self.x[:,2,:]*180/numpy.pi,mark='--',color='r',\
                     labl=currSolLabl)
        plt.grid(True)
        plt.ylabel("gamma [deg]")
        plt.xlabel("t [s]")
        plt.legend(loc="upper left", bbox_to_anchor=(1,1))

        plt.subplot2grid((11,1),(3,0))
        altSol.plotCat(altSol.x[:,3,:],labl=altSolLabl)
        self.plotCat(self.x[:,3,:],mark='--',color='m',labl=currSolLabl)
        plt.grid(True)
        plt.ylabel("m [kg]")
        plt.xlabel("t [s]")
        plt.legend(loc="upper left", bbox_to_anchor=(1,1))

        plt.subplot2grid((11,1),(4,0))
        altSol.plotCat(altSol.u[:,0,:],labl=altSolLabl)
        self.plotCat(self.u[:,0,:],mark='--',color='c',labl=currSolLabl)
        plt.grid(True)
        plt.ylabel("u1 [-]")
        plt.xlabel("t [s]")
        plt.legend(loc="upper left", bbox_to_anchor=(1,1))

        plt.subplot2grid((11,1),(5,0))
        altSol.plotCat(altSol.u[:,1,:],labl=altSolLabl)
        self.plotCat(self.u[:,1,:],mark='--',color='c',labl=currSolLabl)
        plt.grid(True)
        plt.xlabel("t")
        plt.ylabel("u2 [-]")
        plt.xlabel("t [s]")
        plt.legend(loc="upper left", bbox_to_anchor=(1,1))

        ######################################
        alpha,beta = self.calcDimCtrl()
        alpha_alt,beta_alt = altSol.calcDimCtrl()
        plt.subplot2grid((11,1),(6,0))
        altSol.plotCat(alpha_alt*r2d,labl=altSolLabl)
        self.plotCat(alpha*r2d,mark='--',color='k',labl=currSolLabl)

        #plt.hold(True)
        #plt.plot(t,alpha*0+alpha_max*180/numpy.pi,'-.k')
        #plt.plot(t,alpha*0+alpha_min*180/numpy.pi,'-.k')
        plt.grid(True)
        plt.xlabel("t")
        plt.ylabel("alpha [deg]")
        plt.xlabel("t [s]")
        plt.legend(loc="upper left", bbox_to_anchor=(1,1))

        plt.subplot2grid((11,1),(7,0))
        altSol.plotCat(beta_alt,labl=altSolLabl)
        self.plotCat(beta,mark='--',color='k',labl=currSolLabl)

        #plt.hold(True)
        #plt.plot(t,beta*0+beta_max,'-.k')
        #plt.plot(t,beta*0+beta_min,'-.k')
        plt.grid(True)
        plt.xlabel("t [s]")
        plt.ylabel("beta [-]")
        plt.legend(loc="upper left", bbox_to_anchor=(1,1))
        ######################################

        ######################################
        plt.subplot2grid((11,1),(8,0))
        thrust = numpy.empty_like(beta)
        thrust_alt = numpy.empty_like(beta_alt)
        s = self.s
        s_alt = altSol.s
        constants = self.constants
        MaxThrs = constants['Thrust']
        for arc in range(s):
            thrust[:,arc] = beta[:,arc] * MaxThrs[arc]
            thrust_alt[:,arc] = beta_alt[:,arc] * MaxThrs[arc]
        altSol.plotCat(thrust_alt,labl=altSolLabl)
        self.plotCat(thrust,mark='--',color='y',labl=currSolLabl)
        plt.grid(True)
        plt.xlabel("t [s]")
        plt.ylabel("Thrust [kN]")
        plt.legend(loc="upper left", bbox_to_anchor=(1,1))

        plt.subplot2grid((11,1),(9,0))
        solAcc =  self.calcAcc()
        altSolAcc = altSol.calcAcc()
        plt.plot([0.0,self.pi.sum()],\
                  1e3*self.restrictions['acc_max']*numpy.array([1.0,1.0]),\
                  '--',label='accel. limit')
        altSol.plotCat(1e3*altSolAcc,labl=altSolLabl)
        self.plotCat(1e3*solAcc,mark='--',color='y',labl=currSolLabl)
        plt.grid(True)
        plt.xlabel("t [s]")
        plt.ylabel("Tang. accel. [m/s²]")
        plt.legend(loc="upper left", bbox_to_anchor=(1,1))

        ######################################
        ax = plt.subplot2grid((11,1),(10,0))
        position = numpy.arange(s)
        position_alt = numpy.arange(s_alt)
        stages = numpy.arange(1,s+1)
        width = 0.4
        current = ax.bar(position,self.pi,width,color='y')
        initial = ax.bar(position_alt + width,altSol.pi,width)
        ax.set_xticks(position + width)
        ax.set_xticklabels(stages)
        plt.xlabel("Stages")
        plt.ylabel("Duration [s]")
        ax.legend((current[0], initial[0]), (currSolLabl,altSolLabl), \
                  loc="upper left", bbox_to_anchor=(1,1))

        if mustSaveFig:
            self.savefig(keyName='comp',fullName='comparisons')
        else:
            plt.show()
            plt.clf()

        self.log.printL('Final rocket "payload":')
        self.log.printL(currSolLabl+": {:.4E}".format(mPaySol)+" kg.")
        self.log.printL(altSolLabl+": {:.4E}".format(mPayAlt)+" kg.")
        self.log.printL("Difference: {:.4E}".format(paylMassGain)+" kg, "+\
              "{:.4G}".format(paylPercMassGain)+\
              "% more payload!\n")


    def plotTraj(self,compare=False, altSol=None, altSolLabl='altSol', \
                 mustSaveFig=True):
        self.log.printL("\nIn plotTraj!")
        cos = numpy.cos; sin = numpy.sin
        R = self.constants['r_e']
        N, s = self.N, self.s

        # Density for all times/arcs
        dens = numpy.empty((N,s))
        X = numpy.zeros(N*s); Z = numpy.zeros(N*s)
        StgSepPnts = numpy.zeros((s,2))
#        StgInitAcc = numpy.zeros(s)
#        StgFinlAcc = numpy.zeros(s)

        sigma = 0.0 #sigma: range angle
        X[0] = 0.0
        Z[0] = 0.0

        if compare:
            if altSol is None:
                self.log.printL("plotTraj: comparing mode is set to True," + \
                                " but no solution was given to which " + \
                                "compare. Ignoring...")
                compare=False
            else:
                X_alt = X; Z_alt = Z;
                sigma_alt = 0.0
                X_alt[0] = 0.0
                Z_alt[0] = 0.0

        # Propulsive phases' starting and ending times
        # This is implemented with two lists, one for each arc.
        # Each list stores the indices of the points in which the burning
        # begins or ends, respectively.
        isBurn = True
        indBurn = list(); indShut = list()

        indBurn.append(0) # The rocket definitely begins burning at liftoff!
        iCont = 0 # continuous counter (all arcs concatenated)
        strtInd = 1
        for arc in range(s):
            dtd = self.dt * self.pi[arc] # Dimensional dt...
            if compare:
                dtd_alt = altSol.dt * altSol.pi[arc]

            for i in range(strtInd,N):
                dens[i,arc] = rho(self.x[i,0,arc])

                iCont += 1

                if isBurn:
                    if self.u[i,1,arc] < -2.4:#.999:
                        isBurn = False
                        indShut.append(iCont)
                else: #not burning
                    if self.u[i,1,arc] > -2.4:#-.999:
                        isBurn = True
                        indBurn.append(iCont)

                # Propagate the trajectory by Euler method.
                v = self.x[i,1,arc]
                gama = self.x[i,2,arc]
                dsigma = v * cos(gama) / (R + self.x[i,0,arc])
                sigma += dsigma * dtd

                X[iCont] = X[iCont-1] + dtd * v * cos(gama-sigma)
                Z[iCont] = Z[iCont-1] + dtd * v * sin(gama-sigma)


                if compare:
                    v_alt = altSol.x[i,1,arc]
                    gama_alt = altSol.x[i,2,arc]
                    dsigma_alt = v_alt * cos(gama_alt)/(R + altSol.x[i,0,arc])
                    sigma_alt += dsigma_alt * dtd_alt
                    X_alt[iCont] = X_alt[iCont-1] + dtd_alt * v_alt\
                                         * cos(gama_alt-sigma_alt)
                    Z_alt[iCont] = Z_alt[iCont-1] + dtd_alt * v_alt\
                                         * sin(gama_alt-sigma_alt)

            #
            # TODO: this must be readequated to new formulation, where arcs
            # might not be the stage separation poins necessarily
            StgSepPnts[arc,:] = X[iCont],Z[iCont]
            #strtInd = 0
        #
        # The rocket most definitely ends the trajectory with engine shutdown.
        indShut.append(iCont)

        # Remaining points are unused; it is best to repeat the final point
        X[iCont+1 :] = X[iCont]
        Z[iCont+1 :] = Z[iCont]

        # Calculate dynamic pressure, get point of max pdyn
        pDyn = .5 * dens * (self.x[:,1,:]**2)
        indPdynMax = numpy.argmax(pDyn)
        pairIndPdynMax = numpy.unravel_index(indPdynMax,(N,s))
        pDynMax = pDyn[pairIndPdynMax]

        #pairIndPdynMax = numpy.unravel_index(indPdynMax,(N,s))
        #self.log.printL(indPdynMax)
        #self.log.printL("t @ max q (relative to arc):",self.t[pairIndPdynMax[0]])
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
        x = R * cos(.5*numpy.pi - sigVec)
        z = R * (sin(.5*numpy.pi - sigVec) - 1.0)
        plt.plot(x,z,'k',label='Earth surface')

        # Get final orbit parameters
        h,v,gama,M = self.x[N-1,:,s-1]

        self.log.printL("State @burnout time:")
        self.log.printL("h = {:.4E}".format(h)+", v = {:.4E}".format(v)+\
        ", gama = {:.4E}".format(gama)+", m = {:.4E}".format(M))

        GM = self.constants['GM']
        r = R + h
        cosGama, sinGama = cos(gama), sin(gama)
        momAng = r * v * cosGama
        # specific mechanic energy
        en = .5 * v * v - GM / r
        # "Semi-major axis"
        a = - .5 * GM / en
        # Eccentricity
        aux = v * momAng / GM
        e = numpy.sqrt((aux * cosGama - 1.0)**2 + (aux * sinGama)**2)
        # True anomaly
        eccExpr = v * momAng * cosGama / GM - 1.0
        f = numpy.arccos(eccExpr/e)
        ph = a * (1.0 - e) - R
        self.log.printL("Perigee altitude: " + str(ph))
        ah = 2*(a - R) - ph
        self.log.printL("Apogee altitude: " + str(ah))

        # semi-latus rectum
        p = momAng**2 / GM #a * (1.0-e)**2


        # Plot orbit in green over the same range as the Earth shown
        # (and a little bit futher)

        sigVec = numpy.arange(f-1.2*sigma,f+.2*sigma,.01)
        # shifting angle
        sh = sigma - f - .5*numpy.pi
        rOrb = p/(1.0+e*cos(sigVec))
        xOrb = rOrb * cos(-sigVec-sh)
        yOrb = rOrb * sin(-sigVec-sh) - R
        plt.plot(xOrb,yOrb,'g--',label='Target orbit')

        # Draw orbit injection point (green)
        r0 = p / (1.0 + e * cos(f))
        x0 = r0 * cos(-f-sh)
        y0 = r0 * sin(-f-sh) - R
        plt.plot(x0,y0,'og')

        # Plot trajectory in default color (blue)
        plt.plot(X,Z,label='Ballistic flight (coasting)')

        # Plot launching point (black)
        plt.plot(X[0],Z[0],'ok')

        # Plot burning segments in red,
        # label only the first to avoid multiple labels
        mustLabl = True
        for i in range(len(indBurn)):
            ib = indBurn[i]
            ish = indShut[i]
            if mustLabl:
                plt.plot(X[ib:ish],Z[ib:ish],'r',label='Propulsed flight')
                mustLabl = False
            else:
                plt.plot(X[ib:ish],Z[ib:ish],'r')

        # Plot Max Pdyn point in orange
        plt.plot(X[indPdynMax],Z[indPdynMax],marker='o',color='orange',\
                 label='Max dynamic pressure')

        # Plot stage separation points in blue
        mustLabl = True
        for arc in range(s-1):
            if self.isStagSep[arc]:
                # this trick only labels the first segment, to avoid multiple
                # labels afterwards
                if mustLabl:
                    plt.plot(StgSepPnts[arc,0],StgSepPnts[arc,1],marker='o',\
                             color='blue',label='Stage separation point')
                    mustLabl = False
                else:
                    plt.plot(StgSepPnts[arc,0],StgSepPnts[arc,1],marker='o')

        # Plot altSol
        if compare:
            plt.plot(X_alt,Z_alt,'k--',label='Initial Guess')

        # Final plotting commands
        plt.grid(True)
        plt.xlabel("X [km]")
        plt.ylabel("Z [km]")
        plt.axis('equal')
        titlStr = "Rocket trajectory over Earth "
        titlStr += "(grad iter #" + str(self.NIterGrad) + ")\n"
        titlStr += "MaxDynPres = {:.4E} kPa".format(pDynMax*1e-6)
        plt.title(titlStr)
        plt.legend(loc="upper left", bbox_to_anchor=(1,1))

        if mustSaveFig:
            self.savefig(keyName='traj',fullName='trajectory')
        else:
            plt.show()
            plt.clf()

#%%
def calcXdot(td,x,u,constants,arc):
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

    sin = numpy.sin
    cos = numpy.cos

    u1 = u[0]
    u2 = u[1]

    # calculate variables alpha and beta
    alpha = u1
    beta = u2

    # calculate variables CL and CD
    CL = CL0[arc] + CL1[arc]*alpha
    CD = CD0[arc] + CD2[arc]*(alpha)**2

    # calculate L and D

    dens = rho(x[0])
    pDynTimesSref = .5 * dens * (x[1]**2) * s_ref[arc]
    L = CL * pDynTimesSref
    D = CD * pDynTimesSref

    # calculate r
    r = r_e + x[0]

    # calculate grav
    grav = GM/r/r

    # calculate phi:
    dx = numpy.empty(4)

    # example rocket single stage to orbit with Lift and Drag
    sinGama = sin(x[2])
    dx[0] = x[1] * sinGama
    dx[1] = (beta * Thrust[arc] * cos(alpha) - D)/x[3] - grav * sinGama
    dx[2] = (beta * Thrust[arc] * sin(alpha) + L)/(x[3] * x[1]) + \
            cos(x[2]) * ( x[1]/r  -  grav/x[1] )
    # "Gamma factor" in the gamma equation. td is dimensional time
    dx[2] *= .5*(1.0+numpy.tanh(DampSlop*(td-DampCent)))
    dx[3] = -(beta * Thrust[arc])/(grav_e * Isp[arc])
    return dx

