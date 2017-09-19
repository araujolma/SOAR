#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:25:28 2017

@author: levi
"""
import numpy, itsme
from sgra import sgra
from atmosphere import rho
import matplotlib.pyplot as plt

class prob(sgra):
    probName = 'probRock'
        
    def initGues(self,opt={}):
        # matrix sizes
        n = 4
        m = 2
        
        N = 10000 + 1#20000+1#40000+1#20000+1#5000000 + 1 #

        self.N = N
        self.n = n
        self.m = m
                
        dt = 1.0/(N-1)
        t = numpy.arange(0,1.0+dt,dt)
        self.dt = dt
        self.t = t
        
        #prepare tolerances
        tolP = 1.0e-7#8
        tolQ = 1.0e-7#5
        tol = dict()
        tol['P'] = tolP
        tol['Q'] = tolQ
        
        self.tol = tol

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
            
            # Earth constants
            r_e = 6371.0           # km
            GM = 398600.4415       # km^3 s^-2
            grav_e = GM/r_e/r_e    #9.8e-3       km/s^2
        
            # rocket constants
            Thrust = numpy.array([40.0])                 # kg km/s² [= kN] 1.3*m_initial # N
            
            scal = 1.0#e-3#e-6#1.0#1e-2#5.0e-3#7.5e-4# 1.0/2.5e3
            
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
            constants['costScalingFactor'] = scal
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
            self.constants = constants
            
            # restrictions
            alpha_min = -2*(numpy.pi)/180  # in rads
            alpha_max = 2*(numpy.pi)/180   # in rads
            beta_min = 0.0
            beta_max = 1.0
            restrictions = dict()
            restrictions['alpha_min'] = alpha_min
            restrictions['alpha_max'] = alpha_max
            restrictions['beta_min'] = beta_min
            restrictions['beta_max'] = beta_max
            self.restrictions = restrictions
            
            solInit = None
            
        elif initMode == 'naive':
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
    
            t_its, x_its, u_its, tabAlpha, tabBeta, inputDict, tphases, \
            mass0, massJet = itsme.sgra('default2st.its')

            # The inputDict corresponds to the con dictionary from itsme.
            # The con dictionary storages all input information and other
            # informations.
            # massJet: list of jetssoned masses at the beggining of each phase.
            #its1 = itsme.its()
            #t_its,x_its,u_its,tabAlpha,tabBeta = its1.sgra()

            # Number of stages:
            s = inputDict['NStag']
            self.s = s
            
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
            
            scal = 1.0#e-3#e-6#1.0#1e-2#5.0e-3#7.5e-4# 1.0/2.5e3
            
            Isp = inputDict['Isp']*numpy.ones(s)
            s_f = inputDict['efes']*numpy.ones(s)   
            CL0 = inputDict['CL0']*numpy.ones(s)
            CL1 = inputDict['CL1']*numpy.ones(s)
            CD0 = inputDict['CD0']*numpy.ones(s)
            CD2 = inputDict['CD2']*numpy.ones(s)
            s_ref = inputDict['s_ref']*numpy.ones(s)#(numpy.pi*(0.0005)**2)*numpy.ones(s)  # km^2
            DampCent = 3.0#2.0#
            DampSlop = 3.0
    
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
            self.boundary = boundary
        
            constants = dict()
            constants['grav_e'] = grav_e
            constants['Thrust'] = Thrust
            constants['costScalingFactor'] = scal
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
            self.constants = constants
            
            # restrictions
            alpha_min = -inputDict['AoAmax']*(numpy.pi)/180  # in rads
            alpha_max = inputDict['AoAmax']*(numpy.pi)/180   # in rads
            beta_min = 0.0
            beta_max = 1.0
            restrictions = dict()
            restrictions['alpha_min'] = alpha_min
            restrictions['alpha_max'] = alpha_max
            restrictions['beta_min'] = beta_min
            restrictions['beta_max'] = beta_max
            self.restrictions = restrictions

             
            # Find indices for beginning of arc        
            #print("\nSearching indices for beginning of arcs:")
            arcBginIndx = numpy.empty(s+1,dtype='int')
            arc = 0; arcBginIndx[arc] = 0
            j = 0; nt = len(t_its)
            for i in range(len(massJet)):
                #print("i =",i)
                if massJet[i] > 0.0:
                    #print("Jettissoned mass found!")
                    arc += 1
                    #print("arc =",arc)
                    tTarg = tphases[i]
                    #print("Beginning search for tTarg =",tTarg)
                    keepLook = True
                    while (keepLook and (j < nt)):
                        if abs(t_its[j]-tTarg) < 1e-10:
                            keepLook = False
                            #print("Found it! in j =",j,"t_its[j] =",t_its[j],"M[j] =",x_its[j,3])
                            # get the next time for proper initial conditions
                            j += 1
                            arcBginIndx[arc] = j
                        #print("j =",j,"t_its[j] =",t_its[j],"M[j] =",x_its[j,3])
                        j += 1
            #
            #print("Done. Indices:")
            #print(arcBginIndx)
            #print("Times:")
            #print(t_its[arcBginIndx])
            #print("\n\n")
            #print("New initial masses:",x_its[arcBginIndx,3])
            
            
            pi = numpy.empty(s)
            for arc in range(s):
                pi[arc] = t_its[arcBginIndx[arc+1]] - t_its[arcBginIndx[arc]]
            
            self.boundary['m_initial'] = x_its[0,3]

#            solInit = self.copy()
#            solInit.N = len(t_its)
#            solInit.t = t_its.copy()
#            
#            alpha_its = numpy.empty((solInit.N,s)) 
#            beta_its = numpy.empty((solInit.N,s))
#            for arc in range(s):
#                alpha_its[:,arc] = u_its[:,arc]
#                beta_its[:,arc] = u_its[:,arc]
#            u_init = solInit.calcAdimCtrl(alpha_its,beta_its)
#            
#            x_init = numpy.empty((solInit.N,n,s)); x_init[:,:,0] = x_its
#            
#            solInit.x = x_init.copy()
#            solInit.u = u_init.copy()
#            solInit.pi = pi.copy()
            
            # Re-integration of proposed solution. 
            # Only the controls are used, not the integrated state itself       
            for arc in range(s):
                dt = pi[arc]/(N-1); dt6 = dt/6.0
                x[0,:,arc] = x_its[arcBginIndx[arc],:]
                t0arc = t_its[arcBginIndx[arc]]
                uip1 = numpy.array([tabAlpha.value(t0arc),\
                                    tabBeta.value(t0arc)])
                # tt: dimensional time (for integration)
                for i in range(N-1):
                    tt = t0arc + i * dt
                    ui = uip1
                    u[i,:,arc] = ui
                    uipm = numpy.array([tabAlpha.value(tt+.5*dt),\
                                        tabBeta.value(tt+.5*dt)])
                    uip1 = numpy.array([tabAlpha.value(tt+dt),\
                                        tabBeta.value(tt+dt)])
                    if (i == N-2 and arc == s-1):
                        print("Bypassing here...")
                        uip1 = ui
                    f1 = calcXdot(tt,x[i,:,arc],ui,constants,arc)  
                    ttm = tt+.5*dt # time at half the integration interval
                    x2 = x[i,:,arc] + .5*dt*f1 # x at half step, with f1
                    f2 = calcXdot(ttm,x2,uipm,constants,arc)
                    x3 = x[i,:,arc] + .5*dt*f2 # x at half step, with f2
                    f3 = calcXdot(tt+.5*dt,x3,uipm,constants,arc) 
                    x4 = x[i,:,arc] + dt*f3 # x at next step, with f3
                    f4 = calcXdot(tt+dt,x4,uip1,constants,arc)
                    x[i+1,:,arc] = x[i,:,arc] + dt6 * (f1+f2+f2+f3+f3+f4) 

            u[N-1,:,s-1] = u[N-2,:,s-1]#numpy.array([tabAlpha.value(pi[0]),tabBeta.value(pi[0])])


        lam = numpy.zeros((N,n,s))
        mu = numpy.zeros(q)
        u = self.calcAdimCtrl(u[:,0,:],u[:,1,:])
        
        self.x = x
        self.u = u
        self.pi = pi
        self.lam = lam
        self.mu = mu
        
        solInit = self.copy()
        
        self.compWith(solInit,'solZA')
        
        print("\nInitialization complete.\n")        
        return solInit
#%%
    def calcDimCtrl(self):        
        # calculate variables alpha (ang. of att.) and beta (prop. thrust)
        
        restrictions = self.restrictions
        alpha_min = restrictions['alpha_min']
        alpha_max = restrictions['alpha_max']
        beta_min = restrictions['beta_min']
        beta_max = restrictions['beta_max']
        
        alfa = .5*((alpha_max + alpha_min) + \
                (alpha_max - alpha_min)*numpy.tanh(self.u[:,0,:]))
        beta = .5*((beta_max + beta_min) +  \
                (beta_max - beta_min)*numpy.tanh(self.u[:,1,:]))

        return alfa,beta
    
    def calcAdimCtrl(self,alfa,beta):
        #u = numpy.empty((self.N,self.m))
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
            phi[:,1,arc] = (beta[:,arc] * Thrust[arc] * cosAlfa[:,arc] - D[:,arc])/x[:,3,arc] - grav[:,arc] * sinGama[:,arc]
            phi[:,2,arc] = ((beta[:,arc] * Thrust[arc] * sinAlfa[:,arc] + L[:,arc])/(x[:,3,arc] * x[:,1,arc]) + \
                                  cosGama[:,arc] * ( x[:,1,arc]/r[:,arc]  -  grav[:,arc]/x[:,1,arc] )) * \
                                  .5*(1.0+numpy.tanh(DampSlop*(td-DampCent)))
            phi[:,3,arc] = - (beta[:,arc] * Thrust[arc])/(grav_e * Isp[arc])
            phi[:,:,arc] *= pi[arc]
            accDimTime += pi[arc]
        return phi

#%%

    def calcGrads(self):
        Grads = dict()
    
        N,n,m,p,q,s = self.N,self.n,self.m,self.p,self.q,self.s
        
    
        # Pre-assign functions
        sin = numpy.sin
        cos = numpy.cos
        tanh = numpy.tanh
        array = numpy.array
    
        constants = self.constants
        grav_e = constants['grav_e']
        Thrust = constants['Thrust'] 
        Isp = constants['Isp']
        scal = constants['costScalingFactor']
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
    
        restrictions = self.restrictions
        alpha_min = restrictions['alpha_min']
        alpha_max = restrictions['alpha_max']
        beta_min = restrictions['beta_min']
        beta_max = restrictions['beta_max']
        
        u = self.u
        u1 = u[:,0]
        u2 = u[:,1]
        x = self.x
        pi = self.pi
        
        phix = numpy.zeros((N,n,n,s))
        phiu = numpy.zeros((N,n,m,s))
    
        if p>0:
            phip = numpy.zeros((N,n,p,s))
        else:
            phip = numpy.zeros((N,n,1,s))
    
        fx = numpy.zeros((N,n,s))
        fu = numpy.zeros((N,m,s))
        fp = numpy.zeros((N,p,s))
    
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

        # First n rows:
        for i in range(n):
            psiy[i,i] = 1.0

        # Last n-1 rows (no mass eq for end condition of final arc):
        for ind in range(n-1):
            psiy[q-1-ind,2*n*s-2-ind] = 1.0
            
        # Intermediate conditions (except for mass)
        i0 = n
        for arc in range(s-1): 
            # For height, speed and angle:
            for stt in range(n-1):
                psiy[i0+stt,i0+stt] = -1.0  # this state, this arc  (end cond)
                psiy[i0+stt,i0+stt+n] = 1.0 # this state, next arc (init cond)
            # For mass (discontinuities allowed)
            psiy[i0+n-1,i0+n-1+n] = 1.0
            psiy[i0+n-1,i0+n-1] = -1.0/(1.0-s_f[arc])
            psiy[i0+n-1,i0-1] = s_f[arc]/(1.0-s_f[arc])
            i0 += n
         
        psip = numpy.zeros((q,p))
    
        # Calculate variables (arrays) alpha and beta
        aExp = .5*(alpha_max - alpha_min)
        bExp = .5*(beta_max - beta_min)
        alpha,beta = self.calcDimCtrl()
    
        # calculate variables CL and CD
        CL = numpy.empty_like(alpha)
        CD = numpy.empty_like(alpha)
        for arc in range(s):
            CL[:,arc] = CL0[arc] + CL1[arc]*alpha[:,arc]
            CD[:,arc] = CD0[arc] + CD2[arc]*(alpha[:,arc]**2)
    
        # calculate L and D; atmosphere: numerical gradient
        dens = numpy.empty((N,s))
        del_rho = numpy.empty((N,s))
        for arc in range(s):
            for k in range(N):
                dens[k,arc] = rho(x[k,0,arc])
                del_rho[k,arc] = (rho(x[k,0,arc]+.1) - dens[k,arc])/.1
        
        pDynTimesSref = numpy.empty_like(CL)
        for arc in range(s):
            pDynTimesSref[:,arc] = .5 * dens[:,arc] * (x[:,1,arc]**2) * s_ref[arc]    

        L = CL * pDynTimesSref
        D = CD * pDynTimesSref
        
        # calculate r
        r = r_e + x[:,0,:]
    
        # calculate grav
        grav = GM/r/r
    
    #==============================================================================
    # TODO: rewrite this, leaving the : as the time axis.
        for arc in range(s):
            for k in range(N):
                sinGama = sin(x[k,2,arc])
                cosGama = cos(x[k,2,arc])
         
                sinAlpha = sin(alpha[k,arc])
                cosAlpha = cos(alpha[k,arc])
         
                #cosu1 = cos(u1[k])
                #cosu2 = cos(u2[k])
         
                r2 = r[k,arc]**2; r3 = r2*r[k,arc]
                V = x[k,1,arc]; V2 = V*V
                m = x[k,3,arc]; m2 = m*m
                fVel = beta[k,arc]*Thrust[arc]*cosAlpha-D[k,arc] # forces on velocity direction
                fNor = beta[k,arc]*Thrust[arc]*sinAlpha+L[k,arc] # forces normal to velocity
                fdg = .5*(1.0+tanh(DampSlop*(k*pi[arc]/(N-1)-DampCent)))
    
                # Expanded notation:
                DAlfaDu1 = aExp*(1.0-tanh(u1[k,arc])**2)
                DBetaDu2 = bExp*(1.0-tanh(u2[k,arc])**2)
        
                phix[k,:,:,arc] = pi[arc]*array([[0.0                                                              ,sinGama                                                                                        ,V*cosGama                      ,0.0          ],
                                               [2*GM*sinGama/r3 - (0.5*CD[k,arc]*del_rho[k,arc]*s_ref[arc]*V2)/m              ,-CD[k,arc]*dens[k,arc]*s_ref[arc]*V/m                                                                       ,-grav[k,arc]*cosGama               ,-fVel/m2     ],
                                               [cosGama*(-V/r2+2*GM/(V*r3)) + (0.5*CL[k,arc]*del_rho[k,arc]*s_ref[arc]*V)/m   ,-beta[k,arc]*Thrust[arc]*sinAlpha/(m*V2) + cosGama*((1/r[k,arc])+grav[k,arc]/(V2)) + 0.5*CL[k,arc]*dens[k,arc]*s_ref[arc]/m  ,-sinGama*((V/r[k,arc])-grav[k,arc]/V)  ,-fNor/(m2*V) ],
                                               [0.0                                                              ,0.0                                                                                            ,0.0                            ,0.0          ]])
                phix[k,2,:] *= fdg
                phiu[k,:,:,arc] = pi[arc]*array([[0.0                                                        ,0.0                           ],
                     [(-beta[k,arc]*Thrust[arc]*sinAlpha*DAlfaDu1 - CD2[arc]*alpha[k,arc]*dens[k,arc]*s_ref[arc]*V2*DAlfaDu1)/m   ,Thrust[arc]*cosAlpha*DBetaDu2/m    ],
                     [(beta[k,arc]*Thrust[arc]*cosAlpha*DAlfaDu1/V + 0.5*CL1[arc]*dens[k,arc]*s_ref[arc]*(V)*DAlfaDu1)/m      ,Thrust[arc]*sinAlpha*DBetaDu2/(m*V)],
                     [0.0                                                                              ,-Thrust[arc]*DBetaDu2/(grav_e*Isp[arc]) ]])
                phiu[k,2,:,arc] *= fdg
                phip[k,:,:,arc] = array([[V*sinGama                                   ],
                                     [fVel/m - grav[k,arc]*sinGama                    ],
                                     [fNor/(m*V) + cosGama*((V/r[k,arc])-(grav[k,arc]/V)) ],
                                     [-(beta[k,arc]*Thrust[arc])/(grav_e*Isp[arc])              ]])
                phip[k,2,:,arc] *= fdg
                fu[k,:,arc] = array([0.0,(pi[arc]*Thrust[arc]*DBetaDu2)/(grav_e * Isp[arc] * (1.0-s_f[arc]))])
                fp[k,arc,arc] = (Thrust[arc] * beta[k,arc])/(grav_e * Isp[arc] * (1.0-s_f[arc]))
    #==============================================================================
    
        Grads['phix'] = phix
        Grads['phiu'] = phiu
        Grads['phip'] = phip
        Grads['fx'] = fx * scal
        Grads['fu'] = fu * scal
        Grads['fp'] = fp * scal
    #    Grads['gx'] = gx
    #    Grads['gp'] = gp
        Grads['psiy'] = psiy
        Grads['psip'] = psip
        return Grads

#%%
    def calcPsi(self):
        
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

        # interstage conditions
#        strPrnt = "0,1,2,3,"
        for arc in range(s-1):
            i0 = 4 * (arc+1)
#            strPrnt = strPrnt + str(i0) + "," + str(i0+1) + "," + \
#                        str(i0+2) + "," + str(i0+3) + ","
            # four states in order: position, speed, flight angle and mass
            psi[i0]   = x[0,0,arc+1] - x[N-1,0,arc] 
            psi[i0+1] = x[0,1,arc+1] - x[N-1,1,arc]
            psi[i0+2] = x[0,2,arc+1] - x[N-1,2,arc]
            psi[i0+3] = x[0,3,arc+1] - \
            (1.0/(1.0 - s_f[arc])) * (x[N-1,3,arc] - s_f[arc] * x[0,3,arc])

        # End of final subarc
        psi[q-3] = x[N-1,0,s-1] - boundary['h_final']
        psi[q-2] = x[N-1,1,s-1] - boundary['V_final']
        psi[q-1] = x[N-1,2,s-1] - boundary['gamma_final']
#        strPrnt = strPrnt+str(q-3)+","+str(q-2)+","+str(q-1)+","
        print("In calcPsi. Psi =",psi)
#        print("q =",q)
#        print(strPrnt)
        return psi
        
    def calcF(self):
        constants = self.constants
        grav_e = constants['grav_e']
        Thrust = constants['Thrust']
        Isp = constants['Isp']
        s_f = constants['s_f']
        scal = constants['costScalingFactor']
        
        # calculate variable beta
        _,beta = self.calcDimCtrl()
    
        f = numpy.empty((self.N,self.s))
        for arc in range(self.s):
            f[:,arc] = scal * beta[:,arc] * \
                ( (Thrust[arc] * self.pi[arc])/(grav_e * (1.0-s_f[arc]) * Isp[arc]) )
    
        return f

    def calcI(self):
        N,s = self.N,self.s
        f = self.calcF()

        Ivec = numpy.empty(s)
        for arc in range(s):
            Ivec[arc] = .5*(f[0,arc]+f[N-1,arc])
            Ivec[arc] += f[1:(N-1),arc].sum()
            
        Ivec *= 1.0/(N-1)
        
        if self.dbugOptGrad.get('plotF',False):
            print("\nThis is f:")
            for arc in range(s):
                print("Arc =",arc)
                plt.plot(self.t,f[:,arc])
                plt.grid(True)
                plt.xlabel("t")
                plt.ylabel("f")
                plt.show()
                plt.clf()
                plt.close('all')


        if self.dbugOptGrad.get('plotI',False):
            print("I =",Ivec)

        return Ivec.sum()
#%%
    def plotSol(self,opt={},intv=[]):

        x = self.x
        u = self.u
        pi = self.pi
        
#        if len(intv)==0:
#            intv = numpy.arange(0,self.N,1,dtype='int')
#        else:
#             intv = list(intv)   
    
        if len(intv)>0:       
            print("plotSol: Sorry, currently ignoring plotting range.")

        if opt.get('mode','sol') == 'sol':
            I = self.calcI()
            print("In plotSol.")
            print("Initial mass:",x[0,3,0])
            print("I:",I)
            print("CostScalFact:",self.constants['costScalingFactor'])
            print("Payload Mass:",self.mPayl)
            mFinl = x[0,3,0] - I/self.constants['costScalingFactor']
            print("'Final' mass:",mFinl)
            paylPercMassGain = 100.0*(mFinl-self.mPayl)/self.mPayl

            titlStr = "Current solution: I = {:.4E}".format(I) + \
            " P = {:.4E} ".format(self.P) + " Q = {:.4E} ".format(self.Q) + \
            "\nPayload mass gain: {:.4G}%".format(paylPercMassGain)
#            titlStr = "Current solution: I = {:.4E}".format(I)
#            if opt.get('dispP',False):
#                P = opt['P']
#                titlStr = titlStr + " P = {:.4E} ".format(P)
#            if opt.get('dispQ',False):
#                Q = opt['Q']
#                titlStr = titlStr + " Q = {:.4E} ".format(Q)
        elif opt['mode'] == 'var':
            titlStr = "Proposed variations"
        else:
            titlStr = opt['mode']
        #
        plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
        
        plt.subplot2grid((8,1),(0,0))
        self.plotCat(x[:,0,:])
        plt.grid(True)
        plt.ylabel("h [km]")
        plt.title(titlStr)
        
        plt.subplot2grid((8,1),(1,0))
        self.plotCat(x[:,1,:],color='g')
        plt.grid(True)
        plt.ylabel("V [km/s]")
        
        plt.subplot2grid((8,1),(2,0))
        self.plotCat(x[:,2,:]*180/numpy.pi,color='r')
        plt.grid(True)
        plt.ylabel("gamma [deg]")
        
        plt.subplot2grid((8,1),(3,0))
        self.plotCat(x[:,3,:],color='m')
        plt.grid(True)
        plt.ylabel("m [kg]")
        
        plt.subplot2grid((8,1),(4,0))
        self.plotCat(u[:,0,:],color='k')
        plt.grid(True)
        plt.ylabel("u1 [-]")
        
        plt.subplot2grid((8,1),(5,0))
        self.plotCat(u[:,1,:],color='c')
        plt.grid(True)
        plt.xlabel("t")
        plt.ylabel("u2 [-]")
        
        ######################################
        alpha,beta = self.calcDimCtrl()
        alpha *= 180.0/numpy.pi
        plt.subplot2grid((8,1),(6,0))
        self.plotCat(alpha)
        #plt.hold(True)
        #plt.plot(t,alpha*0+alpha_max*180/numpy.pi,'-.k')
        #plt.plot(t,alpha*0+alpha_min*180/numpy.pi,'-.k')
        plt.grid(True)
        plt.xlabel("t")
        plt.ylabel("alpha [deg]")
        
        plt.subplot2grid((8,1),(7,0))
        self.plotCat(beta)
        #plt.hold(True)
        #plt.plot(t,beta*0+beta_max,'-.k')
        #plt.plot(t,beta*0+beta_min,'-.k')
        plt.grid(True)
        plt.xlabel("t")
        plt.ylabel("beta [-]")
        ######################################        

        # TODO: include a plot for visualization of pi!

        self.savefig(keyName='currSol',fullName='solution')
        #if self.save['sol']:
        #    print("Saving solution plot to "+self.probName+"_currSol.pdf!")
        #    plt.savefig(self.probName+"_currSol.pdf",bbox_inches='tight', pad_inches=0.1)
        #else:
        #    plt.show()
        #plt.clf()
        #plt.close('all')
            
        print("pi =",pi)
        print("Final (injected into orbit) rocket mass: "+\
              "{:.4E}\n".format(x[-1,3,self.s-1]))
        print("Ejected mass (1st-2nd stage):",x[-1,3,0]-x[0,3,1])
    #
    
    def compWith(self,altSol,altSolLabl='altSol'):
        print("\nComparing solutions...\n")

        currSolLabl = 'currentSol'

        # Comparing final mass:
        mFinSol = self.x[self.N-1,3,self.s-1]
        mFinAlt = altSol.x[altSol.N-1,3,altSol.s-1]
        paylMassGain = mFinSol-mFinAlt
        paylPercMassGain = 100.0*paylMassGain/self.mPayl
        

        plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
        
        plt.subplot2grid((8,1),(0,0))
        altSol.plotCat(altSol.x[:,0,:],labl=altSolLabl)
        self.plotCat(self.x[:,0,:],mark='--',color='y',labl=currSolLabl)
        plt.grid(True)
        plt.ylabel("h [km]")
        plt.legend()
        plt.title("Comparing solutions: " + currSolLabl + " and " + \
                  altSolLabl+\
                  "\nPayload mass gain: {:.4G}%".format(paylPercMassGain))
        
        plt.subplot2grid((8,1),(1,0))
        altSol.plotCat(altSol.x[:,1,:],labl=altSolLabl)
        self.plotCat(self.x[:,1,:],mark='--',color='g',labl=currSolLabl)
        plt.grid(True)
        plt.ylabel("V [km/s]")
        plt.legend()
        
        plt.subplot2grid((8,1),(2,0))
        altSol.plotCat(altSol.x[:,2,:]*180/numpy.pi,labl=altSolLabl)
        self.plotCat(self.x[:,2,:]*180/numpy.pi,mark='--',color='r',\
                     labl=currSolLabl)
        plt.grid(True)
        plt.ylabel("gamma [deg]")
        plt.legend()
        
        plt.subplot2grid((8,1),(3,0))
        altSol.plotCat(altSol.x[:,3,:],labl=altSolLabl)
        self.plotCat(self.x[:,3,:],mark='--',color='m',labl=currSolLabl)
        plt.grid(True)
        plt.ylabel("m [kg]")
        plt.legend()
        
        plt.subplot2grid((8,1),(4,0))
        altSol.plotCat(altSol.u[:,0,:],labl=altSolLabl)
        self.plotCat(self.u[:,0,:],mark='--',color='k',labl=currSolLabl)
        plt.grid(True)
        plt.ylabel("u1 [-]")
        plt.legend()
        
        plt.subplot2grid((8,1),(5,0))
        altSol.plotCat(altSol.u[:,1,:],labl=altSolLabl)
        self.plotCat(self.u[:,1,:],mark='--',color='c',labl=currSolLabl)
        plt.grid(True)
        plt.xlabel("t")
        plt.ylabel("u2 [-]")
        plt.legend()
        
        ######################################
        alpha,beta = self.calcDimCtrl()
        alpha_alt,beta_alt = altSol.calcDimCtrl()
        plt.subplot2grid((8,1),(6,0))
        altSol.plotCat(alpha_alt*180/numpy.pi,labl=altSolLabl)
        self.plotCat(alpha*180/numpy.pi,mark='--',color='g',labl=currSolLabl)

        #plt.hold(True)
        #plt.plot(t,alpha*0+alpha_max*180/numpy.pi,'-.k')
        #plt.plot(t,alpha*0+alpha_min*180/numpy.pi,'-.k')
        plt.grid(True)
        plt.xlabel("t")
        plt.ylabel("alpha [deg]")
        plt.legend()
        
        plt.subplot2grid((8,1),(7,0))
        altSol.plotCat(beta_alt,labl=altSolLabl)
        self.plotCat(beta,mark='--',color='k',labl=currSolLabl)

        #plt.hold(True)
        #plt.plot(t,beta*0+beta_max,'-.k')
        #plt.plot(t,beta*0+beta_min,'-.k')
        plt.grid(True)
        plt.xlabel("t")
        plt.ylabel("beta [-]")
        plt.legend()
        ######################################
        
        self.savefig(keyName='comp',fullName='comparisons')
        #if self.save['comp']:
        #    print("Saving comparisons plot to "+self.probName+"_comp.pdf!")
        #    plt.savefig(self.probName+"_comp.pdf",bbox_inches='tight', pad_inches=0.1)
        #else:
        #    plt.show()
        #plt.clf()
        #plt.close('all')

#        
#        for arc in range(s):
#        
#            plt.plot(altSol.t,altSol.x[:,0,arc],label=altSolLabl)
#            plt.plot(self.t,self.x[:,0,arc],'--y',label=currSolLabl)
#            plt.grid()
#            plt.ylabel("h [km]")
#            plt.xlabel("t [-]")
#            plt.title('Height')
#            plt.legend()
#            plt.show()
#            plt.clf()
#            plt.close('all')
#
#            
#            plt.plot(altSol.t,altSol.x[:,1,arc],label=altSolLabl)
#            plt.plot(self.t,self.x[:,1,arc],'--g',label=currSolLabl)
#            plt.grid()
#            plt.ylabel("V [km/s]")
#            plt.xlabel("t [-]")
#            plt.title('Absolute speed')
#            plt.legend()
#            plt.show()
#            plt.clf()
#            plt.close('all')
#     
#            plt.plot(altSol.t,altSol.x[:,2,arc]*180/numpy.pi,label=altSolLabl)
#            plt.plot(self.t,self.x[:,2,arc]*180/numpy.pi,'--r',label=currSolLabl)    
#            plt.grid()
#            plt.ylabel("gamma [deg]")
#            plt.xlabel("t [-]")
#            plt.title('Flight path angle')
#            plt.legend()
#            plt.show()
#            plt.clf()
#            plt.close('all')
#       
#            plt.plot(altSol.t,altSol.x[:,3,arc],label=altSolLabl)
#            plt.plot(self.t,self.x[:,3,arc],'--m',label=currSolLabl)
#            plt.grid()
#            plt.ylabel("m [kg]")
#            plt.xlabel("t [-]")
#            plt.title('Rocket mass')
#            plt.legend()
#            plt.show()
#            plt.clf()
#            plt.close('all')
#
#                        
#            alpha,beta = self.calcDimCtrl()
#            alpha_alt,beta_alt = altSol.calcDimCtrl()
#            plt.plot(altSol.t,alpha_alt[:,arc]*180/numpy.pi,label=altSolLabl)
#            plt.plot(self.t,alpha[:,arc]*180/numpy.pi,'--g',label=currSolLabl)
#            plt.grid()
#            plt.xlabel("t [-]")
#            plt.ylabel("alfa [deg]")
#            plt.title('Attack angle')
#            plt.legend()
#            plt.show()
#            plt.clf()
#            plt.close('all')
#            
#            plt.plot(altSol.t,beta_alt[:,arc],label=altSolLabl)
#            plt.plot(self.t,beta[:,arc],'--k',label=currSolLabl)
#            plt.grid()
#            plt.xlabel("t [-]")
#            plt.ylabel("beta [-]")
#            plt.title('Thrust profile')
#            plt.legend()
#            plt.show()
#            plt.clf()
#            plt.close('all')
            
        print("Final rocket mass:")  
        print(currSolLabl+": {:.4E}".format(mFinSol)+" kg.")
        print(altSolLabl+": {:.4E}".format(mFinAlt)+" kg.")
        print("Difference: {:.4E}".format(paylMassGain)+" kg, "+\
              "{:.4G}".format(paylPercMassGain)+\
              "% more payload!\n")
        
     
# TODO: re-implement plotTraj...
        
    def plotTraj(self):
        
        cos = numpy.cos; sin = numpy.sin
        R = self.constants['r_e']
        N, s = self.N, self.s

        X = numpy.empty(N*s); Z = numpy.empty(N*s)

        sigma = 0.0 #sigma: range angle
        X[0] = 0.0
        Z[0] = 0.0

        # Propulsive phases' starting and ending times
        isBurn = True
        indBurn = list(); indShut = list()
        for arc in range(s):
            indBurn.append(list())
            indShut.append(list())

        indBurn[0].append(0)
        iCont = 0 # continuous counter (all arcs concatenated)
        for arc in range(s):
            dtd = self.dt * self.pi[arc] # Dimensional dt...

            for i in range(1,N):
                iCont += 1

                if isBurn:
                    if self.u[i,1,arc] < -.999:
                        isBurn = False
                        indShut[arc].append(i)
                else: #not burning
                    if self.u[i,1,arc] > -.999:
                        isBurn = True
                        indBurn[arc].append(i)

                # Propagate the trajectory by Euler method.
                v = self.x[i,1,arc]
                gama = self.x[i,2,arc]
                dsigma = v * cos(gama) / (R + self.x[i,0,arc])
                sigma += dsigma * dtd

                X[iCont] = X[iCont-1] + dtd * v * cos(gama-sigma)
                Z[iCont] = Z[iCont-1] + dtd * v * sin(gama-sigma)
            #
        indShut[s-1].append(N-1)    
                
        # Draw Earth segment corresponding to flight range
        sigVec = numpy.arange(0,1.01,.01) * sigma
        x = R * cos(.5*numpy.pi - sigVec)
        z = R * (sin(.5*numpy.pi - sigVec) - 1.0)
        plt.plot(x,z,'k')
        
        # Get final orbit parameters
        h,v,gama,M = self.x[N-1,:,s-1]
        
        print("State @burnout time:")
        print("h = {:.4E}".format(h)+", v = {:.4E}".format(v)+\
        ", gama = {:.4E}".format(gama)+", m = {:.4E}".format(M))
        
        GM = self.constants['GM']       
        r = R + h
#        print("Final altitude:",h)
        cosGama = cos(gama)
        sinGama = sin(gama)
        momAng = r * v * cosGama
#        print("Ang mom:",momAng)
        en = .5 * v * v - GM / r
#        print("Energy:",en)
        a = - .5 * GM / en
#        print("Semi-major axis:",a)
        aux = v * momAng / GM
        e = numpy.sqrt((aux * cosGama - 1.0)**2 + (aux * sinGama)**2)
        print("Eccentricity:",e)
        eccExpr = v * momAng * cosGama / GM - 1.0
#        print("r =",r)
        f = numpy.arccos(eccExpr/e)
        print("True anomaly:",f*180/numpy.pi)
        ph = a * (1.0 - e) - R
        print("Perigee altitude:",ph)    
        ah = 2*(a - R) - ph        
        print("Apogee altitude:",ah)

        # semi-latus rectum
        p = momAng**2 / GM #a * (1.0-e)**2
                
        
        # Plot orbit in green over the same range as the Earth shown 
        # (and a little but futher)
        
        sigVec = numpy.arange(f-1.2*sigma,f+.2*sigma,.01)
#        print("s =",sigVec)
        # shifting angle
        sh = sigma - f - .5*numpy.pi
        rOrb = p/(1.0+e*cos(sigVec))
#        print("rOrb =",rOrb)
        xOrb = rOrb * cos(-sigVec-sh)
        yOrb = rOrb * sin(-sigVec-sh) - R
        plt.plot(xOrb,yOrb,'g--')
 
        # Draw orbit injection point       
        r0 = p / (1.0 + e * cos(f))
#        print("r0 =",r0)
        x0 = r0 * cos(-f-sh)
        y0 = r0 * sin(-f-sh) - R
        plt.plot(x0,y0,'og')
        
        # Plot trajectory in default color (blue)
        plt.plot(X,Z)
        
        # Plot launching point in black
        plt.plot(X[0],Z[0],'ok')
        
        # Plot burning segments in red
        for arc in range(s):
            iOS = arc * N # offset index
            for i in range(len(indBurn[arc])):
                ib = indBurn[arc][i]+iOS
                ish = indShut[arc][i]+iOS
                plt.plot(X[ib:ish],Z[ib:ish],'r')
        
        plt.grid(True)
        plt.xlabel("X [km]")
        plt.ylabel("Z [km]")
        plt.axis('equal')
        plt.title("Rocket trajectory over Earth")
        
        self.savefig(keyName='traj',fullName='trajectory')
#        if self.save['traj']:
#            print("Saving trajectory plot to " + self.probName + \
#                  "_traj.pdf!")
#            plt.savefig(self.probName + "_traj.pdf",bbox_inches='tight', pad_inches=0.1)
#        else:
#            plt.show()
#        plt.clf()
#        plt.close('all')
        
#
#%%
def calcXdot(t,x,u,constants,arc):
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
    dx[2] *= .5*(1.0+numpy.tanh(DampSlop*(t-DampCent)))
    dx[3] = -(beta * Thrust[arc])/(grav_e * Isp[arc])
    #if t < 3.0:
#        print(t)
    #    dx[2] = 0.0
    return dx