# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:02:30 2017

@author: munizlgmn
"""

import numpy, itsme
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from atmosphere import rho

def calcXdot(sizes,x,u,constants,restrictions):
    n = sizes['n']
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
    alpha_min = restrictions['alpha_min']
    alpha_max = restrictions['alpha_max']
    beta_min = restrictions['beta_min']
    beta_max = restrictions['beta_max']
    sin = numpy.sin
    cos = numpy.cos
    u1 = u[0]
    u2 = u[1]

    # calculate variables alpha and beta
    alpha = (alpha_max + alpha_min)/2 + sin(u1)*(alpha_max - alpha_min)/2
    beta = (beta_max + beta_min)/2 + sin(u2)*(beta_max - beta_min)/2

    # calculate variables CL and CD
    CL = CL0 + CL1*alpha
    CD = CD0 + CD2*(alpha)**2

    # calculate L and D
    
    dens = rho(x[0])
    pDynTimesSref = .5 * dens * (x[1]**2) * s_ref    
    L = CL * pDynTimesSref
    D = CD * pDynTimesSref
    
    # calculate r
    r = r_e + x[0]

    # calculate grav
    grav = GM/r/r

    # calculate phi:
    dx = numpy.empty(n)

    # example rocket single stage to orbit with Lift and Drag
    sinGama = sin(x[2])
    dx[0] = x[1] * sinGama
    dx[1] = (beta * Thrust * cos(alpha) - D)/x[3] - grav * sinGama
    dx[2] = (beta * Thrust * sin(alpha) + L)/(x[3] * x[1]) + cos(x[2]) * ( x[1]/r  -  grav/x[1] )
    dx[3] = -(beta * Thrust)/(grav_e * Isp)

    return dx

# ##################
# PROBLEM DOMAIN:
# ##################
def declProb(opt=dict()):
# time discretization
    N = 20000+1#20000+1#5000000 + 1 #1000 + 1#
    dt = 1.0/(N-1)
    t = numpy.arange(0,1.0+dt,dt)

# example rocket single stage to orbit with Lift and Drag

# boundary conditions
    h_initial = 0.0            # km
    V_initial = 0.0            # km/s
    gamma_initial = numpy.pi/2 # rad
    m_initial = 50000          # kg
    h_final = 463.0   # km
    V_final = 7.633   # km/s
    gamma_final = 0.0 # rad
    #m_final = free   # kg

# matrix sizes
    n = 4
    m = 2
    p = 1
    q = 3

# Earth constants
    r_e = 6371.0           # km
    GM = 398600.4415       # km^3 s^-2
    grav_e = GM/r_e/r_e#9.8e-3        # km/s^2

# rocket constants
    Thrust = 40.0                 # kg km/s²  1.3*m_initial # N
    Isp = 450.0                   # s
    s_f = 0.05
    CL0 = -0.03                   # (B0 Miele 1998)
    CL1 = 0.8                     # (B1 Miele 1998)
    CD0 = 0.05                    # (A0 Miele 1998)
    CD2 = 0.5                     # (A2 Miele 1998)
    s_ref = numpy.pi*(0.0005)**2  # km^2
    f_scal = 1e-4
    
 # restrictions
    alpha_min = -2*(numpy.pi)/180  # in rads
    alpha_max = 2*(numpy.pi)/180   # in rads
    beta_min = 0
    beta_max = 1

# tolerances
    tolP = 1.0e-7#8
    tolQ = 1.0e-5

# prepare boundary conditions
    boundary = dict()
    boundary['h_initial'] = h_initial
    boundary['V_initial'] = V_initial
    boundary['gamma_initial'] = gamma_initial
    boundary['m_initial'] = m_initial
    boundary['h_final'] = h_final
    boundary['V_final'] = V_final
    boundary['gamma_final'] = gamma_final

# prepare sizes
    sizes = dict()
    sizes['N'] = N
    sizes['n'] = n
    sizes['m'] = m
    sizes['p'] = p
    sizes['q'] = q

# prepare constants
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
    constants['cost_scaling_factor'] = f_scal

# prepare restrictions
    restrictions = dict()
    restrictions['alpha_min'] = alpha_min
    restrictions['alpha_max'] = alpha_max
    restrictions['beta_min'] = beta_min
    restrictions['beta_max'] = beta_max

#prepare tolerances
    tol = dict()
    tol['P'] = tolP
    tol['Q'] = tolQ

# Get initialization mode
    initMode = opt.get('initMode','default')
    x = numpy.zeros((N,n))
    u = numpy.zeros((N,m))

    if initMode == 'default':
        # artesanal handicraft with L and D (Miele 2003)
        x[:,0] = h_final*numpy.sin(numpy.pi*t.copy()/2)
        x[:,1] = 3.793*numpy.exp(0.7256*t) -1.585 -3.661*numpy.cos(3.785*t+0.9552)
        #x[:,1] = V_final*numpy.sin(numpy.pi*t.copy()/2)
        #x[:,1] = 1.0e3*(-0.4523*t.copy()**5 + 1.2353*t.copy()**4-1.1884*t.copy()**3+0.4527*t.copy()**2-0.0397*t.copy())
        x[:,2] = (numpy.pi/2)*(numpy.exp(-(t.copy()**2)/0.017))+0.06419
        x[:,3] = m_initial*((0.7979*numpy.exp(-(t.copy()**2)/0.02))+0.1901*numpy.cos(t.copy()))
        #x[:,3] = m_initial*(1.0-0.89*t.copy())
        #x[:,3] = m_initial*(-2.9*t.copy()**3 + 6.2*t.copy()**2 - 4.2*t.copy() + 1)
        for k in range(N):
            if k<910:
                u[k,1] = (numpy.pi/2)
            else:
                if k>4999:
                    u[k,1] = (numpy.pi/2)*0.27
        pi = 1100*numpy.ones((p,1))

    elif initMode == 'extSol':

        # Factors instervals for aerodynamics
        fsup = numpy.array([0.61 + 0.3,500 + 100,1.94 + 0.3]) # Superior limit
        finf = numpy.array([0.61 - 0.3,500 - 100,1.94 - 0.3]) # Inferior limit

        # Automatic adjustment
        new_factors,t_its,x_its,u_its = itsme.its(fsup, finf, h_final, 100.0, 1.0e-7)
                    
        # Solutions must be made compatible: t_its is dimensional, 
        # u_its consists of the actual controls (alpha and beta), etc.
        # Besides, all arrays are in a different time discretization
        
        pi = numpy.array([t_its[-1]])
        t_its = t_its/pi

        # Perform inverse transformations for u:
        a1 = (alpha_max + alpha_min)/2
        a2 = (alpha_max - alpha_min)/2
        b1 = (beta_max + beta_min)/2
        b2 = (beta_max - beta_min)/2
        u_its[:,0] = numpy.arcsin((u_its[:,0]-a1)/a2)
        u_its[:,1] = numpy.arcsin((u_its[:,1]-b1)/b2)
    
#        for i in range(n):
#            f_x = interp1d(t_its, x_its[:,i])
#            x[:,i] = f_x(t)
        
        for i in range(m):
            f_u = interp1d(t_its,u_its[:,i])
            u[:,i] = f_u(t)
        
        dt = pi[0]/(N-1); dt6 = dt/6
        x[0,:] = x_its[0,:]
        for i in range(N-1):
            k1 = calcXdot(sizes,x[i,:],u[i,:],constants,restrictions)  
            k2 = calcXdot(sizes,x[i,:]+.5*dt*k1,.5*(u[i,:]+u[i+1,:]),constants,restrictions)
            k3 = calcXdot(sizes,x[i,:]+.5*dt*k2,.5*(u[i,:]+u[i+1,:]),constants,restrictions)  
            k4 = calcXdot(sizes,x[i,:]+dt*k3,u[i+1,:],constants,restrictions)
            x[i+1,:] = x[i,:] + dt6 * (k1+k2+k2+k3+k3+k4) 

    lam = 0.0*x.copy()
    mu = numpy.zeros(q)

    ###
    print("\nUn-interpolated solution from Souza's propagation:")

    plt.plot(t_its,x_its[:,0])
    plt.grid(True)
    plt.ylabel("h [km]")
    plt.xlabel("t_its [-]")
    plt.show()

    plt.plot(t_its,x_its[:,1],'g')
    plt.grid(True)
    plt.ylabel("V [km/s]")
    plt.xlabel("t_its [-]")
    plt.show()

    plt.plot(t_its,x_its[:,2]*180/numpy.pi,'r')
    plt.grid(True)
    plt.ylabel("gamma [deg]")
    plt.xlabel("t_its [-]")
    plt.show()

    plt.plot(t_its,x_its[:,3],'m')
    plt.grid(True)
    plt.ylabel("m [kg]")
    plt.xlabel("t_its [-]")
    plt.show()

    print("\nUn-interpolated control profiles:")

    plt.plot(t_its,(numpy.sin(u_its[:,0])*a2 + a1)*180/numpy.pi,'k')
    plt.grid(True)
    plt.xlabel("t_its [-]")
    plt.ylabel("Attack angle [deg]")
    plt.show()

    plt.plot(t_its,(numpy.sin(u_its[:,1])*b2 + b1),'c')
    plt.grid(True)
    plt.xlabel("t_its [-]")
    plt.ylabel("Thrust profile [-]")
    plt.show()
    ###

    print("\nInitialization complete.\n")
    return sizes,t,x,u,pi,lam,mu,tol,constants,boundary,restrictions


def calcPhi(sizes,x,u,pi,constants,restrictions):
    N = sizes['N']
    n = sizes['n']
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
    alpha_min = restrictions['alpha_min']
    alpha_max = restrictions['alpha_max']
    beta_min = restrictions['beta_min']
    beta_max = restrictions['beta_max']
    sin = numpy.sin
    cos = numpy.cos
    u1 = u[:,0]
    u2 = u[:,1]

    # calculate variables alpha and beta
    alpha = (alpha_max + alpha_min)/2 + sin(u1)*(alpha_max - alpha_min)/2
    beta = (beta_max + beta_min)/2 + sin(u2)*(beta_max - beta_min)/2

    # calculate variables CL and CD
    CL = CL0 + CL1*alpha
    CD = CD0 + CD2*(alpha)**2

    # calculate L and D
    # TODO: making atmosphere.rho vectorized (array compatible) would increase 
    # performance significantly!
    
    dens = numpy.empty(N)
    for k in range(N):
        dens[k] = rho(x[k,0])
    
    pDynTimesSref = .5 * dens * (x[:,1]**2) * s_ref    
    L = CL * pDynTimesSref
    D = CD * pDynTimesSref
    
    # calculate r
    r = r_e + x[:,0]

    # calculate grav
    grav = GM/r/r

    # calculate phi:
    phi = numpy.empty((N,n))

    # example rocket single stage to orbit with Lift and Drag
    sinGama = sin(x[:,2])
    phi[:,0] = pi[0] * x[:,1] * sinGama
    phi[:,1] = pi[0] * ((beta * Thrust * cos(alpha) - D)/x[:,3] - grav * sinGama)
    phi[:,2] = pi[0] * ((beta * Thrust * sin(alpha) + L)/(x[:,3] * x[:,1]) + cos(x[:,2]) * ( x[:,1]/r  -  grav/x[:,1] ))
    phi[0,2] = 0.0
    phi[:,3] = - (pi[0] * beta * Thrust)/(grav_e * Isp)

    return phi

def calcPsi(sizes,x,boundary):
    N = sizes['N']
    h_final = boundary['h_final']
    V_final = boundary['V_final']
    gamma_final = boundary['gamma_final']

    # example rocket single stage to orbit with Lift and Drag
    psi = numpy.array([x[N-1,0]-h_final,x[N-1,1]-V_final,x[N-1,2]-gamma_final])

    return psi

def calcF(sizes,x,u,pi,constants,restrictions):
    N = sizes['N']
    f = numpy.empty(N)
    grav_e = constants['grav_e']
    Thrust = constants['Thrust']
    Isp = constants['Isp']
    s_f = constants['s_f']
    beta_min = restrictions['beta_min']
    beta_max = restrictions['beta_max']
    #u1 = u[:,0]
    u2 = u[:,1]

    # calculate variable beta
    beta = (beta_max + beta_min)/2 + numpy.sin(u2)*(beta_max - beta_min)/2

    # example rocket single stage to orbit with Lift and Drag
    f = ((Thrust * pi[0])/(grav_e * (1-s_f) * Isp)) * beta
    return f

def calcGrads(sizes,x,u,pi,constants,restrictions):
    Grads = dict()

    N = sizes['N']
    n = sizes['n']
    m = sizes['m']
    p = sizes['p']
    #q = sizes['q']

    # Pre-assign functions
    sin = numpy.sin
    cos = numpy.cos
    array = numpy.array

    grav_e = constants['grav_e']
    Thrust = constants['Thrust']
    Isp = constants['Isp']
    r_e = constants['r_e']
    GM = constants['GM']
    s_f = constants['s_f']
    CL0 = constants['CL0']
    CL1 = constants['CL1']
    CD0 = constants['CD0']
    CD2 = constants['CD2']
    s_ref = constants['s_ref']
    f_scal = constants['cost_scaling_factor']

    alpha_min = restrictions['alpha_min']
    alpha_max = restrictions['alpha_max']
    beta_min = restrictions['beta_min']
    beta_max = restrictions['beta_max']
    u1 = u[:,0]
    u2 = u[:,1]

    Grads['dt'] = 1.0/(N-1)

    phix = numpy.zeros((N,n,n))
    phiu = numpy.zeros((N,n,m))

    if p>0:
        phip = numpy.zeros((N,n,p))
    else:
        phip = numpy.zeros((N,n,1))

    fx = numpy.zeros((N,n))
    fu = numpy.zeros((N,m))
    fp = numpy.zeros((N,p))

    # Gradients from example rocket single stage to orbit with Lift and Drag
    psix = array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0]])
    psip = array([[0.0],[0.0],[0.0]])

    # Calculate variables (arrays) alpha and beta
    aExp = .5*(alpha_max - alpha_min)
    alpha = (alpha_max + alpha_min)/2 + sin(u1)*aExp
    bExp = .5*(beta_max - beta_min)
    beta = (beta_max + beta_min)/2 + sin(u2)*bExp

    # calculate variables CL and CD
    CL = CL0 + CL1*alpha
    CD = CD0 + CD2*(alpha)**2

    # calculate L and D; atmosphere: numerical gradient
    dens = numpy.empty(N)
    del_rho = numpy.empty(N)
    for k in range(N):
        dens[k] = rho(x[k,0])
        del_rho[k] = (rho(x[k,0]+.1) - dens[k])/.1
    
    pDynTimesSref = .5 * dens * (x[:,1]**2) * s_ref    
    L = CL * pDynTimesSref
    D = CD * pDynTimesSref
    
    # calculate r
    r = r_e + x[:,0]

    # calculate grav
    grav = GM/r/r

#==============================================================================
    for k in range(N):
        sinGama = sin(x[k,2])
        cosGama = cos(x[k,2])
 
        sinAlpha = sin(alpha[k])
        cosAlpha = cos(alpha[k])
 
        cosu1 = cos(u1[k])
        cosu2 = cos(u2[k])
 
        r2 = r[k]**2; r3 = r2*r[k]
        V = x[k,1]; V2 = V*V
        m = x[k,3]; m2 = m*m
        fVel = beta[k]*Thrust*cosAlpha-D[k] # forces on velocity direction
        fNor = beta[k]*Thrust*sinAlpha+L[k] # forces normal to velocity
 
        # Expanded notation:
        DAlfaDu1 = aExp*cosu1
        DBetaDu2 = bExp*cosu2
        if k<int(N/100):# 1:#
            phix[k,:,:] = pi[0]*array([[0.0                                                  ,sinGama                   ,V*cosGama         ,0.0      ],
                                       [2*GM*sinGama/r3 - (0.5*CD[k]*del_rho[k]*s_ref*V2)/m  ,-CD[k]*dens[k]*s_ref*V/m  ,-grav[k]*cosGama  ,-fVel/m2 ],
                                       [0.0                                                  ,0.0                       ,0.0               ,0.0      ],
                                       [0.0                                                  ,0.0                       ,0.0               ,0.0      ]])
        
            phiu[k,:,:] = pi[0]*array([[0.0                                  ,0.0                          ],
                                       [-beta[k]*Thrust*sinAlpha*DAlfaDu1/m  ,Thrust*cosAlpha*DBetaDu2/m   ],
                                       [0.0                                  ,0.0                          ],
                                       [0.0                                  ,-Thrust*DBetaDu2/(grav_e*Isp)]])

        else:
            phix[k,:,:] = pi[0]*array([[0.0                                                              ,sinGama                                                                                        ,V*cosGama                      ,0.0          ],
                                       [2*GM*sinGama/r3 - (0.5*CD[k]*del_rho[k]*s_ref*V2)/m              ,-CD[k]*dens[k]*s_ref*V/m                                                                       ,-grav[k]*cosGama               ,-fVel/m2     ],
                                       [cosGama*(-V/r2+2*GM/(V*r3)) + (0.5*CL[k]*del_rho[k]*s_ref*V)/m   ,-beta[k]*Thrust*sinAlpha/(m*V2) + cosGama*((1/r[k])+grav[k]/(V2)) + 0.5*CL[k]*dens[k]*s_ref/m  ,-sinGama*((V/r[k])-grav[k]/V)  ,-fNor/(m2*V) ],
                                       [0.0                                                              ,0.0                                                                                            ,0.0                            ,0.0          ]])
     
            phiu[k,:,:] = pi[0]*array([[0.0                                                                                ,0.0                           ],
                                       [(-beta[k]*Thrust*sinAlpha*DAlfaDu1 - CD2*alpha[k]*dens[k]*s_ref*V2*aExp*cosu1)/m   ,Thrust*cosAlpha*DBetaDu2/m    ],
                                       [(beta[k]*Thrust*cosAlpha*DAlfaDu1 + 0.5*CL1*dens[k]*s_ref*(V)*aExp*cosu1)/m        ,Thrust*sinAlpha*DBetaDu2/(m*V)],
                                       [0.0                                                                                ,-Thrust*DBetaDu2/(grav_e*Isp) ]])
        #
        phip[k,:,:] = array([[V*sinGama                                                           ],
                            [fVel/m - grav[k]*sinGama                   ],
                            [fNor/(m*V) + cosGama*((V/r[k])-(grav[k]/V))],
                            [-(beta[k]*Thrust)/(grav_e*Isp)                                       ]])
    
        #print("fu without scaling = ",fu)
        fu[k,:] = array([0.0,(pi[0]*Thrust*DBetaDu2)/(grav_e * Isp * (1-s_f))])
        fu *= f_scal
        #print("fu with scaling = ",fu)
        fp[k,0] = (Thrust * beta[k])/(grav_e * Isp * (1-s_f))
#==============================================================================

    Grads['phix'] = phix
    Grads['phiu'] = phiu
    Grads['phip'] = phip
    Grads['fx'] = fx
    Grads['fu'] = fu
    Grads['fp'] = fp
#    Grads['gx'] = gx
#    Grads['gp'] = gp
    Grads['psix'] = psix
    Grads['psip'] = psip
    return Grads

def calcI(sizes,x,u,pi,constants,restrictions):
    N = sizes['N']
    dt = 1.0/(N-1)
    I = 0.0
# example rocket single stage to orbit with Lift and Drag    
    f = calcF(sizes,x,u,pi,constants,restrictions)
    for t in range(1,N-1):
        I += f[t]
    I += 0.5*f[0]
    I += 0.5*f[N-1]
    I *= dt        
    return I

def plotSol(sizes,t,x,u,pi,constants,restrictions,opt=dict()):

    alpha_min = restrictions['alpha_min']
    alpha_max = restrictions['alpha_max']
    beta_min = restrictions['beta_min']
    beta_max = restrictions['beta_max']

    plt.subplot2grid((8,4),(0,0),colspan=5)
    plt.plot(t,x[:,0],)
    plt.grid(True)
    plt.ylabel("h [km]")
    if opt.get('mode','sol') == 'sol':
        I = calcI(sizes,x,u,pi,constants,restrictions)
        titlStr = "Current solution: I = {:.4E}".format(I)
        if opt.get('dispP',False):
            P = opt['P']
            titlStr = titlStr + " P = {:.4E} ".format(P)
        if opt.get('dispQ',False):
            Q = opt['Q']
            titlStr = titlStr + " Q = {:.4E} ".format(Q)
    elif opt['mode'] == 'var':
        titlStr = "Proposed variations"
    else:
        titlStr = opt['mode']
    #
    plt.title(titlStr)
    plt.subplot2grid((8,4),(1,0),colspan=5)
    plt.plot(t,x[:,1],'g')
    plt.grid(True)
    plt.ylabel("V [km/s]")
    plt.subplot2grid((8,4),(2,0),colspan=5)
    plt.plot(t,x[:,2]*180/numpy.pi,'r')
    plt.grid(True)
    plt.ylabel("gamma [deg]")
    plt.subplot2grid((8,4),(3,0),colspan=5)
    plt.plot(t,x[:,3],'m')
    plt.grid(True)
    plt.ylabel("m [kg]")
    plt.subplot2grid((8,4),(4,0),colspan=5)
    plt.plot(t,u[:,0],'k')
    plt.grid(True)
    plt.ylabel("u1 [-]")
    plt.subplot2grid((8,4),(5,0),colspan=5)
    plt.plot(t,u[:,1],'c')
    plt.grid(True)
    plt.xlabel("t")
    plt.ylabel("u2 [-]")
    ######################################
    alpha = (alpha_max + alpha_min)/2 + numpy.sin(u[:,0])*(alpha_max - alpha_min)/2
    alpha *= 180/numpy.pi
    plt.subplot2grid((8,4),(6,0),colspan=5)
    plt.plot(t,alpha,'b')
    #plt.hold(True)
    #plt.plot(t,alpha*0+alpha_max*180/numpy.pi,'-.k')
    #plt.plot(t,alpha*0+alpha_min*180/numpy.pi,'-.k')
    plt.grid(True)
    plt.xlabel("t")
    plt.ylabel("alpha [deg]")
    beta = (beta_max + beta_min)/2 + numpy.sin(u[:,1])*(beta_max - beta_min)/2
    plt.subplot2grid((8,4),(7,0),colspan=5)
    plt.plot(t,beta,'b')
    #plt.hold(True)
    #plt.plot(t,beta*0+beta_max,'-.k')
    #plt.plot(t,beta*0+beta_min,'-.k')
    plt.grid(True)
    plt.xlabel("t")
    plt.ylabel("beta [-]")
    ######################################
    plt.subplots_adjust(0.0125,0.0,0.9,2.5,0.2,0.2)
    plt.show()
    print("pi =",pi,"\n")
#
