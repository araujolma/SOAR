# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:02:30 2017

@author: munizlgmn
"""

import numpy, rockProp, itsme
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ##################
# PROBLEM DOMAIN:
# ##################
def declProb(opt=dict()):
# time discretization
    N = 5000 + 1#20000 + 1 #
    dt = 1.0/(N-1)
    t = numpy.arange(0,1.0+dt,dt)

# example rocket single stage to orbit L=0 D=0

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
    q = 3  # (Miele 1970)  # 7 (Miele 2003)

# Earth constants
    r_e = 6371.0           # km
    GM = 398600.4415       # km^3 s^-2
    grav_e = GM/r_e/r_e#9.8e-3        # km/s^2

# rocket constants
    Thrust = 40.0          # kg km/s²  1.3*m_initial # N
    Isp = 450.0            # s
    s_f = 0.05

 # restrictions
    alpha_min = -2*(numpy.pi)/180  # in rads
    alpha_max = 2*(numpy.pi)/180   # in rads
    beta_min = 0
    beta_max = 1

# tolerances
    tolP = 1.0e-8
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

        # OLD VERSION:

#        # adapt solution
#        t_rp,x_rp,u_rp,pi = rockProp.getRockTraj(boundary=boundary, constants=constants, restrictions=restrictions)
#        for i in range(n):
#            f_x = interp1d(t_rp, x_rp[:,i])
#            x[:,i] = f_x(t)
#        for i in range(m):
#            f_u = interp1d(t_rp,u_rp[:,i])
#            u[:,i] = f_u(t)


        # NEW VERSION:

    # Factors intervals
        fsup = numpy.array([0.68 + 0.3,486 + 100,1.77 + 0.3]) # Superior limit
        finf = numpy.array([0.68 - 0.3,486 - 100,1.77 - 0.3]) # Inferior limit


        # Automatic adjustament
        new_factors,t_rp,x_rp,u_rp = itsme.its(fsup, finf, h_final, 100.0, 1.0e-10)
        pi = numpy.array([t_rp[-1]])
        t_rp = t_rp/pi
        for i in range(n):
            f_x = interp1d(t_rp, x_rp[:,i])
            x[:,i] = f_x(t)

#        # Get proper transition times
#        tAlfaLow = 0.0
#        tAlfaHigh = 0.0
#        tBetaLow = 0.0
#        tBetaHigh = 0.0
#
#        N_rp = len(t_rp)
#
#        for k in range(N_rp):
#            if tAlfaLow == 0.0:
#                # find tAlfaLow
#                if u_rp[k,0] < 0.0:
#                    tAlfaLow = t_rp[k]#/N_rp
#            if tAlfaHigh == 0.0:
#                # find tAlfaHigh
#                if tAlfaLow>0 and u_rp[k,0] == 0.0:
#                    tAlfaHigh = t_rp[k]#/N_rp
#            if tBetaLow == 0.0:
#                if u_rp[k,1] < 0.5:
#                    tBetaLow = t_rp[k]#/N_rp
#            if tBetaHigh == 0.0:
#                if tBetaLow>0.0 and u_rp[k,1] > .5:
#                    tBetaHigh = t_rp[k]#/N_rp

        print("\nUn-interpolated control profiles:")

        plt.plot(t_rp,u_rp[:,0]*180/numpy.pi)#,'ob')
        plt.xlabel("t_rp")
        plt.ylabel("Attack angle [deg]")
        plt.title("u[0,:]")
        plt.show()

        plt.plot(t_rp,u_rp[:,1])#,'ob')
        plt.xlabel("t_rp")
        plt.ylabel("Propulsion profile [-]")
        plt.title("u[1,:]")
        plt.show()

#        print("tAlfaLow =",tAlfaLow)
#        print("tAlfaHigh =",tAlfaHigh)
#        print("tBetaLow =",tBetaLow)
#        print("tBetaHigh =",tBetaHigh)

        for i in range(m):
            f_u = interp1d(t_rp,u_rp[:,i])
            u[:,i] = f_u(t)

        # Perform inverse transformations for u:
        a1 = (alpha_max + alpha_min)/2
        a2 = (alpha_max - alpha_min)/2
        b1 = (beta_max + beta_min)/2
        b2 = (beta_max - beta_min)/2
        u[:,0] = numpy.arcsin((u[:,0]-a1)/a2)
        u[:,1] = numpy.arcsin((u[:,1]-b1)/b2)


    lam = 0.0*x.copy()
    mu = numpy.zeros(q)

    ###
    print("\nUn-interpolated solution from Souza's propagation:")

    plt.plot(t_rp,x_rp[:,0])
    plt.grid(True)
    plt.ylabel("h [km]")
    plt.xlabel("t_rp [-]")
    plt.show()

    plt.plot(t_rp,x_rp[:,1],'g')
    plt.grid(True)
    plt.ylabel("V [km/s]")
    plt.xlabel("t_rp [-]")
    plt.show()

    plt.plot(t_rp,x_rp[:,2]*180/numpy.pi,'r')
    plt.grid(True)
    plt.ylabel("gamma [deg]")
    plt.xlabel("t_rp [-]")
    plt.show()

    plt.plot(t_rp,x_rp[:,3],'m')
    plt.grid(True)
    plt.ylabel("m [kg]")
    plt.xlabel("t_rp [-]")
    plt.show()

    plt.plot(t_rp,u_rp[:,0]*180/numpy.pi,'k')
    plt.grid(True)
    plt.xlabel("t_rp [-]")
    plt.ylabel("u1 [deg]")
    plt.show()

    plt.plot(t_rp,u_rp[:,1],'c')
    plt.grid(True)
    plt.xlabel("t_rp [-]")
    plt.ylabel("u2 [-]")
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

# calculate r
    r = r_e + x[:,0]

# calculate grav
    grav = GM/r/r

# calculate phi:
    phi = numpy.empty((N,n))

# example rocket single stage to orbit L=0 D=0
    phi[:,0] = pi[0] * x[:,1] * sin(x[:,2])
    phi[:,1] = pi[0] * (beta * Thrust * cos(alpha)/x[:,3] - grav * sin(x[:,2]))
    phi[0,2] = 0.0
    for k in range(1,N):
        phi[k,2] = pi[0] * (beta[k] * Thrust * sin(alpha[k])/(x[k,3] * x[k,1]) + cos(x[k,2]) * ( x[k,1]/r[k]  -  grav[k]/x[k,1] ))
    phi[:,3] = - (pi[0] * beta * Thrust)/(grav_e * Isp)

    return phi

def calcPsi(sizes,x,boundary):
    N = sizes['N']
    h_final = boundary['h_final']
    V_final = boundary['V_final']
    gamma_final = boundary['gamma_final']

# example rocket single stage to orbit L=0 D=0
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

# example rocket single stage to orbit L=0 D=0
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

    # Gradients from example rocket single stage to orbit L=0 D=0
    psix = array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0]])
    psip = array([[0.0],[0.0],[0.0]])

#   calculate variables (arrays) alpha and beta
    aExp = .5*(alpha_max - alpha_min)
    alpha = (alpha_max + alpha_min)/2 + sin(u1)*aExp
    bExp = .5*(beta_max - beta_min)
    beta = (beta_max + beta_min)/2 + sin(u2)*bExp

    # calculate r
    r = r_e + x[:,0]

    # calculate grav
    grav = GM/r/r

    for k in range(N):
        sinGama = sin(x[k,2])
        cosGama = cos(x[k,2])

        sinAlpha = sin(alpha[k])
        cosAlpha = cos(alpha[k])

        cosu1 = cos(u1[k])
        cosu2 = cos(u2[k])

        V = x[k,1]
        m = x[k,3]

        # Expanded notation:
        DCosAlfaDu1 = -sinAlpha*aExp*cosu1
        DSinAlfaDu1 = cosAlpha*aExp*cosu1
        DBetaDu2 = bExp*cosu2
        if k==0:
            phix[k,:,:] = pi[0]*array([[0.0                           ,sinGama         ,V*cosGama               ,0.0                            ],
                                       [2*GM*sinGama/(r[k]**3)        ,0.0             ,-grav[k]*cosGama        ,-beta[k]*Thrust*cosAlpha/(m**2)],
                                       [0.0                           ,0.0             ,0.0                     ,0.0                            ],
                                       [0.0                           ,0.0             ,0.0                     ,0.0                            ]])

            phiu[k,:,:] = pi[0]*array([[0.0                           ,0.0                            ],
                                       [beta[k]*Thrust*DCosAlfaDu1/m  ,Thrust*cosAlpha*bExp*cosu2/m   ],
                                       [0.0                           ,0.0                            ],
                                       [0.0                           ,-Thrust*bExp*cosu2/(grav_e*Isp)]])

            phip[k,:,:] = array([[V*sinGama                                  ],
                                 [beta[k]*Thrust*cosAlpha/m - grav[k]*sinGama],
                                 [0.0                                        ],
                                 [-(beta[k]*Thrust)/(grav_e*Isp)             ]])
        else:
            phix[k,:,:] = pi[0]*array([[0.0                                        ,sinGama                                                                  ,V*cosGama                      ,0.0                                ],
                                       [2*GM*sinGama/(r[k]**3)                     ,0.0                                                                      ,-grav[k]*cosGama               ,-beta[k]*Thrust*cosAlpha/(m**2)    ],
                                       [cosGama*(-V/(r[k]**2)+2*GM/(V*(r[k]**3)))  ,(-beta[k]*Thrust*sinAlpha/(m*V**2)+cosGama*((1/r[k]) + grav[k]/(V**2)))  ,-sinGama*((V/r[k])-grav[k]/V)  ,-beta[k]*Thrust*sinAlpha/(V*(m**2))],
                                       [0.0                                        ,0.0                                                                      ,0.0                            ,0.0                                ]])

            phiu[k,:,:] = pi[0]*array([[0.0                               ,0.0                           ],
                                       [beta[k]*Thrust*DCosAlfaDu1/m      ,Thrust*cosAlpha*DBetaDu2/m    ],
                                       [beta[k]*Thrust*DSinAlfaDu1/(m*V)  ,Thrust*sinAlpha*DBetaDu2/(m*V)],
                                       [0.0                               ,-Thrust*DBetaDu2/(grav_e*Isp) ]])

            phip[k,:,:] = array([[V*sinGama                                                       ],
                                [beta[k]*Thrust*cosAlpha/m - grav[k]*sinGama                     ],
                                [(beta[k]*Thrust*sinAlpha)/(m*V) + cosGama*((V/r[k])-(grav[k]/V))],
                                [-(beta[k]*Thrust)/(grav_e*Isp)                                  ]])

        fu[k,:] = array([0.0,(pi[0]*Thrust*DBetaDu2)/(grav_e * Isp * (1-s_f))])
        fp[k,0] = (Thrust * beta[k])/(grav_e * Isp * (1-s_f))

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
# example rocket single stage to orbit L=0 D=0
    f = calcF(sizes,x,u,pi,constants,restrictions)
    I = f.sum()

    return I
