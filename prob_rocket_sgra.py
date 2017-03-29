# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:02:30 2017

@author: munizlgmn
"""

import numpy, rockProp
from scipy.interpolate import interp1d

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
    h_final = 463     # km
    V_final = 7.633   # km/s    
    gamma_final = 0.0 # rad
    #m_final = free   # kg

# matrix sizes    
    n = 4
    m = 2
    p = 1 
    q = 3  # (Miele 1970)  # 7 (Miele 2003)     

# Earth constants
    grav_e = 9.8e-3        # km/s^2
    r_e = 6371             # km
    GM = 398600.4415       # km^3 s^-2
    
# rocket constants     
    Thrust = 40.0          # kg km/s²  1.3*m_initial # N
    Isp = 450              # s
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
        # adapt solution        
        t_rp,x_rp,u_rp,pi = rockProp.getRockTraj()        
        for i in range(n):
            f_x = interp1d(t_rp, x_rp[:,i])
            x[:,i] = f_x(t)
        for i in range(m):
            f_u = interp1d(t_rp,u_rp[:,i])
            u[:,i] = f_u(t)
                   
    lam = 0.0*x.copy()
    mu = numpy.zeros(q)   

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
    alpha = (alpha_max + alpha_min)/2 + numpy.sin(u1)*(alpha_max - alpha_min)/2
    beta = (beta_max + beta_min)/2 + numpy.sin(u2)*(beta_max - beta_min)/2

# calculate r
    r = r_e + x[:,0]
    
# calculate grav
    grav = GM/r/r

# calculate phi:
    phi = numpy.empty((N,n))
    
# example rocket single stage to orbit L=0 D=0    
    phi[:,0] = pi[0] * x[:,1] * sin(x[:,2])
    phi[:,1] = pi[0] * (beta * Thrust * cos(u[:,0])/x[:,3] - grav * sin(x[:,2]))
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
    u1 = u[:,0]
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
    psix = numpy.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0]])        
    psip = numpy.array([[0.0],[0.0],[0.0]])
    
#   calculate variables alpha and beta
    alpha = (alpha_max + alpha_min)/2 + numpy.sin(u1)*(alpha_max - alpha_min)/2
    beta = (beta_max + beta_min)/2 + numpy.sin(u2)*(beta_max - beta_min)/2
      
    # calculate r
    r = r_e + x[:,0]
    
    # calculate grav
    grav = GM/r/r
        
    for k in range(N):  
        sinGama = numpy.sin(x[k,2])
        cosGama = numpy.cos(x[k,2])
        
        sinAlpha = numpy.sin(alpha[k])
        cosAlpha = numpy.cos(alpha[k])
        
        cosu1 = numpy.cos(u1[k])
        cosu2 = numpy.cos(u2[k])
        
        V = x[k,1]        
        m = x[k,3]

    # Expanded notation:
        if k==0:
            phix[k,:,:] = numpy.array([[0.0                           ,pi[0]*sinGama   ,pi[0]*V*cosGama         ,0.0                                  ],
                                       [pi[0]*2*GM*sinGama/(r[k]**3)  ,0.0             ,-pi[0]*grav[k]*cosGama  ,-pi[0]*beta[k]*Thrust*cosAlpha/(m**2)],
                                       [0.0                           ,0.0             ,0.0                     ,0.0                                  ],
                                       [0.0                           ,0.0             ,0.0                     ,0.0                                  ]])
            
            phiu[k,:,:] = numpy.array([[0.0                                                                   ,0.0                                                       ],
                                       [-pi[0]*beta[k]*Thrust*sinAlpha*((alpha_max - alpha_min)/2)*cosu1/(m)  ,pi[0]*Thrust*cosAlpha*((beta_max - beta_min)/2)*cosu2/(m) ],
                                       [0.0                                                                   ,0.0                                                       ],
                                       [0.0                                                                   ,-pi[0]*Thrust*((beta_max - beta_min)/2)*cosu2/(grav_e*Isp)]])
            
            phip[k,:,:] = numpy.array([[V*sinGama                                    ],
                                       [beta[k]*Thrust*cosAlpha/(m) - grav[k]*sinGama],
                                       [0.0                                          ],
                                       [-(beta[k]*Thrust)/(grav_e*Isp)               ]])
        else:
            phix[k,:,:] = numpy.array([[0.0                                              ,pi[0]*sinGama                                                                    ,pi[0]*V*cosGama                      ,0.0                                      ],
                                       [pi[0]*2*GM*sinGama/(r[k]**3)                     ,0.0                                                                              ,-pi[0]*grav[k]*cosGama               ,-pi[0]*beta[k]*Thrust*cosAlpha/(m**2)    ],
                                       [pi[0]*cosGama*(-V/(r[k]**2)+2*GM/(V*(r[k]**3)))  ,pi[0]*(-beta[k]*Thrust*sinAlpha/(m*V**2)+cosGama*((1/r[k])+grav[k]/(V**2)))      ,-pi[0]*sinGama*((V/r[k])-grav[k]/V)  ,-pi[0]*beta[k]*Thrust*sinAlpha/(V*(m**2))],
                                       [0.0                                              ,0.0                                                                              ,0.0                                  ,0.0                                      ]])
        
            phiu[k,:,:] = numpy.array([[0.0                                                                    ,0.0                                                        ],
                                       [-pi[0]*beta[k]*Thrust*sinAlpha*((alpha_max - alpha_min)/2)*cosu1/(m)   ,pi[0]*Thrust*cosAlpha*((beta_max - beta_min)/2)*cosu2/(m)  ],
                                       [pi[0]*beta[k]*Thrust*cosAlpha*((alpha_max - alpha_min)/2)*cosu1/(m*V)  ,pi[0]*Thrust*sinAlpha*((beta_max - beta_min)/2)*cosu2/(m*V)],
                                       [0.0                                                                    ,-pi[0]*Thrust*((beta_max - beta_min)/2)*cosu2/(grav_e*Isp)]])
            
            phip[k,:,:] = numpy.array([[V*sinGama                                                     ],
                                       [beta[k]*Thrust*cosAlpha/(m) - grav[k]*sinGama                 ],
                                       [(beta[k]*Thrust*sinAlpha)/(m*V)+cosGama*((V/r[k])-(grav[k]/V))],
                                       [-(beta[k]*Thrust)/(grav_e*Isp)                               ]])
   
        fu[k,:] = numpy.array([0.0,(pi[0]*Thrust*((beta_max - beta_min)/2)*cosu2)/(grav_e * Isp * (1-s_f))])
        fp[k,0] =(Thrust * beta[k])/(grav_e * Isp * (1-s_f))
    
    Grads['phix'] = phix.copy()
    Grads['phiu'] = phiu.copy()
    Grads['phip'] = phip.copy()
    Grads['fx'] = fx.copy()
    Grads['fu'] = fu.copy()
    Grads['fp'] = fp.copy()
#    Grads['gx'] = gx.copy()
#    Grads['gp'] = gp.copy()
    Grads['psix'] = psix.copy()
    Grads['psip'] = psip.copy()        
    return Grads
    
def calcI(sizes,x,u,pi,constants,restrictions):
# example rocket single stage to orbit L=0 D=0
    f = calcF(sizes,x,u,pi,constants,restrictions)
    I = f.sum()
    
    return I