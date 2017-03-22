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

# boundary conditions:
# initial condition
    h_initial = 0.0            # km  
    V_initial = 0.0            # km/s
    gamma_initial = numpy.pi/2 # rad
    m_initial = 50000          # kg
# final condition
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
    Thrust = 40.0   # kg km/s²  1.3*m_initial # N
    Isp = 450              # s
    s_f = 0.05
    
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

# Get initialization mode
    initMode = opt.get('initMode','default')
    x = numpy.zeros((N,n))
    u = numpy.zeros((N,m))    
        
    if initMode == 'default':
        # artesanal handicraft with L and D (Miele 2003)        
        x[:,0] = h_final*numpy.sin(numpy.pi*t.copy()/2)
        x[:,1] = 1.0e3*(-0.4523*t.copy()**5 + 1.2353*t.copy()**4-1.1884*t.copy()**3+0.4527*t.copy()**2-0.0397*t.copy())
        x[:,2] = (numpy.pi/2)*(1.0-t.copy()*t.copy())
        x[:,3] = m_initial*(1.0-0.89*t.copy())
        u[:,1] = numpy.ones(N)
        #u2 = (numpy.pi/2)*numpy.ones(N)
        #u[:,1] = 0.5 + 0.5*numpy.sin(u2[:])        
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
        #u2 = numpy.arcsin(2*u[:,1] - 1)
            
    lam = 0.0*x.copy()
    mu = numpy.zeros(q)        

    tol = dict()
    tol['P'] = 1.0e-8
    tol['Q'] = 1.0e-5

    return sizes,t,x,u,pi,lam,mu,tol,constants,boundary
    
    
def calcPhi(sizes,x,u,pi,constants):
    N = sizes['N']
    n = sizes['n']
    grav_e = constants['grav_e']
    Thrust = constants['Thrust']
    Isp = constants['Isp']
    r_e = constants['r_e']
    GM = constants['GM']
    sin = numpy.sin
    cos = numpy.cos
    
# defining auxiliary variable control u2
    u2 = numpy.arcsin(2*u[:,1] - 1)
    new_beta = 0.5 + 0.5*sin(u2)

# calculate r
    r = r_e + x[:,0]
    
# calculate grav
    grav = GM/r/r

# calculate phi:
    phi = numpy.empty((N,n))
    
# example rocket single stage to orbit L=0 D=0    
    phi[:,0] = pi[0] * x[:,1] * sin(x[:,2])
    phi[:,1] = pi[0] * (new_beta * Thrust * cos(u[:,0])/x[:,3] - grav * sin(x[:,2]))
    phi[0,2] = 0.0
    for k in range(1,N):
        phi[k,2] = pi[0] * (new_beta[k] * Thrust * sin(u[k,0])/(x[k,3] * x[k,1]) + cos(x[k,2]) * ( x[k,1]/r[k]  -  grav[k]/x[k,1] ))
    phi[:,3] = - (pi[0] * new_beta * Thrust)/(grav_e * Isp)
   
    return phi
    
def calcPsi(sizes,x,boundary):    
    N = sizes['N']
    h_final = boundary['h_final']
    V_final = boundary['V_final']
    gamma_final = boundary['gamma_final']
  
# example rocket single stage to orbit L=0 D=0
    psi = numpy.array([x[N-1,0]-h_final,x[N-1,1]-V_final,x[N-1,2]-gamma_final])

    return psi
    
def calcF(sizes,x,u,pi,constants):
    N = sizes['N']
    f = numpy.empty(N)
    grav_e = constants['grav_e']
    Thrust = constants['Thrust']
    Isp = constants['Isp']
    s_f = constants['s_f']

# defining auxiliary variable control u2
    u2 = numpy.arcsin(2*u[:,1] - 1)
    new_beta = 0.5 + 0.5*numpy.sin(u2)
    
# example rocket single stage to orbit L=0 D=0    
    f = ((Thrust * pi[0])/(grav_e * (1-s_f) * Isp)) * new_beta
    
    return f

def calcGrads(sizes,x,u,pi,constants):
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
    
    # defining auxiliary variable control u2
    u2 = numpy.arcsin(2*u[:,1] - 1)
    new_beta = 0.5 + 0.5*numpy.sin(u2)
    
    #beta = u[k,1]
    
    # calculate r
    r = r_e + x[:,0]
    
    # calculate grav
    grav = GM/r/r
        
    for k in range(N):  
        sinGama = numpy.sin(x[k,2])
        cosGama = numpy.cos(x[k,2])
        
        sinAlfa = numpy.sin(u[k,0])
        cosAlfa = numpy.cos(u[k,0])
        
        cosu2 = numpy.cos(u2[k])
        
        V = x[k,1]
        
        m = x[k,3]

    # Expanded notation:
        if k==0:
            phix[k,:,:] = numpy.array([[0.0                           ,pi[0]*sinGama   ,pi[0]*V*cosGama         ,0.0                                     ],
                                       [pi[0]*2*GM*sinGama/(r[k]**3)  ,0.0             ,-pi[0]*grav[k]*cosGama  ,-pi[0]*new_beta[k]*Thrust*cosAlfa/(m**2)],
                                       [0.0                           ,0.0             ,0.0                     ,0.0                                     ],
                                       [0.0                           ,0.0             ,0.0                     ,0.0                                     ]])
            
            phiu[k,:,:] = numpy.array([[0.0                                    ,0.0                                 ],
                                       [-pi[0]*new_beta[k]*Thrust*sinAlfa/(m)  ,pi[0]*Thrust*cosAlfa*0.5*cosu2/(m)  ],
                                       [0.0                                    ,0.0                                 ],
                                       [0.0                                    ,-pi[0]*Thrust*0.5*cosu2/(grav_e*Isp)]])
            
            phip[k,:,:] = numpy.array([[V*sinGama                                       ],
                                       [new_beta[k]*Thrust*cosAlfa/(m) - grav[k]*sinGama],
                                       [0.0                                             ],
                                       [-(new_beta[k]*Thrust)/(grav_e*Isp)              ]])
        else:
            phix[k,:,:] = numpy.array([[0.0                                              ,pi[0]*sinGama                                                                    ,pi[0]*V*cosGama                      ,0.0                                         ],
                                       [pi[0]*2*GM*sinGama/(r[k]**3)                     ,0.0                                                                              ,-pi[0]*grav[k]*cosGama               ,-pi[0]*new_beta[k]*Thrust*cosAlfa/(m**2)    ],
                                       [pi[0]*cosGama*(-V/(r[k]**2)+2*GM/(V*(r[k]**3)))  ,pi[0]*(-new_beta[k]*Thrust*sinAlfa/(m*V**2)+cosGama*((1/r[k])+grav[k]/(V**2)))   ,-pi[0]*sinGama*((V/r[k])-grav[k]/V)  ,-pi[0]*new_beta[k]*Thrust*sinAlfa/(V*(m**2))],
                                       [0.0                                              ,0.0                                                                              ,0.0                                  ,0.0                                         ]])
        
            phiu[k,:,:] = numpy.array([[0.0                                      ,0.0                                 ],
                                       [-pi[0]*new_beta[k]*Thrust*sinAlfa/(m)    ,pi[0]*Thrust*cosAlfa*0.5*cosu2/(m)  ],
                                       [pi[0]*new_beta[k]*Thrust*cosAlfa/(m*V)   ,pi[0]*Thrust*sinAlfa*0.5*cosu2/(m*V)],
                                       [0.0                                      ,-pi[0]*Thrust*0.5*cosu2/(grav_e*Isp)]])
            
            phip[k,:,:] = numpy.array([[V*sinGama                                                        ],
                                       [new_beta[k]*Thrust*cosAlfa/(m) - grav[k]*sinGama                 ],
                                       [(new_beta[k]*Thrust*sinAlfa)/(m*V)+cosGama*((V/r[k])-(grav[k]/V))],
                                       [-(new_beta[k]*Thrust)/(grav_e*Isp)                               ]])
   
        fu[k,:] = numpy.array([0.0,(pi[0]*Thrust*0.5*cosu2)/(grav_e * Isp * (1-s_f))])
        fp[k,0] =(Thrust * new_beta[k])/(grav_e * Isp * (1-s_f))
    
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
    
def calcI(sizes,x,u,pi,constants):
# example rocket single stage to orbit L=0 D=0
    f = calcF(sizes,x,u,pi,constants)
    I = f.sum()
    
    return I