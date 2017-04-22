# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 16:09:05 2017

@author: munizlgmn
"""
import math

constant = 0.03416319
R = 287.04

def cal(p0, t0, a0, h0, h1):
    if a0 != 0:
        t1 = t0 + a0*(h1 - h0)
        p1 = p0*(t0/(t0 + a0*(h1 - h0)))**(constant/a0)
    else:
        t1 = t0
        p1 = p0*math.exp(-constant*(h1 - h0)/t0)
    return t1, p1
   
def rho(altitude):
    altitude *= 1000    # converting from km to m
    a = [-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002, 0.0]
    h_ref = [0, 11000, 20000, 32000, 47000, 51000, 71000, 84852]
    p_ref = [101325, 22632.1, 5474.8, 868.01, 110.90, 66.938, 3.9564, 0.37338]
    t_ref = [288.15, 216.65,  216.65, 228.65, 270.65, 270.65, 214.65, 186.8673]
    if altitude < 0 or altitude > 84852:
        density = 0.0
#        print("altitude must be in [0, 84852]\n")
    else:
        for i in range(0, 8):
            if altitude <= h_ref[i]:
                if i == 0:
                    temperature, pressure = cal(p_ref[i], t_ref[i], a[i], h_ref[i], altitude)            
                else:
                    temperature, pressure = cal(p_ref[i], t_ref[i], a[i-1], h_ref[i], altitude)
                break;
        density = pressure / (R * temperature)     # kg/m^3
        density *= 1e9                             # converting to kg/km^3
#    strformat = 'Altitude: {0:.1f} \nTemperature: {1:.3f} \nPressure: {2:.3f} \nDensity: {3:.6f}\n'
#    print(strformat.format(altitude, temperature, pressure, density))
    return density
    
# test cases:
#teste1 = rho(-10)
#teste2 = rho(0)
#teste3 = rho(10000)
#teste4 = rho(20000)
#teste5 = rho(30000)
#teste6 = rho(40000)
#teste7 = rho(50000)
#teste8 = rho(60000)
#teste9 = rho(70000)
#teste10 = rho(71001)
#teste11 = rho(80000)
#teste12 = rho(84852)
#teste13 = rho(84853)
#teste14 = rho(9000000)