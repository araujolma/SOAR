# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 16:09:05 2017

@author: munizlgmn
"""
import math
import numpy

constant = 0.03416319
R = 287.04
a = [-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002, 0.0]
h_ref = [0., 11000., 20000., 32000., 47000., 51000., 71000., 84852.]
p_ref = [101325., 22632.1, 5474.8, 868.01, 110.90, 66.938, 3.9564, 0.37338]
t_ref = [288.15, 216.65,  216.65, 228.65, 270.65, 270.65, 214.65, 186.8673]

def cal(p0, t0, a0, h0, h1):
    if a0 != 0:
        t1 = t0 + a0*(h1 - h0)
        p1 = p0*(t0/(t0 + a0*(h1 - h0)))**(constant/a0)
    else:
        t1 = t0
        p1 = p0*math.exp(-constant*(h1 - h0)/t0)
    return t1, p1

# def rhoFast(altitude):
#     altitude *= 1000    # converting from km to m
#     if altitude <= 0.:
#         pressure, temperature = p_ref[0], t_ref[0]
#     elif altitude >= 84852.:
#         pressure, temperature = p_ref[-1], t_ref[-1]
# #        print("altitude must be in [0, 84852]\n")
#     else:
#         for i in range(0, 8):
#             if altitude <= h_ref[i]:
#                 if i == 0:
#                     temperature, pressure = cal(p_ref[i], t_ref[i], a[i], h_ref[i], altitude)
#                 else:
#                     temperature, pressure = cal(p_ref[i], t_ref[i], a[i-1], h_ref[i], altitude)
#                 break
#     density = pressure / (R * temperature)     # kg/m^3
#     density *= 1e9                             # converting to kg/km^3
# #    strformat = 'Altitude: {0:.1f} \nTemperature: {1:.3f} \nPressure: {2:.3f} \nDensity: {3:.6f}\n'
# #    print(strformat.format(altitude, temperature, pressure, density))
#     return density


def rhoFast(altitude):
    altitude *= 1000    # converting from km to m
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
                break
        density = pressure / (R * temperature)     # kg/m^3
        density *= 1e9                             # converting to kg/km^3
#    strformat = 'Altitude: {0:.1f} \nTemperature: {1:.3f} \nPressure: {2:.3f} \nDensity: {3:.6f}\n'
#    print(strformat.format(altitude, temperature, pressure, density))
    return density

def atm(altitude):
    # This is a function based on rhoFast implementation that returns rho,
    # pressure, temperature and sound velocity

    altitude *= 1000  # converting from km to m
    if altitude < h_ref[0]:
        pressure = p_ref[0]
        temperature = t_ref[0]
    elif altitude > h_ref[-1]:
        pressure = p_ref[-1]
        temperature = t_ref[-1]
    else:
        for i in range(0, 8):
            if altitude <= h_ref[i]:
                if i == 0:
                    temperature, pressure = cal(p_ref[i], t_ref[i], a[i],
                                                h_ref[i], altitude)
                else:
                    temperature, pressure = cal(p_ref[i], t_ref[i], a[i-1],
                                                h_ref[i], altitude)
                break

    density = pressure / (R * temperature)          # kg/m^3
    density *= 1e9                                  # converting to kg/km^3
    asound = math.sqrt(1.4 * R * temperature)*1e-3  # [km/s]
    pressure = pressure*1e-3                        # [kPa]

    return density, pressure, temperature, asound


def rho(altitude):
    # rho implementation for list inputs
    ta = type(altitude)
    if ta is numpy.float64:
        # more frequent situation during integration
        ans = rhoFast(altitude)
    elif (ta is int) or (ta is float):
        # other possible scalar situations situation during integration
        ans = rhoFast(altitude)
    else:
        # vetorial situation
        ans = list(map(rhoFast, altitude))

    return ans

def calSGRA(indx, h):
    """Calculate atmospheric temperature and pressure according to the model.

    Designed for use with rhoSGRA.
    """


    if indx == 2 or indx == 5:# or indx == 8:
        # These are the ones that use a=0
        T = t_ref[indx]
        p = p_ref[indx] * numpy.exp(-constant * (h-h_ref[indx])/T)
    else:
        # the others
        t0, a0, h0 = t_ref[indx], a[indx-1], h_ref[indx]
        T = t0 + a0*(h - h0)
        p = p_ref[indx] * (t0/T) ** (constant/a0)

    return T, p

# works for non-sorted, reduction to 62-63% in that case, 63% sorted as well...
# def rhoSGRA(altArray):
#     """ Density implementation, optimized for SGRA.
#
#     Some of the ideas in this optimization are:
#         - avoid floating point comparisons
#         - take advantage of the non- remember the index"""
#
#     pArray = numpy.empty_like(altArray)
#     TArray = numpy.empty_like(altArray)
#
#     altArray *= 1000. # converting to m
#
#     # index for starting the search
#     #print("h_ref = ",str(h_ref))
#     for i, h in enumerate(altArray):
#         #print("i = {}, h = {} km".format(i,h/1000.))
#         if h <= 0.:
#             TArray[i], pArray[i] = t_ref[0], p_ref[0]
#         elif h > 84852.:
#             TArray[i], pArray[i] = t_ref[-1], 0.#p_ref[-1]#
#         else:
#             for indx in range(1, 8):
#                 if h <= h_ref[indx]:
#                     TArray[i], pArray[i] = calSGRA(indx, h)
#                     #print("indx = {}, h_ref = {} km".format(indx, h_ref[indx]/1000.))
#                     break
#
#     dens = pArray * 1e9 / (R * TArray)        # kg/km³
#     #dens = pArray / (R * TArray)              # kg/m³
#     #dens *= 1e9                               # converting to kg/km³
#     altArray *= 1e-3                          # converting back to km
#     return dens

# Sorting and unsorting! Reduces to 55% in unsorted cases. 51-52% if sorted.
def rhoSGRA(altArray):
    """ Density implementation, optimized for SGRA.

    Some of the ideas in this optimization are:
        - avoid floating point comparisons
        - sorting the input altitude array
        - taking advantage of the sorted input altitude array when searching for the
        reference altitude."""

    # declare the arrays
    pArray, TArray = numpy.empty_like(altArray), numpy.empty_like(altArray)

    # get the indexes for sorting the altitude array
    indxSort = numpy.argsort(altArray)
    #print("indxSort = {}".format(indxSort))

    altArray = 1000. * altArray[indxSort] # sorting and converting to m

    # index for starting the search. Since the first one is zero, let's go with 1
    last_indx = 1
    for i, h in enumerate(altArray):
        if h <= 0.:
            TArray[i], pArray[i] = t_ref[0], p_ref[0]
        elif h > 84852.:
            # TODO: according to this, the density vanishes after 84.852 km;
            #  It is just not right!
            TArray[i], pArray[i] = t_ref[-1], 0.#p_ref[-1]#
        else:
            if h <= h_ref[last_indx]:
                TArray[i], pArray[i] = calSGRA(last_indx, h)
            else:
                # Last index did not work; since the array is sorted, look up
                for indx in range(last_indx+1, 8):
                    if h <= h_ref[indx]:
                        # reset the index to save time for the next altitude
                        last_indx = indx
                        TArray[i], pArray[i] = calSGRA(last_indx, h)
                        break

    dens = pArray * 1e9 / (R * TArray)         # kg/km³
    dens = dens[numpy.argsort(indxSort)]       # unsorting back
    return dens

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

if __name__ == "__main__":
    # declare an array with altitudes
    altVec = numpy.arange(0.,200.,0.1)
    #altVec = numpy.array([0.,10.,20.,50.,100.,200.])
    altVec = numpy.random.permutation(altVec)
    print("altVec = {} km".format(altVec))

    # calculate atmospheric density in old method
    dens_base = rho(altVec)
    #print("dens_base = {} kg/m³".format(numpy.array(dens_base)/1e9))
    # calculate atmospheric density in new method
    dens = rhoSGRA(altVec)
    #print("dens = {} kg/m³".format(dens/1e9))
    # calculate error, print it
    error = dens-dens_base
    print("error = {}".format(sum(error ** 2.)))
    # plot it
    # import matplotlib.pyplot as plt
    # plt.semilogy(altVec, dens, label='new')
    # plt.semilogy(altVec,dens_base, '--',label='old')
    # #plt.plot(altVec,error, label='error')
    # #plt.plot(error, label='error')
    # plt.grid(True)
    # plt.legend()
    # plt.xlabel("Altitude [km]")
    # plt.ylabel("Density [kg/km³]")
    # plt.show()

    # Test the difference (with a timing application such as cProfile, naturally)
    NTest = 10000
    for i in range(NTest):
        altVec = numpy.random.permutation(altVec)
        dens_base = rho(altVec)
        dens = rhoSGRA(altVec)
        error = dens - dens_base
        print("error = {}".format(sum(error ** 2.)))
    print(altVec)