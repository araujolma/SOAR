#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 17:49:40 2019

@author: levi

A module for automating the generation of ITSME case files (.its).
"""

import itsme, pprint
import matplotlib.pyplot as plt
from utils import getNowStr

def write_parameter(file_handle,par_name,par_value):
    """Write a given parameter value to the file."""

    if par_name == 'Thrust':
        msg = "\n# Thrust [kN]\nT = {}\n".format(par_value)
    elif par_name == 'Payload':
        msg = "\n# Payload Mass [kg]\nMu = {}\n".format(par_value)
    elif par_name == 'Area':
        msg = "\n# Reference area [km²]\ns_ref = {}\n".format(par_value)
    else:
        raise Exception('write_parameter: Unknown parameter "'+par_name+'"')

    file_handle.write(msg)

def make_case(Thrust,Payload,Area,base_file,out_file):
    """Prepare the case file. The change is only in Thrust, Payload mass and
    Reference area. All other parameters from the base case are kept equal."""

    msg = "\nPreparing a case with:" + \
          "\nThrust = {:.4F} kN, ".format(Thrust) + \
          "Payload mass = {:.3F} kg, ".format(Payload) + \
          "Ref. area = {:4E} km²".format(Area)
    print(msg)

    # Opening file handles
    base_file_hand = open(base_file, 'r')
    outp_file_hand = open(out_file, 'w+')

    # The prefix is for commenting out the "original" value of the parameter
    prefix = ''
    for line in base_file_hand:
        if line.startswith('# Thrust [kN]'):
            write_parameter(outp_file_hand,'Thrust',Thrust)
            prefix = '# original value: '
        elif line.startswith('# Payload mass [kg]'):
            write_parameter(outp_file_hand,'Payload',Payload)
            prefix = '# original value: '
        elif line.startswith('# Reference area'):
            write_parameter(outp_file_hand,'Area',Area)
            prefix = '# original value: '
        else:
            outp_file_hand.write(prefix+line)
            prefix = ''

    input("IAE? ")

def try_case(file):
    """Run a given case and calculate the obtained values of interest
    (initial mass, thrust-to-weight, wing loading)."""

    # Run the case
    _, _, _, _, _, inputDict, _, mass0, _ = itsme.sgra(file)

    # Get the parameters:
    M0 = mass0[0] # Initial rocket mass
    W0 = M0 * inputDict['g0'] # Initial rocket weight [kN]
    TWR = inputDict['Tlist'][0] / W0 # Thrust to weight ratio [-]
    WL = M0 / (inputDict['s_ref'] * 1e6) # Wing loading [kgf/m²]
    pars = {'M0': M0, 'TWR': TWR, 'WL':WL}

    msg = "\nObtained values:\n" + \
          "M0 = {:.1F}\n".format(M0) + \
          "TWR = {:.4G}\nWL = {:.4E} kgf/m²".format(TWR,WL)
    print(msg)
    return pars, inputDict

def generate_case(Targets,Tols,base_file='',outp_file=''):
    """ This is the main function, which generates the case by
    iteration until the parameter values are met."""

    # Base file (for the parameters that don't need to change)
    if base_file == '':
        base_file = 'defaults/probRock.its'

    # Output file
    if outp_file == '':
        outp_file = 'caseGen_' + getNowStr() + '.its'


    # Try the base case first
    pars,inputDict = try_case(base_file)

    # Find out which variables are the targets,
    # in order to decide the convergence scheme
    Targ_keys = Targets.keys()

    if 'TWR' in Targ_keys and 'WL' in Targ_keys:
        # TODO: this is not working yet...
        # Iteration loops for matching ratios TWR and WL

        #while abs(WL_obt/WL-1.) > 1e-3:
        Area = inputDict['s_ref']
        # Inner loop for Thrust-to-Weight Ratio
        while abs(pars['TWR']/Targets['TWR']-1.) > Tols['TWR']:
            Thr = inputDict['Tlist'][0]
            mPayl = inputDict['Mu']

            mPayl *= 0.9 * pars['TWR']/Targets['TWR']

            make_case(Thr,mPayl,Area,base_file,outp_file)
            pars,inputDict = try_case(outp_file)
            input("IAE? ")
    elif 'M0' in Targ_keys:

        # Iteration loop for matching the initial mass:
        # Just adjust the payload mass until the initial mass is as required.
        Area = inputDict['s_ref']
        Thr = inputDict['Tlist'][0]
        Mu = inputDict['Mu']

        # Surely this will result in a vehicle lighter than necessary
        Mu_low = 0.0
        # Surely this will result in a vehicle heavier than necessary
        Mu_high = Mu * (Targets['M0']/pars['M0'])

        while abs(pars['M0']/Targets['M0']-1.) > Tols['M0']:

            # Perform bisection
            if pars['M0'] > Targets['M0']:
                Mu_high = Mu
            else:
                Mu_low = Mu
            Mu = 0.5 * (Mu_high+Mu_low)

            print("\nPayload mass bounds:\n"
                  "-low = {:.4F}, -high = {:.4F}".format(Mu_low,Mu_high))
            make_case(Thr,Mu,Area,base_file,outp_file)
            pars,inputDict = try_case(outp_file)

    print("\n\nExecution finished.\n"
          "Check file {} for results.\n".format(outp_file))

if __name__ == "__main__":
    # Basic Miele (2003) case
    generate_case({'M0': 3000.},{'M0': 1e-5})
    #generate_case({'TWR':1.3,'WL':1890.})
