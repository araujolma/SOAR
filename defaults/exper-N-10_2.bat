###############################################################################
# SOAR batch configuration file (.bat)
#
# author: Levi Araújo
#
###############################################################################

[main] # Main parameters

# Batch name
name = prob10_2-N-experiment

# Mode ('explicit', 'variations', 'closed-loop')
mode = variations

# Logger mode ('file', 'screen', 'both')
logMode = both

# Data to retrieve from the solution for post processing 
postProcKeyList =  NIterGrad,
                   GSStotObjEval,
                   GSSavgObjEval,
                   N,
                   relErrI_opt,
                   rmsErr_x_opt,
                   rmsErr_u_opt,
                   rmsErr_pi_opt,
                   timer

###############################################################################
[variations_mode] # Parameters for batch runner in variations mode

# Problem name ('probRock', 'probLand', etc)
probName = prob10_2

# Configuration file (.its) for base case
baseFile = defaults/prob10_2.its

# Initial guess for subsequent cases ('scratch', 'base', '')
initGuesMode = 'scratch'

# Variations on input parameters to be run in each case (other than the first one)
#	 (each variation must be a | enclosed triple containing:
#         section, parameter name as in the file, value
#         separate items for each case via ', ' and 'enter')
vars = sgra,N,41 ,
       sgra,N,21 ,
       sgra,N,11 ,
       sgra,N,5
#vars = sgra,GSS_PLimCte,1.0e5   |  accel,acc_max,100 ,  
#       sgra,GSS_PLimCte,1.0e10  
#vars = accel,PFtol,1.1e-2 | sgra,GSS_PLimCte,1.0e4, 
#       accel,PFtol,1.2e-2 | sgra,GSS_PLimCte,1.0e4 | sgra,N,1001 ,
#       accel,PFtol,1.3e-2 | sgra,GSS_PLimCte,1.0e4 | sgra,N,1001 ,
#       accel,PFtol,1.4e-2 | sgra,GSS_PLimCte,1.0e4 | sgra,N,1001 ,
#       accel,PFtol,1.5e-2 | sgra,GSS_PLimCte,1.0e4 | sgra,N,1001 ,
#       accel,PFtol,1.6e-2 | sgra,GSS_PLimCte,1.0e4 | sgra,N,1001 ,
#       accel,PFtol,1.7e-2 | sgra,GSS_PLimCte,1.0e4 | sgra,N,1001 ,
#       accel,PFtol,1.8e-2 | sgra,GSS_PLimCte,1.0e4 | sgra,N,1001 ,
#       accel,PFtol,1.9e-2 | sgra,GSS_PLimCte,1.0e4 | sgra,N,1001
