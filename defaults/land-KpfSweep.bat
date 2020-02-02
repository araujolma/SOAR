###############################################################################
# SOAR batch configuration file (.bat)
#
# author: Levi Araújo
#
###############################################################################

[main] # Main parameters

# Batch name
name = probLand-KpfSweep

# Mode ('explicit', 'variations', 'closed-loop')
mode = variations

# Logger mode ('file', 'screen', 'both')
logMode = both


###############################################################################
[variations_mode] # Parameters for batch runner in variations mode

# Problem name ('probRock', 'probLand', etc)
probName = probLand

# Configuration file (.its) for base case
baseFile = defaults/probLand.its

# Initial guess for subsequent cases ('scratch', 'base', '')
initGuesMode = 'scratch'

# Variations on input parameters to be run in each case (other than the first one)
#	 (each variation must be a | enclosed triple containing:
#         section, parameter name as in the file, value
#         separate items for each case via ', ' and 'enter')
#vars = sgra,GSS_PLimCte,1.0e5   |  accel,acc_max,100 ,  
#       sgra,GSS_PLimCte,1.0e10  
vars = accel,PFtol,1.1e-2 | sgra,GSS_PLimCte,1.0e4, 
       accel,PFtol,1.2e-2 | sgra,GSS_PLimCte,1.0e4 | sgra,N,1001 ,
       accel,PFtol,1.3e-2 | sgra,GSS_PLimCte,1.0e4 | sgra,N,1001 ,
       accel,PFtol,1.4e-2 | sgra,GSS_PLimCte,1.0e4 | sgra,N,1001 ,
       accel,PFtol,1.5e-2 | sgra,GSS_PLimCte,1.0e4 | sgra,N,1001 ,
       accel,PFtol,1.6e-2 | sgra,GSS_PLimCte,1.0e4 | sgra,N,1001 ,
       accel,PFtol,1.7e-2 | sgra,GSS_PLimCte,1.0e4 | sgra,N,1001 ,
       accel,PFtol,1.8e-2 | sgra,GSS_PLimCte,1.0e4 | sgra,N,1001 ,
       accel,PFtol,1.9e-2 | sgra,GSS_PLimCte,1.0e4 | sgra,N,1001
