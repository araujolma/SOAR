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

# List of keys for post-processing
postProcKeyList = constants['Kpf'],
                  I,
                  Q,
                  tol['Q'],
                  NIterRest, 
                  N,
                  constants['GSS_PLimCte']

###############################################################################
[variations_mode] # Parameters for batch runner in variations mode

# Problem name ('probRock', 'probLand', etc)
probName = probLand

# Configuration file (.its) for base case
baseFile = probLand-alt.its

# Initial guess for subsequent cases ('scratch', 'base', '')
initGuesMode = scratch

# Variations on input parameters to be run in each case (other than the first one)
#	 (each variation must be a | enclosed triple containing:
#         section, parameter name as in the file, value
#         separate items for each case via ', ' and 'enter')
vars = sgra, N, 201,
       sgra, N, 1001,
       sgra, GSS_PLimCte, 1.0e7,
       sgra, GSS_PLimCte, 1.0e6,
       sgra, GSS_PLimCte, 1.0e5,
       sgra, GSS_PLimCte, 1.0e4,
       sgra, GSS_PLimCte, 1.0e3,
       sgra, GSS_PLimCte, 1.0e2     


#vars = accel,PFtol,1e-3,
#       accel,PFtol,5e-3,
#       accel,PFtol,5e-3 | sgra,tolQ,1e-4 ,
#       accel,PFtol,1e-2,
#       accel,PFtol,1.1e-2,
#       accel,PFtol,1.2e-2,
#       accel,PFtol,1.4e-2,
#       accel,PFtol,1.6e-2,
#       accel,PFtol,1.8e-2,
#       accel,PFtol,2e-2,

#vars = accel,PFtol,2e-2 | sgra,GSS_PLimCte,1.0e4 | sgra,N,1001 ,
#       accel,PFtol,5e-2 | sgra,GSS_PLimCte,1.0e4 | sgra,N,1001 ,
#       accel,PFtol,1e-1 | sgra,GSS_PLimCte,1.0e4 | sgra,N,1001 ,
#       accel,PFtol,1e-1 | sgra,GSS_PLimCte,1.0e4 | sgra,N,2001 ,
