###############################################################################
# SOAR batch configuration file (.bat)
#
# author: Levi Araújo
#
###############################################################################

[main] # Main parameters

# Batch name
name = test

# Mode ('explicit', 'variations', 'closed-loop')
mode = explicit

# Logger mode ('file', 'screen', 'both')
logMode = both


###############################################################################
[explicit_mode] # Parameters for batch runner in explicit mode

# number of cases to be run 
#	 (all parameters marked with * must be either a single
#	  value or a tuple with the same size as this number,
#         in the case of a tuple, separate items via , and 'enter')
NCases = 5

# Problem names('probRock', 'probLand', etc) *
probs = prob9_1, 
        prob9_2, 
        prob10_1, 
        prob10_2, 
        probBrac

# Configuration file (.its) (one for each case) *
baseFiles = defaults/prob9_1.its, 
            defaults/prob9_2.its, 
            defaults/prob10_1.its, 
            defaults/prob10_2.its, 
            defaults/probBrac.its

# Initial guess mode for subsequent cases ('scratch', 'base', '') *
initGues = scratch

###############################################################################
[variations_mode] # Parameters for batch runner in variations mode

# Problem name ('probRock', 'probLand', etc)
probName = probLand

# Configuration file (.its) for base case
baseFile = defaults/probLand.its

# Initial guess for subsequent cases ('scratch', 'base', '')

