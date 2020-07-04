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
#         in the case of a tuple, separate items via ', ' and 'enter')
NCases = 3

# Problem names('probRock', 'probLand', etc) *
probs = probRock

# Configuration file (.its) (one for each case) *
baseFiles = defaults/probRock-noPlot.its

###############################################################################
[variations_mode] # Parameters for batch runner in variations mode

# Problem name ('probRock', 'probLand', etc)
probName = probLand

# Configuration file (.its) for base case
baseFile = defaults/probLand.its

# Initial guess for subsequent cases ('scratch', 'base', '')

