# SOAR

# OBJECTIVES
A software for optimizing thrust profiles and trajectories of launching vehicles (for academic purposes).

# GLOBAL INPUT DATA:
- As expected, Earth and universal constants such as G, R, g0, etc;
- For the main mission: the desired orbit, i.e., altitude and speed at last stage burnout, number of stages for rocket;

- PAYLOAD MASS??

- For each stage: maximum thrust, specific impulse and structural efficiency;

# GLOBAL OUTPUT DATA:
- The propellant masses in each stage;
- The control functions, i.e., the thrust and angle of attack applied to the rocket in each time.

# METHOD
The optimization method is based on Miele’s MSGRA algorithm. Currently, the considerations apply only to single stage rockets, which corresponds to SGRA algorithm.


# FLOW OF THE PROGRAM
The main file (which is run in the highest level) is sgra_simple_rocket.py. 
In this module there are the functions to perform Miele’s gradient and restoration processes on the ongoing solution (time, state, controls) until optimality conditions are satisfied.

Some problem specific functions are in prob_rocket_sgra.py.
Some general utility functions are found in utils.py.

As it turns out, SGRA requires an initial guess very close to an actual solution of the problem (that is, a solution which satisfies the constraints, but not necessarily the optimality condition). 
Hence, the rockProp.py module was written to provide a trajectory that corresponds to a propagated solution of the problem. However, just guessing the control profiles to match a given orbit was found too hard, and the automatic_Trajectory_Design was written to solve this problem.


- MAJOR TODO’S
@munizlgmn: first aerodynamic model
@TBD: Implement maximum acceleration limitation, staging…

- MINOR TODO’S
@araujolma: Create option in plotSol to announce solution and correction

@araujolma: encapsulate all constants in a dictionary to be passed in function calls.



