# SOAR

# OBJECTIVES
A software for optimizing thrust profiles and trajectories of launching vehicles (for academic purposes).

# GLOBAL INPUT DATA:
- As expected, Earth and universal constants such as G, R, g0, etc;
- For the main mission: the desired orbit, i.e., altitude and speed at last stage burnout, minimum payload mass, number of stages for rocket;
- For each stage: maximum thrust, specific impulse and structural efficiency;

# GLOBAL OUTPUT DATA:
- The propellant masses in each stage;
- The control functions, i.e., the thrust and angle of attack applied to the rocket in each time.

# METHOD
The optimization method is based on Angelo Miele’s MSGRA algorithm. 

# FLOW OF THE PROGRAM
As it is, many types of optimal control problems can be solved using this program. For each of them,
there must be a module (typically called prob<Problem Name>.py) which implements a given set of methods that shall be explained later. Therefore, this section explains the general flow of the program, not necessarily for the rocket problem.

The overall module-calling scheme is presented below:

The main module (which is run in the highest level) is main.py. This module imports the three basic modules:
interf.py: user interfacing and iteration managing module, 
sgra.py: MSGRA functions that are not specific to any particular instance of problem;
prob<X>.py: methods that are specific to the problem.

------------
prob<X>.py:
------------
A running solution of the program is an object of the class prob. This class is, very appropriately, defined in prob<X>.py, and must implement methods that are required by sgra:
- calcPhi (for the dynamic function);
- calcGrads (for the gradients);
- calcPsi (for the error function associated to the boundary conditions)
- calcF (for the integral term of the cost function);
- calcI (for the cost function itself);

The prob<X> class may also implement other methods for visualizing the solution (plotSol), the trajectory (plotTraj), etc. 

------------
sgra.py:
------------
All the methods that are not specific to the problem are defined here. Actually, in order to prevent this module from becoming gigantic, it was split into three: 
- sgra.py, 
- rest_sgra.py, and
- grad_sgra.

The sgra.py module calls the other two. The overall guideline is that if a module is specific to restoration procedures, it goes into rest_sgra.py,
and if it is specific to gradient procedures, it goes into grad_sgra.py.

------------
utils.py:
------------
Some general utility methods go here. Only the time derivative method is being used currently.

------------
interf.py:
------------
This module contains the object that performs user interface, and manages solution loading, saving, as well as the gradient and restoration iterations.

