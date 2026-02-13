**4D Topology Optimization of a Soft Body.**
This project focuses on the structural and actuation optimization of soft bodies designed for dynamic tasks. By using the Material Point Method (MPM) for stable discretization and the ADAM optimizer for iterative design refinement, the system evolves into a soft silicone like structure capable of complex movement.

**Project Structure**
*Core Optimization & Simulation*
_4Dtopopt.py: The primary entry point. It handles the optimization loops, physical simulation, and generates initial visual outputs (final configurations and actuation signals).

particle_actuation.py: Helper class for plotting actuation curves.

particle_bin_plot.py: Utility for tracking specific material points and binning data for analysis.

*Visualization & Analysis*
plot_stacked_actuation.py: Generates stacked visualizations of actuator responses.

plot_desvar_vs_iteration.py: Tracks the evolution of design variables across the optimization iterations.

plot_obj_cons_vs_iteration.py: Plots the objective function and constraint convergence.

**Getting Started**
*Prerequisites*
The simulation relies heavily on Taichi for GPU-accelerated physics. You will need the following Python environment:

Bash
pip install taichi numpy matplotlib pandas pillow pyevtk numpy_ml

Execution Flow
To replicate the results, follow this sequence:

Run the Optimization:

Bash
python _4Dtopopt.py
This will perform the simulation and save the raw data/pickles.

*Generate Post-Processing Plots*:
Run the specific plotting scripts to visualize the convergence and actuation:

Bash
python plot_stacked_actuation.py
python plot_desvar_vs_iteration.py

**Dependencies**
The project utilizes a robust stack of scientific libraries:

Simulation: taichi

Math/Data: numpy, math, pandas, pickle

Optimization: numpy_ml.neural_nets.optimizers.Adam

Visualization: matplotlib, PIL (Pillow)

Export: pyevtk (for VTK/Paraview integration)