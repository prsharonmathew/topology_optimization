import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

class ParticleActuationAnalyzer:
    def __init__(self, n_actuators, dt, act_strength, matd, penal_act):
        """
        Initializes the analyzer with the static physics/simulation constants.
            - n_actuators: number of actuators in the system
            - dt: time step size for the simulation
            - act_strength: scalar multiplier for the actuation force
            - matd: minimum material density for SIMP scaling
            - penal_act: penalization factor for SIMP scaling
        """
        self.n_actuators = n_actuators
        self.dt = dt
        self.act_strength = act_strength
        self.matd = matd
        self.penal_act = penal_act

    def compute_curves(
        self,
        p,
        total_steps,
        actuation_np,      # (max_steps+1, n_actuators)
        act_w_soft_np,     # (n_particles, n_actuators+1)
        is_design_np,
        design_id_np,
        rho_np             # (n_design_particles,)
    ):
        """
        Calculates the effective actuation forces for a specific particle 'p'.
            - p: particle index to analyze
            - total_steps: total number of time steps in the simulation
            -actuation_np:array of actuation signals over time for all actuators
        """
        T = int(total_steps)
        t = np.arange(T) * self.dt

        if is_design_np[p] != 1:
            return t, None, None, None, None, None

        des_id = int(design_id_np[p])
        
        # Calculate SIMP scaling
        simp_act = float((1.0 - self.matd) * (rho_np[des_id] ** self.penal_act) + self.matd)

        # Get weights and actuation signal
        w = act_w_soft_np[p, :self.n_actuators].astype(float)
        a = actuation_np[:T, :self.n_actuators].astype(float)

        # Calculate effective force u_hat
        # u_hat shape: (T, n_actuators)
        u_hat = a * w[None, :] * self.act_strength * simp_act
        u_sum = np.sum(u_hat, axis=1)

        target_j = int(np.argmax(w))
        act_w_soft_target = w.tolist() # Returning full weight list as requested

        return t, u_hat, u_sum, target_j, act_w_soft_target, simp_act

    def plot_single_iter(
        self,
        particle_id,
        iter_idx,
        t,
        u_hat,
        u_sum,
        act_w_soft_target,
        target_j,
        outpath
    ):
        """
        Generates a plot for a single iteration showing u_hat components and
        u_sum.
            - particle_id: ID of the particle being plotted
            - iter_idx: iteration index for title annotation
            - t: time array
            - u_hat: candidate actuation forces (T, n_actuators)
            - u_sum: sum of effective forces over actuators (T,)
            - act_w_soft_target: list of weights for all actuators for the 
                target particle
            - target_j: index of the actuator with the highest weight
            - outpath: file path to save the plot
        """
        fig, ax = plt.subplots(figsize=(9, 3.5))

        # Plot individual actuator lines
        for j in range(u_hat.shape[1]):
            ax.plot(t, u_hat[:, j], linewidth=1.6,\
            label=rf'$\hat{{u}}_{j+1}(t)$')

        # Plot sum line
        ax.plot(t, u_sum, linewidth=2.0, label=r'$\hat{u}_{\mathrm{sum}}(t)$')

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Actuation scalar (Pa)")
        ax.set_xlim([t[0], t[-1]])

        # Scientific notation for Y axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        # Title and Legend
        ax.set_title(
            f"particle_id = {particle_id} | iter={iter_idx:04d} | "
            f"target actuator = {target_j+1} | act_w_soft_target = {act_w_soft_target}"
        )
        ax.legend(loc="best", ncols=3, fontsize=9)

        plt.tight_layout()
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath)
        plt.close(fig)

    def plot_stacked_history(self, particle_folder, particle_id):
        """
        Scans a folder for 'u_hat_iterXXXX.npz' files and creates a vertically 
        stacked history plot of the actuation evolution.
            - particle_folder: directory containing the npz files for the 
                particle
            - particle_id: ID of the particle being analyzed (for title 
                annotation)
        """
        npz_files = sorted(
            [f for f in os.listdir(particle_folder) if re.match(r"u_hat_iter\d{4}\.npz", f)]
        )
        if not npz_files:
            print(f"No npz files found in {particle_folder}")
            return

        # Load all data
        items = []
        for fn in npz_files:
            # Extract iteration number from filename
            it = int(re.findall(r"\d{4}", fn)[0])
            data = np.load(os.path.join(particle_folder, fn))
            items.append((it, data))

        nrows = len(items)
        fig_h = max(2.2 * nrows, 3.0)
        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(10, fig_h),\
         sharex=True)
        
        # Handle case where there is only 1 file (axes is not a list)
        if nrows == 1:
            axes = [axes]

        for ax, (it, data) in zip(axes, items):
            t = data["t"]
            u_hat = data["u_hat"]
            u_sum = data["u_sum"]
            
        # Extract metadata safely
        # Note: Handling potential difference between list/float storage in NPZ
            try:
                if data["act_w_soft_target"].size > 1:
                     # If stored as list/array, take max or specific one? 
                     # Code snippet implies taking first element or scalar
                     act_w_soft_target = float(data["act_w_soft_target"][0])
                else:
                     act_w_soft_target = float(data["act_w_soft_target"])
            except:
                act_w_soft_target = 0.0

            target_j = int(data["target_j"]) if data["target_j"].shape==()\
             else int(data["target_j"][0])

            # Plot lines
            for j in range(u_hat.shape[1]):
                ax.plot(t, u_hat[:, j], linewidth=1.2, label=rf'$\hat{{u}}_{j+1}(t)$')
            ax.plot(t, u_sum, linewidth=1.8, label=r'$\hat{u}_{\mathrm{sum}}(t)$')

            # Formatting
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            ax.set_title(
                f"iter={it:04d} | target={target_j+1} | act_w_soft_target={act_w_soft_target:.4f}",
                fontsize=10,
            )
            ax.grid(alpha=0.15)

        # Global labels
        axes[-1].set_xlabel("Time (s)")
        axes[0].set_ylabel("Actuation scalar (Pa)")

        # Single unified legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncols=5, fontsize=9)

        fig.suptitle(f"particle_id = {particle_id} | stacked actuation curves", y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        outpath = os.path.join(particle_folder, "STACKED_u_hat.png")
        plt.savefig(outpath)
        plt.close(fig)