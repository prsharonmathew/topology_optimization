import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class ParticleGridPlotter:
    def __init__(
        self,
        x0, y0, w, h,
        nx=3, ny=3,
        dx=None,
        n_grid=None,
        bound_x_min_np=None,
        bound_x_max_np=None,
        bound_y_min_np=None,
        bound_y_max_np=None
    ):
        """
        Initialize the plotter with grid and boundary information.
         - (x0, y0): bottom-left corner of the bin grid
         - w, h: width and height of the bin grid
         - nx, ny: number of bins in x and y directions
         - dx: grid spacing for the computational grid
         - n_grid: number of grid points in each direction for the computational
             grid 
         - bound_x_min_np, bound_x_max_np: arrays of x-boundary values along 
            y-axis 
         - bound_y_min_np, bound_y_max_np: arrays of y-boundary values along 
            x-axis
        """
        self.x0 = x0
        self.y0 = y0
        self.w = w
        self.h = h
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.n_grid = n_grid
        
        # Store boundary arrays
        self.bound_x_min_np = bound_x_min_np
        self.bound_x_max_np = bound_x_max_np
        self.bound_y_min_np = bound_y_min_np
        self.bound_y_max_np = bound_y_max_np

        # Pre-calculate bin edges since geometry doesn't change
        self.bx = w / nx
        self.by = h / ny
        self.x_edges = [x0 + i * self.bx for i in range(nx + 1)]
        self.y_edges = [y0 + j * self.by for j in range(ny + 1)]
        self.x_centers = [x0 + (i + 0.5) * self.bx for i in range(nx)]
        self.y_centers = [y0 + (j + 0.5) * self.by for j in range(ny)]

    def _which_bin(self, xy):
        """
            Determine which bin a point (x, y) belongs to.
            Return (ix, iy) or (-1, -1) if outside.
        """
        x, y = float(xy[0]), float(xy[1])
        if not (self.x0 <= x < self.x0 + self.w and self.y0 <= y < self.y0 + self.h):
            return -1, -1
        
        ix = int((x - self.x0) / self.bx)
        iy = int((y - self.y0) / self.by)
        
        ix = max(0, min(self.nx - 1, ix))
        iy = max(0, min(self.ny - 1, iy))
        return ix, iy

    def plot(
        self,
        x_pos_np,           # (n_particles, 2)
        is_design_np,       # (n_particles,)
        tracked_particles,  # list[int]
        outpath=None,
        title="",
        annotate_counts=True,
        show_comp_grid=True
    ):
        """
        Plot the particle distribution with bin grid, computational grid, and 
        boundaries.
         - x_pos_np: numpy array of shape (n_particles, 2) with particle 
            positions
         - is_design_np: boolean array indicating which particles are design 
            variables
         - tracked_particles: list of particle indices to highlight
         - outpath: if provided, save the figure to this path
         - title: plot title
         - annotate_counts: whether to show particle counts in each bin
         - show_comp_grid: whether to draw the computational grid in the 
            background     
        
        Main plotting method.
        """

        # Data preparation
        x_pos_np = np.asarray(x_pos_np, dtype=float)
        is_design_np = np.asarray(is_design_np, dtype=int)
        design_idx = np.where(is_design_np == 1)[0]
        design_pos = x_pos_np[design_idx]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect("equal", adjustable="box")

        # 1. Draw computational grid
        if show_comp_grid and (self.dx is not None) and (self.n_grid is not None):
            self._draw_computational_grid(ax)

        # 2. Draw bin grid
        for xe in self.x_edges:
            ax.plot([xe, xe], [self.y0, self.y0 + self.h], linewidth=1, color='k')
        for ye in self.y_edges:
            ax.plot([self.x0, self.x0 + self.w], [ye, ye], linewidth=1, color='k')

        # 3. Scatter particles
        ax.scatter(design_pos[:, 0], design_pos[:, 1], s=6, alpha=0.35)

        # 4. Annotate counts
        if annotate_counts:
            self._annotate_bin_counts(ax, design_idx, x_pos_np)

        # 5. Highlight tracked particles
        self._highlight_tracked(ax, tracked_particles, x_pos_np, show_comp_grid)

        # Final setup
        # ax.set_xlim(self.x0 - 0.02, self.x0 + self.w + 0.02)
        # ax.set_ylim(self.y0 - 0.02, self.y0 + self.h + 0.02)
        if self.dx is not None and self.n_grid is not None:
            total_size = self.n_grid * self.dx
            # Set view to 0 -> total_size
            ax.set_xlim(0, total_size)
            ax.set_ylim(0, total_size)
        else:
            # Fallback if grid info isn't available
            ax.set_xlim(self.x0, self.x0 + self.w)
            ax.set_ylim(self.y0, self.y0 + self.h)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)

        plt.tight_layout()
        if outpath is not None:
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            plt.savefig(outpath)
            plt.close(fig)
        
        return fig, ax

    def _draw_computational_grid(self, ax):
        """
        Helper to draw the fine computational background grid and boundaries.
         - ax: matplotlib axis to draw on"""
        # x_min, x_max = self.x0 - 0.02, self.x0 + self.w + 0.02
        # # x_min, x_max = self.x0 , self.x0
        # # y_min, y_max = self.y0, self.y0 
        # y_min, y_max = self.y0 - 0.02, self.y0 + self.h + 0.02

        # i0 = max(0, int(np.floor(x_min / self.dx)))
        # i1 = min(self.n_grid, int(np.ceil(x_max / self.dx)))
        # j0 = max(0, int(np.floor(y_min / self.dx)))
        # j1 = min(self.n_grid, int(np.ceil(y_max / self.dx)))

        total_width = self.n_grid * self.dx
        x_min, x_max = 0, total_width
        y_min, y_max = 0, total_width

        i0 = 0 
        i1 = self.n_grid
        j0 = 0
        j1 = self.n_grid

        # Draw boundary curves
        if self.bound_x_min_np is not None:
            js = np.arange(j0, j1 + 1)
            y_line = js * self.dx
            # Ensure indices are within bounds
            valid_j = js < len(self.bound_x_min_np)
            if np.any(valid_j):
                safe_js = js[valid_j]
                safe_y = y_line[valid_j]
                ax.plot(self.bound_x_min_np[safe_js] * self.dx, safe_y, \
                linewidth=2.0)
                ax.plot(self.bound_x_max_np[safe_js] * self.dx, safe_y, \
                linewidth=2.0)

        if self.bound_y_min_np is not None:
            is_ = np.arange(i0, i1 + 1)
            x_line = is_ * self.dx
            valid_i = is_ < len(self.bound_y_min_np)
            if np.any(valid_i):
                safe_is = is_[valid_i]
                safe_x = x_line[valid_i]
                ax.plot(safe_x, self.bound_y_min_np[safe_is] * self.dx, \
                linewidth=2.0)
                ax.plot(safe_x, self.bound_y_max_np[safe_is] * self.dx, \
                linewidth=2.0)

        # Draw grid lines
        for i in range(i0, i1 + 1):
            xx = i * self.dx
            ax.plot([xx, xx], [y_min, y_max], linewidth=0.4, alpha=0.25, \
            color='gray')
        for j in range(j0, j1 + 1):
            yy = j * self.dx
            ax.plot([x_min, x_max], [yy, yy], linewidth=0.4, alpha=0.25, \
            color='gray')

    def _annotate_bin_counts(self, ax, design_idx, x_pos_np):
        """Helper to calculate and display particle counts per bin.
            - ax: matplotlib axis to draw on
            - design_idx: indices of particles that are design variables
            - x_pos_np: numpy array of shape (n_particles, 2) with particle 
                positions       
        """
        counts = np.zeros((self.ny, self.nx), dtype=int)
        for p in design_idx:
            ix, iy = self._which_bin(x_pos_np[p])
            if ix >= 0:
                counts[iy, ix] += 1
        
        for iy in range(self.ny):
            for ix in range(self.nx):
                ax.text(
                    self.x_centers[ix], self.y_centers[iy],
                    f"{counts[iy, ix]}",
                    ha="left", va="top", fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.15", alpha=0.35, fc="white")
                )

    def _highlight_tracked(self, ax, tracked_particles, x_pos_np, show_comp_grid):
        """
        Helper to highlight specific particles and their grid data.
         - ax: matplotlib axis to draw on
         - tracked_particles: list of particle indices to highlight
         - x_pos_np: numpy array of shape (n_particles, 2) with particle 
            positions
         - show_comp_grid: whether the computational grid is shown (to decide 
            if we draw the red box)
        """
        for p in tracked_particles:
            xp, yp = x_pos_np[p, 0], x_pos_np[p, 1]
            ax.scatter([xp], [yp], s=60, marker="x", color='red')

            ix = int(np.floor(xp / self.dx)) if self.dx is not None else -1
            iy = int(np.floor(yp / self.dx)) if self.dx is not None else -1

            if show_comp_grid and self.dx is not None:
                rect = Rectangle((ix * self.dx, iy * self.dx), self.dx, self.dx, 
                                 fill=False, linewidth=1.2, alpha=0.9, edgecolor='red')
                ax.add_patch(rect)

            bx_ix, bx_iy = self._which_bin([xp, yp])
            
            label = f"p={p}"
            ax.text(xp, yp, label, fontsize=8, ha="left", va="bottom")