    """plot design variables against the number of iterations. 
    """

import re
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE_DIR = r"/home/*****/4dtopopt" #replace with the dir where csv's are located.
DES_VAR_SUBDIR = "des_var"

PARTICLES = [246, 259, 273, 766, 779, 793, 1326, 1339, 1353]

ONE_BASED_LINE_NUMBERS = False

ACT_W_DIM = 5          # 4 actuators + 1 no-act
ACTUATION_DIM = 4      # 4 actuators

OUT_DIR = Path(BASE_DIR) / "particle_plots_combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

iter_re = re.compile(r"iter(\d+)$", re.IGNORECASE)

def list_iterations(base_dir: str):
    """Returns a sorted list of (iter_num, iter_path) for all iterXXXX folders
         under base_dir.
         - base_dir: the directory to search for iterXXXX folders
    """
    base = Path(base_dir)
    iters = []
    for p in base.glob("iter*"):
        if p.is_dir():
            m = iter_re.search(p.name)
            if m:
                iters.append((int(m.group(1)), p))
    iters.sort(key=lambda x: x[0])
    return iters  # list of (iter_num, iter_path)

def find_file(des_dir: Path, stem: str, it: int):
    """ 
    Given a design variable directory (des_dir) and a stem (e.g. "phi"), 
    find the file for the given iteration.
    """
    candidates = [
        des_dir / f"{stem}_iter{it:04d}.csv",
        des_dir / f"{stem}_iter{it:04d}",
    ]
    for c in candidates:
        if c.exists():
            return c
    hits = sorted(des_dir.glob(f"{stem}_iter{it:04d}.*"))
    return hits[0] if hits else None

def parse_csv_line(line: str):
    """
    Parses a line of CSV, returning a list of floats. Handles empty lines
     and ignores empty values.
    """
    line = line.strip()
    if not line:
        return []
    return [float(x) for x in line.split(",") if x.strip() != ""]

def read_selected_lines(file_path: Path, target_lines_zero_based):
    """
    Reads only the requested line indices (0-based) from the file, in one pass.
    Returns dict: {line_idx: list_of_values}
    """
    targets = set(target_lines_zero_based)
    max_target = max(targets)
    out = {}

    with file_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in targets:
                out[i] = parse_csv_line(line)
                if len(out) == len(targets):
                    break
            if i > max_target and len(out) == len(targets):
                break
    return out

def pad_or_slice(vals, dim):
    vals = list(vals)
    if len(vals) >= dim:
        return vals[:dim]
    return vals + [math.nan] * (dim - len(vals))


iters = list_iterations(BASE_DIR)
if not iters:
    raise RuntimeError(f"No iterXXXX folders found under: {BASE_DIR}")

iter_nums = [it for it, _ in iters]
n_iter = len(iter_nums)

# map particle -> line index in file
particle_lines = [(p - 1 if ONE_BASED_LINE_NUMBERS else p) for p in PARTICLES]
pcount = len(PARTICLES)
line_to_pidx = {line: i for i, line in enumerate(particle_lines)}

# arrays
material_density_design = np.full((pcount, n_iter), np.nan, dtype=float)  # phi
material_density = np.full((pcount, n_iter), np.nan, dtype=float)         # rho

actuator_density_design = np.full((pcount, n_iter, ACT_W_DIM), np.nan, dtype=float)  # act_w
actuator_densities = np.full((pcount, n_iter, ACT_W_DIM), np.nan, dtype=float)       # act_w_soft

candidate_actuation = np.full((pcount, n_iter, ACTUATION_DIM), np.nan, dtype=float)  # actuation

stems = ["phi", "rho", "act_w", "act_w_soft", "actuation"]

for t_idx, (it, it_path) in enumerate(iters):
    des_dir = it_path / DES_VAR_SUBDIR
    if not des_dir.exists():
        continue

    paths = {stem: find_file(des_dir, stem, it) for stem in stems}

    # material density design variable (phi)
    if paths["phi"] is not None:
        got = read_selected_lines(paths["phi"], particle_lines)
        for line, vals in got.items():
            pidx = line_to_pidx[line]
            material_density_design[pidx, t_idx] = vals[0] if len(vals) else np.nan

    # material density (rho)  (NO scaling)
    if paths["rho"] is not None:
        got = read_selected_lines(paths["rho"], particle_lines)
        for line, vals in got.items():
            pidx = line_to_pidx[line]
            material_density[pidx, t_idx] = vals[0] if len(vals) else np.nan

    # actuator density design variable (act_w)
    if paths["act_w"] is not None:
        got = read_selected_lines(paths["act_w"], particle_lines)
        for line, vals in got.items():
            pidx = line_to_pidx[line]
            actuator_density_design[pidx, t_idx, :] = pad_or_slice(vals, ACT_W_DIM)

    # actuator densities (act_w_soft)
    if paths["act_w_soft"] is not None:
        got = read_selected_lines(paths["act_w_soft"], particle_lines)
        for line, vals in got.items():
            pidx = line_to_pidx[line]
            actuator_densities[pidx, t_idx, :] = pad_or_slice(vals, ACT_W_DIM)

    # candidate actuation signals (actuation)
    if paths["actuation"] is not None:
        got = read_selected_lines(paths["actuation"], particle_lines)
        for line, vals in got.items():
            pidx = line_to_pidx[line]
            candidate_actuation[pidx, t_idx, :] = pad_or_slice(vals, ACTUATION_DIM)

iters_x = np.array(iter_nums, dtype=int)

# 5 colors from matplotlib default cycle (no hardcoding)
default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors5 = (default_colors * 2)[:ACT_W_DIM]

region_labels = ["Actuator 1", "Actuator 2", "Actuator 3", "Actuator 4", "No-actuator region"]
actuation_labels = ["Actuator 1", "Actuator 2", "Actuator 3", "Actuator 4"]

for p_i, particle in enumerate(PARTICLES):
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # ---- (1) Material density design variable + material density ----
    ax = axs[0]
    ax.plot(iters_x, material_density_design[p_i], label="Material density design variable", linewidth=1.2)
    ax.set_ylabel("Material density design variable")
    ax.set_title("Material density design variable and material density vs iteration")

    ax_r = ax.twinx()
    ax_r.plot(iters_x, material_density[p_i], linestyle="--", label="Material density", linewidth=1.2)
    ax_r.set_ylabel("Material density")

    # combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best")

    # ---- (2) Actuator density design variable (solid) + actuator densities (dashed), 5 colors ----
    ax = axs[1]
    for k in range(ACT_W_DIM):
        ax.plot(iters_x, actuator_density_design[p_i, :, k], color=colors5[k], linestyle="-", linewidth=1.0)
        ax.plot(iters_x, actuator_densities[p_i, :, k],       color=colors5[k], linestyle="--", linewidth=1.0)

    ax.set_ylabel("Actuator density")
    ax.set_title("Actuator density design variable (solid) and actuator densities (dashed)")

    # Legend: colors -> region meaning
    color_handles = [Line2D([0], [0], color=colors5[k], lw=2, linestyle="-") for k in range(ACT_W_DIM)]
    leg1 = ax.legend(color_handles, region_labels, title="Region", loc="upper right")
    ax.add_artist(leg1)

    # Legend: line style meaning
    style_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle="-"),
        Line2D([0], [0], color="black", lw=2, linestyle="--"),
    ]
    ax.legend(style_handles, ["Actuator density design variable", "Actuator densities"],
              title="Line type", loc="upper left")

    # ---- (3) Candidate actuation signals ----
    # ax = axs[2]
    # for k in range(ACTUATION_DIM):
    #     ax.plot(iters_x, candidate_actuation[p_i, :, k], label=actuation_labels[k], linewidth=1.2)

    # ax.set_xlabel("Iteration number")
    # ax.set_ylabel("Candidate actuation signal")
    # ax.set_title("Candidate actuation signals vs iteration")
    # ax.legend(loc="best")

    fig.suptitle(f"Particle {particle}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(OUT_DIR / f"particle_{particle:04d}_combined.png", dpi=200)
    plt.close(fig)

print(f"Done. Combined plots saved to: {OUT_DIR}")
