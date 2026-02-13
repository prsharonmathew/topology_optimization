"""
    Plot objective (as absolute value) and constraints vs iteration from log files.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FOLDER = r"/home/*****/log" #update this to log folder path
LOSS_FILE = os.path.join(FOLDER, "_losses.csv")

# Define files and their corresponding custom labels
CONSTRAINT_FILES_MAP = {
    "_l_l2_pens": r"$C_{act}$",        # was _l_l2_pens.csv
    "_l_quad_pens": r"$C_{mat}$",      # was _l_quad_pens.csv
    "_l_pw_sgn_pens": r"$C_{pul,sgn}$", # was _l_pw_sgn_pens.csv
    "_l_pw_abs_pens": r"$C_{pul,abs}$"  # was _l_pw_abs_pens.csv
}

# Construct full paths based on the keys above
CONSTRAINT_FILES = [os.path.join(FOLDER, f"{k}.csv") \
    for k in CONSTRAINT_FILES_MAP.keys()]

USE_TWIN_AXIS = True

def read_csv_flexible(path: str) -> pd.DataFrame:
    """Read csv; if it has no header, create generic column names.
        - path: file path to read
        Returns: DataFrame with appropriate columns
    """
    try:
        return pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, header=None)
        df.columns = [f"c{i}" for i in range(df.shape[1])]
        return df

def find_iter_col(df: pd.DataFrame):
    """
    Find the iteration column in a DataFrame by checking common names.
        - Returns the column name if found, else None.
    """
    for c in ["iter", "iteration", "it", "step", "k", "epoch"]:
        if c in df.columns:
            return c
    return None

def find_objective_col(df: pd.DataFrame):
    """
    Find the objective column in a DataFrame by checking common names.
    """
    # common objective column names
    for c in ["objective", "obj", "loss", "total_loss", "J"]:
        if c in df.columns:
            return c
    # fallback: pick the last mostly-numeric column
    numeric_cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.9:
            numeric_cols.append(c)
    if not numeric_cols:
        raise ValueError("Could not find a numeric objective column in _losses.csv")
    return numeric_cols[-1]

def numeric_columns(df: pd.DataFrame, exclude=None):
    """
    Identify mostly-numeric columns in a DataFrame, excluding specified ones.
        - df: input DataFrame
        - exclude: list of column names to ignore
        Returns: list of column names that are mostly numeric
    """
    exclude = set(exclude or [])
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.5:  # allow some NaNs but mostly numeric
            cols.append(c)
    return cols

# -------- Load objective --------
loss_df = read_csv_flexible(LOSS_FILE)
loss_iter_col = find_iter_col(loss_df)
obj_col = find_objective_col(loss_df)

obj = pd.to_numeric(loss_df[obj_col], errors="coerce")

if loss_iter_col is not None:
    it_obj = pd.to_numeric(loss_df[loss_iter_col], errors="coerce")
else:
    it_obj = pd.Series(np.arange(len(obj)))

# clean
mask = it_obj.notna() & obj.notna()
it_obj = it_obj[mask].astype(int).to_numpy()
obj = obj[mask].to_numpy()

# --- CHANGE 1: Convert objective to absolute (positive) values ---
obj = np.abs(obj)

# Store objective in a dataframe for easy alignment later
obj_df = pd.DataFrame({"iter": it_obj, "objective": obj}).drop_duplicates("iter")

# -------- Load constraints (each col -> one line) --------
constraint_lines = []  # list of (label, df with columns iter + value)

for fpath in CONSTRAINT_FILES:
    if not os.path.exists(fpath):
        print(f"[WARN] Missing: {fpath}")
        continue

    df = read_csv_flexible(fpath)
    it_col = find_iter_col(df)

    if it_col is not None:
        it = pd.to_numeric(df[it_col], errors="coerce")
        df_vals = df.drop(columns=[it_col])
    else:
        it = pd.Series(np.arange(len(df)))
        df_vals = df

    # keep numeric cols
    cols = numeric_columns(df_vals)
    if not cols:
        print(f"[WARN] No numeric columns found in {os.path.basename(fpath)}")
        continue

    # Get the base filename without extension to look up the custom label
    base_filename = os.path.splitext(os.path.basename(fpath))[0]
    
    # --- CHANGE 2: Use custom label map ---
    # Default to filename if not in map, otherwise use the mapped label
    base_label = CONSTRAINT_FILES_MAP.get(base_filename, base_filename)

    for i, c in enumerate(cols):
        y = pd.to_numeric(df_vals[c], errors="coerce")
        m = it.notna() & y.notna()
        tmp = pd.DataFrame({"iter": it[m].astype(int), "value": y[m].astype(float)})

        # If there is only 1 data column, just use the label (e.g. "C_act")
        # If multiple, append index/name (e.g. "C_act:c0") to distinguish them
        if len(cols) == 1:
            label = base_label
        else:
            label = f"{base_label}:{c}"
            
        constraint_lines.append((label, tmp))

# -------- Align everything by iteration (inner join set) --------
# Compute the common iteration set: intersection of objective + all constraint series
common_iters = set(obj_df["iter"].to_list())
for _, d in constraint_lines:
    common_iters &= set(d["iter"].to_list())

common_iters = np.array(sorted(common_iters), dtype=int)
# Ensure iterations are positive (sometimes logs have -1 for init)
common_iters = np.abs(common_iters) 

if len(common_iters) == 0:
    # Fallback: if intersection is empty, try union or just objective iters? 
    # Usually intersection is safest for plotting x-y lines together.
    common_iters = obj_df["iter"].to_numpy()
    common_iters.sort()

# Re-fetch objective aligned to these iters
# We use reindex/merge logic here to be safe
obj_aligned_series = obj_df.set_index("iter").reindex(common_iters)["objective"]
obj_aligned = obj_aligned_series.to_numpy()

# -------- Plot --------
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Objective (Absolute Value)
ax1.plot(common_iters, obj_aligned, linewidth=2, color='black', label="Objective $|J|$")
ax1.set_xlabel("Iteration")
ax1.set_title("Objective and Constraints vs Iteration")
ax1.grid(True, alpha=0.25)

if USE_TWIN_AXIS:
    ax1.set_ylabel("Objective Value (abs)")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Constraint Values")

    # Cycle through some colors for constraints
    colors = plt.cm.tab10(np.linspace(0, 1, len(constraint_lines)))

    for idx, (label, d) in enumerate(constraint_lines):
        # Align constraint data to common_iters
        series = d.set_index("iter").reindex(common_iters)["value"]
        y = series.to_numpy()
        
        # skip plotting if all NaNs after alignment
        if np.isnan(y).all():
            continue

        ax2.plot(common_iters, y, linestyle="--", linewidth=1.5, alpha=0.85, 
                 label=label, color=colors[idx])

    # combined legend
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=10, frameon=True)

else:
    ax1.set_ylabel("Objective / Constraint Values")
    for label, d in constraint_lines:
        series = d.set_index("iter").reindex(common_iters)["value"]
        y = series.to_numpy()
        if np.isnan(y).all(): continue
        ax1.plot(common_iters, y, linestyle="--", linewidth=1.5, alpha=0.85, label=label)

    ax1.legend(loc="best", fontsize=10)

plt.tight_layout()
out = os.path.join(FOLDER, "objective_abs_and_constraints.png")
plt.savefig(out, dpi=200)
plt.close(fig)

print("Saved:", out)