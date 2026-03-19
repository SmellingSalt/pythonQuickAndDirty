# RL-style explorer for the user's Excel sheet (screenshot-based schema).
# This code creates reusable functions to parse the wide Excel layout into episodes
# and generate a few model-agnostic visualizations. 
#
# After you upload your Excel file to this chat session, set EXCEL_PATH below and re-run.
# The code will save plots under /mnt/data/rl_viz and also write a tidy CSV for reuse.

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- User input: set this path after uploading your file ----------
EXCEL_PATH = "/mnt/data/your_file.xlsx"  # <- replace with your uploaded Excel file path
SHEET_NAME = 0  # or the sheet name string, e.g., "Data"

# --------- Helpers ----------

def find_episode_ids(columns: List[str]) -> List[str]:
    """
    Detect episode IDs by scanning column names like:
    'experiment1', 'input1_3', 'output1_2', 'input2_1', 'output2_3', etc.
    Returns sorted unique episode ids as strings: ['1','2',...]
    """
    ids = set()
    for c in columns:
        m = re.match(r"^(?:experiment|input|output)(\d+)", str(c))
        if m:
            ids.add(m.group(1))
    return sorted(ids, key=lambda x: int(x))

def episode_column_groups(columns: List[str], eid: str) -> Dict[str, List[str]]:
    """
    For a given episode id, collect columns:
    - experiment column: exact match 'experiment{eid}' if present
    - input columns: 'input{eid}_k'
    - output columns: 'output{eid}_k'
    Returns dict with keys: 'tcol', 'inputs', 'outputs'
    """
    tcol = None
    inputs = []
    outputs = []
    for c in columns:
        s = str(c)
        if s == f"experiment{eid}":
            tcol = s
        m_in = re.match(rf"^input{eid}_(\d+)$", s)
        if m_in:
            inputs.append(s)
        m_out = re.match(rf"^output{eid}_(\d+)$", s)
        if m_out:
            outputs.append(s)
    # stable sort by index number
    key_idx = lambda name: int(re.search(r"_(\d+)$", name).group(1))
    inputs = sorted(inputs, key=key_idx)
    outputs = sorted(outputs, key=key_idx)
    return {"tcol": tcol, "inputs": inputs, "outputs": outputs}

def to_long_episodes(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the wide sheet into a tidy long dataframe with columns:
    ['episode', 't'] + input columns + output columns.
    Rows with all-NaN for an episode are dropped. Handles variable lengths.
    """
    episode_ids = find_episode_ids(list(df_wide.columns))
    long_frames = []
    for eid in episode_ids:
        groups = episode_column_groups(list(df_wide.columns), eid)
        cols = []
        if groups["tcol"] is not None:
            cols.append(groups["tcol"])
        cols += groups["inputs"] + groups["outputs"]
        if not cols:
            continue
        sub = df_wide[cols].copy()
        # rename columns to generic names
        ren = {}
        if groups["tcol"] is not None:
            ren[groups["tcol"]] = "t"
        for i, c in enumerate(groups["inputs"], start=1):
            ren[c] = f"input_{i}"
        for j, c in enumerate(groups["outputs"], start=1):
            ren[c] = f"output_{j}"
        sub = sub.rename(columns=ren)
        # drop rows fully NaN across inputs and outputs
        io_cols = [c for c in sub.columns if c.startswith("input_") or c.startswith("output_")]
        mask_all_nan = sub[io_cols].isna().all(axis=1)
        sub = sub.loc[~mask_all_nan].copy()
        # if t missing, create a step index starting at 1
        if "t" not in sub.columns:
            sub["t"] = np.arange(1, len(sub) + 1)
        # keep only rows where at least one of inputs/outputs is not NaN
        sub["episode"] = int(eid)
        # ensure sorted by t, then reset index
        sub = sub.sort_values("t").reset_index(drop=True)
        long_frames.append(sub)
    if not long_frames:
        return pd.DataFrame()
    long_df = pd.concat(long_frames, ignore_index=True)
    # add deltas within each episode
    long_df = long_df.sort_values(["episode", "t"]).reset_index(drop=True)
    # compute delta_output_3 if exists
    if "output_3" in long_df.columns:
        long_df["delta_output_3"] = long_df.groupby("episode")["output_3"].diff()
    # generic deltas for inputs
    for col in [c for c in long_df.columns if c.startswith("input_")]:
        long_df[f"delta_{col}"] = long_df.groupby("episode")[col].diff()
    return long_df

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# --------- Visualization functions ----------

def plot_episode_output3(df_long: pd.DataFrame, save_dir: str):
    """
    For each episode, plot output_3 vs t.
    """
    if "output_3" not in df_long.columns:
        print("No column 'output_3' found. Skipping output_3 plots.")
        return
    ensure_dir(save_dir)
    for eid, g in df_long.groupby("episode"):
        plt.figure()
        plt.plot(g["t"], g["output_3"])
        plt.xlabel("t")
        plt.ylabel("output_3")
        plt.title(f"Episode {eid}: output_3 over time")
        out_path = os.path.join(save_dir, f"episode_{eid}_output3.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

def scatter_input_vs_delta_output3(df_long: pd.DataFrame, save_dir: str):
    """
    For each input_k present, scatter input_k vs next-step change in output_3 aggregated across episodes.
    """
    if "output_3" not in df_long.columns:
        print("No column 'output_3' found. Skipping scatter plots.")
        return
    ensure_dir(save_dir)
    cols_in = [c for c in df_long.columns if c.startswith("input_")]
    if not cols_in:
        print("No input_* columns found.")
        return
    # use change in output_3 at next step
    d = df_long.copy()
    d["delta_output_3_next"] = d.groupby("episode")["output_3"].shift(-1) - d["output_3"]
    for col in cols_in:
        plt.figure()
        plt.scatter(d[col], d["delta_output_3_next"], s=10)
        plt.xlabel(col)
        plt.ylabel("next Δ output_3")
        plt.title(f"{col} vs next Δ output_3")
        out_path = os.path.join(save_dir, f"{col}_vs_delta_output3.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

def hexbin_state_change(df_long: pd.DataFrame, save_dir: str, x_col: str = "input_1", y_col: str = "input_2"):
    """
    Hexbin of two inputs vs next-step reduction in output_3. 
    If output_3 not present, function returns early.
    """
    if "output_3" not in df_long.columns:
        print("No column 'output_3' found. Skipping hexbin plot.")
        return
    if x_col not in df_long.columns or y_col not in df_long.columns:
        print(f"Columns {x_col} and/or {y_col} not found. Skipping hexbin.")
        return
    ensure_dir(save_dir)
    d = df_long.copy()
    d["delta_output_3_next"] = d.groupby("episode")["output_3"].shift(-1) - d["output_3"]
    plt.figure()
    hb = plt.hexbin(d[x_col], d[y_col], C=d["delta_output_3_next"], gridsize=30, reduce_C_function=np.nanmean)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    cbar = plt.colorbar(hb)
    cbar.set_label("mean next Δ output_3")
    plt.title(f"Hexbin: {x_col}, {y_col} vs mean next Δ output_3")
    out_path = os.path.join(save_dir, f"hexbin_{x_col}_{y_col}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_tidy_csv(df_long: pd.DataFrame, path: str):
    df_long.to_csv(path, index=False)

# --------- Run if file exists ----------

viz_dir = "/mnt/data/rl_viz"
ensure_dir(viz_dir)

outputs = []
if os.path.exists(EXCEL_PATH):
    try:
        raw = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
        # keep only columns that look like the pattern plus any metadata you might need
        long_df = to_long_episodes(raw)
        tidy_path = os.path.join(viz_dir, "tidy_episodes.csv")
        save_tidy_csv(long_df, tidy_path)
        # Show a preview to user
        from ace_tools import display_dataframe_to_user
        preview = long_df.head(50)
        display_dataframe_to_user("Episode preview (first 50 rows)", preview)
        # Create visuals
        plot_episode_output3(long_df, viz_dir)
        scatter_input_vs_delta_output3(long_df, viz_dir)
        # try hexbin on first two inputs if present
        possible_inputs = [c for c in long_df.columns if c.startswith("input_")]
        if len(possible_inputs) >= 2:
            hexbin_state_change(long_df, viz_dir, possible_inputs[0], possible_inputs[1])
        outputs.append(f"Tidy CSV saved to {tidy_path}")
    except Exception as e:
        outputs.append(f"Error reading or processing Excel: {e}")
else:
    outputs.append(
        "Excel file not found. Upload your file and set EXCEL_PATH accordingly, then re-run this cell."
    )

# Create a small README with instructions
readme_path = "/mnt/data/rl_viz/README.txt"
with open(readme_path, "w") as f:
    f.write(
        "Instructions:\n"
        "1) Upload your Excel file to this chat session.\n"
        "2) Edit EXCEL_PATH at the top of the script to point to your uploaded file.\n"
        "3) Re-run the cell. It will parse episodes, write rl_viz/tidy_episodes.csv, and save plots in rl_viz/.\n"
        "4) Key plots:\n"
        "   - episode_{id}_output3.png: output_3 vs t for each episode.\n"
        "   - input_k_vs_delta_output3.png: scatter of input vs next Δ output_3.\n"
        "   - hexbin_input_1_input_2.png: mean next Δ output_3 across input_1 and input_2 grid.\n"
    )

outputs, readme_path
