import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---- EDIT THESE TWO if needed ----
PICKLE_PATH = Path(r"runs\Final_20260211_1205_132934\test\model_epoch008\test_results.p")
BASIN_ID = "11"   # change to any key you saw, e.g. "111"

# ---------------------------------
with PICKLE_PATH.open("rb") as f:
    results = pickle.load(f)

# results[basin][freq]['xr'] is usually an xarray Dataset
# We'll just take the first freq that exists
freq = next(iter(results[BASIN_ID].keys()))
ds = results[BASIN_ID][freq]["xr"]

df = ds.to_dataframe().reset_index()

# ---- find likely prediction/obs column names ----
cols = [c for c in df.columns if c not in ("date", "time", "basin")]
print("Columns:", cols)

# Try common names used by NeuralHydrology
obs_candidates = ["y_obs", "obs", "QObs", "target", "labels"]
pred_candidates = ["y_hat", "y_sim", "pred", "prediction", "output"]

def pick_col(cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

obs_col = pick_col(obs_candidates)
pred_col = pick_col(pred_candidates)

# If not found, just pick two numeric columns automatically
if obs_col is None or pred_col is None:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # pick first two numeric columns
    obs_col = obs_col or numeric_cols[0]
    pred_col = pred_col or numeric_cols[1]

print("Using obs:", obs_col, "pred:", pred_col)

# ---- 1) Time series plot ----
plt.figure()
plt.plot(df["date"], df[obs_col], label="Observed")
plt.plot(df["date"], df[pred_col], label="Predicted")
plt.title(f"Basin {BASIN_ID} - Time series")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()

# ---- 2) Scatter plot (pred vs obs) ----
plt.figure()
plt.scatter(df[obs_col], df[pred_col], s=8)
plt.title(f"Basin {BASIN_ID} - Predicted vs Observed")
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.tight_layout()
plt.show()
