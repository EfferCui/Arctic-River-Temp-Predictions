

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./runs/Final_20260211_1205_132934/test/model_epoch008/test_metrics.csv")
metrics = ["NSE", "RMSE", "KGE"]

for m in metrics:
    s = pd.to_numeric(df[m], errors="coerce").dropna()
    plt.figure(figsize=(6,4))
    plt.boxplot(s.values, labels=[m])
    plt.title(f"Boxplot of {m}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
