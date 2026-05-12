import pandas as pd

df = pd.read_csv("./runs/Final_20260211_1205_132934/test/model_epoch008/test_metrics.csv")

median_values = df[["NSE", "RMSE", "KGE"]].median()
print(median_values)

