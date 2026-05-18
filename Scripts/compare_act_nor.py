# -*- coding: utf-8 -*-
"""
Compare actual-scale and normalized-scale alpha paths for one random basin.

For each alpha stage, the script first builds the intermediate actual value:
    actual_alpha = alpha * actual_value

Then it normalizes that intermediate actual value:
    normalized_alpha = (actual_alpha - center) / scale

This is intentionally different from alpha * normalized_value.
"""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.utils.config import Config
from torch.utils.data import DataLoader


RUN_DIR = Path("./runs/Final_20260211_1205_132934")
CONFIG_FILE = Path("./Data/Hyperparameter_18.yml")
OUTPUT_FILE = RUN_DIR / "actual_vs_normalized_alpha_paths.png"
RANDOM_SEED = 42
ALPHAS = np.linspace(0, 1, 5)

FEATURE_LABELS = {
    "snow_depth_water_equivalent_mean": "SWE",
    "surface_net_solar_radiation_mean": "Solar Radiation",
    "surface_net_thermal_radiation_mean": "Thermal Radiation",
    "temperature_2m_mean": "Air Temperature",
    "u_component_of_wind_10m_mean": "Wind U",
    "v_component_of_wind_10m_mean": "Wind V",
    "volumetric_soil_water_layer_1_mean": "Soil Water Content",
    "soil_temperature_level_1_mean": "Soil Temperature",
    "total_precipitation_sum": "Precipitation",
    "potential_evaporation_sum": "PET",
    "surface_runoff_sum": "Surface Runoff",
    "sub_surface_runoff_sum": "Subsurface Runoff",
}


def _read_basin_ids(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as fp:
        return [line.strip() for line in fp if line.strip()]


def _as_1d_numpy(sample, feature: str) -> np.ndarray:
    return sample["x_d"][feature].detach().cpu().numpy().reshape(-1)


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    cfg = Config(CONFIG_FILE)
    scaler = load_scaler(RUN_DIR)

    basin_file = Path(cfg.test_basin_file)
    basin_ids = _read_basin_ids(basin_file)
    basin = random.choice(basin_ids)

    dataset = get_dataset(cfg, basin=basin, is_train=False, period="test", scaler=scaler)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    samples = list(dataloader)
    sample_index = random.randrange(len(samples))
    sample = samples[sample_index]

    feature_keys = list(sample["x_d"].keys())
    lookback_days = len(_as_1d_numpy(sample, feature_keys[0]))
    days_before_present = np.arange(lookback_days)
    prediction_date = str(sample["date"][0, -1])[:10]

    center = scaler["xarray_feature_center"]
    scale = scaler["xarray_feature_scale"]

    fig, axs = plt.subplots(
        nrows=len(feature_keys),
        ncols=2,
        figsize=(14, 2.4 * len(feature_keys)),
        facecolor="w",
        sharex=True,
    )

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(ALPHAS)))

    for row, feature in enumerate(feature_keys):
        normalized_actual = _as_1d_numpy(sample, feature)[::-1]
        feature_center = float(center[feature].values)
        feature_scale = float(scale[feature].values)
        actual_value = normalized_actual * feature_scale + feature_center

        ax_actual = axs[row, 0]
        ax_normalized = axs[row, 1]

        for alpha, color in zip(ALPHAS, colors):
            actual_alpha = alpha * actual_value
            normalized_alpha = (actual_alpha - feature_center) / feature_scale

            label = f"{int(alpha * 100)}%"
            ax_actual.plot(days_before_present, actual_alpha, color=color, linewidth=1.3, label=label)
            ax_normalized.plot(days_before_present, normalized_alpha, color=color, linewidth=1.3, label=label)

        label = FEATURE_LABELS.get(feature, feature)
        ax_actual.set_ylabel(label, fontsize=9)
        ax_actual.grid(alpha=0.25)
        ax_normalized.grid(alpha=0.25)

        if row == 0:
            ax_actual.set_title("Actual alpha path", fontsize=13)
            ax_normalized.set_title("Normalized actual-alpha path", fontsize=13)

        for ax in (ax_actual, ax_normalized):
            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(0.8)

    axs[-1, 0].set_xlabel("Days before present (d)", fontsize=12)
    axs[-1, 1].set_xlabel("Days before present (d)", fontsize=12)
    axs[0, 1].legend(title="alpha", frameon=False, fontsize=8, title_fontsize=9, loc="best")

    fig.suptitle(
        f"Actual vs normalized alpha paths | basin {basin} | sample {sample_index} | prediction date {prediction_date}",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to {OUTPUT_FILE}")
    print(f"Basin: {basin}")
    print(f"Sample index: {sample_index}")
    print(f"Prediction date: {prediction_date}")


if __name__ == "__main__":
    main()
