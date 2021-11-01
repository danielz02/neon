import rasterio
import numpy as np
import pandas as pd
from scipy.stats import kde
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_squared_error, mean_absolute_error


def des_scatter_plot(y_true: np.ndarray, y_pred: np.ndarray, label_name: str, save_path: str = None, ax: plt.Axes = None):
    # assert (save_path is not None and ax is None) or (save_path is None and ax is not None)
    save_fig = ax is None and save_path is not None

    # Evaluate a gaussian kde on a regular grid of n_bins x n_bins over data extents
    # Calculate the point density
    xy = np.vstack([y_pred, y_true])
    colors = kde.gaussian_kde(xy)(xy)

    # r2 = r2_score(y_true=y_true, y_pred=y_pred)
    r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
    rmse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    bias = float(np.mean(y_pred - y_true))
    range_ = np.max(y_true) - np.min(y_true)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    # ax.scatter(x=y_pred, y=y_true, c=colors, s=80, alpha=1)
    # ax.scatter(x=y_pred, y=y_true, s=80, alpha=1, edgecolors="k")
    ax.scatter(x=y_pred, y=y_true, s=80, c=colors, alpha=1)
    # ax.set_title(label_name)

    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Actual Value")
    txt = f"n = {len(y_true)}\n" \
          f"RMSE = {rmse:.2f} ({(rmse / range_ * 100):.2f}%)\n" \
          f"Bias = {bias:.2f} ({(bias / range_ * 100):.2f}%)\n" \
          f"$R^2$ = {r2:.2f}\n"
    # f"MAE = {round(mae, 2)} ({round(mae / y_true.mean() * 100, 2)} %)\n"
    axis_min, axis_max = 0, max(np.max(y_true), np.max(y_pred))
    # axis_min = max(5 * round(axis_min - np.abs(axis_max - axis_min) * 0.05) / 5, 0)
    # axis_max = 5 * round((axis_max + np.abs(axis_max - axis_min) * 0.3) / 5 + 1)
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.plot(
        [-1000, 1000],
        [-1000, 1000],
        linestyle='--', color='k'
    )
    ya, xa = ax.get_yaxis(), ax.get_xaxis()
    locator = MaxNLocator(nbins=5, min_n_ticks=3, steps=[2, 5, 10], integer=True)
    locator.view_limits(axis_min, axis_max)
    ya.set_major_locator(locator)
    xa.set_major_locator(locator)
    ax.annotate(text=txt, xy=(0.02, 0.70), xycoords='axes fraction', color="red")
    if save_fig:
        plt.savefig(save_path, dpi=500)
    elif not ax:
        plt.show()


def visualize_spectrum(df: pd.DataFrame):
    with rasterio.open(
        "/home/danielz/PycharmProjects/neon/data/DP3.30006.001/YELL/2020-07/"
        "3758_YELL_008_D12_DP3_530000_4978000_reflectance.tif"
    ) as f:
        wls = np.array([float(x.split(" ")[0]) for x in f.descriptions])
    wls_idx = list(range(8, 191)) + list(range(216, 265)) + list(range(321, 403))
    spec_orig = df.sample(1)[[str(x) for x in wls_idx]].to_numpy().reshape(-1).tolist()
    spec = list(savgol_filter(spec_orig, window_length=5, polyorder=3, mode="interp"))
    spec_wls = wls[wls_idx].copy().tolist()
    spec.insert(183, np.nan)
    spec.insert(233, np.nan)
    spec_orig.insert(183, np.nan)
    spec_orig.insert(233, np.nan)
    spec_wls.insert(183, np.nan)
    spec_wls.insert(233, np.nan)
    plt.plot(spec_wls, spec_orig, label="Original")
    plt.plot(spec_wls, spec, label="SG")
    plt.legend()
    plt.show()


class WandbImageCallback(pl.Callback):
    """Logs the input and output images of a module.

    Images are stacked into a mosaic, with output on the top
    and input on the bottom."""

    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        trainer.logger.experiment.log({
            "val/example": pl_module.ax_val,
            "global_step": trainer.global_step
        })

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        trainer.logger.experiment.log({
            "test/example": pl_module.ax_test,
            "global_step": trainer.global_step
        })
