import os.path
import rasterio
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from task_pool import MPITaskPool


def work(folder_path):
    fs = glob(os.path.join(folder_path, "*.tif"))

    c = None
    spectra = []
    meta = {"FID": [], "domainID": [], "siteID": [], "plotID": [], "flightDate": []}
    for f in tqdm(fs):
        _, domain_id, site_id, _, yyyyddmm, _, _, ids = os.path.basename(f).replace(".tif", "").split("_")
        plot_id, fid = ids.split("-")

        with rasterio.open(f) as src:
            c, h, w = src.count, src.height, src.width
            if not (36 <= h <= 42 and 36 <= w <= 42):
                continue
            arr = src.read().astype(np.float32)
            arr[arr <= 0] = np.nan
            spec = np.nanmedian(arr, axis=(1, 2)).reshape(1, -1)
            if np.all(np.isnan(spec)):
                continue
            spectra.append(spec)
            meta["FID"].append(fid)
            meta["siteID"].append(site_id)
            meta["domainID"].append(domain_id)
            meta["flightDate"].append(yyyyddmm)
            meta["plotID"].append(f"{site_id}_{plot_id}")
    spectra = np.vstack(spectra)

    df = pd.concat([pd.DataFrame(meta), pd.DataFrame(spectra, columns=range(c))], axis=1)
    df.to_csv(os.path.join(folder_path, "roi.csv"), index=False)
    print(os.path.join(folder_path, "roi.csv"))


if __name__ == "__main__":
    exe = MPITaskPool()
    if exe.is_parent():
        args = glob("/taiga/illinois/aces_guan/sheng/Airborne/Processed_old/NEON/DP1.30006.001/*/*/ROI/")
    else:
        args = None
    exe.run(args, work, log_freq=1)

