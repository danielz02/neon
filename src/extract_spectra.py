import rasterio
import numpy as np
import pandas as pd
from glob import glob
from pandas.tseries.offsets import DateOffset

df = pd.read_csv("/scratch/sciteam/ILL_bbdv/NEON/data/NEON.DP1.10026.001.cfc_carbonNitrogen.expanded.csv")
df["collectDate"] = pd.to_datetime(df["collectDate"])

rows = []
for i in range(len(df)):
    row = df.loc[i]
    yymm = row["collectDate"].strftime("%Y-%m")
    fid = str(row["FID"]).zfill(4)
    site_id = row["siteID"]
    p = sorted(glob(
        "/scratch/sciteam/ILL_bbdv/NEON/DP3.30006.001/{site}/{yymm}/{fid}_*.tif"
        .format(site=site_id, yymm=yymm, fid=fid)
    ))
    if len(p) == 0:
        yymm = (row["collectDate"] + DateOffset(20)).strftime("%Y-%m")
        p = sorted(glob(
            "/scratch/sciteam/ILL_bbdv/NEON/DP3.30006.001/{site}/{yymm}/{fid}_*.tif"
            .format(site=site_id, yymm=yymm, fid=fid)
        ))
    if len(p) == 0:
        yymm = (row["collectDate"] - DateOffset(20)).strftime("%Y-%m")
        p = sorted(glob(
            "/scratch/sciteam/ILL_bbdv/NEON/DP3.30006.001/{site}/{yymm}/{fid}_*.tif"
            .format(site=site_id, yymm=yymm, fid=fid)
        ))
    if len(p) == 0:
        print("Not Found:", fid, row["plotID"], row["collectDate"].strftime("%Y-%m-%d"))
    else:
        print("Found:", fid, row["plotID"], row["collectDate"].strftime("%Y-%m-%d"))
    for f in p:
        arrs = []
        with rasterio.open(f) as src:
            arr = src.read()
            c, h, w = arr.shape
            arr = arr.reshape(c, -1).T
            arrs.append(arr)
        arrs = np.concatenate(arrs, axis=0)
        df_plot = pd.concat([
            pd.DataFrame({"UID": [row["UID"]] * len(arrs), "FID": [row["FID"]] * len(arrs), "Date": [yymm] * len(arr)}),
            pd.DataFrame(arrs, columns=range(c)).reset_index(drop=True)
        ], axis=1).reindex()
        rows.append(df_plot)
rows = pd.concat(rows, axis=0)
rows.to_csv(
    "/scratch/sciteam/ILL_bbdv/NEON/data/NEON.DP1.10026.001.cfc_carbonNitrogen.expanded.spectra.csv",
    index=False
)
