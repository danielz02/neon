from __future__ import annotations

import os
import zlib
from glob import glob
from typing import Optional, List, Tuple

import geopandas
import numpy
import pandas as pd
import h5py
import fiona
import pytorch_lightning as pl
import rasterio
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from affine import Affine
from osgeo import ogr, osr
from rasterio.crs import CRS
from collections import namedtuple
from rasterio.features import bounds
from rasterio.coords import disjoint_bounds
from sklearn.model_selection import train_test_split
from rasterio.windows import from_bounds, transform, intersection
from torch.utils.data import TensorDataset, random_split, Dataset, DataLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS


Bounds = namedtuple("Bounds", ["left", "bottom", "right", "top"])


class NeonDataset:
    def __init__(self, path: str, comm=None):
        self.path = path
        if not comm:
            self.h5 = h5py.File(path, "r")
        else:
            self.comm = comm
            self.h5 = h5py.File(path, "r", driver="mpio", comm=self.comm)
        self.site_name = list(self.h5.keys())[0]
        self.data = self.h5[self.site_name]["Reflectance"]["Reflectance_Data"]
        self.meta = self.h5[self.site_name]["Reflectance"]["Metadata"]
        self.h, self.w, self.c = self.data.shape

        mapinfo = self.meta['Coordinate_System']['Map_Info'][()].decode("utf-8").split(',')
        self.transform = Affine.from_gdal(
            float(mapinfo[3]), float(mapinfo[1]), 0, float(mapinfo[4]), 0, -float(mapinfo[2])
        )
        left, top = self.transform * (0, 0)
        right, bottom = self.transform * (self.w, self.h)
        self.bounds = Bounds(left=left, bottom=bottom, right=right, top=top)

        # TODO: Check wkt
        self.crs = CRS.from_epsg(self.meta["Coordinate_System"]["EPSG Code"][()])
        assert self.crs.is_valid

        self.fwhm = self.meta['Spectral_Data']['FWHM'][()]
        self.wavelengths = self.meta['Spectral_Data']['Wavelength'][()]
        self.wavelength_units = self.meta['Spectral_Data']['Wavelength'].attrs['Units']

        self.raster_meta = {
            "crs": self.crs,
            "nodata": -9999,
            "width": self.w,
            "height": self.h,
            "dtype": np.int16,
            "driver": "GTiff",
            "transform": self.transform
        }

        # TODO: More metadata fields under ["Reflectance"]["Metadata"]

    def crop_roi(self, shapefile: str, dest_dir: str):
        with fiona.open(shapefile) as shp:
            for feature in tqdm(shp, total=len(shp)):
                polygon, *_ = feature["geometry"]["coordinates"]
                if shp.crs["init"].split(":")[-1] != str(self.crs.to_epsg()):
                    feature["geometry"]["coordinates"] = [[
                        self.coord_trans(
                            int(shp.crs["init"].split(":")[-1]), self.crs.to_epsg(), *x
                        ) for x in polygon
                    ]]
                dest_path = os.path.join(
                    dest_dir, "{fid}_{pid}_{name}.tif".format(
                        fid=str(feature["id"]).zfill(4),
                        pid=feature["properties"]["plotID"],
                        name=os.path.basename(self.path).strip(".h5").strip("NEON_")
                                    .replace("{}_".format(self.site_name), "")
                    )
                )
                if os.path.exists(dest_path):
                    continue
                if np.any(np.isinf(feature["geometry"]["coordinates"])):
                    continue
                feat_bounds = bounds(feature)
                if disjoint_bounds(feat_bounds, self.bounds):
                    continue
                feat_window = from_bounds(*feat_bounds, transform=self.transform)
                feat_window = intersection(feat_window, from_bounds(*self.bounds, transform=self.transform))
                if feat_window.height < 1 or feat_window.width < 1:
                    continue
                feat_transform = transform(feat_window, self.transform)
                feat_arr = self.data[
                    int(feat_window.row_off):int(feat_window.row_off + feat_window.height),
                    int(feat_window.col_off):int(feat_window.col_off + feat_window.width),
                    :
                ]
                feat_meta = self.raster_meta.copy()
                feat_meta["height"], feat_meta["width"], feat_meta["count"] = feat_arr.shape
                feat_meta["transform"] = feat_transform
                with rasterio.open(dest_path, "w", **feat_meta) as dest:
                    dest.update_tags(**feature["properties"])
                    dest.write(feat_arr.transpose(2, 0, 1))
                    for i, wls in enumerate(self.wavelengths):
                        dest.set_band_description(i + 1, "{} {}".format(wls, self.wavelength_units))
                    print(dest_path)

    def to_tiff(self, dest_path: str = None):
        dest_path = self.path.replace(".h5", ".tif") if not dest_path else dest_path
        with rasterio.open(dest_path, "w", **self.raster_meta) as dest:
            dest.write(self.data[()].transpose(2, 0, 1))
            for i, wls in enumerate(self.wavelengths):
                dest.set_band_description(i + 1, wls)

    @staticmethod
    def coord_trans(src_epsg: int, dest_epsg: int, src_x: float, src_y: float):
        src_sr = osr.SpatialReference()
        src_sr.ImportFromEPSG(src_epsg)
        dest_sr = osr.SpatialReference()
        dest_sr.ImportFromEPSG(dest_epsg)

        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(src_x, src_y)  # use your coordinates here
        point.AssignSpatialReference(src_sr)  # tell the point what coordinates it's in
        point.TransformTo(dest_sr)  # project it to the out spatial reference
        return point.GetX(), point.GetY()

    def __del__(self):
        self.h5.close()


class SpectralDataset(pl.LightningDataModule):
    def __init__(
        self, path: str, seq_len: int, target_name: str | List[str], batch_size: int = 2048,
        split_ratios: List[float] = None, da_set: str = None, scale: float = None, transforms=None,
        position_encode=False, manual_splits: List = None
    ):
        super().__init__()
        self.scale = scale
        self.seq_len = seq_len
        self.df = pd.read_csv(path)
        self.batch_size = batch_size
        self.transforms = transforms
        self.target_name = target_name
        self.position_encode = position_encode
        if da_set is not None:
            self.target_domain = pd.read_csv(da_set)
            self.df["domain"] = 0
            self.target_domain["domain"] = 1
            self.df = pd.concat([self.df, self.target_domain], axis=0)
        else:
            self.target_domain = None

        try:
            self.wls = torch.tensor(self.df.columns[:seq_len].astype(dtype=np.float32), dtype=torch.float32)
        except:
            print("")

        idx = list(range(len(self.df)))
        if not split_ratios:
            self.test_idx = idx
            self.test_set = TensorDataset(*self.__getitem__(self.test_idx))
        elif len(split_ratios) == 2:
            train_ratio, test_ratio = split_ratios
            self.train_idx, self.test_idx = train_test_split(idx, train_size=train_ratio, test_size=test_ratio)
            self.train_set = TensorDataset(*self.__getitem__(self.train_idx))
            self.test_set = TensorDataset(*self.__getitem__(self.test_idx))
        elif len(split_ratios) == 3:
            train_ratio, val_ratio, test_ratio = split_ratios
            train_size, val_size = int(len(self.df) * train_ratio), int(len(self.df) * val_ratio)
            test_size = len(self.df) - train_size - val_size

            self.train_idx, self.test_idx = train_test_split(idx, test_size=test_size, train_size=(train_size + val_size))
            self.train_idx, self.val_idx = train_test_split(self.train_idx, train_size=train_size, test_size=val_size)
            self.train_set = TensorDataset(*self.__getitem__(self.train_idx))
            self.val_set = TensorDataset(*self.__getitem__(self.val_idx))
            self.test_set = TensorDataset(*self.__getitem__(self.test_idx))

        if manual_splits is not None:
            self.train_idx, self.val_idx, self.test_idx, self.pred_idx = manual_splits
            if self.train_idx is not None:
                self.train_set = TensorDataset(*self.__getitem__(self.train_idx))
            if self.val_idx is not None:
                self.val_set = TensorDataset(*self.__getitem__(self.val_idx))
            if self.test_idx is not None:
                self.test_set = TensorDataset(*self.__getitem__(self.test_idx))
            if self.pred_idx is not None:
                self.pred_set = TensorDataset(*self.__getitem__(self.pred_idx))

        # self.save_hyperparameters()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        assert self.train_idx is not None
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

    def val_dataloader(self) -> EVAL_DATALOADERS:
        assert self.val_idx is not None
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=os.cpu_count())

    def test_dataloader(self) -> EVAL_DATALOADERS:
        assert self.test_idx is not None
        print("Drop Last")
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=os.cpu_count(), drop_last=True)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        assert self.pred_idx is not None
        return DataLoader(self.pred_set, batch_size=self.batch_size, num_workers=os.cpu_count())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor] | Tuple[Tensor, Tensor]:
        x_tensor = torch.from_numpy(self.df.iloc[idx, :self.seq_len].to_numpy().astype(np.float32))
        y_tensor = torch.tensor(self.df[self.target_name].iloc[idx].to_numpy().astype(np.float32))
        if self.target_domain is not None:
            domain_tensor = torch.tensor(self.df["domain"].iloc[idx].to_numpy().astype(np.float32))
        else:
            domain_tensor = torch.zeros_like(y_tensor) * np.nan
        if self.transforms:
            x_tensor = self.transforms(x_tensor)
        if self.scale:
            x_tensor *= self.scale
        if self.position_encode:
            x_tensor = torch.vstack([x_tensor, self.wls]).T
        else:
            x_tensor = x_tensor.unsqueeze(dim=2)
        if self.target_domain is not None:
            return x_tensor, y_tensor, domain_tensor
        else:
            return x_tensor, y_tensor


class RasterDataset(pl.LightningDataModule):
    def __init__(self, path: str, scale: float = None, batch_size: int = 2048, nodata: int = None):
        super().__init__()
        self.raster = rasterio.open(path)
        if self.raster.count == 360:
            self.arr = self.raster.read(range(5, 361))
        else:
            self.arr = self.raster.read()
        self.c, self.h, self.w = self.arr.shape

        if self.raster.dtype == "uint16":
            scale = 1e-4
        if nodata:
            self.nodata = nodata
        else:
            self.nodata = self.raster.nodata
        self.batch_size = batch_size

        self.arr = self.arr.transpose(1, 2, 0).reshape(-1, self.c).astype(np.float32)
        if scale:
            self.arr *= self.arr * scale

        self.arr = torch.from_numpy(self.arr)
        self.dataset = TensorDataset(self.arr)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=os.cpu_count())

    def __len__(self) -> int:
        return self.h * self.w

    def __getitem__(self, idx):
        return self.arr[idx]

    def __del__(self):
        self.raster.close()


class IndianaDataset(pl.LightningDataModule):
    def __init__(self, shp_folder: str, field_spectra: str, attr: str, batch_size: int):
        super().__init__()
        self.attr = attr
        self.batch_size = batch_size
        self.spectra = pd.read_csv(field_spectra)
        self.spectra["Year"] = self.spectra["Date"].apply(lambda x: int(x[:4]))
        self.spectra.sort_values(["Year", "Feature ID", "Date"], inplace=True)

        mask = self.spectra.groupby(["Year", "Feature ID"])["Date"].transform(lambda x: len(x) == 215)
        self.spectra = self.spectra[mask].copy()

        self.spectra["Day"] = self.spectra.groupby(["Year", "Feature ID"])["Date"].transform(
            lambda x: list(range(len(x)))
        )
        self.spectra.set_index(["Year", "Feature ID"], inplace=True)
        self.spectra.sort_index(inplace=True)
        self.index = self.spectra.index.copy().drop_duplicates()
        self.spectra.set_index("Day", append=True, inplace=True)
        self.spectra.sort_index(inplace=True)

        p = sorted(glob(os.path.join(shp_folder, "Indiana_Fall_Transect_Data_V1_*_closest.shp")))
        self.shapes = pd.concat([
            geopandas.read_file(x).reset_index().rename({"index": "FID"}, axis=1).set_index(["Year", "FID"]) for x in p
        ])
        self.labels = self.shapes[self.attr].unique()
        self.label_encodings = {x: i for i, x in enumerate(self.labels)}

        # assert len(self.shapes.index) == len(self.spectra.index)

        self.stair_bands = ["blue", "green", "red", "nir", "swir1", "swir2"]

        self.train_size = int(self.__len__() * 0.6)
        self.val_size = int(self.__len__() * 0.2)
        self.test_size = self.__len__() - self.train_size - self.val_size

        self.train_idx, self.test_idx = train_test_split(
            self.index, test_size=self.test_size, train_size=(self.train_size + self.val_size)
        )
        self.train_idx, self.val_idx = train_test_split(
            self.index, train_size=self.train_size, test_size=self.val_size
        )

        self.train_set, self.val_set, self.test_set = [
            TensorDataset(*self.__getitem__(self.train_idx)),
            TensorDataset(*self.__getitem__(self.val_idx)),
            TensorDataset(*self.__getitem__(self.test_idx))
        ]

        self.save_hyperparameters()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if isinstance(idx, pd.MultiIndex):
            idx_spectra = np.repeat(np.array(idx.to_list()), len(idx))
            idx_spectra = np.hstack([idx_spectra, np.repeat(np.array(range(215)).reshape(-1, 1), repeats=len(idx))])
        else:
            idx_spectra = idx
        spectra = self.spectra[self.stair_bands].loc[idx_spectra].to_numpy()
        if isinstance(idx, pd.MultiIndex):
            spectra = np.vsplit(spectra, len(idx))
        spectra = torch.from_numpy(spectra).float()
        if isinstance(idx, pd.MultiIndex):
            labels = torch.from_numpy(self.shapes[self.attr][idx].replace(self.label_encodings))
        else:
            labels = torch.tensor(self.label_encodings[self.shapes[self.attr][idx]])

        return spectra, labels


class GaussianNoise(object):
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        return batch + torch.normal(0, self.std, size=batch.shape)


