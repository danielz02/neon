import os
import zlib
import pandas as pd
import h5py
import fiona
import rasterio
import numpy as np
import torch
from tqdm import tqdm
from affine import Affine
from osgeo import ogr, osr
from rasterio.crs import CRS
from collections import namedtuple
from rasterio.features import bounds
from rasterio.coords import disjoint_bounds
from rasterio.windows import from_bounds, transform, intersection
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, random_split, Dataset

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


class SpectralDataset(Dataset):
    def __init__(self, path: str, seq_len: int, target_name: str, scale: float = None, transforms=None, position_encode=False):
        super().__init__()
        self.scale = scale
        self.seq_len = seq_len
        self.df = pd.read_csv(path)
        self.transforms = transforms
        self.target_name = target_name
        self.position_encode = position_encode
        try:
            self.wls = torch.tensor(self.df.columns[:seq_len].astype(dtype=np.float32), dtype=torch.float32)
        except:
            print("")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x_tensor = torch.from_numpy(self.df.iloc[idx, :self.seq_len].to_numpy().astype(np.float32))
        y_tensor = torch.tensor(self.df[self.target_name].iloc[idx].astype(np.float32))
        if self.transforms:
            x_tensor = self.transforms(x_tensor)
        if self.scale:
            x_tensor *= self.scale
        if self.position_encode:
            x_tensor = torch.vstack([x_tensor, self.wls]).T
        else:
            x_tensor = x_tensor.unsqueeze(dim=1)
        return x_tensor, y_tensor


class GaussianNoise(object):
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        return batch + torch.normal(0, self.std, size=batch.shape)


def read_dataset(
    path: str, target_name: str, seq_len, split_ratio: float = None, scale: float = None,
        stratify: bool = False
):
    """
    Read any tabular dataset (with the last column being labels) as PyTorch TensorDataset
    :param stratify: whether to stratify the dataset while splitting
    :param scale: scale factor for the spectra
    :param seq_len: the length of the input features
    :param target_name: the name of the label column in the CSV
    :param path: path to the csv file
    :param split_ratio: Train/Validation split ratio
    :return: Train and validation dataset if split_ratio is define, the entire dataset otherwise
    """
    if split_ratio:
        assert split_ratio >= 0
        assert split_ratio <= 1

    df = pd.read_csv(path)
    x_arr = df.iloc[:, :seq_len]
    y_arr = df[target_name]

    x_tensor = torch.tensor(x_arr.to_numpy(), dtype=torch.float32)
    if scale:
        x_tensor *= scale
    x_tensor = x_tensor.unsqueeze(1)  # Add channel dimension -> (N, C, seq_len)
    y_tensor = torch.tensor(y_arr, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)

    if split_ratio:
        train_size = int(len(dataset) * split_ratio)
        val_size = len(dataset) - train_size
        if stratify:
            bin_edges = np.histogram_bin_edges(y_tensor, bins="auto")
            print(bin_edges)
            y_binned = np.digitize(y_tensor, bin_edges, right=True)
            train_set, val_set = train_test_split(
                dataset, train_size=train_size, test_size=val_size, random_state=0,
                shuffle=True, stratify=y_binned
            )
        else:
            train_set, val_set = random_split(
                dataset=dataset, lengths=[train_size, val_size], generator=torch.manual_seed(0)
            )
        return train_set, val_set
    else:
        return dataset
