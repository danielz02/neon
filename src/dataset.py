import os
import zlib
import h5py
import fiona
import rasterio
import numpy as np
from tqdm import tqdm
from affine import Affine
from osgeo import ogr, osr
from rasterio.crs import CRS
from collections import namedtuple
from rasterio.features import bounds
from rasterio.coords import disjoint_bounds
from rasterio.windows import from_bounds, transform, intersection

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
