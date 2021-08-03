import os
import h5py
import fiona
import rasterio
import numpy as np
from affine import Affine
from rasterio.crs import CRS
from pyproj import Transformer
from collections import namedtuple
from rasterio.features import bounds
from fiona.collection import Collection
from rasterio.coords import disjoint_bounds
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, transform, intersection


Bounds = namedtuple("Bounds", ["left", "bottom", "right", "top"])


class NeonDataset:
    def __init__(self, path: str):
        self.path = path
        self.h5 = h5py.File(path, "r")
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
            "height": self.h,
            "width": self.w,
            "crs": self.crs,
            "transform": self.transform
        }

        # TODO: More metadata fields under ["Reflectance"]["Metadata"]

    def crop_roi(self, shapefile: str, dest: str):
        with fiona.open(shapefile) as shp:
            trans = Transformer.from_crs(shp.crs["init"], self.crs) \
                if shp.crs["init"].split(":")[-1] != str(self.crs.to_epsg()) else lambda x, y: (x, y)
            for feature in shp:
                polygon, *_ = feature["coordinates"]
                feature["coordinates"] = [[trans(*x) for x in polygon]]
                feat_bounds = bounds(feature)
                if disjoint_bounds(feat_bounds, self.bounds):
                    continue
                feat_window = from_bounds(*feat_bounds, transform=self.transform)
                feat_window = intersection(feat_window, from_bounds(*self.bounds, transform=self.transform))
                feat_transform = transform(feat_window, self.transform)
                feat_arr = self.data[
                    int(feat_window.row_off):int(feat_window.row_off + feat_window.height),
                    int(feat_window.col_off):int(feat_window.col_off + feat_window.width),
                    :
                ]
                feat_meta = self.meta.copy()
                feat_meta["height"], feat_meta["width"], feat_meta["count"] = feat_arr.shape
                feat_meta["transform"] = feat_transform
                with rasterio.open(
                    os.path.join(dest, "{}.tif".format(feature["properties"]["plotID"])), "w", **feat_meta
                ) as dest:
                    dest.write(self.data[()].transpose(2, 0, 1))
                    for i, wls in enumerate(self.wavelengths):
                        dest.set_band_description(wls, i)

    def to_tiff(self):
        pass

    def __del__(self):
        self.h5.close()
