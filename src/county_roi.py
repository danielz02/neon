import os
from functools import partial
from glob import glob
from typing import Union

from task_pool import MPITaskPool
from argparse import ArgumentParser

import rasterio
from tqdm import tqdm
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.features import bounds
from rasterio.coords import disjoint_bounds
from rasterio._err import CPLE_OpenFailedError


import fiona
import geopandas
from fiona.transform import transform_geom
from packaging import version
from pyproj import CRS
from pyproj.enums import WktVersion
from shapely.geometry import mapping, shape


def crs_to_fiona(proj_crs):
    proj_crs = CRS.from_user_input(proj_crs)
    if version.parse(fiona.__gdal_version__) < version.parse("3.0.0"):
        fio_crs = proj_crs.to_wkt(WktVersion.WKT1_GDAL)
    else:
        # GDAL 3+ can use WKT2
        fio_crs = proj_crs.to_wkt()
    return fio_crs


def base_transformer(geom, src_crs, dst_crs):
    return shape(
        transform_geom(
            src_crs=crs_to_fiona(src_crs),
            dst_crs=crs_to_fiona(dst_crs),
            geom=mapping(geom),
            antimeridian_cutting=True,
        )
    )


def crop_tif(arg):
    path, _ = arg
    try:
        f = rasterio.open(path)
        dest_epsg = f.crs.to_epsg() 
    except Union[rasterio.errors.RasterioIOError, CPLE_OpenFailedError] as e:
        print("Failed to Open", e)
        return
    field_bounds = f.bounds

    shp = geopandas.read_file(args.shapefile)
    destination_crs = f"EPSG:{dest_epsg}"
    forward_transformer = partial(base_transformer, src_crs=shp.crs, dst_crs=destination_crs)

    with fiona.Env(OGR_ENABLE_PARTIAL_REPROJECTION="YES"):
        shp = shp.set_geometry(shp.geometry.apply(forward_transformer), crs=destination_crs)

    for feature in tqdm(shp, total=len(shp)):
        bbox = bounds(feature)
        if disjoint_bounds(field_bounds, bbox):
            continue
        cropped_image, cropped_transform = mask(f, [feature["geometry"]], crop=True)
        out_meta = f.meta
        out_meta.update({
            "driver": "GTiff", "height": cropped_image.shape[1], "width": cropped_image.shape[2],
            "transform": cropped_transform, "nodata": 0
        })
        fips = feature["properties"]["STATEFP"] + feature["properties"]["COUNTYFP"]
        dest_basename = os.path.basename(path).replace(".tif", f'_{fips}.tif')
        dest_path = os.path.join(os.path.dirname(path), "ROI", dest_basename)
        if not os.path.exists(os.path.join(os.path.dirname(path), "ROI")):
            os.makedirs(os.path.join(os.path.dirname(path), "ROI"))
        with rasterio.open(dest_path, "w", **out_meta) as dest:
            dest.write(cropped_image)
            dest.update_tags(**feature["properties"])
        print("Saving to:", dest_path)
    print("Finished", path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("shapefile", type=str)
    args = parser.parse_args()

    exe = MPITaskPool()
    if exe.is_parent():
        fs = glob(
            "/taiga/illinois/aces_guan/sheng/stair/fusion_*/*/*stack.tif",
            recursive=True
        )
        jobs = [(x, None) for x in sorted(fs)]
    else:
        jobs = []
    exe.run(jobs, crop_tif, log_freq=1)
