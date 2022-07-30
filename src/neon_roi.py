import os
from glob import glob
from typing import Union

from osgeo import ogr, osr
from task_pool import MPITaskPool
from argparse import ArgumentParser


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


def crop_h5(arg):
    from dataset import NeonDataset

    path, comm = arg
    ds = NeonDataset(path=path)
    ds.crop_roi(shapefile=args.shapefile, dest_dir=os.path.dirname(path))


def crop_tif(arg):
    import fiona
    import rasterio
    import h5py
    from tqdm import tqdm
    from rasterio.crs import CRS
    from rasterio.mask import mask
    from rasterio.features import bounds
    from rasterio.coords import disjoint_bounds
    from rasterio._err import CPLE_OpenFailedError

    path, _ = arg
    _, _, site_id, *_ = os.path.basename(path).split("_")
    with h5py.File(path.replace("_BRDF_Corrected.tif", ".h5"), "r") as f:
        meta = f[site_id]["Reflectance"]["Metadata"]
        dest_epsg = int(meta["Coordinate_System"]["EPSG Code"][()])
    try:
        f = rasterio.open(path)
    except Union[rasterio.errors.RasterioIOError, CPLE_OpenFailedError] as e:
        print("Failed to Open", e)
        return
    field_bounds = f.bounds

    with fiona.open(args.shapefile) as shp:
        for feature in tqdm(shp, total=len(shp)):
            if feature["properties"]["siteID"] != site_id:
                continue
            polygon, *_ = feature["geometry"]["coordinates"]
            feature["geometry"]["coordinates"] = [[
                coord_trans(CRS.from_dict(shp.crs).to_epsg(), dest_epsg, *reversed(x)) for x in polygon
            ]]

            bbox = bounds(feature)
            if disjoint_bounds(field_bounds, bbox):
                continue
            cropped_image, cropped_transform = mask(f, [feature["geometry"]], crop=True)
            out_meta = f.meta
            out_meta.update({
                "driver": "GTiff", "height": cropped_image.shape[1], "width": cropped_image.shape[2],
                "transform": cropped_transform, "nodata": 0
            })
            dest_basename = os.path.basename(path).replace(
                "BRDF_Corrected", f'{feature["properties"]["plotID"].split("_")[1]}-{str(feature["id"]).zfill(4)}'
            )
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
            "/taiga/illinois/aces_guan/sheng/Airborne/Processed_old/NEON/DP1.30006.001/**/*_BRDF_Corrected.tif",
            recursive=True
        )
        jobs = [(x, None) for x in sorted(fs)]
    else:
        jobs = []
    exe.run(jobs, crop_tif, log_freq=1)
