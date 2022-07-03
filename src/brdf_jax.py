# -*- coding: utf-8 -*-
"""
    BRDF Correction
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]

import jax
import jax.numpy as jnp
from jax import jit, vmap

import time
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from dataset import NeonDataset

try:
    from osgeo import gdal
except ImportError:
    import gdal


def writeTiff(im_data, im_width, im_height, im_bands, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = jnp.asarray([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if dataset is not None:
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).SetNoDataValue(0)
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


# calc_geom_kernel(sra,sza,vra,vza,'li_sparse')
# vza = Sensor view zenith angle.
# vra = Sensor view azimuth angle.
# sza = Solar zenith angle.
# sra = Solar azimuth angle.
@jit
def calc_geom_kernel_li_sparse(solar_az, solar_zn, sensor_az, sensor_zn, b_r=1., h_b=2.):
    """Calculate geometric scattering kernel.
       Constants b_r (b/r) and h_b (h/b) from Colgan et al. RS 2012
       Alternatives include MODIS specification:
           b/r : sparse: 1, dense: 2.5
           h/b : sparse, dense : 2
       All input geometry units must be in radians.
    Args:
        solar_az (numpy.ndarray): Solar azimuth angle.
        solar_zn (numpy.ndarray): Solar zenith angle.
        sensor_az (numpy.ndarray): Sensor view azimuth angle.
        sensor_zn (numpy.ndarray): Sensor view zenith angle.
        kernel (str): Li geometric scattering kernel type [li_dense,li_sparse, roujean].
        b_r (float, optional): Object height. Defaults to 10.
        h_b (float, optional): Object shape. Defaults to 2.
    Returns:
        numpy.ndarray: Geometric scattering kernel.
    """

    relative_az = sensor_az - solar_az

    # Eq. 37,52. Wanner et al. JGRA 1995
    solar_zn_p = jnp.arctan(b_r * jnp.tan(solar_zn))
    sensor_zn_p = jnp.arctan(b_r * jnp.tan(sensor_zn))
    # Eq 50. Wanner et al. JGRA 1995
    D = jnp.sqrt((jnp.tan(solar_zn_p) ** 2) + (jnp.tan(sensor_zn_p) ** 2) - 2 * jnp.tan(solar_zn_p) * jnp.tan(
        sensor_zn_p) * jnp.cos(relative_az))
    # Eq 49. Wanner et al. JGRA 1995
    t_num = h_b * jnp.sqrt(D ** 2 + (jnp.tan(solar_zn_p) * jnp.tan(sensor_zn_p) * jnp.sin(relative_az)) ** 2)
    t_denom = (1 / jnp.cos(solar_zn_p)) + (1 / jnp.cos(sensor_zn_p))
    t = jnp.arccos(jnp.clip(t_num / t_denom, -1, 1))
    # Eq 33,48. Wanner et al. JGRA 1995
    O = (1 / jnp.pi) * (t - jnp.sin(t) * jnp.cos(t)) * t_denom
    # Eq 51. Wanner et al. JGRA 1995
    cos_phase_p = jnp.cos(solar_zn_p) * jnp.cos(sensor_zn_p) + jnp.sin(solar_zn_p) * jnp.sin(sensor_zn_p) * jnp.cos(
        relative_az)

    k_geom = O - (1 / jnp.cos(solar_zn_p)) - (1 / jnp.cos(sensor_zn_p)) + .5 * (1 + cos_phase_p) * (
            1 / jnp.cos(sensor_zn_p))

    return k_geom


@jit
def calc_geom_kernel_li_dense(solar_az, solar_zn, sensor_az, sensor_zn, b_r=1., h_b=2.):
    """Calculate geometric scattering kernel.
       Constants b_r (b/r) and h_b (h/b) from Colgan et al. RS 2012
       Alternatives include MODIS specification:
           b/r : sparse: 1, dense: 2.5
           h/b : sparse, dense : 2
       All input geometry units must be in radians.
    Args:
        solar_az (numpy.ndarray): Solar azimuth angle.
        solar_zn (numpy.ndarray): Solar zenith angle.
        sensor_az (numpy.ndarray): Sensor view azimuth angle.
        sensor_zn (numpy.ndarray): Sensor view zenith angle.
        kernel (str): Li geometric scattering kernel type [li_dense,li_sparse, roujean].
        b_r (float, optional): Object height. Defaults to 10.
        h_b (float, optional): Object shape. Defaults to 2.
    Returns:
        numpy.ndarray: Geometric scattering kernel.
    """

    relative_az = sensor_az - solar_az

    # Eq. 37,52. Wanner et al. JGRA 1995
    solar_zn_p = jnp.arctan(b_r * jnp.tan(solar_zn))
    sensor_zn_p = jnp.arctan(b_r * jnp.tan(sensor_zn))
    # Eq 50. Wanner et al. JGRA 1995
    D = jnp.sqrt((jnp.tan(solar_zn_p) ** 2) + (jnp.tan(sensor_zn_p) ** 2) - 2 * jnp.tan(solar_zn_p) * jnp.tan(
        sensor_zn_p) * jnp.cos(relative_az))
    # Eq 49. Wanner et al. JGRA 1995
    t_num = h_b * jnp.sqrt(D ** 2 + (jnp.tan(solar_zn_p) * jnp.tan(sensor_zn_p) * jnp.sin(relative_az)) ** 2)
    t_denom = (1 / jnp.cos(solar_zn_p)) + (1 / jnp.cos(sensor_zn_p))
    t = jnp.arccos(jnp.clip(t_num / t_denom, -1, 1))
    # Eq 33,48. Wanner et al. JGRA 1995
    O = (1 / jnp.pi) * (t - jnp.sin(t) * jnp.cos(t)) * t_denom
    # Eq 51. Wanner et al. JGRA 1995
    cos_phase_p = jnp.cos(solar_zn_p) * jnp.cos(sensor_zn_p) + jnp.sin(solar_zn_p) * jnp.sin(sensor_zn_p) * jnp.cos(
        relative_az)

    k_geom = (((1 + cos_phase_p) * (1 / jnp.cos(sensor_zn_p))) / (t_denom - O)) - 2

    return k_geom


@jit
def calc_volume_kernel_ross_thin(solar_az, solar_zn, sensor_az, sensor_zn):
    # Eq 13. Wanner et al. JGRA 1995
    relative_az = sensor_az - solar_az

    # Eq 2. Schlapfer et al. IEEE-TGARS 2015
    phase = jnp.arccos(
        jnp.cos(solar_zn) * jnp.cos(sensor_zn) + jnp.sin(solar_zn) * jnp.sin(sensor_zn) * jnp.cos(relative_az))
    k_vol = ((jnp.pi / 2 - phase) * jnp.cos(phase) + jnp.sin(phase)) / (jnp.cos(sensor_zn) * jnp.cos(solar_zn)) - (
            jnp.pi / 2)

    return k_vol


@jit
def calc_volume_kernel_ross_thick(solar_az, solar_zn, sensor_az, sensor_zn):
    # Eq 13. Wanner et al. JGRA 1995
    relative_az = sensor_az - solar_az

    # Eq 2. Schlapfer et al. IEEE-TGARS 2015
    phase = jnp.arccos(
        jnp.cos(solar_zn) * jnp.cos(sensor_zn) + jnp.sin(solar_zn) * jnp.sin(sensor_zn) * jnp.cos(relative_az))
    # Eq 7. Wanner et al. JGRA 1995
    k_vol = ((jnp.pi / 2 - phase) * jnp.cos(phase) + jnp.sin(phase)) / (jnp.cos(sensor_zn) * jnp.cos(solar_zn)) - (
            jnp.pi / 4)

    return k_vol


geom_kernel = {
    "li_dense": calc_geom_kernel_li_dense,
    "li_sparse": calc_geom_kernel_li_sparse
}

volume_kernel = {
    "ross_thin": calc_volume_kernel_ross_thin,
    "ross_thick": calc_volume_kernel_ross_thick,
}


def brdf_band(band_arr, masks, sra, sza, vra, vza):
    y1 = band_arr
    tmp = y1.copy()
    for ni in range(5):
        if ni <= 3:
            k_geo = calc_geom_kernel_li_sparse(sra, sza, vra, vza)
            k_vol = calc_volume_kernel_ross_thin(sra, sza, vra, vza)

            k_geo_nadir = calc_geom_kernel_li_sparse(0, sza, 0, 0, )
            k_vol_nadir = calc_volume_kernel_ross_thin(0, sza, 0, 0)
        else:
            k_geo = calc_geom_kernel_li_dense(sra, sza, vra, vza)
            k_vol = calc_volume_kernel_ross_thick(sra, sza, vra, vza)

            k_geo_nadir = calc_geom_kernel_li_dense(0, sza, 0, 0)
            k_vol_nadir = calc_volume_kernel_ross_thick(0, sza, 0, 0)
        mask = masks[ni]

        p1 = k_vol[mask]
        p2 = k_geo[mask]
        x = jnp.c_[p1, p2, jnp.ones(p2.shape)]
        y = y1[mask]
        if len(y1) > 10000:
            y = y[0:10000]
            x = x[0:10000, :]
        coeffs = [1, 1, 1]
        try:
            coeffs = jnp.linalg.lstsq(x, y, rcond=None)[0]
        except Exception:
            pass

        brdf = coeffs[0] * k_vol + coeffs[1] * k_geo + 1 * coeffs[2]
        brdf_nadir = coeffs[0] * k_vol_nadir + coeffs[1] * k_geo_nadir + 1 * coeffs[2]

        correction_factor = brdf_nadir / brdf
        tmp = tmp.at[mask].set(correction_factor[mask] * y1[mask])

    return tmp


def work(image):
    start_time = time.time()
    out_img = os.path.splitext(image)[0] + '_BRDF_Corrected.tif'

    if os.path.exists(out_img):
        print(f"[{os.environ['OMPI_COMM_WORLD_RANK']}/{os.environ['OMPI_COMM_WORLD_SIZE']}] File Exists: ", out_img)
        if os.path.exists(image) and not image.endswith(".h5"):
            print(f"[{os.environ['OMPI_COMM_WORLD_RANK']}/{os.environ['OMPI_COMM_WORLD_SIZE']}] Removing: ", image)
            os.remove(image)
        return
    else:
        print(f"[{os.environ['OMPI_COMM_WORLD_RANK']}/{os.environ['OMPI_COMM_WORLD_SIZE']}] Processing: ", image)
        try:
            shutil.copy(image, "/tmp")
        except shutil.SameFileError:
            pass
        if image.endswith(".h5"):
            data = NeonDataset(os.path.join("/tmp", os.path.basename(image)))
            tiff_tmp_path = os.path.join("/tmp", os.path.basename(image.replace(".h5", ".tif")))
            data.to_tiff(tiff_tmp_path)
            print(f"[{os.environ['OMPI_COMM_WORLD_RANK']}/{os.environ['OMPI_COMM_WORLD_SIZE']}]", tiff_tmp_path)
            image = image.replace(".h5", ".tif")

        ds = gdal.Open(os.path.join("/tmp", os.path.basename(image)), gdal.GA_ReadOnly)
        if ds is None:
            print(f"[{os.environ['OMPI_COMM_WORLD_RANK']}/{os.environ['OMPI_COMM_WORLD_SIZE']}]",
                  f"GDAL cannot open input raster {image} !!")
            return
            # raise RuntimeError("GDAL cannot open input raster!!")
        im_geotrans = list(ds.GetGeoTransform())
        im_proj = ds.GetProjection()
        im_width = ds.RasterXSize  # samples
        im_height = ds.RasterYSize  # lines
        im_band = ds.RasterCount  # bands

        bands = ds.ReadAsArray()
        if bands is None:
            print(f"[{os.environ['OMPI_COMM_WORLD_RANK']}/{os.environ['OMPI_COMM_WORLD_SIZE']}]",
                  f"GDAL cannot open input raster {image} !!")
            return

        out_data = bands[0:5]
        shadow_mask = out_data[-1].copy()

        deg2rad = 3.1415926 / 180
        vza = jnp.asarray(out_data[0] * deg2rad / 100)
        vra = jnp.asarray(out_data[1] * deg2rad / 100)
        sza = jnp.asarray(out_data[2] * deg2rad / 100)
        sra = jnp.asarray(out_data[3] * deg2rad / 100)

        red = bands[51 + 5 - 1].copy() * 1e-5
        nir = bands[70 + 5 - 1].copy() * 1e-5
        NDVI = (nir - red) / (nir + red + 0.000001)
        NDVI[NDVI < 0] = 0
        NDVI[NDVI == 1] = 0.99
        NDVI = NDVI * 5
        NDVI = NDVI.astype(int)

        bands = jnp.asarray(bands)
        masks = [jnp.where(NDVI == i) for i in range(5)]
        # brdf_vmap = vmap(brdf_band)
        brdf_out = []
        for band in tqdm(range(5, im_band)):
            brdf_out.append(np.asarray(brdf_band(bands[band], masks, sra, sza, vra, vza)))
        del bands

        out_data = np.concatenate([out_data, np.stack(brdf_out)]).astype(np.uint16)
        out_data[5:, shadow_mask != 1] = 0
        out_data[5:, nir > 0.06] = 0

        out_img_tmp = os.path.join("/tmp", os.path.basename(out_img))
        writeTiff(out_data, im_width, im_height, im_band, im_geotrans, im_proj, out_img_tmp)
        shutil.copy(out_img_tmp, out_img)
        os.remove(out_img_tmp)
        print(f"[{os.environ['OMPI_COMM_WORLD_RANK']}/{os.environ['OMPI_COMM_WORLD_SIZE']}]", out_img)

        index = [51 + 5, 32 + 5, 13 + 5]
        out_RGB = out_data[index, :, :].astype(np.uint16)
        out_img = os.path.splitext(out_img)[0] + '_RGB.tif'
        out_img_tmp = os.path.join("/tmp", os.path.basename(out_img))
        writeTiff(out_RGB, im_width, im_height, 3, im_geotrans, im_proj, out_img_tmp)
        shutil.copy(out_img_tmp, out_img)
        if os.path.exists(image):
            os.remove(image)
            print(f"[{os.environ['OMPI_COMM_WORLD_RANK']}/{os.environ['OMPI_COMM_WORLD_SIZE']}] Removing: ", image)
        os.remove(out_img_tmp)
        print(f"[{os.environ['OMPI_COMM_WORLD_RANK']}/{os.environ['OMPI_COMM_WORLD_SIZE']}]", out_img)
        print(f"[{os.environ['OMPI_COMM_WORLD_RANK']}/{os.environ['OMPI_COMM_WORLD_SIZE']}]",
              f"--- {time.time() - start_time} seconds ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-name", type=str, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if os.path.isfile(args.image_name) and not args.image_name.endswith(".txt"):
        work(args.image_name)
    else:
        import mpi4py
        from glob import glob
        from task_pool import MPITaskPool

        exe = MPITaskPool()
        if exe.is_parent():
            if args.image_name.endswith(".txt"):
                with open(args.image_name) as f:
                    arg_list = [x.strip("\n") for x in f.readlines()]
            else:
                arg_list = glob(args.image_name)[::-1]
                # arg_list = arg_list[:(len(arg_list) // 2)]
        else:
            arg_list = None
        print(f"[{os.environ['OMPI_COMM_WORLD_RANK']}/{os.environ['OMPI_COMM_WORLD_SIZE']}]", jax.devices())
        if "Gpu" not in str(jax.devices()) and not args.cpu:
            import sys

            print(f"[{os.environ['OMPI_COMM_WORLD_RANK']}/{os.environ['OMPI_COMM_WORLD_SIZE']}]", "GPU not available!")
            sys.exit(0)
        exe.run(arg_list, work, log_freq=1)
