#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 18/01/2024

import os
import glob

import xarray as xr
import rioxarray
import rasterio
import numpy as np
import pyresample
from pyresample.bucket import BucketResampler

import vortex
from vortex import toolbox
import cen

toolbox.active_now = True

# This script upscales 30m wind fields produced by Louis Le Toumelin's method to an EDELWEISS's 250m grid
# Input data are monthly Netcdf files at 30m resolution with wind 'speed' and 'direction' variables.
# Output data is one single yearly Netcdf file at 250 resolution with FORCING-compatible 'Wind' and 'Wind_dir' variables
#
# METHOD (from Ange Haddjeri) :
# -----------------------------
# 1. Concatenate all data into 1 single xarray dataset to avoid duplicates or missing data
# 2. Apply 'wind2comp' method to get U and V wind components
# 3. Upscale each component separately with rioxarray "reproject_match" function and "average" option (https://corteva.github.io/rioxarray/html/examples/reproject_match.html)
# 4. Convert U and V wind components into 'Wind' and 'Wind_dir' with 'comp2speed' et 'comp2dir' methods
# 5. Update netcdf attributes
# 6. Archive output netcdf file on hendrix with Vortex

datadir = '/home/merzisenh/NO_SAVE/arome_2021_2022_downscaled_devine/downscaled'  # Extraction HM de decembre 2023 from 2021070106 to 2022073123
#workdir = '/home/vernaym/workdir/EDELWEISS/wind'
workdir = '/cnrm/cen/users/NO_SAVE/vernaym/workdir/EDELWEISS/wind'

# Fonction LLT*
def wind2comp(uv, dir, unit_direction="radian"):
    if unit_direction == "degree":
        dir = np.deg2rad(dir)
    u = -np.sin(dir) * uv
    v = -np.cos(dir) * uv
    return u, v

# Fonction LLT*
def comp2dir(u, v, unit_output="degree"):
    if unit_output == "degree":
        return np.mod(180 + np.rad2deg(np.arctan2(u, v)), 360)
    else:
        raise NotImplementedError

# Fonction LLT*
def comp2speed(u, v, w=None):
    if w is None:
        return np.sqrt(u ** 2 + v ** 2)
    else:
        return np.sqrt(u ** 2 + v ** 2 + w ** 2)

# * Source des fonctions LLT : https://github.com/louisletoumelin/bias_correction/blob/12e806af084d086d30e429b21deb8ab7f243a381/bias_correction/train/wind_utils.py

def upscale(filename):
    """
    Upscale 30m resolution wind fields from LTT to 250m Wind/Wind_dir FORCING-like variables
    """

    #if not (os.path.exists(os.path.join(workdir, 'u30m.zarr')) and os.path.exists(os.path.join(workdir, 'v30m.zarr'))):
    # 1. Open monthly 30m wind Netcdf files produced by HM into 1 single xarray dataset and add projection
    suffix = '_'.join(file.split('.')[0].split('_')[-2:])
    wind30m = xr.open_dataset(filename)
    #wind30m = xr.open_mfdataset(os.path.join(datadir, '*'))
    #wind30m = xr.open_mfdataset(os.path.join(datadir, '*'), chunks={'x': 500, 'y': 500})
    wind30m = wind30m.rio.write_crs(2154)  # 3857 ?
    #wind30m = wind30m.chunk({'time':100, 'x': 10, 'y': 10})
    wind30m = wind30m.chunk({'time':100})
    # TODO : optimisation of chunk sizes necessary
    #wind30m.to_zarr(os.path.join(workdir, 'wind30m.zarr'), mode='w')

    # 2. Compute U and V wind components
    u30m, v30m = wind2comp(wind30m.speed, wind30m.direction, unit_direction="degree")

    #u30m.to_zarr(os.path.join(workdir, 'u30m.zarr'))
    #v30m.to_zarr(os.path.join(workdir, 'v30m.zarr'))

    #print('Read u30m')
    #u30m = xr.open_zarr(os.path.join(workdir, 'u30m.zarr'), chunks={'time':1000, 'x': 500, 'y': 500})
    #u30m = xr.open_zarr(os.path.join(workdir, 'u30m.zarr'))
    #print('Read v30m')
    #v30m = xr.open_zarr(os.path.join(workdir, 'v30m.zarr'), chunks={'time':1000, 'x': 500, 'y': 500})
    #v30m = xr.open_zarr(os.path.join(workdir, 'v30m.zarr'))

    # 3. Project 30m U and V wind components on the 250m grid

    # Open 250m grid reference file
    #gr250m = xr.open_dataarray('/home/vernaym/workdir/EDELWEISS/SAFRAN_to_grid/meteo/FORCING_2021080106_2022080106_gr250m.nc')  # FORCING projected from SAFRAN reanalysis
    gr250m = xr.open_dataset('/home/vernaym/These/DATA/MNTLouisGRoussecorrected.nc')  # Ange's MNT with elevation adapted to SARAN geometry
    gr250m = gr250m.rio.write_crs(2154)

    # With pyresample (NOT WORKING)
#    source_area = pyresample.geometry.AreaDefinition(  # https://pyresample.readthedocs.io/en/latest/api/pyresample.html#module-pyresample.geometry
#            'gr30m',  # area_id
#            'Grandes Rousses domain at 30m resolution',  # description
#            'WGS84',  #proj_id
#            'Europe_Lambert_Conformal_Conic',  # projection
#            len(u30m.x.data),  # width
#            len(u30m.y.data),  # height
#            (u30m.x.data[0], u30m.y.data[0], u30m.x.data[-1], u30m.y.data[-1]),  #area_extent
#        )
#
#    target_area = pyresample.geometry.AreaDefinition(  # https://pyresample.readthedocs.io/en/latest/api/pyresample.html#module-pyresample.geometry
#            'gr250m',  # area_id
#            'Grandes Rousses domain at 250m resolution',  # description
#            'WGS84',  #proj_id
#            'Europe_Lambert_Conformal_Conic',  # projection
#            len(gr250m.x.data),  # width
#            len(gr250m.y.data),  # height
#            (gr250m.x.data[0], gr250m.y.data[0], gr250m.x.data[-1], gr250m.y.data[-1]),  #area_extent
#        )

    # Project 30m fields on 250m grid
    #lons, lats = source_area.get_lonlats()
    #resampler = BucketResampler(target_area, lons, lats)  # https://pyresample.readthedocs.io/en/latest/api/pyresample.html
    print(f'Projection U {suffix}')
    u250m = u30m.rio.reproject_match(gr250m, resamplin="average")  # rioxarray does not support dask : https://github.com/corteva/rioxarray/issues/119 --> memory limit
    #u250m = resampler.get_average(u30m)
    print(f'Projection V {suffix}')
    v250m = v30m.rio.reproject_match(gr250m, resamplin="average")

    # 4. Convert U/V into Wind/Wind_dir
    wind250m = comp2speed(u250m, v250m)
    wind250m.to_netcdf(os.path.join(workdir, f'Wind_gr250m_{suffix}.nc'))
    wind250m.close()
    wdir250m = comp2dir(u250m, v250m)
    wdir250m.to_netcdf(os.path.join(workdir, f'Wind_dir_gr250m_{suffix}.nc'))
    wdir250m.close()

if __name__ == '__main__':

    outname = os.path.join(workdir, 'Wind_gr250m_2021070106_2022073123.nc')
    if not os.path.exists(outname):

        # 1. Upscale 30m resolution wind fields from LTT to 250m Wind/Wind_dir FORCING-like variables
        #for file in glob.glob(os.path.join(datadir, '*')):  # Loop over files to avoid memory limitations
        #    upscale(file)

        # 2. Read the (multiple) files created and concatenate data into 1 single netcf file
        wind250m = xr.open_mfdataset(glob.glob(os.path.join(workdir, 'Wind_gr250m*.nc')))
        wind250m = wind250m.rename({'__xarray_dataarray_variable__':'Wind'})
        wdir250m = xr.open_mfdataset(glob.glob(os.path.join(workdir, 'Wind_dir_gr250m*.nc')))
        wdir250m = wdir250m.rename({'__xarray_dataarray_variable__':'Wind_dir'})
        wind250m = wind250m.merge(wdir250m)
        wind250m.to_netcdf(outname)

    tbout = toolbox.output(
        role        = 'Wind',
        kind        = 'Wind',
        vapp        = 'edelweiss',
        vconf       = '[geometry:area]',
        source_app  = 'arome',
        source_conf = '4dvarfr',
        cutoff      = 'assimilation',
        filename    = outname,
        experiment  = 'WHM01@vernaym',  # First experiment produced with Hugo.M wind
        geometry    = 'GrandesRousses250m',
        nativefmt   = 'netcdf',
        namebuild   = 'flat@cen',
        model       = 'devine',
        #date        = enddate.ymd6h,
        #datebegin   = startdate.ymd6h,
        #dateend     = enddate.ymd6h,
        date        = '2022080106',
        datebegin   = '2021070106',
        dateend     = '2022073123',
        namespace   = 'vortex.multi.fr',
        block       = 'analysis',
        intent      = 'inout',
    )

