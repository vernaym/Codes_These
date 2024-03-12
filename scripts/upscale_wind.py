#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 18/01/2024

import os
import glob

import xarray as xr
import pandas as pd
import numpy as np

from snowtools.scripts.extract.vortex import vortexIO


DEFAULT_NETCDF_FORMAT = 'NETCDF4_CLASSIC'

# This script upscales 30m wind fields produced by Louis Le Toumelin's method to an EDELWEISS's 250m grid
# Input data are monthly Netcdf files at 30m resolution with wind 'speed' and 'direction' variables.
# Output data is one single yearly Netcdf file at 250 resolution with FORCING-compatible 'Wind' and 'Wind_dir' variables
#
# METHOD (from Ange Haddjeri) :
# -----------------------------
# 1. Concatenate all data into 1 single xarray dataset to avoid duplicates or missing data
# 2. Apply 'wind2comp' method to get U and V wind components
# 3. Upscale each component separately with rioxarray "reproject_match" function and "average" option
#    (https://corteva.github.io/rioxarray/html/examples/reproject_match.html)
# 4. Convert U and V wind components into 'Wind' and 'Wind_dir' with 'comp2speed' et 'comp2dir' methods
# 5. Update netcdf attributes
# 6. Archive output netcdf file on hendrix with Vortex

# Extraction HM de decembre 2023 from 2021070106 to 2022073123
datadir = '/home/merzisenh/NO_SAVE/arome_2021_2022_downscaled_devine/downscaled'
# workdir = '/home/vernaym/workdir/EDELWEISS/wind'
workdir = '/cnrm/cen/users/NO_SAVE/vernaym/workdir/EDELWEISS/wind'

# * Source des fonctions LLT :
# https://github.com/louisletoumelin/bias_correction/blob/12e806af084d086d30e429b21deb8ab7f243a381/bias_correction/train/wind_utils.py


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


def upscale(filename):
    """
    Upscale 30m resolution wind fields from LTT to 250m Wind/Wind_dir FORCING-like variables
    """

    suffix = '_'.join(filename.split('.')[0].split('_')[-2:])
    if not (os.path.exists(os.path.join(workdir, f'Wind_gr250m_{suffix}.nc')) or
            os.path.exists(os.path.join(workdir, f'Wind_dir_gr250m_{suffix}.nc'))):
        # 1. Open monthly 30m wind Netcdf files produced by HM into 1 single xarray dataset and add projection
        wind30m = xr.open_dataset(filename)
        wind30m = wind30m.rio.write_crs(2154)  # 3857 ?
        wind30m = wind30m.chunk({'time': 100})

        # 2. Compute U and V wind components
        u30m, v30m = wind2comp(wind30m.speed, wind30m.direction, unit_direction="degree")

        # 3. Project 30m U and V wind components on the 250m grid

        # Open 250m grid reference file
        # Use Ange's MNT with elevation adapted to SARAN geometry
        gr250m = xr.open_dataset('/home/vernaym/These/DATA/MNTLouisGRoussecorrected.nc')
        gr250m = gr250m.rio.write_crs(2154)

        # Project 30m fields on 250m grid
        # lons, lats = source_area.get_lonlats()
        print(f'Projection U {suffix}')
        # rioxarray does not support dask : https://github.com/corteva/rioxarray/issues/119 --> memory limit
        u250m = u30m.rio.reproject_match(gr250m, resamplin="average")
        print(f'Projection V {suffix}')
        v250m = v30m.rio.reproject_match(gr250m, resamplin="average")

        # 4. Convert U/V into Wind/Wind_dir
        wind250m = comp2speed(u250m, v250m)
        wind250m.to_netcdf(os.path.join(workdir, f'Wind_gr250m_{suffix}.nc'), format=DEFAULT_NETCDF_FORMAT)
        wind250m.close()
        wdir250m = comp2dir(u250m, v250m)
        wdir250m.to_netcdf(os.path.join(workdir, f'Wind_dir_gr250m_{suffix}.nc'), format=DEFAULT_NETCDF_FORMAT)
        wdir250m.close()


if __name__ == '__main__':

    datebegin = '2021073106'
    dateend   = '2022080106'
    outname = os.path.join(workdir, f'Wind_gr250m_{datebegin}_{dateend}.nc')
    if os.path.exists(outname):
        os.remove(outname)

    # 1. Upscale 30m resolution wind fields from LTT to 250m Wind/Wind_dir FORCING-like variables
    for file in glob.glob(os.path.join(datadir, '*')):  # Loop over files to avoid memory limitations
        if not os.path.basename(file) == 'arome_downscaled_completion_2022_07.nc':
            upscale(file)

    # 2. Read the (multiple) files created and concatenate data into 1 single netcf file
    wind250m = xr.open_mfdataset(glob.glob(os.path.join(workdir, 'Wind_gr250m_????_??.nc')))
    wind250m = wind250m.rename({'__xarray_dataarray_variable__': 'Wind'})
    wdir250m = xr.open_mfdataset(glob.glob(os.path.join(workdir, 'Wind_dir_gr250m_????_??.nc')))
    wdir250m = wdir250m.rename({'__xarray_dataarray_variable__': 'Wind_dir'})
    wind250m = wind250m.merge(wdir250m)
    start = pd.to_datetime(datebegin, format='%Y%m%d%H').to_numpy()
    end = pd.to_datetime(dateend, format='%Y%m%d%H').to_numpy()
    dates = xr.date_range(start=start, end=end, freq='H')
    wind250m = wind250m.sel({'time': np.intersect1d(dates, wind250m.time)})
    wind250m.to_netcdf(outname, format=DEFAULT_NETCDF_FORMAT)

    wind_xpid = 'WHM01@vernaym',  # First experiment produced with Hugo.M wind
    geometry  = 'GrandesRousses250m'
    vortexIO.put_wind(datebegin, dateend, wind_xpid, geometry, filename=outname)
