#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 30/11/2023

import os
import xarray as xr

import vortex
from cen.data import flow
import footprints
from vortex import toolbox
from bronx.stdtypes.date import Date, Period

toolbox.active_now = True

# Define VORTEX cache if not already in the environment (local only)
#t = vortex.ticket()
#t.env.setvar('WORKDIR', '/home/vernaym/workdir')

datebegin = '2021080206'
dateend   = '2022080106'

home = '/home/vernaym'  #local/sxcen
#home = '/home/mrns/vernaym'  # soprano
#home = '/home/cnrm_other/cen/mrns/vernaym'  # HPC

datadir =  os.path.join(home, 'workdir/EDELWEISS')


# 1. Radiation variables come from SAFRAN reanalysis in the initial stage (similar for all ensemble members)
# TODO : à récupérer avec Vortex
#forcing = xr.open_dataset(os.path.join(datadir, 'SAFRAN_to_grid', 'meteo', 'SAFRAN_RADIATION_2021122706_2021123006_gr250ls.nc'))
#forcing = xr.open_dataset(os.path.join(datadir, 'SAFRAN_to_grid', 'meteo', 'FORCING_2021122706_2021123006_gr250ls.nc'))
forcing = xr.open_dataset(os.path.join(datadir, 'SAFRAN_to_grid', 'meteo', 'FORCING_2021080106_2022080106_gr250ls.nc'))

# 2. Precipitation come from MV's ensemble analysis, downscaled by SR with the method from VV
dirname = os.path.join(datadir, 'meteo')
filename = 'Precipitation_2021080206_2022080106.nc'
tbin = toolbox.input(
        role        = 'Precipitation analysis',
        kind        = 'Precipitation',
        vapp        = 'edelweiss',
        vconf       = '[geometry:area]',
        source_app  = 'antilope',
        source_conf = 'RandomSampling',
        cutoff      = 'assimilation',
        filename    = f'{dirname}/mb[member]/{filename}',
        experiment  = 'XP25@radanovicss',
        geometry    = 'GrandesRousses250m',
        nativefmt   = 'netcdf',
        namebuild   = 'flat@cen',
        model       = 'edelweiss',
        #date        = enddate.ymd6h,
        #datebegin   = startdate.ymd6h,
        #dateend     = enddate.ymd6h,
        date        = '2022080106',
        datebegin   = '2021080206',
        dateend     = '2022080106',
        namespace   = 'vortex.multi.fr',
        member      = footprints.util.rangex(1, 16, 1),
        block       = 'analysis',
        intent      = 'inout',
    )

# Concatenation of all FORCING variables into the final FORCING files
outname = 'FORCING_2021080206_2022080106_gr250ls.nc'
for member in range(1,17):
    fullname = os.path.join(datadir, 'meteo', f'mb{member:03d}', outname)
    if not os.path.exists(fullname):
        #precipitation = xr.open_dataset(os.path.join(datadir, 'hourly_precipitation_analysis', 'precipitation.antilope-randomsampling_2021122706_2021123006.nc'))
        precipitation = xr.open_dataset(os.path.join(dirname, f'mb{member:03d}', filename))
        precipitation=precipitation.rename({'xx':'x', 'yy':'y'})

        # Replace Rainf/Snowf variables by the ones from the ensemble analysis
        forcing['Rainf'] = precipitation['Rainf_ds']
        forcing['Snowf'] = precipitation['Snowf_ds']

        forcing.to_netcdf(fullname)

tbin = toolbox.input(
        role        = 'Forcing file',
        kind        = 'MeteorologicalForcing',
        vapp        = 'edelweiss',
        vconf       = '[geometry:area]',
        source_app  = 'antilope',
        source_conf = 'RandomSampling',
        cutoff      = 'assimilation',
        filename    = f'{datadir}/meteo/mb[member]/{outname}',
        experiment  = 'XP25@vernaym',
        geometry    = 'GrandesRousses250m',
        nativefmt   = 'netcdf',
        namebuild   = 'flat@cen',
        model       = 'edelweiss',
        #date        = enddate.ymd6h,
        #datebegin   = startdate.ymd6h,
        #dateend     = enddate.ymd6h,
        date        = '2022080106',
        datebegin   = '2021080206',
        dateend     = '2022080106',
        namespace   = 'vortex.multi.fr',
        member      = footprints.util.rangex(1, 16, 1),
        block       = 'analysis',
        intent      = 'inout',
    )

