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


datadir =  '/home/vernaym/workdir/EDELWEISS'

# 1. Radiation variables come from SAFRAN reanalysis in the initial stage (similar for all ensemble members)
# TODO : à récupérer avec Vortex
#radiation = xr.open_dataset(os.path.join(datadir, 'SAFRAN_to_grid', 'meteo', 'SAFRAN_RADIATION_2021122706_2021123006_gr250ls.nc'))
radiation = xr.open_dataset(os.path.join(datadir, 'SAFRAN_to_grid', 'meteo', 'FORCING_2021122706_2021123006_gr250ls.nc'))

# 2. Precipitation come from MV's ensemble analysis, downscaled by SR with the method from VV
# TODO : à récupérer avec Vortex
# TODO : Loop over ensemble members
#filename = os.path.join(datadir, 'hourly_precipitation_analysis', 'Precipittaion_2021122706_2021123006.nc')
#tbin = toolbox.output(
#        role        = 'Precipitation analysis',
#        kind        = 'Precipitation',
#        vapp        = 'edelweiss',
#        vconf       = '[geometry:area]',
#        source_app  = 'antilope',
#        source_conf ='RandomSampling',
#        cutoff      = 'assimilation',
#        filename    = filename,
#        experiment  = 'XP25@radanovicss',
#        geometry    = 'GrandesRousses250m',
#        nativefmt   = 'netcdf',
#        model       = 'edelweiss',
#        date        = enddate.ymd6h,
#        datebegin   = startdate.ymd6h,
#        dateend     = enddate.ymd6h,
#        namespace   = 'vortex.multi.fr',
#        member      = footprints.util.rangex(1, 16, 1),
#        block       = 'analysis',
#        intent      = 'inout',
#    )
precipitation = xr.open_dataset(os.path.join(datadir, 'hourly_precipitation_analysis', 'precipitation.antilope-randomsampling_2021122706_2021123006.nc'))
precipitation=precipitation.rename({'xx':'x', 'yy':'y'})

# Replace Rainf/Snowf variables by the ones from the ensemble analysis
radiation['Rainf'] = precipitation['Rainf_ds']
radiation['Snowf'] = precipitation['Snowf_ds']

radiation.to_netcdf(os.path.join(datadir, 'meteo', 'FORCING_2021122706_2021123006_gr250ls.nc'))

import pdb
pdb.set_trace()
