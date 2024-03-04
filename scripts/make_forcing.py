#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 30/11/2023

import os
import shutil

import xarray as xr
import numpy as np
import pandas as pd

import vortex
from cen.data import flow
import footprints
from vortex import toolbox
from bronx.stdtypes.date import Date, Period

toolbox.active_now = True

DEFAULT_NETCDF_FORMAT = 'NETCDF3_CLASSIC'

# Define VORTEX cache if not already in the environment (local only)
#t = vortex.ticket()
#t.env.setvar('WORKDIR', '/home/vernaym/workdir')

datebegin = '2021080206'
dateend   = '2022080106'

#home = '/home/vernaym'  # local
home = '/cnrm/cen/users/NO_SAVE/vernaym'  # sxcen
#home = '/home/mrns/vernaym'  # soprano
#home = '/home/cnrm_other/cen/mrns/vernaym'  # HPC

datadir =  os.path.join(home, 'workdir/EDELWEISS')
forcingname = f'FORCING_{datebegin}_{dateend}_gr250m'
precipitation_xpid = 'RS27@radanovicss'


def update(forcing):

    # 1 Open wind produced by HM with LLT method
    windname = get_wind()
    wind = xr.open_dataset(os.path.join(datadir, 'wind', windname), chunks='auto')

    dates = np.intersect1d(forcing.time, wind.time)
    datedeb = pd.to_datetime(str(dates[0]))
    datefin = pd.to_datetime(str(dates[-1]))
    forcing = forcing.sel({'time':dates})
    wind = wind.sel({'time':dates})

    print(f'Update wind')
    forcing['Wind'].data = wind['Wind'].data
    forcing['Wind_DIR'].data = wind['Wind_dir'].data

    wind.close()

    forcing.time.encoding['dtype'] = 'int32'

    # 2. Precipitation come from MV's ensemble analysis, downscaled by SR with the method from VV
    dirname = os.path.join(datadir, 'meteo')
    deb = '2021080206'
    end = '2022080106'
    #datedeb = pd.to_datetime(deb, format='%Y%m%d%H')
    #datefin = pd.to_datetime(end, format='%Y%m%d%H')
    filename = f'Precipitation_{deb}_{end}.nc'
    tbin = toolbox.input(
            role        = 'Precipitation analysis',
            kind        = 'Precipitation',
            vapp        = 'edelweiss',
            #vconf       = '[geometry:area]',
            vconf       = '[geometry:tag]',
            source_app  = 'antilope',
            source_conf = 'RandomSampling',
            cutoff      = 'assimilation',
            filename    = f'{dirname}/mb[member]/{filename}',
            experiment  = precipitation_xpid,
            geometry    = 'GrandesRousses250m',
            nativefmt   = 'netcdf',
            namebuild   = 'flat@cen',
            model       = 'edelweiss',
            #date        = enddate.ymd6h,
            #datebegin   = startdate.ymd6h,
            #dateend     = enddate.ymd6h,
            date        = end,
            datebegin   = deb,
            dateend     = end,
            namespace   = 'vortex.multi.fr',
            member      = footprints.util.rangex(1, 16, 1),
            block       = 'analysis',
            #intent      = 'inout',
        )

    # Concatenation of all FORCING variables into the final FORCING files
    outname = f'{forcingname}_out.nc'
    for member in range(1,17):
        print(f'Update precipitation : member {member}')
        fullname = os.path.join(datadir, 'meteo', f'mb{member:03d}', outname)
        if not os.path.exists(fullname):
            #precipitation = xr.open_dataset(os.path.join(datadir, 'hourly_precipitation_analysis', 'precipitation.antilope-randomsampling_2021122706_2021123006.nc'))
            precipitation = xr.open_dataset(os.path.join(dirname, f'mb{member:03d}', filename), drop_variables=['Precipitation', 'snowfrac_ds', 'z_snowlim_ds'], chunks='auto')

            # Output file on common dates only
            dates = np.intersect1d(forcing.time, precipitation.time)
            datedeb = pd.to_datetime(str(dates[0]))
            datefin = pd.to_datetime(str(dates[-1]))
            forcing = forcing.sel({'time':dates})
            # Set time variable attributes
            forcing.time.encoding['units'] = f'hours since {forcing.time.data[0]}'

            precipitation = precipitation.sel({'time':dates})
            precipitation = precipitation.rename({'xx':'x', 'yy':'y'})

            # Replace Rainf/Snowf variables by the ones from the ensemble analysis
            forcing['Rainf'] = precipitation['Rainf_ds'] / 3600.
            forcing['Snowf'] = precipitation['Snowf_ds'] / 3600.

            forcing.to_netcdf(fullname, unlimited_dims={'time': True}, format=DEFAULT_NETCDF_FORMAT)
            #forcing.to_netcdf(fullname, unlimited_dims={'time': True}, format=DEFAULT_NETCDF_FORMAT, encoding={"time":{"dtype": "int32"}})

    return datedeb, datefin


def get_forcings():

    tbin = toolbox.input(
            role        = 'Forcing file',
            kind        = 'MeteorologicalForcing',
            vapp        = 'edelweiss',
            vconf       = '[geometry:tag]',
            #source_app  = 'antilope',
            #source_conf = 'RandomSampling',
            cutoff      = 'assimilation',
            filename    = f'{datadir}/meteo/mb[member]/{forcingname}_in.nc',
            experiment  = xpid,
            #experiment  = 'XP25@vernaym',
            geometry    = 'GrandesRousses250m',
            nativefmt   = 'netcdf',
            namebuild   = 'flat@cen',
            model       = 'edelweiss',
            #date        = enddate.ymd6h,
            #datebegin   = startdate.ymd6h,
            #dateend     = enddate.ymd6h,
            date        = dateend,
            datebegin   = datebegin,
            dateend     = dateend,
            namespace   = 'vortex.multi.fr',
            member      = footprints.util.rangex(1, 16, 1),
            block       = 'precipitation',
            #intent      = 'inout',
        )

def get_wind():

    start = '2021073106'
    stop = '2022080106'
    windname = f'Wind_gr250m_{start}_{stop}.nc'
    tbin = toolbox.input(
        role        = 'Wind',
        kind        = 'Wind',
        vapp        = 'edelweiss',
        vconf       = '[geometry:tag]',
        source_app  = 'arome',
        source_conf = '4dvarfr',
        cutoff      = 'assimilation',
        filename    = f'{datadir}/wind/{windname}',
        experiment  = 'WHM01@vernaym',  # First experiment produced with Hugo.M wind
        geometry    = 'GrandesRousses250m',
        nativefmt   = 'netcdf',
        namebuild   = 'flat@cen',
        model       = 'devine',
        date        = stop,
        datebegin   = start,
        dateend     = stop,
        namespace   = 'vortex.multi.fr',
        block       = 'analysis',
        #intent      = 'inout',
    )
    return windname

def save(begin, end, block='meteo'):

    tbout = toolbox.output(
            role        = 'Forcing file',
            kind        = 'MeteorologicalForcing',
            vapp        = 'edelweiss',
            vconf       = '[geometry:tag]',
            #source_app  = 'antilope',
            #source_conf = 'RandomSampling',
            cutoff      = 'assimilation',
            filename    = f'{datadir}/meteo/mb[member]/{forcingname}_out.nc',
            experiment  = 'XP25@vernaym',
            geometry    = 'GrandesRousses250m',
            nativefmt   = 'netcdf',
            namebuild   = 'flat@cen',
            model       = 'edelweiss',
            #date        = enddate.ymd6h,
            #datebegin   = startdate.ymd6h,
            #dateend     = enddate.ymd6h,
            date        = end.strftime('%Y%m%d%H'),  # end is a pandas 'Timestamp' object
            datebegin   = begin.strftime('%Y%m%d%H'),  # begin is a pandas 'Timestamp' object
            dateend     = end.strftime('%Y%m%d%H'),  # end is a pandas 'Timestamp' object
            namespace   = 'vortex.multi.fr',
            member      = footprints.util.rangex(1, 16, 1),
            block       = block,
            #intent      = 'inout',
        )

def clean():
    for member in range(1,17):
        shutil.rmtree(os.path.join(datadir, 'meteo', f'mb{member:03d}'))


if __name__ == '__main__':

    #forcing = xr.open_dataset(os.path.join(datadir, 'SAFRAN_to_grid', 'meteo', 'FORCING_2021080106_2022080106_gr250m.nc'), drop_variables=['ZS', 'aspect', 'slope', 'massif_number'], chunks='auto')
    forcing = xr.open_dataset(os.path.join(datadir, 'SAFRAN_to_grid', 'meteo', 'FORCING_2021080106_2022080106_gr250m.nc'), drop_variables=['massif_number'], chunks='auto')

    datedeb, datefin = update(forcing)  # Single SAFRAN FORCING file --> 16 FORCINGs

    save(datedeb, datefin)

    clean()

