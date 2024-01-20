#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 30/11/2023

import os
import shutil

import xarray as xr
import numpy as np

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

#home = '/home/vernaym'  # local
home = '/cnrm/cen/users/NO_SAVE/vernaym'  # sxcen
#home = '/home/mrns/vernaym'  # soprano
#home = '/home/cnrm_other/cen/mrns/vernaym'  # HPC

datadir =  os.path.join(home, 'workdir/EDELWEISS')
forcingname = 'FORCING_2021080206_2022080106_gr250m'


def update_precipitation(forcing):

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
            #intent      = 'inout',
        )

    # Concatenation of all FORCING variables into the final FORCING files
    outname = f'{forcingname}_out.nc'
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

def update_wind(windname):

    wind = xr.open_dataset(os.path.join(datadir, 'wind', windname))
    for member in range(1,17):
        print(f'Member {member}')
        filename = os.path.join(datadir, 'meteo', f'mb{member:03d}', f'{forcingname}_in.nc')
        outname = os.path.join(datadir, 'meteo', f'mb{member:03d}', f'{forcingname}_out.nc')
        forcing = xr.open_dataset(filename)

        dates = np.intersect1d(forcing.time, wind.time)
        forcing = forcing.sel({'time':dates})
        wind = wind.sel({'time':dates})

        forcing['Wind'].data = wind['Wind'].data
        forcing['Wind_DIR'].data = wind['Wind_dir'].data
        forcing.to_netcdf(outname, mode='w')
        forcing.close()

def get_forcings():

    tbin = toolbox.input(
            role        = 'Forcing file',
            kind        = 'MeteorologicalForcing',
            vapp        = 'edelweiss',
            vconf       = '[geometry:area]',
            source_app  = 'antilope',
            source_conf = 'RandomSampling',
            cutoff      = 'assimilation',
            filename    = f'{datadir}/meteo/mb[member]/{forcingname}_in.nc',
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
            #intent      = 'inout',
        )

def get_wind():

    windname = 'Wind_gr250m_2021070106_2022073123.nc'
    tbin = toolbox.input(
        role        = 'Wind',
        kind        = 'Wind',
        vapp        = 'edelweiss',
        vconf       = '[geometry:area]',
        source_app  = 'arome',
        source_conf = '4dvarfr',
        cutoff      = 'assimilation',
        filename    = f'{datadir}/wind/{windname}',
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
        #intent      = 'inout',
    )
    return windname

def save():

    tbout = toolbox.output(
            role        = 'Forcing file',
            kind        = 'MeteorologicalForcing',
            vapp        = 'edelweiss',
            vconf       = '[geometry:area]',
            source_app  = 'antilope',
            source_conf = 'RandomSampling',
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
            date        = '2022080106',
            datebegin   = '2021080206',
            dateend     = '2022080106',
            namespace   = 'vortex.multi.fr',
            member      = footprints.util.rangex(1, 16, 1),
            block       = 'analysis',
            #intent      = 'inout',
        )

def clean():
    for member in range(1,17):
        shutil.rmtree(os.path.join(datadir, 'meteo', f'mb{member:03d}'))


if __name__ == '__main__':

    #forcing = xr.open_dataset(os.path.join(datadir, 'SAFRAN_to_grid', 'meteo', 'FORCING_2021080106_2022080106_gr250m.nc'))
    #update_precipitation()  # Single SAFRAN FORCING file --> 16 FORCINGs
    #save()

    get_forcings()  # optionnal if previous steps uncommented

    windname = get_wind()

    update_wind(windname)  # Update 16 FORCING files

    save()

    clean()

