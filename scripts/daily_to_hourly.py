import os, sys
import pandas as pd
import numpy as np
import xarray as xr
#import rioxarray
import pytz

#from snowtools.scripts.extract.vortex import vortexIO as io
from snowtools.scripts.extract.vortex import vortex_get as io

import vortex
from cen.data import flow
import footprints
from vortex import toolbox
from bronx.stdtypes.date import Date, Period

toolbox.active_now = True

if len(sys.argv) == 5:
    datebegin = Date(sys.argv[1])
    dateend   = Date(sys.argv[2])
    xpid      = sys.argv[3]
    geometry  = sys.argv[4]
elif len(sys.argv) == 4:
    datebegin = Date(sys.argv[1])
    dateend   = Date(sys.argv[2])
    xpid      = sys.argv[3]
    geometry  = 'GrandesRousses1km'
else:
    print('ERROR : missing arguments')
    print('USAGE : daily_to_hourly.py datebegin dateend xpid')
    sys.exit()

local_tz = pytz.timezone("Europe/Paris")

if xpid.startswith('RS'):
    block = 'RandomSampling'
    source_conf = 'RandomSampling'
elif xpid.startswith('EnKF'):
    block = 'EnsembleKalmanFilter'
    source_conf = 'EnsembleKalmanFilter'
elif xpid.startswith('PF'):
    block = 'ParticleFilter'
    source_conf = 'ParticleFilter'
else:
    source_conf = None

workdir = f'{os.environ["HOME"]}/workdir/EDELWEISS/precipitation_analysis/{block}/{xpid}'
if not os.path.exists(workdir):
    os.makedirs(workdir)
os.chdir(workdir)

xp_number = xpid[-2:]

# Read ensemble analysis
# All members are stored in the same netcdf file !
io.get(
    kind           = 'Precipitation',
    geometry       = geometry,
    xpid           = f'{xpid}@vernaym',
    vapp           = 'edelweiss',
    block          = 'daily',
    datebegin      = datebegin.ymd6h,
    dateend        = dateend.ymd6h,
    filename       = 'ANALYSIS.nc',
)
analysis = xr.open_dataset('ANALYSIS.nc')

# Read ANTILOPE raw hourly precipitation
io.get(
    kind           = 'Precipitation',
    geometry       = geometry if geometry == 'GrandesRousses1km' else 'Alp1km',
    #xpid           = 'RawData@vernaym',
    xpid           = 'raw@vernaym',
    vapp           = 'antilope',
    block          = 'hourly',
    datebegin      = datebegin.ymd6h,
    dateend        = dateend.ymd6h,
    filename       = 'ANTILOPE.nc',
)
antilope = xr.open_dataset('ANTILOPE.nc')  # sxcen
antilope = antilope.assign_coords({'lat': np.round(antilope.lat.data, 2), 'lon': np.round(antilope.lon.data, 2)})
antilope = antilope.sel(lat=analysis.lat.data, lon=analysis.lon.data)

dailyfiles = False

if not dailyfiles:
    sel_time = np.arange(datebegin-Period(hours=24), dateend + Period(hours=1), dtype='datetime64[h]')

    antilope = antilope.sel(time=sel_time)
    antilope = antilope.fillna(0)  # Fill missing values with 0s. TODO : interpolate bettewen previous and next time step values ?
    antilope['time'] = antilope.time-np.timedelta64(7, 'h')
    # Compute daily ANTILOPE chronology
    print('Resampling hourly ANTILOPE data in progress...')
    tmp = antilope.resample(time='D').sum(dim='time')  # !!! VERY SLOW !!!
    print('Resampling hourly ANTILOPE data over !')
    tmp = tmp.transpose('lat', 'lon', 'time')  # reorder data
    tmp = tmp.reindex_like(antilope).ffill('time')  # Fill hourly time steps with daily precipitation (https://stackoverflow.com/questions/54452336/xarray-resample-time-series-data-from-daily-to-hourly)
    chronology = antilope.rr / (tmp.rr + 0.00001)  # Avoid division by 0 Warnings
    chronology.where(tmp.rr==0, 1/24.)  # Avoid to remove precipitation when/where the analysis transformed null precipitation into >0 ones.
    chronology = chronology.transpose('time', 'lat', 'lon')

    analysis['time'] = analysis.time-np.timedelta64(6, 'h')-np.timedelta64(1, 'D')
    # Is it really necessary to duplicate daily precipitation 24 times ?
    daily_ana = analysis.resample(time='1D').pad()  # Remove hours from time coord
    hourly_ana = daily_ana.reindex_like(antilope).ffill('time')  # Fill hourly time steps with daily precipitation (https://stackoverflow.com/questions/54452336/xarray-resample-time-series-data-from-daily-to-hourly)
    hourly_ana = hourly_ana.transpose('member', 'time', 'lat', 'lon')


    for member in hourly_ana.member.data:
        array = hourly_ana.sel({'member':member}).rr.data * chronology.data
        #array = hourly_ana.sel({'member':member}).rr.data * chronology
        output = xr.Dataset(
            data_vars = dict(Precipitation=(["time", "latitude", "longitude"], array)),
            coords    = dict(longitude=('longitude', analysis.lon.data), latitude=('latitude', analysis.lat.data), time=sel_time),
            )
        #output.rio.write_grid_mapping(inplace=True)
        #output = output.rio.write_crs("EPSG:4326", "grid_mapping", inplace=True)
        #output.rio.write_crs("EPSG:4326", inplace=True)
        output['Precipitation'].attrs = dict(coordinates="latitude longitude", grid_mapping="spatial_ref")
        outdir = f'mb{member:03d}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outname = f'{outdir}/hourly_precipitation_{datebegin.ymd6h}_{dateend.ymd6h}.nc'
        output.to_netcdf(outname, mode='w')

    # Use put_meteo because this is not a FORCING-ready resource
    # TODO : do not archive on Hendrix !
    io.put(
        kind           = 'Precipitation',
        geometry       = geometry,
        xpid           = f'{xpid}@vernaym',
        member         = footprints.util.rangex(0, len(hourly_ana.member) - 1),
        vapp           = 'edelweiss',
        datebegin      = datebegin.ymd6h,
        dateend        = dateend.ymd6h,
        block          = 'hourly',
        filename       = 'hourly_precipitation_[datebegin:ymd6h]_[dateend:ymd6h].nc'
    )

    for member in hourly_ana.member.data:
        os.remove(os.path.join(f'mb{member:03d}', f'hourly_precipitation_{datebegin.ymd6h}_{dateend.ymd6h}.nc'))

else:

    date = start
    while date <= stop:
        print(date)
        datebegin = date.replace(hour=6)
        dateend = date + Period(days=1)
        deb = np.datetime64(date)  # D (7h)
        fin = np.datetime64(date+Period(hours=23))  # D+1 (6h)
        sel_time = np.arange(deb, fin+np.timedelta64(1, 'h'), dtype='datetime64[h]')
        tmp = antilope.sel(time=sel_time)
        tmp = tmp.transpose('time', 'lat','lon')  # reorder data
        tmp['time'] = tmp.time-np.timedelta64(7, 'h')

        # Compute daily ANTILOPE chronology
        tmpd = tmp.resample(time='D').sum(dim='time')  # !!! VERY SLOW !!!
        tmph = tmpd.reindex_like(tmp).ffill('time')  # Fill hourly time steps with daily precipitation (https://stackoverflow.com/questions/54452336/xarray-resample-time-series-data-from-daily-to-hourly)

        chronology = tmp.rr.data / tmph.rr.data

        # Select analysis date and convert to hourly data
        ana = analysis.sel(time=fin, drop=False).transpose('member', 'lat', 'lon')
        #analysis['time'] = analysis.time-np.timedelta64(6, 'h')-np.timedelta64(1, 'D')
        #daily_ana = ana.resample(time='1D').pad()  # Remove hours from time coord
        #hourly_ana = daily_ana.reindex_like(tmp).ffill('time')  # Fill hourly time steps with daily precipitation (https://stackoverflow.com/questions/54452336/xarray-resample-time-series-data-from-daily-to-hourly)
        #hourly_ana = hourly_ana.transpose('member', 'lat', 'lon', 'time')

        for member in ana.member.data:
            array = ana.sel({'member':member}).rr.data * chronology
            #array = hourly_ana.sel({'member':member}).rr.data * chronology
            output = xr.Dataset(
                #name     = 'Precipitation',
                #data     = array,
                data_vars = dict(Precipitation=(["time", "yy", "xx"], array, dict(coordinates="latitude longitude"))),
                #dims      = ["yy", "xx", "time"],
                coords    = dict(longitude=('xx', analysis.lon.data), latitude=('yy', analysis.lat.data), time=sel_time),
                #attrs     = dict(coordinates="latitude longitude"),
                #attrs    = dict(description="Difference between each pixel cumul and the max of its neighbours"),
                )
            output = output.rio.write_crs("EPSG:4326", "grid_mapping", inplace=True)
            outname = f'/home/vernaym/workdir/EDELWEISS/hourly_precipitation_analysis/precipitation_{datebegin.ymd6h}_{dateend.ymd6h}_mb{member:03d}.nc'
            output.to_netcdf(outname)

        tbout = toolbox.output(
            role           = 'Precipitation analysis',
            kind           = 'Precipitation',
            vapp           = 'edelweiss',
            vconf          = '[geometry:tag]',
            source_app     = 'antilope',
            source_conf    = 'RandomSampling',
            cutoff         = 'assimilation',
            filename       = f'/home/vernaym/workdir/EDELWEISS/hourly_precipitation_analysis/precipitation_[datebegin:ymd6h]_[dateend:ymd6h]_mb[member].nc',
            #filename       = f'precipitation_[datebegin]_[dateend]_mb[member].nc',
            experiment     = xpid,
            geometry       = geometry,
            nativefmt      = 'netcdf',
            model          = 'edelweiss',
            date           = dateend.ymd6h,
            datebegin      = datebegin.ymd6h,
            dateend        = dateend.ymd6h,
            namespace      = 'vortex.multi.fr',
            member         = footprints.util.rangex(1, 16, 1),
            block          = 'analysis',
            intent         = 'inout',
        ),
        print(tbout)

        date = date + Period(days=1)


