import os
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import pytz

import vortex
from cen.data import flow
import footprints
from vortex import toolbox
from bronx.stdtypes.date import Date, Period

toolbox.active_now = True

# Define VORTEX cache
t = vortex.ticket()
t.env.setvar('WORKDIR', '/home/vernaym/workdir')


local_tz = pytz.timezone("Europe/Paris")

start = Date(2021, 8, 2, 7)
stop = Date(2022, 8, 1, 6)

# Read ensemble analysis
#filename = os.path.join('/home/vernaym/workdir/ASSIMILATION/RandomSampling/XP25', 'Random_Sampling_2021122806_2021123006_daily_GrandesRousses.nc')
filename = os.path.join('/home/vernaym/workdir/ASSIMILATION/RandomSampling/XP26', 'Random_Sampling_2021080206_2022080106_daily_GrandesRousses.nc')
analysis = xr.open_dataset(filename)
analysis = analysis.sel(member=range(1,17))

# Read ANTILOPE raw hourly precipitation
#filename = 'ANTILOPEH_2021103000_2022060200_alp.nc'
filename = 'ANTILOPEH_2021080106_2022080106_GrandesRousses.nc'
#filename = 'ANTILOPEH_2021073106_2022070106_GrandesRousses.nc'
antilope = xr.open_dataset(os.path.join('/home/vernaym/These/DATA', filename))
antilope = antilope.sel(lat=analysis.lat.data, lon=analysis.lon.data)

dailyfiles = False

if not dailyfiles:
    sel_time = np.arange(start, stop+Period(hours=1), dtype='datetime64[h]')

    antilope = antilope.sel(time=sel_time)
    antilope['time'] = antilope.time-np.timedelta64(7, 'h')
    # Compute daily ANTILOPE chronology
    tmp = antilope.resample(time='D').sum(dim='time')  # !!! VERY SLOW !!!
    tmp = tmp.transpose('lat','lon','time')  # reorder data
    tmp = tmp.reindex_like(antilope).ffill('time')  # Fill hourly time steps with daily precipitation (https://stackoverflow.com/questions/54452336/xarray-resample-time-series-data-from-daily-to-hourly)
    chronology = antilope.rr.data / (tmp.rr.data+0.00001)  # Avoid division by 0 Warnings
    chronology[tmp.rr.data==0] = 1/24.  # Avoid to remove precipitation when/where the analysis transformed null precipitation into >0 ones. TODO : Find a better solution

    analysis['time'] = analysis.time-np.timedelta64(6, 'h')-np.timedelta64(1, 'D')
    # Is it really necessary to duplicate daily precipitation 24 times ?
    daily_ana = analysis.resample(time='1D').pad()  # Remove hours from time coord
    hourly_ana = daily_ana.reindex_like(antilope).ffill('time')  # Fill hourly time steps with daily precipitation (https://stackoverflow.com/questions/54452336/xarray-resample-time-series-data-from-daily-to-hourly)
    hourly_ana = hourly_ana.transpose('member', 'lat', 'lon', 'time')

    for member in hourly_ana.member.data:
        array = hourly_ana.sel({'member':member}).rr.data * chronology
        #array = hourly_ana.sel({'member':member}).rr.data * chronology
        output = xr.Dataset(
            #name     = 'Precipitation',
            #data     = array,
            data_vars = dict(Precipitation=(["yy", "xx", "time"], array)),
            #data_vars = dict(Precipitation=(["yy", "xx", "time"], array, dict(coordinates="latitude longitude", grid_mapping="spatial_ref"))),
            #data_vars = dict(Precipitation=(["yy", "xx", "time"], array, dict(coordinates="latitude longitude"))),
            #dims      = ["yy", "xx", "time"],
            coords    = dict(longitude=('xx', analysis.lon.data), latitude=('yy', analysis.lat.data), time=sel_time),
            #attrs     = dict(coordinates="latitude longitude"),
            #attrs    = dict(description="Difference between each pixel cumul and the max of its neighbours"),
            )
        #output.rio.write_grid_mapping(inplace=True)
        #output = output.rio.write_crs("EPSG:4326", "grid_mapping", inplace=True)
        #output.rio.write_crs("EPSG:4326", inplace=True)
        output['Precipitation'].attrs = dict(coordinates="latitude longitude", grid_mapping="spatial_ref")
        outname = f'/home/vernaym/workdir/EDELWEISS/hourly_precipitation_analysis/precipitation_{start.ymd6h}_{stop.ymd6h}_mb{member:03d}.nc'
        output.to_netcdf(outname, mode='w')

    tbout = toolbox.output(
        role           = 'Precipitation analysis',
        kind           = 'Precipitation',
        vapp           = 'edelweiss',
        vconf          = '[geometry:area]',
        source_app     = 'antilope',
        source_conf    = 'RandomSampling',
        cutoff         = 'assimilation',
        filename       = f'/home/vernaym/workdir/EDELWEISS/hourly_precipitation_analysis/precipitation_[datebegin:ymd6h]_[dateend:ymd6h]_mb[member].nc',
        #filename       = f'precipitation_[datebegin]_[dateend]_mb[member:03d].nc',
        experiment     = 'XP25',
        geometry       = 'GrandesRousses1km',
        nativefmt      = 'netcdf',
        model          = 'edelweiss',
        #date           = dateend.ymd6h,
        namebuild      = 'flat@cen',
        date           = stop.ymd6h,
        datebegin      = start.ymd6h,
        dateend        = stop.ymd6h,
        namespace      = 'vortex.multi.fr',
        member         = footprints.util.rangex(1,16,1),
        block          = 'analysis',
        intent         = 'inout',
    ),
    print(tbout)

else:

    date = start
    while date <= stop:
        print(date)
        # Filter date
        #deb = np.datetime64('2021-12-27T07:00:00')
        #fin = np.datetime64('2021-12-30T06:00:00')
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
            vconf          = '[geometry:area]',
            source_app     = 'antilope',
            source_conf    = 'RandomSampling',
            cutoff         = 'assimilation',
            filename       = f'/home/vernaym/workdir/EDELWEISS/hourly_precipitation_analysis/precipitation_[datebegin:ymd6h]_[dateend:ymd6h]_mb[member].nc',
            #filename       = f'precipitation_[datebegin]_[dateend]_mb[member:03d].nc',
            experiment     = 'XP25',
            geometry       = 'GrandesRousses1km',
            nativefmt      = 'netcdf',
            model          = 'edelweiss',
            date           = dateend.ymd6h,
            datebegin      = datebegin.ymd6h,
            dateend        = dateend.ymd6h,
            namespace      = 'vortex.multi.fr',
            member         = footprints.util.rangex(1,16,1),
            block          = 'analysis',
            intent         = 'inout',
        ),
        print(tbout)

        date = date + Period(days=1)


