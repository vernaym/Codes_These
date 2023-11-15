import os
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import pytz

import vortex
import toolbox
from bronx.stdtypes.date import Datet, Period

local_tz = pytz.timezone("Europe/Paris")

# Read ensemble analysis
filename = os.path.join('/home/vernaym/workdir/ASSIMILATION/RandomSampling/XP25', 'Random_Sampling_2021122806_2021123006_daily_GrandesRousses.nc')
analysis = xr.open_dataset(filename)
analysis = analysis.sel(member=range(1,17))
analysis['time'] = analysis.time-np.timedelta64(6, 'h')-np.timedelta64(1, 'D')
daily_ana = analysis.resample(time='1D').pad()  # Remove hours from time coord
#hourly_ana = daily_ana.resample(time='1H').pad()  # copy daily precipitation into 24 values (https://stackoverflow.com/questions/54452336/xarray-resample-time-series-data-from-daily-to-hourly)

# Read ANTILOPE raw hourly precipitation
filename = 'ANTILOPEH_2021103000_2022060200_alp.nc'
antilope = xr.open_dataset(os.path.join('/home/vernaym/These/DATA', filename))
antilope = antilope.sel(lat=analysis.lat.data, lon=analysis.lon.data)

date = datebegin
while date <= dateend:
    # Filter date
    #deb = np.datetime64('2021-12-27T07:00:00')
    #fin = np.datetime64('2021-12-30T06:00:00')
    deb = np.datetime64(date)
    fin = np.datetime64(date+Period(days=1))
    sel_time = np.arange(deb, fin+np.timedelta64(1, 'h'), dtype='datetime64[h]')
    antilope = antilope.sel(time=sel_time)
    antilope['time'] = antilope.time-np.timedelta64(7, 'h')

    # Compute daily ANTILOPE chronology
    tmp = antilope.copy()  # TODO : find a way to avoid a copy
    tmp = tmp.resample(time='D').sum(dim='time')  # !!! VERY SLOW !!!
    tmp = tmp.transpose('lat','lon','time')  # reorder data
    tmp = tmp.reindex_like(antilope).ffill('time')  # Fill hourly time steps with daily precipitation (https://stackoverflow.com/questions/54452336/xarray-resample-time-series-data-from-daily-to-hourly)
    chronology = antilope.rr.data / tmp.rr.data

    hourly_ana = daily_ana.reindex_like(antilope).ffill('time')  # Fill hourly time steps with daily precipitation (https://stackoverflow.com/questions/54452336/xarray-resample-time-series-data-from-daily-to-hourly)
    hourly_ana = hourly_ana.transpose('member', 'lat', 'lon', 'time')


    for member in hourly_ana.member.data:
        array = hourly_ana.sel({'member':member}).rr.data * chronology
        output = xr.Dataset(
            #name     = 'Precipitation',
            #data     = array,
            data_vars = dict(Precipitation=(["yy", "xx", "time"], array, dict(coordinates="latitude longitude"))),
            #dims      = ["yy", "xx", "time"],
            coords    = dict(longitude=('xx', analysis.lon.data), latitude=('yy', analysis.lat.data), time=sel_time),
            #attrs     = dict(coordinates="latitude longitude"),
            #attrs    = dict(description="Difference between each pixel cumul and the max of its neighbours"),
            )
        output = output.rio.write_crs("EPSG:4326", inplace=True)
        outname = f'/home/vernaym/workdir/hourly_precipitation_analysis/precipitation_{pd.to_datetime(deb).strftime("%Y%m%d%H")}_{pd.to_datetime(fin).strftime("%Y%m%d%H")}_mb{member:03d}.nc'
        output.to_netcdf(outname)

    self.sh.title('Toolbox output')
    tb27 = toolbox.output(
        role           = 'Precipitation analysis',
        kind           = 'PrecipitationForcing',
        source_app     = 'ANTILOPE',
        source_conf    = 'RandomSampling',
        cutoff         = 'assimilation',
        local          = f'/home/vernaym/workdir/hourly_precipitation_analysis/precipitation_[datebegin:ymdh]_[dateend:ymdh]_mb[member:03d].nc',
        experiment     = 'XP25',
        geometry        = 'GrandesRousses',
        nativefmt      = 'netcdf',
        model          = 'EDELWEISS',
        datebegin      = datebegin.ymd6h,
        dateend        = dateend.ymd6h,
        namespace      = 'vortex.multi.fr',
        member         = footprints.util.rangex(1,16,1),
    ),
    print(t.prompt, 'tb27 =', tb27)
    print()






