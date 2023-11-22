import os
import pandas as pd
import numpy as np
import xarray as xr
import pytz

#https://stackoverflow.com/questions/15799162/resampling-within-a-pandas-multiindex

local_tz = pytz.timezone("Europe/Paris")

csv = True

if csv:
    #df=pd.read_csv("/home/vernaym/These/DATA/obs_horaires_clim_RR.data", sep=';', parse_dates=['dat'])
    #df=pd.read_csv("/home/vernaym/These/DATA/obs_horaires_auto_RR_20211101_20220430.csv", sep=';', parse_dates=['date'])
    df=pd.read_csv("/home/vernaym/These/DATA/obs_horaires_RR_auto_pyr_20211110_20220430.csv", sep=';', parse_dates=['date'])

    #newdates=df['dat']-pd.Timedelta(7, 'H')  # Shift time to sum rr over 6h J --> 6h J+1 period
    #df = df.assign(date=newdates).drop(columns=['dat'])  # Replace date column
    newdates=df['date']-pd.Timedelta(7, 'H')  # Shift time to sum rr over 6h J --> 6h J+1 period
    #df = df.assign(date=newdates).drop(columns=['date'])  # Replace date column
    df = df.assign(date=newdates)  # Replace date column
    #df = df.set_index(['num_poste', 'lat', 'lon', 'alti', 'poste', 'reseau_poste', 'date']).sort_index()
    df = df.set_index(['num_poste', 'lat', 'lon', 'alti', 'nom', 'reseau_poste', 'date']).sort_index()
    level_values = df.index.get_level_values
    df = df.groupby([level_values(i) for i in range(len(df.index.names)-1)]+[pd.Grouper(freq='1D', level=-1)]).sum()
    df = df.reset_index()
    newdates = df['date']+pd.Timedelta(1, 'D') + pd.Timedelta(6, 'H')  # Shift dates : the cumul is between 6h YYYYMM{D-1}06 and YYYYMMDD06
    df = df.assign(date=newdates)  # Replace date column
    #df = df.set_index(['num_poste', 'lat', 'lon', 'alti', 'poste', 'reseau_poste', 'date']).sort_index()
    df = df.set_index(['num_poste', 'lat', 'lon', 'alti', 'nom', 'reseau_poste', 'date']).sort_index()

    #df.to_csv("/home/vernaym/These/DATA/obs_quotidienne_clim_RR.data", sep=';')
    df.to_csv("/home/vernaym/These/DATA/obs_quotidienne_auto_RR_pyr_20211101_20220430.csv", sep=';')

else:

    #files=['ANTILOPEH_2021073106_2021102923_alp.nc', 'ANTILOPEH_2021103000_2022060200_alp.nc', 'ANTILOPEH_2022060201_2022080106_alp.nc', 'ANTILOPEH_2022080107_2022123123_alp.nc', 'ANTILOPEH_2023010100_2023042306_alp.nc']
    #antilope = xr.open_mfdataset([os.path.join('/home/vernaym/These/DATA', f) for f in files])

    filename = "ANTILOPEH_2021080106_2022080106_GrandesRousses.nc"
    antilope = xr.open_dataset(filename)

    # Convert hourly precipitation into 24h precipitation covering the same period as nivometeo observations
    # Problems :
    # 1. nivometeo observations are made at 7am CET (6am UTC in winter, 5am UTC in summer)
    # 2. the xarray tools to do that allows only accumulations between  0h and 24h.
    #
    # solutions :
    # 1. Xarray does not manage local time series index (see code bellow) --> convert only in winter time
    #    It could be bossible to separate the dataset into winter-time and summer-time periods and assemble it back...
#    ds = antilope.copy()
#    cet_time = pd.to_datetime(antilope['time']).tz_localize('UTC').tz_convert('Europe/Paris')  # convert time index from UTC to CET
#    tmptime = cet_time - np.timedelta64(7, 'h')  # Shift 7 am CET (nivometeo observation time) at 0 am for xarray accumulation
#    local_series = tmptime.to_series()  # Convert DatetimeIndex into pandas TimeSerie
#    ds['time'] = local_series  # PROBLEM : dates are converted back in UTC by xarray...
#    ds=ds.drop("dim_0").rename({"dim_0": "time"})
    # 2. shift time serie by 6h, compute 24h accumulations and shift back !
    #
    antilope['time'] = antilope.time-np.timedelta64(7, 'h')  # Winter time
    #antilope['time'] = antilope.time-np.timedelta64(6, 'h')  # Spring time
    antilope = antilope.resample(time='D').sum(dim='time')  # !!! VERY SLOW !!!
    antilope['time'] = antilope.time+np.timedelta64(30, 'h')

    antilope = antilope.sel(time=antilope.time>np.datetime64('2021-08-02'))  # Le 1er jour est incomplet si l'extraction a débuté à 6h : elle correspond au cumul 1h entre 5h et 6h du J1...

    antilope.to_netcdf("ANTILOPED_2021080106_2022080106_GrandesRousses.nc")

    #return antilope

