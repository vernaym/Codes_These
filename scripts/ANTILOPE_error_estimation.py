#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 5/03/2025

import os
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from pykrige.uk import UniversalKriging
from scipy.ndimage import uniform_filter

from These.scripts import make_mask

savedir = '/home/vernaym/workdir/ASSIMILATION/mask/alp'

# datebegin = '2021-07-31'
# dateend   = '2022-08-01'
# datebegin = '2021-12-01'
# dateend   = '2022-03-31'
datebegin = '2021-12-15'
dateend   = '2022-03-31'
latmax = 46.45
latmin = 44.1
lonmin = 5.4
lonmax = 7.2

gauge_tolerance = 0.1  # *100% of difference from ANTILOPE


def read_obs_auto_accumulation():
    """
    TODO : Add check for missing values
    """

    src = '/home/vernaym/These/NO_TRANSFER/DATA/obs_horaires_RR_20210731_20230423.csv'
    df = pd.read_csv(src, sep=';', parse_dates=['date'])
    # Filter out stations outside of the target domain
    df = df[(df.lon >= lonmin) & (df.lon <= lonmax) & (df.lat >= latmin) & (df.lat <= latmax)]
    df = df[(df['date'] >= '2021-07-31') & (df['date'] <= '2022-08-01')]  # Same period as AROME/ANTILOPE accumulations
    year = df.groupby('num_poste').agg({'rr': 'sum', 'lat': 'min', 'lon': 'min', 'alti': 'min', 'date': "count"})
    # Filter out stations with too many missing values
    year = year[year.date > year.date.max() * 0.95]
    df = df[df['num_poste'].isin(year.index.values)]

    winter = df[(df['date'] >= datebegin) & (df['date'] <= dateend)]
    winter = winter.groupby('num_poste').agg({'rr': 'sum', 'lat': 'min', 'lon': 'min', 'alti': 'min', 'date': "count"})

    out = winter.rename(columns={'rr': 'winter_cumul'}).drop(columns='date')
    out['summer_cumul'] = year['rr'] - out['winter_cumul']

    return out


def read_obs_nivometeo_accumulation():
    """
    TODO : Add check for missing values
    """

    src = '/home/vernaym/These/NO_TRANSFER/DATA/obs_nivometeo_daily_RR_20210801_20220801.csv'
    df = pd.read_csv(src, sep=';', parse_dates=['Q.dat'])
    df = df.rename(columns={'Q.dat': 'date', 'poste_nivo.alti': 'alti', 'poste_nivo.lat_dg': 'lat',
        'poste_nivo.lon_dg': 'lon', 'Q.rr': 'rr', 'Q.num_poste': 'num_poste'})
    # Filter out stations outside of the target domain
    df = df[(df.lon >= lonmin) & (df.lon <= lonmax) & (df.lat >= latmin) & (df.lat <= latmax)]
    tmp = df[(df['date'] >= datebegin) & (df['date'] <= dateend)]  # Same period as AROME accumulation
    # tmp = tmp[tmp['date'].dt.month.isin([11, 12, 1, 2, 3, 4])]  # Same period as AROME accumulation
    # tmp = tmp[tmp['date'].dt.month.isin([12, 1, 2, 3])]  # Same period as AROME accumulation
    out = tmp.groupby('num_poste').agg({'rr': 'sum', 'lat': 'min', 'lon': 'min', 'alti': 'min', 'date': "count"})
    # out = out[out.date > 8700]  # Filter out stations with too many missing values
    out = out[out.date > out.date.max() * 0.95]  # Filter out stations with too many missing values
    out = out.rename(columns={'rr': 'winter_cumul'}).drop(columns='date')

    return out


def read_AROME_accumulation():
    src = '/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_AROME_2021073106_2022080106_alp.nc'
    # src = '/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_AROME_2021073106_2022080106_nov-apr_alp.nc'
    # src = '/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_AROME_20211201_20220415_alp.nc'
    # src = '/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_AROME_20211201_20220331_alp.nc'
    ds_year = xr.open_dataset(src, engine='netcdf4')
    ds_year['lat'] = ds_year.lat.round(2)
    ds_year['lon'] = ds_year.lon.round(2)
    ds_year = xr.where(ds_year.cumul == 0., np.nan, ds_year)

    src = f'/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_AROME_{datebegin}_{dateend}_alp.nc'
    ds_winter = xr.open_dataset(src, engine='netcdf4')
    ds_winter['lat'] = ds_winter.lat.round(2)
    ds_winter['lon'] = ds_winter.lon.round(2)
    ds_winter = xr.where(ds_winter.cumul == 0., np.nan, ds_winter)

    out = ds_winter.rename({'cumul': 'winter_cumul'})
    out['summer_cumul'] = ds_year.cumul - ds_winter.cumul

    return out


def read_ANTILOPE_accumulation():
    src = '/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_ANTILOPE_alp_2021080106_2022080106.nc'
    ds_year = xr.open_dataset(src, engine='netcdf4')
    ds_year['lat'] = ds_year.lat.round(2)
    ds_year['lon'] = ds_year.lon.round(2)
    ds_year = xr.where(ds_year.cumul == 0., np.nan, ds_year)

    # src = '/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_ANTILOPE_2021073106_2022080106_nov-apr_alp.nc'
    # src = '/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_ANTILOPE_20211201_20220331_alp.nc'
    src = f'/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_ANTILOPE_{datebegin}_{dateend}_alp.nc'
    ds_winter = xr.open_dataset(src, engine='netcdf4')
    ds_winter['lat'] = ds_winter.lat.round(2)
    ds_winter['lon'] = ds_winter.lon.round(2)
    ds_winter = xr.where(ds_winter.cumul == 0., np.nan, ds_winter)

    # ds = ds.isel(lat=slice(None, None, -1))  # Flip upside down

    out = ds_winter.rename({'cumul': 'winter_cumul'})
    out['summer_cumul'] = ds_year.cumul - ds_winter.cumul

    return out


def compute_reference_field(df, da):
    """
    refs :
    https://github.com/GeoStat-Framework/PyKrige/issues/155
    https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/generated/pykrige.uk.UniversalKriging.html
    """

    UK = UniversalKriging(
        df.lon,
        df.lat,
        # df.cumul * 1.1,
        # df.cumul * 0.9,
        df.cumul,
        variogram_model  = 'spherical',
        drift_terms      = ['external_Z'],
        external_drift   = da,
        external_drift_x = da.lon,
        external_drift_y = da.lat,
    )
    kg, sd = UK.execute("grid", da.lon, da.lat)

    # im = plt.imshow(np.flipud(np.sqrt(sd) / kg))
    # plt.colorbar(im)
    # plt.show()

    kg = xr.DataArray(
        data = kg,
        dims = ["lat", "lon"],
        coords = dict(
            lon = (('lon'), da.lon.data),
            lat = (('lat'), da.lat.data),
        ),
    )

    err = xr.DataArray(
        data = np.sqrt(sd) / kg,
        dims = ["lat", "lon"],
        coords = dict(
            lon = (('lon'), da.lon.data),
            lat = (('lat'), da.lat.data),
        ),
    )

    return kg, err


def plot_ratio(field, filename, cmap=plt.cm.YlGnBu, origin='lower', vmin=None, vmax=None):

    fig, ax = plt.subplots(1, 1, figsize=(14, 14), subplot_kw=dict(projection=ccrs.PlateCarree()), layout='compressed')
    ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
    ax.set_title('')
    make_mask.plot_field(fig, ax, field, cmap=cmap, vmin=vmin, vmax=vmax, elevation=True, coords=False,
            colorbar=True)
    fig.savefig(os.path.join(savedir, filename), format='pdf')
    # fig.savefig(os.path.join(savedir, 'Estimated_ratio_from_kriging_may-sep.pdf'), format='pdf')


def plot_fields(arome, reference_field, antilope, gauges, season):
    fig, ax = plt.subplots(1, 3, figsize=(14, 6), sharex=True, sharey=True,
            subplot_kw=dict(projection=ccrs.PlateCarree()), layout='compressed')
    for axis in ax:
        # axis.set_extent([antilope.lon.min(), antilope.lon.max(), antilope.lat.min(), antilope.lat.max()],
        axis.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
    vmax = max(np.nanmax(reference_field), antilope.max(), arome.max())
    cmap = plt.cm.YlGnBu
    make_mask.plot_field(fig, ax[0], arome, cmap=cmap, vmin=0, vmax=vmax, elevation=True, coords=False,
            colorbar=False, categories=False)
    ax[0].set_title('AROME')
    make_mask.plot_field(fig, ax[1], reference_field, cmap=cmap, vmin=0, vmax=vmax, elevation=True, coords=False,
            colorbar=False, categories=False)
    gauges.plot.scatter('lon', 'lat', c='cumul', edgecolor='black', cmap=plt.cm.YlGnBu, vmin=0, vmax=vmax, ax=ax[1],
            colorbar=False)
    ax[1].set_title('Reference Field')
    im = make_mask.plot_field(fig, ax[2], antilope, cmap=cmap, vmin=0, vmax=vmax, elevation=True, coords=False,
            colorbar=False, categories=False)
    ax[2].set_title('ANTILOPE')
    plt.colorbar(im, ax=ax, label=f'Precipitation accumulation {datebegin}h - {dateend}h (kg/m²)')
    fig.savefig(os.path.join(savedir, f'AROME_reference_ANTILOPE_{datebegin}_{dateend}_{season}.pdf'), format='pdf')
    # plt.colorbar(im, ax=ax, label='Precipitation accumulation May - September (kg/m²)')
    # fig.savefig(os.path.join(savedir, 'AROME_reference_ANTILOPE_may-sep.pdf'), format='pdf')


if __name__ == '__main__':

    nivometeo = read_obs_nivometeo_accumulation()
    obs_auto = read_obs_auto_accumulation()
    arome = read_AROME_accumulation()
    antilope = read_ANTILOPE_accumulation()

    arome = arome.where((arome.lat == antilope.lat) & (arome.lon == antilope.lon))
    antilope = antilope.where((antilope.lat == arome.lat) & (antilope.lon == arome.lon))

    # Compare gauge accumulation to ANTILOPE to filter out gauges
    tmp = antilope.sel(lat=xr.DataArray(obs_auto.lat, dims='num_poste'),
            lon=xr.DataArray(obs_auto.lon, dims='num_poste'), method='nearest').to_dataframe()
    obs_auto['winter_cumul'] = obs_auto['winter_cumul'].where(
        (tmp.winter_cumul / obs_auto.winter_cumul < (1 + gauge_tolerance)) &
        (tmp.winter_cumul / obs_auto.winter_cumul > (1 - gauge_tolerance)),
    )
    obs_auto['summer_cumul'] = obs_auto['summer_cumul'].where(
        (tmp.summer_cumul / obs_auto.summer_cumul < (1 + gauge_tolerance)) &
        (tmp.summer_cumul / obs_auto.summer_cumul > (1 - gauge_tolerance)),
    )

    obs = pd.concat([obs_auto, nivometeo])

    for season in ['summer', 'winter']:

        obs_season = obs[[f'{season}_cumul', 'lat', 'lon']].rename(columns={f'{season}_cumul': 'cumul'}).dropna()
        arome_season = arome[f'{season}_cumul']
        antilope_season = antilope[f'{season}_cumul']

        reference_field, error = compute_reference_field(obs_season, arome_season)

        # plot reference field over the Grandes Rousses
        tmp = reference_field.sel(lon=slice(6.0, 6.5), lat=slice(45.0, 45.24))
        fig, ax = plt.subplots(figsize=(12, 6))
        tmp.plot(ax=ax, cmap=plt.cm.YlGnBu)
        fig.savefig(os.path.join(savedir, f'reference_field_{season}_GrandesRousses.pdf'))
        plt.close(fig)

        plot_fields(arome_season, reference_field, antilope_season, obs_season, season)

        smooth = uniform_filter(reference_field, 20)
        gradient = reference_field / smooth
        gradient.rename('gradient')
        gradient.to_netcdf(os.path.join(savedir, f'Estimated_{season}_gradient_from_kriging_{datebegin}_{dateend}.nc'))
        plot_ratio(gradient, f'Estimated_{season}_gradient_from_kriging_{datebegin}_{dateend}.pdf', origin='lower',
                cmap=plt.cm.RdBu_r, vmin=0.7, vmax=1.3)

        ratio = antilope_season / reference_field
        ratio = ratio.rename('ratio')
        ratio.to_netcdf(os.path.join(savedir, f'Estimated_{season}_ratio_from_kriging_{datebegin}_{dateend}.nc'))
        ratio = ratio.rename('ANTILOPE / reference ratio')
        plot_ratio(ratio, f'Estimated_{season}_ratio_from_kriging_{datebegin}_{dateend}.pdf', origin='lower',
                cmap=plt.cm.RdBu_r, vmin=0.3, vmax=1.7)
