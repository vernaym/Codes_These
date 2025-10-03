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
from These.scripts.tools import uniform_filter

from snowtools.utils import xarray_snowtools  # noqa
from snowtools.plots.maps import plot2D

# savedir = '/home/vernaym/workdir/ASSIMILATION/mask/alp'
datadir = '/cnrm/cen/users/NO_SAVE/vernaym/workdir/EDELWEISS/DATA'
savedir = '.'

# datebegin = '2021-07-31'
# dateend   = '2022-08-01'
# datebegin = '2021-12-01'
# dateend   = '2022-03-31'
datebegin = '2018-08-01'
dateend   = '2021-08-01'
plot_domain = dict(
    xmin = 5.4,
    xmax = 7.2,
    ymin = 44.1,
    ymax = 46.45,
)
latmax = 46.9
latmin = 43
lonmin = 4.5
lonmax = 8

gauge_tolerance = 0.2  # *100% of difference from ANTILOPE


def read_obs_auto_accumulation():
    """
    TODO : Add check for missing values
    """

    # src = '/home/vernaym/These/NO_TRANSFER/DATA/obs_horaires_RR_20210731_20230423.csv'
    src = os.path.join(datadir, f'obs_horaires_RR_{datebegin}_{dateend}.csv')
    df = pd.read_csv(src, sep=';', parse_dates=['date'])
    # Filter out rows where lat / lon coordinates can not be read as float
    # df = df[df['lon'].apply(isinstance, args=[float]) & df['lat'].apply(isinstance, args=[float])]
    # Filter out stations outside of the target domain
    df = df[(df.lon >= lonmin) & (df.lon <= lonmax) & (df.lat >= latmin) & (df.lat <= latmax)]
    df = df[(df['date'] >= datebegin) & (df['date'] <= dateend)]  # Same period as AROME/ANTILOPE accumulations
    year = df.groupby('num_poste').agg({'rr': 'sum', 'lat': 'min', 'lon': 'min', 'alti': 'min', 'date': "count"})
    # Filter out stations with too many missing values
    year = year[year.date > year.date.max() * 0.80]
    df = df[df['num_poste'].isin(year.index.values)]

    # winter = df[(df['date'] >= datebegin) & (df['date'] <= dateend)]
    winter = df[df['date'].dt.month.isin([12, 1, 2, 3])]
    winter = winter.groupby('num_poste').agg({'rr': 'sum', 'lat': 'min', 'lon': 'min', 'alti': 'min', 'date': "count"})

    out = winter.rename(columns={'rr': 'winter_cumul'}).drop(columns='date')
    out['summer_cumul'] = year['rr'] - out['winter_cumul']
    out['annual_cumul'] = year['rr']

    return out


def read_obs_nivometeo_accumulation():
    """
    TODO : Add check for missing values
    """

    # WARNING : No nivometeo observations in 2019/2020 and 2020/2021

    # src = '/home/vernaym/These/NO_TRANSFER/DATA/obs_nivometeo_daily_RR_20210801_20220801.csv'
    src = os.path.join(datadir, f'obs_nivometeo_daily_RR_{datebegin}_{dateend}.csv')
    df = pd.read_csv(src, sep=';', parse_dates=['date'])
    # df = df.rename(columns={'Q.dat': 'date', 'poste_nivo.alti': 'alti', 'poste_nivo.lat_dg': 'lat',
    #    'poste_nivo.lon_dg': 'lon', 'Q.rr': 'rr', 'Q.num_poste': 'num_poste'})
    # Filter out stations outside of the target domain
    df = df[(df.lon >= lonmin) & (df.lon <= lonmax) & (df.lat >= latmin) & (df.lat <= latmax)]
    tmp = df[(df['date'] >= datebegin) & (df['date'] <= dateend)]  # Same period as AROME accumulation
    # tmp = tmp[tmp['date'].dt.month.isin([11, 12, 1, 2, 3, 4])]  # Same period as AROME accumulation
    tmp = tmp[tmp['date'].dt.month.isin([12, 1, 2, 3])]  # Same period as AROME accumulation
    out = tmp.groupby('num_poste').agg({'rr': 'sum', 'lat': 'min', 'lon': 'min', 'alti': 'min', 'date': "count"})
    # out = out[out.date > 8700]  # Filter out stations with too many missing values
    out = out[out.date > out.date.max() * 0.75]  # Filter out stations with too many missing values
    out = out.rename(columns={'rr': 'winter_cumul'}).drop(columns='date')

    return out


def read_AROME_accumulation():
    # src = '/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_AROME_2021073106_2022080106_alp.nc'
    # src = '/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_AROME_2021073106_2022080106_nov-apr_alp.nc'
    # src = '/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_AROME_20211201_20220415_alp.nc'
    # src = '/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_AROME_20211201_20220331_alp.nc'
    src = os.path.join(datadir, 'CUMUL_AROME_alp_2018073106_2021080106.nc')
    # src = os.path.join(datadir, 'CUMUL_AROME_SUMMER_alp_2018_2021.nc')
    ds_year = xr.open_dataset(src, engine='snowtools')
    ds_year['yy'] = ds_year.yy.round(2)
    ds_year['xx'] = ds_year.xx.round(2)
    ds_year = xr.where(ds_year.cumul == 0., np.nan, ds_year)

    # src = f'/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_AROME_{datebegin}_{dateend}_alp.nc'
    src = os.path.join(datadir, 'CUMUL_AROME_WINTER_alp_2018_2021.nc')
    ds_winter = xr.open_dataset(src, engine='snowtools')
    ds_winter['yy'] = ds_winter.yy.round(2)
    ds_winter['xx'] = ds_winter.xx.round(2)
    ds_winter = xr.where(ds_winter.cumul == 0., np.nan, ds_winter)

    out = ds_winter.rename({'cumul': 'winter_cumul'})
    out['summer_cumul'] = ds_year.cumul - ds_winter.cumul
    out['annual_cumul'] = ds_year.cumul

    return out


def read_ANTILOPE_accumulation():
    # src = '/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_ANTILOPE_alp_2021080106_2022080106.nc'
    src = os.path.join(datadir, 'CUMUL_ANTILOPE_alp_2018080106_2021080106.nc')
    ds_year = xr.open_dataset(src, engine='snowtools')
    ds_year['yy'] = ds_year.yy.round(2)
    ds_year['xx'] = ds_year.xx.round(2)
    ds_year = xr.where(ds_year.cumul == 0., np.nan, ds_year)

    # src = '/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_ANTILOPE_2021073106_2022080106_nov-apr_alp.nc'
    # src = '/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_ANTILOPE_20211201_20220331_alp.nc'
    # src = f'/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_ANTILOPE_{datebegin}_{dateend}_alp.nc'
    src = os.path.join(datadir, 'CUMUL_ANTILOPE_WINTER_alp_2018_2021.nc')
    ds_winter = xr.open_dataset(src, engine='snowtools')
    ds_winter['yy'] = ds_winter.yy.round(2)
    ds_winter['xx'] = ds_winter.xx.round(2)
    ds_winter = xr.where(ds_winter.cumul == 0., np.nan, ds_winter)

    # ds = ds.isel(lat=slice(None, None, -1))  # Flip upside down
    out = ds_winter.rename({'cumul': 'winter_cumul'})
    out['summer_cumul'] = ds_year.cumul - ds_winter.cumul
    out['annual_cumul'] = ds_year.cumul

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
        # external_drift   = np.flipud(da.data),
        # external_drift_y = np.flipud(da.yy.data),
        external_drift   = da,
        external_drift_x = da.xx,
        external_drift_y = da.yy,
    )
    kg, sd = UK.execute("grid", da.xx, da.yy)

    # im = plt.imshow(np.flipud(np.sqrt(sd) / kg))
    # plt.colorbar(im)
    # plt.show()

    kg = xr.DataArray(
        data = kg,
        dims = ["yy", "xx"],
        coords = dict(
            xx = (('xx'), da.xx.data),
            yy = (('yy'), da.yy.data),
        ),
    )

    err = xr.DataArray(
        data = np.sqrt(sd) / kg,
        dims = ["yy", "xx"],
        coords = dict(
            xx = (('xx'), da.xx.data),
            yy = (('yy'), da.yy.data),
        ),
    )

    return kg, err


def plot_ratio(field, filename, cmap=plt.cm.YlGnBu, origin='lower', vmin=None, vmax=None):

    fig, ax = plt.subplots(1, 1, figsize=(14, 14), subplot_kw=dict(projection=ccrs.PlateCarree()), layout='compressed')
    ax.set_extent(plot_domain.values(), crs=ccrs.PlateCarree())
    ax.set_title('')
    plot2D.plot_field(field, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=True, **plot_domain)
    fig.savefig(os.path.join(savedir, filename), format='pdf')
    # fig.savefig(os.path.join(savedir, 'Estimated_ratio_from_kriging_may-sep.pdf'), format='pdf')


def plot_fields(arome, reference_field, antilope, gauges, season):
    fig, ax = plt.subplots(1, 3, figsize=(14, 6), sharex=True, sharey=True,
            subplot_kw=dict(projection=ccrs.PlateCarree()), layout='compressed')
    for axis in ax:
        # axis.set_extent([antilope.lon.min(), antilope.lon.max(), antilope.lat.min(), antilope.lat.max()],
        axis.set_extent(plot_domain.values(), crs=ccrs.PlateCarree())
    vmax = max(np.nanmax(reference_field), antilope.max(), arome.max())
    cmap = plt.cm.YlGnBu
    plot2D.plot_field(arome, ax=ax[0], cmap=cmap, vmin=0, vmax=vmax, add_colorbar=False, **plot_domain)
    gauges.plot.scatter('lon', 'lat', c='cumul', edgecolor='black', cmap=plt.cm.YlGnBu, vmin=0, vmax=vmax, ax=ax[0],
            colorbar=False)
    ax[0].set_title('AROME')
    plot2D.plot_field(reference_field, ax=ax[1], cmap=cmap, vmin=0, vmax=vmax, add_colorbar=False, **plot_domain)
    # gauges.plot.scatter('lon', 'lat', c='cumul', edgecolor='black', cmap=plt.cm.YlGnBu, vmin=0, vmax=vmax, ax=ax[1],
    #         colorbar=False)
    ax[1].set_title('Reference Field')
    im = plot2D.plot_field(antilope, ax=ax[2], cmap=cmap, vmin=0, vmax=vmax, add_colorbar=False, **plot_domain)
    ax[2].set_title('ANTILOPE')

    # Add colorbar
    plt.colorbar(im, ax=ax, label=f'Precipitation accumulation {datebegin}h - {dateend}h (kg/m²)')
    fig.savefig(os.path.join(savedir, f'AROME_reference_ANTILOPE_{datebegin}_{dateend}_{season}.pdf'), format='pdf')
    # plt.colorbar(im, ax=ax, label='Precipitation accumulation May - September (kg/m²)')
    # fig.savefig(os.path.join(savedir, 'AROME_reference_ANTILOPE_may-sep.pdf'), format='pdf')


if __name__ == '__main__':

    nivometeo = read_obs_nivometeo_accumulation()
    obs_auto = read_obs_auto_accumulation()
    arome = read_AROME_accumulation()
    antilope = read_ANTILOPE_accumulation()

    # arome = arome.where((arome.yy == antilope.yy) & (arome.xx == antilope.xx))
    # antilope = antilope.where((antilope.yy == arome.yy) & (antilope.xx == arome.xx))
    arome = arome.interp(yy=antilope.yy, xx=antilope.xx, method='linear')

    # Compare gauge accumulation to ANTILOPE to filter out gauges
    tmp = antilope.sel(yy=xr.DataArray(obs_auto.lat, dims='num_poste'),
            xx=xr.DataArray(obs_auto.lon, dims='num_poste'), method='nearest').to_dataframe()
    obs_auto['annual_cumul'] = obs_auto['annual_cumul'].where(
        (tmp.annual_cumul / obs_auto.annual_cumul < (1 + gauge_tolerance)) &
        (tmp.annual_cumul / obs_auto.annual_cumul > (1 - gauge_tolerance)),
    )
    obs_auto['winter_cumul'] = obs_auto['winter_cumul'].where(
        (tmp.winter_cumul / obs_auto.winter_cumul < (1 + gauge_tolerance)) &
        (tmp.winter_cumul / obs_auto.winter_cumul > (1 - gauge_tolerance)),
    )
    obs_auto['summer_cumul'] = obs_auto['summer_cumul'].where(
        (tmp.summer_cumul / obs_auto.summer_cumul < (1 + gauge_tolerance)) &
        (tmp.summer_cumul / obs_auto.summer_cumul > (1 - gauge_tolerance)),
    )

    # obs = pd.concat([obs_auto, nivometeo])
    obs = obs_auto

    # for season in ['annual']:
    for season in ['summer', 'winter', 'annual']:

        obs_season = obs[[f'{season}_cumul', 'lat', 'lon']].rename(columns={f'{season}_cumul': 'cumul'}).dropna()
        arome_season = arome[f'{season}_cumul']
        antilope_season = antilope[f'{season}_cumul']

        reference_field, error = compute_reference_field(obs_season, arome_season)

        # plot reference field over the Grandes Rousses
        # tmp = reference_field.sel(lon=slice(6.0, 6.5), lat=slice(45.0, 45.24))
        # fig, ax = plt.subplots(figsize=(12, 6))
        # tmp.plot(ax=ax, cmap=plt.cm.YlGnBu)
        # fig.savefig(os.path.join(savedir, f'reference_field_{season}_GrandesRousses.pdf'))
        # plt.close(fig)

        plot_fields(arome_season, reference_field, antilope_season, obs_season, season)

        smooth = uniform_filter(reference_field, 20)
        gradient = reference_field / smooth
        gradient.rename('Precipitation ratio between the reference and its local mean')
        gradient.to_netcdf(os.path.join(savedir, f'Estimated_{season}_gradient_from_kriging_{datebegin}_{dateend}.nc'))
        plot_ratio(gradient, f'Estimated_{season}_gradient_from_kriging_{datebegin}_{dateend}.pdf', origin='lower',
                cmap=plt.cm.PuOr, vmin=0.7, vmax=1.3)

        ratio = antilope_season / reference_field
        ratio = ratio.rename('ratio')
        ratio.to_netcdf(os.path.join(savedir, f'Estimated_{season}_ratio_from_kriging_{datebegin}_{dateend}.nc'))
        ratio = ratio.rename('ANTILOPE / reference ratio')
        plot_ratio(ratio, f'Estimated_{season}_ratio_from_kriging_{datebegin}_{dateend}.pdf', origin='lower',
                cmap=plt.cm.RdBu_r, vmin=0.3, vmax=1.7)
