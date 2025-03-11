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

# datebegin = '2021-07-31 07',
# dateend   = '2022-08-01 06',
datebegin = '2021-11-01'
dateend   = '2022-04-30'
latmax = 46.45
latmin = 44.1
lonmin = 5.4
lonmax = 7.2


def read_obs_auto_accumulation():
    """
    TODO : Add check for missing values
    """

    src = '/home/vernaym/These/NO_TRANSFER/DATA/obs_horaires_RR_20210731_20230423.csv'
    df = pd.read_csv(src, sep=';', parse_dates=['date'])
    tmp = df[(df['date'] >= '2021-07-31 07') & (df['date'] <= '2022-08-01 06')]  # Same period as AROME accumulation
    tmp = tmp[tmp['date'].dt.month.isin([11, 12, 1, 2, 3, 4])]  # Same period as AROME accumulation
    out = tmp.groupby('num_poste').agg({'rr': 'sum', 'lat': 'min', 'lon': 'min', 'alti': 'min', 'date': "count"})
    # out = out[out.date > 8700]  # Filter out stations with too many missing values
    out = out[out.date > 3600]  # Filter out stations with too many missing values

    return out


def read_AROME_accumulation():
    # src = '/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_AROME_2021073106_2022080106_alp.nc'
    src = '/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_AROME_2021073106_2022080106_nov-apr_alp.nc'
    ds = xr.open_dataset(src, engine='netcdf4')
    ds['lat'] = ds.lat.round(2)
    ds['lon'] = ds.lon.round(2)

    return ds


def read_ANTILOPE_accumulation():
    # src = '/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_ANTILOPE_alp_2021080106_2022080106.nc'
    src = '/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_ANTILOPE_2021073106_2022080106_nov-apr_alp.nc'
    ds = xr.open_dataset(src, engine='netcdf4')
    ds['lat'] = ds.lat.round(2)
    ds['lon'] = ds.lon.round(2)
    ds = ds.isel(lat=slice(None, None, -1))  # Flip upside down

    return ds


def compute_reference_field(df, ds):
    """
    refs :
    https://github.com/GeoStat-Framework/PyKrige/issues/155
    https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/generated/pykrige.uk.UniversalKriging.html
    """
    UK = UniversalKriging(
        df.lon,
        df.lat,
        df.rr * 1.1,
        # df.rr,
        drift_terms      = ['external_Z'],
        external_drift   = ds,
        external_drift_x = ds.lon,
        external_drift_y = ds.lat,
    )
    kg, sd = UK.execute("grid", ds.lon, ds.lat)

    # im = plt.imshow(np.flipud(np.sqrt(sd) / kg))
    # plt.colorbar(im)
    # plt.show()

    kg = xr.DataArray(
        data = kg,
        dims = ["lat", "lon"],
        coords = dict(
            lon = (('lon'), ds.lon.data),
            lat = (('lat'), ds.lat.data),
        ),
    )

    err = xr.DataArray(
        data = np.sqrt(sd) / kg,
        dims = ["lat", "lon"],
        coords = dict(
            lon = (('lon'), ds.lon.data),
            lat = (('lat'), ds.lat.data),
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


def plot_fields(arome, reference_field, antilope, gauges):
    fig, ax = plt.subplots(1, 3, figsize=(14, 6), sharex=True, sharey=True,
            subplot_kw=dict(projection=ccrs.PlateCarree()), layout='compressed')
    for axis in ax:
        # axis.set_extent([antilope.lon.min(), antilope.lon.max(), antilope.lat.min(), antilope.lat.max()],
        axis.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
    vmax = max(np.nanmax(reference_field), antilope.max(), arome.max())
    cmap = plt.cm.YlGnBu
    make_mask.plot_field(fig, ax[0], arome, cmap=cmap, vmin=400, vmax=vmax, elevation=True, coords=False,
            colorbar=False, categories=False)
    ax[0].set_title('AROME')
    make_mask.plot_field(fig, ax[1], reference_field, cmap=cmap, vmin=400, vmax=vmax, elevation=True, coords=False,
            colorbar=False, categories=False)
    gauges.plot.scatter('lon', 'lat', c='rr', edgecolor='black', cmap=plt.cm.YlGnBu, vmin=400, vmax=vmax, ax=ax[1],
            colorbar=False)
    ax[1].set_title('Reference Field')
    im = make_mask.plot_field(fig, ax[2], antilope, cmap=cmap, vmin=400, vmax=vmax, elevation=True, coords=False,
            colorbar=False, categories=False)
    ax[2].set_title('ANTILOPE')
    plt.colorbar(im, ax=ax, label=f'Precipitation accumulation {datebegin}h - {dateend}h (kg/m²)')
    fig.savefig(os.path.join(savedir, f'AROME_reference_ANTILOPE_{datebegin}_{dateend}.pdf'), format='pdf')
    # plt.colorbar(im, ax=ax, label='Precipitation accumulation May - September (kg/m²)')
    # fig.savefig(os.path.join(savedir, 'AROME_reference_ANTILOPE_may-sep.pdf'), format='pdf')


if __name__ == '__main__':

    obs_auto = read_obs_auto_accumulation()
    arome = read_AROME_accumulation()
    antilope = read_ANTILOPE_accumulation()

    arome = arome.where((arome.lat == antilope.lat) & (arome.lon == antilope.lon))
    arome = xr.where(arome.cumul == 0., np.nan, arome.cumul)
    antilope = antilope.where((antilope.lat == arome.lat) & (antilope.lon == arome.lon))
    antilope = xr.where(antilope.cumul == 0., np.nan, antilope.cumul)

    reference_field, error = compute_reference_field(obs_auto, arome)

    plot_fields(arome, reference_field, antilope, obs_auto)

    smooth = uniform_filter(reference_field, 10)
    gradient = reference_field / smooth
    gradient.rename('gradient')
    gradient.to_netcdf(os.path.join(savedir, f'Estimated_gradient_from_kriging_{datebegin}_{dateend}.nc'))
    plot_ratio(gradient, f'Estimated_gradient_from_kriging_{datebegin}_{dateend}.pdf', origin='lower', cmap=plt.cm.RdBu_r, vmin=0.7, vmax=1.3)

    # ratio = np.where(error < 1, error, 1) * 1 + (1 - np.where(error < 1, error, 1)) * antilope.cumul / reference_field
    ratio = antilope / reference_field
    ratio = ratio.rename('ratio')
    ratio.to_netcdf(os.path.join(savedir, f'Estimated_ratio_from_kriging_{datebegin}_{dateend}.nc'))
    ratio = ratio.rename('ANTILOPE / reference ratio')
    plot_ratio(ratio, f'Estimated_ratio_from_kriging_{datebegin}_{dateend}.pdf', origin='lower', cmap=plt.cm.RdBu_r, vmin=0.3, vmax=1.7)
