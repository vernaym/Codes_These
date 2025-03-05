#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 5/03/2025

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

from pykrige.uk import UniversalKriging

datebegin = '2021-07-31 06'
dateend   = '2022-08-01 06'


def read_obs_auto_accumulation():
    """
    TODO : Add check for missing values
    """

    src = '/home/vernaym/These/NO_TRANSFER/DATA/obs_horaires_RR_20210731_20230423.csv'
    df = pd.read_csv(src, sep=';')
    tmp = df[(df['date'] > datebegin) & (df['date'] <= dateend)]  # Same period as AROME accumulation
    out = tmp.groupby('num_poste').agg({'rr': 'sum', 'lat': 'min', 'lon': 'min', 'alti': 'min', 'date': "count"})
    out = out[out.date > 8700]  # Filter out stations with too many missing values

    return out


def read_AROME_accumulation():
    src = '/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_AROME_2021073106_2022080106_alp.nc'
    ds = xr.open_dataset(src, engine='netcdf4')
    ds['lat'] = ds.lat.round(2)
    ds['lon'] = ds.lon.round(2)

    return ds


def read_ANTILOPE_accumulation():
    src = '/home/vernaym/These/NO_TRANSFER/DATA/CUMUL_ANTILOPE_alp_2021080106_2022080106.nc'
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
        df.rr * 1.5,
        drift_terms      = ['external_Z'],
        external_drift   = ds.cumul,
        external_drift_x = ds.lon,
        external_drift_y = ds.lat,
    )
    out, ss = UK.execute("grid", ds.lon, ds.lat)

    return out


def plot_field(field, cmap=plt.cm.YlGnBu, origin='lower', label=None, vmin=None, vmax=None):

    im = plt.imshow(field, origin=origin, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=label)
    plt.show()


def plot_fields(arome, reference_field, antilope):
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    vmax = max(np.max(reference_field), antilope.cumul.max(), arome.cumul.max())
    im = ax[0].imshow(arome.cumul, origin='lower', cmap=plt.cm.YlGnBu, vmin=200, vmax=vmax)
    ax[0].set_title('AROME')
    im = ax[1].imshow(reference_field, cmap=plt.cm.YlGnBu, vmin=200, vmax=vmax)
    ax[1].set_title('Reference Field')
    im = ax[2].imshow(antilope.cumul, origin='lower', cmap=plt.cm.YlGnBu, vmin=200, vmax=vmax)
    ax[2].set_title('ANTILOPE')
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.28, 0.02, 0.43])
    fig.colorbar(im, cax=cbar_ax, label=f'Precipitation accumulation {datebegin} - {dateend} (kg/m²)')
    plt.show()


if __name__ == '__main__':

    obs_auto = read_obs_auto_accumulation()
    arome = read_AROME_accumulation()
    antilope = read_ANTILOPE_accumulation()

    arome = arome.where((arome.lat == antilope.lat) & (arome.lon == antilope.lon))
    antilope = antilope.where((antilope.lat == arome.lat) & (antilope.lon == arome.lon))

    reference_field = compute_reference_field(obs_auto, arome)
    # plot_field(reference_field, label='Precipitation accumulation between 2021080106 and 2022080106 (kg/m²)')

    plot_fields(arome, reference_field, antilope)

    ratio = antilope.cumul / reference_field
    # ratio = antilope.cumul / np.flipud(reference_field)
    plot_field(ratio, label='ANTILOPE / reference ratio', origin='lower', cmap=plt.cm.RdBu_r, vmin=0.4, vmax=1.6)
