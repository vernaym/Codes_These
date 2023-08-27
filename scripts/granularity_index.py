#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 31/03/2023

import os, sys
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import pandas as pd
import xarray as xr

import matplotlib as mpl
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import palettable

from scipy.stats import norm, gamma
from scipy import signal
import random
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from scipy.sparse import csc_matrix, csr_matrix, dia_matrix, diags
from scipy.ndimage import uniform_filter
from sklearn.linear_model import LinearRegression

from These.radar import Preprocessing_ANTILOPE
import make_mask
import scores

if len(sys.argv) > 1:
    domain = sys.argv[1]
else:
    domain = 'MontBlanc'

domain_coords = dict(
        GrandesRousses = dict(latmax=45.4, latmin=44.9, lonmin=5.8, lonmax = 6.6),
        NorthernAlps   = dict(lonmin=6.0, lonmax=6.9, latmin=45.6, latmax=46.35),
        CentralAlps    = dict(lonmin=5.6, lonmax=7.0, latmin=45.0, latmax=45.6),
        SouthernAlps   = dict(lonmin=5.7, lonmax=7.0, latmin=44.2, latmax=45.0),
        HauteSavoie    = dict(lonmin=6.45, lonmax=6.95, latmin=45.67, latmax=46.35),
        MontBlanc      = dict(lonmin=6.45, lonmax=7.1, latmin=45.65, latmax=46.1),
        Savoie         = dict(lonmin=6.0, lonmax=7.2, latmin=45.1, latmax=45.9),
        Isere          = dict(lonmin=5.54, lonmax=6.19, latmin=44.89, latmax=45.16),
        Brianconnais   = dict(lonmin=6.48, lonmax=6.95, latmin=44.67, latmax=44.95),
        HautesAlpes    = dict(lonmin=6.1, lonmax=7.1, latmin=44.4, latmax=45.2),
        AlpesSud       = dict(lonmin=6.56, lonmax=6.92, latmin=44.18, latmax=44.49),
        alp            = dict(latmax=46.450, latmin=44.100, lonmin=5.400, lonmax=7.200),
)

figsize = dict(
        alp            = (14,16),
        GrandesRousses = (15,7),
        HauteSavoie    = (12,12),
        HautesAlpes    = (16,9),
        MontBlanc      = (16,10),
        Savoie         = (16,8),
        Isere          = (16,8),
)

savedir = '/home/vernaym/These/figures/illustration/methode_correction'

extract_lat = np.round(np.arange(domain_coords[domain]['latmin'], domain_coords[domain]['latmax'], 0.01, dtype=float), 2)
extract_lon = np.round(np.arange(domain_coords[domain]['lonmin'], domain_coords[domain]['lonmax'], 0.01, dtype=float), 2)
# Coordonnées du Mont Blanc :
lat = 45.83
lon = 6.87

def plot(field, name, cmap=plt.cm.YlGnBu, vmin=None, vmax=None, scores=None):
    if vmin is None:
        vmin = np.min(field)
    if vmax is None:
        vmax = np.max(field)
    fig, ax = plt.subplots(figsize=figsize[domain])
    make_mask.plot_field(fig, ax, field, cmap=cmap, vmin=vmin, vmax=vmax, scores=scores)
    circle = plt.Circle((lon, lat), 0.14, color='red', fill=False, linewidth=3)
    ax.plot(lon, lat, color='red', marker='+', markersize=10)
    ax.add_artist(circle)
    #plt.Circle((lon, lat), 0.15, color='k', fill=False, linewidth=2)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, f'{name}.pdf'), format='pdf')
    plt.close(fig)

def granularity_index(field):
    """
    dispersion of the field around its moving average
    """
    smooth = uniform_filter(field, size=15)
    diff = field - smooth
    diff = diff.flatten()
    mean = np.mean(diff)
    rmse = np.sqrt(np.mean(diff**2))
    return rmse


if __name__ == "__main__":

    # Extract and plot initial field
    filename = f'CUMUL_ANTILOPE_{domain}.nc'
    if os.path.exists(os.path.join(savedir, filename)):
        field = xr.open_dataarray(os.path.join(savedir, filename))
    else:
        tmp = '/home/vernaym/These/DATA/CUMUL_ANTILOPEH_alp_2021103000_2022060200.nc'
        field = xr.open_dataarray(tmp)
        field = field.sel({'lat':np.intersect1d(extract_lat, field.lat), 'lon':np.intersect1d(extract_lon, field.lon)})
        field.to_netcdf(os.path.join(savedir, filename))
    vmax = np.nanmax(field.data)
    field = field.rename('Precipitation (mm)')

    smooth = field - uniform_filter(field.data, size=15)
    vmax = np.max(smooth.data)
    plot(smooth, 'field_minus_smooth', cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    raw_index = granularity_index(field.data)

    # Extract correlation window
    coords=[(lon,lat) for lat in field.lat.data for lon in field.lon.data]
    codist = Preprocessing_ANTILOPE.codistances(coords)

    # Extract observation uncertainty / confidence
    fic_error = os.path.join('/home/vernaym/workdir/ASSIMILATION/mask/alp', f'Observation_uncertainty_0.15_alp.nc')
    error = xr.open_dataarray(fic_error)
    error = error.sel({'lat':np.intersect1d(extract_lat, error.lat), 'lon':np.intersect1d(extract_lon, error.lon)})
    confidence = 1 / error
    pond = codist.dot(diags(1/error.data.flatten(), 0))

    # Dynaic correction alone
    new, mean, sd = Preprocessing_ANTILOPE.dynamic_correction(field.data, pond)
    dyn = make_mask.to_xarray(new.reshape((len(field.lat), len(field.lon))), field).rename('Precipitation (mm)')
    dyn_index = granularity_index(dyn.data)


    # Debiasing
    fic_ratio = os.path.join('/home/vernaym/workdir/ASSIMILATION/mask/alp', f'Estimated_ratio_alp_0.15.nc')
    ratio = xr.open_dataarray(fic_ratio)
    ratio = ratio.sel({'lat':np.intersect1d(extract_lat, ratio.lat), 'lon':np.intersect1d(extract_lon, ratio.lon)})
    deb = field / ratio.data

    deb_index = granularity_index(deb.data)

    # Debiasing + field correction
    pond = codist.dot(diags(1/error.data.flatten(), 0))
    new, mean, sd = Preprocessing_ANTILOPE.dynamic_correction(deb.data, pond, plot=True)
    correctedfield = make_mask.to_xarray(new.reshape((len(field.lat), len(field.lon))), field).rename('Precipitation (mm)')

    corr_index = granularity_index(correctedfield.data)

    # Add relief mean vertical gradient (Not implemented in experiments)
#    fic_gradient = os.path.join('/home/vernaym/workdir/ASSIMILATION/mask/alp', f'model_gradient.nc')
#    gradient = xr.open_dataarray(fic_gradient)
#    gradient = gradient.sel({'lat':np.intersect1d(extract_lat, gradient.lat), 'lon':np.intersect1d(extract_lon, gradient.lon)})
#    final_field = correctedfield * gradient.data
#    final_index = granularity_index(final_field.data)

    # smoothing
    smooth = field.copy()
    smooth.data = uniform_filter(deb, size=15)
    plot(smooth, 'smoothed_field', vmax=vmax)

    smooth_index = granularity_index(smooth.data)

    print('raw_index=', raw_index)
    print('deb_index=', deb_index)
    print('dyn_index=', dyn_index)
    print('corr_index=', corr_index)
#    print('final_index=', final_index)
    print('smooth_index=', smooth_index)










