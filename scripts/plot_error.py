#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os, sys
from datetime import datetime,timedelta
import pandas as pd  # Version 0.25.3
import numpy as np
import xarray as xr
import scipy
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import shapefile
from metpy.interpolate import cross_section

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

#plt.rcParams["figure.figsize"] = [7.50, 3.50]
#plt.rcParams["axes.grid"] = False
plt.rcParams["figure.autolayout"] = True


##############################################################################################
##############################################################################################

datadir = '/home/vernaym/These/DATA'
savedir = '/home/vernaym/These/figures'

def read_nivometeo_obs():

    nivometeo = pd.read_csv(os.path.join(datadir, 'obs_nivometeo_daily_RR_20211201_20220430.csv'), sep=';', parse_dates=['date'], header=0,
            names=['date', 'num_poste', 'nom', 'alti', 'lat', 'lon', 'massif', 'obs', 'unused'],
            usecols=['date', 'num_poste', 'nom', 'alti', 'lat', 'lon', 'obs'],
            dtype={'num_poste':int, 'nom':str, 'alti':int, 'lat':float, 'lon':float, 'obs':float},
        )

#    latmax, latmin, lonmin, lonmax = np.array(coords[domain]).astype(float)/1000.
#    nivometeo = nivometeo.loc[(nivometeo['lat']>=latmin) & (nivometeo['lat']<=latmax) & (nivometeo['lon']>=lonmin) & (nivometeo['lon']<=lonmax)]  # Select area
    nivometeo.date = nivometeo.date + pd.Timedelta("1d6h")   #BDClim extraction for date ymd is the observation from ymd6h to ym(d+1)6h
    nivometeo.set_index(['num_poste','date'], inplace=True)

    return nivometeo.to_xarray()

antilope  = xr.open_dataset(os.path.join('/home/vernaym/These/DATA', 'ANTILOPEQ_2021120106_2022050106_alp_corrected.nc'))
nivometeo = read_nivometeo_obs()

fig, ax = plt.subplots()

for num_poste in np.intersect1d(antilope.num_poste.data, nivometeo.num_poste.data)[:1]:
    print(num_poste)
    mod = antilope.sel({'num_poste':num_poste})
    mod = mod.rename({'time':'date'})
    obs = nivometeo.sel({'num_poste':num_poste})
    obs = obs.dropna(dim='date')
    #obs = obs.where(~np.isnan(obs.obs.data))
    dates = obs.date.data
    mod = mod.sel({'date':dates})
    #real_error = np.sqrt((mod.rr.data-obs.obs.data)**2)
    #real_error = mod.rr.data-obs.obs.data
    real_error = np.abs(mod.rr.data-obs.obs.data)
    model_error = mod.error.data
    model_error = model_error[~np.isnan(real_error)]
    real_error = real_error[~np.isnan(real_error)]
    real_error_mean = np.mean(real_error)
    real_error_std = np.mean(np.sqrt((real_error-real_error_mean)**2))
    model_error_mean = np.mean(model_error)
    model_error_std = np.mean(np.sqrt((model_error-model_error_mean)**2))
    #ax.errorbar(np.mean(model_error), np.mean(real_error), yerr=np.max(real_error), xerr=np.max(model_error), color='k')
    #ax.errorbar(np.mean(model_error), np.mean(real_error), yerr=real_error_std, xerr=model_error_std, color='k')
    ax.plot(real_error, model_error, linestyle='', marker='+', markersize=2)

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

# Plot bissectrice and adjuste axes limits
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

import pdb
pdb.set_trace()

