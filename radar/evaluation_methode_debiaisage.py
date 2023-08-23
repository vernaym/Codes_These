#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os, sys
import glob
import time
from datetime import datetime,timedelta
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import rankdata
import CRPS.CRPS as pscore

from These.scripts import scores

import argparse

import matplotlib as mpl
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

datadir = '/home/vernaym/These/DATA'
workdir = '/home/vernaym/workdir/ASSIMILATION/mask/evaluation'

def read_nivometeo_obs():
#        nivometeo = pd.read_csv(os.path.join(datadir, 'obs_nivometeo_daily_RR_20211201_20220430.csv'), sep=';', parse_dates=['Q.dat'],
#            dtype={'Q.num_poste':int, 'poste_nivo.nom_usuel':str, 'poste_nivo.alti':int, 'poste_nivo.lat_dg':float, 'poste_nivo.lon_dg':float, 'poste_nivo.massif_nivo':int, 'Q.rr':float},)
#            rename={'Q.date':'date', 'Q.num_poste':'num_poste', 'poste_nivo.nom_usuel':'nom', 'poste_nivo.alti':'alti', 'poste_nivo.lat_dg':'lat', 'poste_nivo.lon_dg':'lon', 'Q.rr':'obs'})

        #Q.dat;Q.num_poste;poste_nivo.nom_usuel;poste_nivo.alti;poste_nivo.lat_dg;poste_nivo.lon_dg;poste_nivo.massif_nivo;Q.rr;hist_reseau_poste.reseau_poste

    nivometeo = pd.read_csv(os.path.join(datadir, 'obs_nivometeo_daily_RR_20211201_20220430.csv'), sep=';', parse_dates=['date'], header=0,
            names=['date', 'num_poste', 'nom', 'alti', 'lat', 'lon', 'massif', 'obs', 'unused'],
            usecols=['date', 'num_poste', 'nom', 'alti', 'lat', 'lon', 'obs'],
            dtype={'num_poste':int, 'nom':str, 'alti':int, 'lat':float, 'lon':float, 'obs':float},
        )

    coords = ['46450', '44100', '5400', '7200']  # Alp domain
    latmax, latmin, lonmin, lonmax = np.array(coords).astype(float)/1000.
    nivometeo = nivometeo.loc[(nivometeo['lat']>=latmin) & (nivometeo['lat']<=latmax) & (nivometeo['lon']>=lonmin) & (nivometeo['lon']<=lonmax)]  # Select area
    #nivometeo = nivometeo[nivometeo['num_poste'].isin(indep)]  # Select evaluation stations
    nivometeo.date = nivometeo.date + pd.Timedelta("1d6h")   #BDClim extraction for date ymd is the observation from ymd6h to ym(d+1)6h
    #nivometeo.groupby('num_poste')['nom', 'lat', 'lon', 'alti'].agg(set)
    #nivometeo = nivometeo.set_index(['num_poste', 'lat', 'lon', 'nom', 'alti', 'date'])  # Utilité de passer en index ?
    nivometeo.set_index(['num_poste','date'], inplace=True)

    return nivometeo.to_xarray()

def read_obs_clim():

    obs = pd.read_csv(os.path.join(datadir, "obs_quotidienne_clim_RR.data"), sep=';', parse_dates=['date'], header=0,
            names = ['num_poste', 'lat', 'lon', 'alti', 'nom', 'reseau_poste', 'date', 'obs'],
            usecols=['num_poste', 'lat', 'lon', 'alti', 'nom', 'date', 'obs'],
            dtype={'num_poste':int, 'nom':str, 'alti':int, 'lat':float, 'lon':float, 'obs':float},
            )
    latmax, latmin, lonmin, lonmax = np.array(coords[domain]).astype(float)/1000.
    obs = obs.loc[(obs['lat']>=latmin) & (obs['lat']<=latmax) & (obs['lon']>=lonmin) & (obs['lon']<=lonmax)]  # Select area
    obs.set_index(['num_poste','date'], inplace=True)

    return obs.to_xarray()

def read_lpn():
    lpn = pd.read_csv(os.path.join(datadir, 'LPN_nivometeo_20210801_20220801.csv'), sep=';', parse_dates=['H_NIVO.DAT'], dtype={'H_NIVO.NUM_POSTE':int, 'H_NIVO.ALTI_LPNX':int})
    lpn.rename(columns={'H_NIVO.ALTI_LPNX':'LPNX', 'H_NIVO.NUM_POSTE':'num_poste', 'H_NIVO.DAT':'date'}, inplace=True)

    return lpn

def read_antilope():
    #filename = 'ANTILOPEQ_2021073106_2022070106_GrandesRousses.nc'
    filename = 'ANTILOPEH_2021103000_2022060200_alp.nc'
    antilope = xr.open_dataset(os.path.join(datadir, filename))
    if filename.startswith('ANTILOPEH'):
        # Convert hourly precipitation into 24h precipitation between 6h J-1 and 6h J
        # Problem : the xarray tools to do that allows only accumulations between
        # 0h and 23h.
        # solution : shift time serie by 7h, compute 24h accumulations and
        # shift back !
        antilope['time'] = antilope.time-np.timedelta64(7, 'h')
        antilope = antilope.resample(time='1D').sum(dim='time')  # !!! VERY SLOW !!! WARNING : does not work with pandas>=2.0.0
        antilope['time'] = antilope.time+np.timedelta64(30, 'h')

    return antilope

def read_corrected_antilope():
    filename = 'ANTILOPEQ_2021120106_2022050106_alp_corrected.nc'
    antilope = xr.open_dataset(os.path.join(datadir, filename))
    if filename.startswith('ANTILOPEH'):
        # Convert hourly precipitation into 24h precipitation between 6h J-1 and 6h J
        # Problem : the xarray tools to do that allows only accumulations between
        # 0h and 23h.
        # solution : shift time serie by 7h, compute 24h accumulations and
        # shift back !
        antilope['time'] = antilope.time-np.timedelta64(7, 'h')
        antilope = antilope.resample(time='1D').sum(dim='time')  # !!! VERY SLOW !!! WARNING : does not work with pandas>=2.0.0
        antilope['time'] = antilope.time+np.timedelta64(30, 'h')

    return antilope

if __name__ == "__main__":

    # Read reference observations
    nivometeo = read_nivometeo_obs()
    nivometeo = nivometeo.transpose()
    lats = nivometeo.lat.groupby('num_poste').mean('date').data
    lons = nivometeo.lon.groupby('num_poste').mean('date').data

    # Read raw ANTILOPE data
    antilope = read_antilope()
    # Extraction of evaluation points
    ant = antilope.sel(lat=xr.DataArray(lats, dims='num_poste'), lon=xr.DataArray(lons, dims='num_poste'), method='nearest').sel(time=nivometeo.date)

    labels = list()

    position = 1  # Positions of violinplots
    fig, ax = plt.subplots(figsize=(10,10))

    #bias = scores.bias(ant.rr.data, nivometeo.obs.data)  # Raw ANTILOPE bias
    bias = np.nanmean(ant.rr.data-nivometeo.obs.data, axis=0)  # Raw ANTILOPE bias
    #mask = np.where(nivometeo.obs.data>0)
    #rat = np.nanmean(ant.rr.data[mask]/nivometeo.obs.data[mask], axis=0)  # Raw ANTILOPE ratio
    labels.append(scores.violinplot(ax, position, bias, 'Raw ANTILOPE'))

    suffix = 'ref'
    suffix = 'v2'
    suffix = 'v3'
    suffix = 'v4'
    suffix = 'v5'
    product = 'arome'
    #for product in ['arome', 'sans_arome', 'nivometeo_arome', 'nivometeo_sans_arome']:
    for suffix in ['v2', 'v4', 'v5', 'v6']:
        position = position + 1

        ratio  = xr.open_dataset(os.path.join(workdir, f"Estimated_ratio_alp_0.15_{product}_{suffix}.nc"))
        #error = xr.open_dataset(os.path.join(workdir, "Observation_error_0.15_alp.nc"))

        rat = ratio.sel(lat=xr.DataArray(lats, dims='num_poste'), lon=xr.DataArray(lons, dims='num_poste'), method='nearest').ratio
        #err = error.sel(lat=xr.DataArray(lats, dims='num_poste'), lon=xr.DataArray(lons, dims='num_poste'), method='nearest').error

        #bias = scores.bias(ant.rr.data/rat.data, nivometeo.obs.data)  # ANTILOPE bias after debiasing
        bias = np.nanmean(ant.rr.data/rat.data-nivometeo.obs.data, axis=0)  # ANTILOPE bias after debiasing
        #labels.append(scores.violinplot(ax, position, bias, f'Method {product}'))
        labels.append(scores.violinplot(ax, position, bias, f'Method {suffix}'))

    ax.set_ylabel(f'Bias', fontsize=28)
    #ax.set_xticklabels([''] + products, fontsize=28)
    ax.set_xticks(range(position))
    ax.yaxis.set_tick_params(labelsize=28)
    ax.legend(*zip(*labels), fontsize=18)
    #fig.savefig(f'{workdir}/bias_{suffix}.pdf', format='pdf',  bbox_inches='tight')
    fig.savefig(f'{workdir}/bias_{product}.pdf', format='pdf',  bbox_inches='tight')

