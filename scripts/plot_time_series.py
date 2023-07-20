#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 05/11/2022

import os, sys
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
import xarray as xr

import argparse

import tools

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Annotation

datadir = '/home/vernaym/These/DATA'
savedir = '/home/vernaym/These/figures'

nivometeo = pd.read_csv(os.path.join(datadir, 'obs_nivometeo_daily_RR_20211201_20220430.csv'), sep=';', parse_dates=['Q.dat'],
    dtype={'Q.num_poste':int, 'poste_nivo.nom_usuel':str, 'poste_nivo.alti':int, 'poste_nivo.lat_dg':float, 'poste_nivo.lon_dg':float, 'poste_nivo.massif_nivo':int, 'Q.rr':float})

os.chdir(datadir)
filenames = ['ANTILOPEH_2021073106_2021102923_alp.nc', 'ANTILOPEH_2021103000_2022060200_alp.nc', 'ANTILOPEH_2022060201_2022080106_alp.nc', 'ANTILOPEH_2022080107_2022123123_alp.nc', 'ANTILOPEH_2023010100_2023042306_alp.nc']
antilope = xr.open_mfdataset(filenames, chunks={'lat':1, 'lon':1}, combine='nested', concat_dim='time')
#antilope.to_netcdf('ANTILOPEH_2021073106_2023042306_alp.nc')
#antilope = tools.hourly_to_daily(antilope)
#antilope.to_netcdf('ANTILOPEQ_2021073106_2023042306_alp.nc')

#antilope = xr.Dataset(os.path.join(datadir, 'ANTILOPEQ_2021103000_2022060200_alp.nc'))
mtblanc = antilope.sel({'lat':45.83, 'lon':6.86}).compute()
mtblanc = tools.hourly_to_daily(mtblanc)
#chamonix = antilope.sel({'lat':45.93, 'lon':6.88}).compute()
chamonix = antilope.sel({'lat':46.0, 'lon':6.95}).compute()  # Le tour nivo (ratio 0.98)
chamonix = tools.hourly_to_daily(chamonix)
#plt.plot(mtblanc.time, mtblanc.rr)
#plt.plot(chamonix.time, chamonix.rr)

ratio = chamonix / mtblanc
y = ratio.rr.data.copy()
#y = np.where(np.isfinite(y), y, 1)  # Ignore Nan and inf values
#y = np.where(y==0, 1, y)  # Ignore null values
#y = np.where(y<1, -1/y, y)
#y = np.where(y==1, 0, y)
pos = np.where(y>1, y, np.nan)
neg = np.where((y<1) & (y>0), -1/y, np.nan)
fig, ax = plt.subplots(figsize=(16,10))
ax.bar(ratio.time, pos, color='blue', label="Chamonix / Mont-Blanc")
ax.bar(ratio.time, neg, color='red', label="- Mont-Blanc / Chamonix")
plt.legend(fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_ylim(-60, 60)
ax.set_ylabel('Precipitation ratio', fontsize=16)
#plt.plot(ratio.time, ratio.rr)
#plt.hist(ratio.rr[np.isfinite(ratio.rr)], bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#plt.hist(ratio.rr[np.isfinite(ratio.rr)], bins=np.arange(0.1, 10, 0.1))

fig.savefig(os.path.join(savedir, 'temporal_plot_ratio_mtblanc_chamonix.pdf'), format='pdf', bbox_inches='tight')
