#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 11/05/2023

import os, sys
from datetime import datetime,timedelta
import numpy as np
import xarray as xr

import argparse

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Annotation

###########################################
# USAGE : p plot_field.py $filename [cumul]
###########################################


radars = dict(
    moucherotte = dict(lat=45.14776, lon=5.63933, alt=1920,name='Moucherotte'),
    colombis    = dict(lat=44.49664, lon=6.21729, alt=1742, name='Colombis'),
    ladole      = dict(lat=46.42565, lon=6.10001, alt=1677, name='La Dole'),
)

error = xr.open_dataset(os.path.join("/home/vernaym/workdir/ASSIMILATION/mask/alp", "Observation_error.nc"))

mnt = xr.open_dataset(os.path.join("/home/vernaym/QGIS/MNT", "DEM_ALPES_WGS84_250m_bilinear.nc"))
mnt = mnt.interp(lat=error.lat, lon=error.lon)

# 1. Plot error as a function of elevation
plt.plot(mnt.Band1.data.flatten(), error['Observation error (mm)'].data.flatten(), linestyle='', marker='.')
plt.show()

# 2. Plot error as a function of the distance to the nearest radar
lons, lats = np.meshgrid(error.lon.data, error.lat.data)
dist = dict()
for radar, infos in radars.items():
    dist[radar] = np.sqrt((lats-infos['lat'])**2+(lons-infos['lon'])**2)  # Euclidian horizontal distance

distmin = np.min([item for item in dist.values()], axis=0)
plt.plot(distmin.flatten(), error['Observation error (mm)'].data.flatten(), linestyle='', marker='.')
plt.show()

