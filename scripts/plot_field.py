#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 05/11/2022

import os, sys
from datetime import datetime,timedelta
import numpy as np
import xarray as xr
import rioxarray

import argparse

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Annotation

from snowtools.plots.maps import plot2D
from snowtools.scripts.extract.vortex import vortexIO as io
import snowtools.tools.xarray_preprocess as xrp

###########################################
# USAGE : p plot_field.py $filename [cumul]
###########################################

latmax = 46.45
latmin = 44.1
lonmin = 5.4
lonmax = 7.2

filename = sys.argv[1]
field = xr.open_dataarray(filename)
field = xrp.preprocess(field)
field = field.where((field.lon >= lonmin) & (field.lon <= lonmax) & (field.lat <= latmax) & (field.lat >= latmin), drop=True)
field = field.rio.write_crs('EPSG:4326')

# High-resolution (25m) DEM for fancy figures (set shade=True in plot_field calls)
io.get_const('uenv:dem.1@vernaym', 'relief', 'alp', filename='TARGET_RELIEF.tif',
    gvar='DEM_FRANCE25M_L93')
dem = rioxarray.open_rasterio('TARGET_RELIEF.tif')
dem = xrp.preprocess(dem)
dem = dem.squeeze()
dem = dem.rename({'xx': 'x', 'yy': 'y'})
dem = dem.rio.reproject_match(field.rename({'lon': 'x', 'lat': 'y'}))
dem = dem.rename({'x': 'xx', 'y': 'yy'})


cumul = False
figname = filename.replace('.nc', '.pdf')
if len(sys.argv)>2:
    cumul=True
    figname = f"CUMUL_{figname}"
    if 'member' in field.dims:
        field = field.sum(dim=['time']).mean(dim=['member'])
    else:
        field = field.sum(dim=['time'])


#fig, ax = plt.subplots(figsize=(12,6))
fig, ax = plt.subplots(figsize=(14,16))
#field.plot(ax=ax, cbar_kwargs={"label":'Total precipitation between {0:s} and {1:s} (mm)'.format(args.datebegin.strftime('%Y%m%d'), args.dateend.strftime('%Y%m%d'))}, cmap=plt.cm.coolwarm)
#field.plot(ax=ax, cmap=plt.cm.Greys)
#field.plot(ax=ax, cmap=plt.cm.YlGnBu)
plot2D.plot_field(field, ax=ax, cmap=plt.cm.YlGnBu, dem=dem, shade=False, vmin=200, vmax=1200)

#add_landmarks()
#add_radar_positions(ax)
#add_scores()
plt.show()
fig.savefig(figname, format='pdf', bbox_inches='tight')




