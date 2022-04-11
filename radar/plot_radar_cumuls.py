#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os
import datetime
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import hvplot
import hvplot.xarray
from pyproj import Proj, transform

extract_dom = ['45240', '44990', '6010', '6490']
mnt   = xr.open_dataset("scriptMNTLouis.nc")

## First convert all csv files in netcdf file in order to read the with xarray
#datebegin = datetime.datetime(2018, 12, 1, 6, 0, 0, 0) 
#date = datebegin
#while date <= datetime.datetime(2019, 4, 30, 6, 0, 0, 0):
##while date <= datetime.datetime(2018, 12, 2, 6, 0, 0, 0):
#    print(date)
#    csvfic = '{0:s}_010000_DATA.text'.format(date.strftime('%Y%m%d%H%M')) # ex: 201904030600_010000_DATA.text
#    if os.path.exists(csvfic):
#        #df = pd.read_csv(cvsfic, names=['y', 'x', 'rr'], header=None, dtype={'x':float, 'y':float, 'rr':float}, sep=' ', comment='#', index_col=['x', 'y'])
#        tmp = pd.read_csv(csvfic, names=['y', 'x', 'rr'], header=None, dtype={'x':float, 'y':float, 'rr':float}, sep=' ', comment='#', index_col=['x', 'y'])
#        tmp = tmp.query('x>6.0 & x<6.5 & y<45.25 & y>44.9')
#        if 'df' in locals():
#            df = df + tmp
#        else:
#            df = tmp
#    date = date + datetime.timedelta(days=1)
##df = df.rename(index=lambda val: round(val, 2))
#df.to_csv('PANTHERE_CUMUL_201811300600_201904300600.csv')

df=pd.read_csv('PANTHERE_CUMUL_201811300600_201904300600.csv')
outProj = Proj(init='epsg:4326')
inProj = Proj(init='epsg:2154')
df['x2'],df['y2'] = transform(outProj,inProj,df['x'].values,df['y'].values)
#df.set_index(['x2', 'y2'], inplace=True)
#radar = xr.Dataset.from_dataframe(df)
#radar.set_coords({'x':'x2', 'y':'y2'})
#radar['rr'].expand_dims({'y':radar['y2']})

nx = 45 # ou 39 ?
ny = 34 #  
rr = []
x = []
y = []
#for i in range(nx):
for lon in mnt['x'].values:
    rr.append(list())
    for lat in mnt['y'].values:
        df['dist'] = ((df['x2']-lon)**2+(df['y2']-lat)**2)**0.5
        rr[-1].append(df['rr'][df['dist'].idxmin()])

radar = xr.DataArray(rr, coords=[("x", mnt['x']), ("y", mnt['y'])])

plot=radar.hvplot(x='x',y='y')
hvplot.show(plot)
#hvplot.save(plot, 'CUMUL_PANTHERE.png')
