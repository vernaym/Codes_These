#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os,sys
import datetime
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#import hvplot
#import hvplot.xarray
from pyproj import Proj, transform


print('USAGE : plot_radar_cumuls.py inputfile')

filename = sys.argv[1]

extract_dom = ['45240', '44990', '6010', '6490']  # Domaine des Grandes Rousses

norm = plt.Normalize()

if filename.startswith('PANTHERE'):

    mnt = xr.open_dataset("scriptMNTLouis.nc")  # MNT uniquement sur les grandes Rousses
    df=pd.read_csv(filename)
    product = 'PANTHERE'
    outProj = Proj(init='epsg:4326')
    inProj = Proj(init='epsg:2154')
    df['x2'],df['y2'] = transform(outProj,inProj,df['x'].values,df['y'].values)
    #df.set_index(['x2', 'y2'], inplace=True)
    #radar = xr.Dataset.from_dataframe(df)
    #radar.set_coords({'x':'x2', 'y':'y2'})
    #radar['rr'].expand_dims({'y':radar['y2']})

    rr = []
    for lon in mnt['x'].values:
        rr.append(list())
        for lat in mnt['y'].values:
            df['dist'] = ((df['x2']-lon)**2+(df['y2']-lat)**2)**0.5
            rr[-1].append(df['rr'][df['dist'].idxmin()])

    radar = xr.DataArray(rr, coords=[("x", mnt['x']), ("y", mnt['y'])])

    #plot=radar.hvplot(x='x',y='y')
    #hvplot.show(plot)
    ##hvplot.save(plot, 'CUMUL_PANTHERE.png')
    x, y = np.meshgrid(mnt['x'], mnt['y'])
    X, Y = transform(inProj, outProj, x, y)
    Z = mnt['ZS']
    colors = plt.cm.coolwarm(norm(np.transpose(radar.values)))

else:

    ds = xr.open_dataset(filename)
    product = 'ANTILOPE'
    mnt = xr.open_dataset('/home/vernaym/QGIS/MNT/DEM_ALPES_WGS84_250m_bilinear.nc')  # Pour tracer sur toutes les Alpes
    mnt = xr.open_dataset("scriptMNTLouis.nc")  # MNT uniquement sur les grandes Rousses
    #tmp = ds.interp(X=mnt.lon,Y=mnt.lat,method='nearest') # Pour interpoller la grille ANTILOPE sur le MNT
    tmp = mnt.interp(lon=ds.longitude,lat=ds.latitude,method='nearest')  # Pour interpoller le MNT sur la grille ANTILOPE
    # Il reste a selectionner le sous domaine d'intéret avant de plotter...
    X = ds['longitude'].values
    Y = ds['latitude'].values
    tmp['Z'] =tmp['Band1']
    tmp['Z'].values = np.nan_to_num(tmp['Z'].values)
    Z = tmp['Z']
    radar = ds.rr_cumul
    colors = plt.cm.coolwarm(norm(np.nan_to_num(radar.values)))
    # TODO il y a surement un problème de dimension...


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X=X, Y=Y, Z=Z, linewidth=0, antialiased=False, facecolors=colors)
ax.xaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.fill = False
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.fill = False
ax.zaxis.pane.set_edgecolor('white')
ax.grid(False)
ax.set_xlabel('Longitude', labelpad=20)
ax.set_ylabel('Latitude', labelpad=20)
ax.set_zlabel('Elevation (m)')
#ax.set_zlim(0., np.max(Z))
ax.set_zlim(0., 3500.)
#fig.colorbar(surf, shrink=0.5, aspect=5)
#fig.colorbar(colorbar=colors, shrink=0.5, aspect=5)
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.cm.coolwarm), ax=ax, shrink=0.75, aspect=8, label=f'{product} cumulated precipitation between 2018113006 and 2019043006 (mm)')
plt.show()
#fig.save("CUMULS_PANTHERE_3D.svg", format='svg')
