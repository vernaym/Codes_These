#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os,sys
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # F401 unused import --> to ignore !

#import hvplot
#import hvplot.xarray
from pyproj import Proj, transform

# This script plot accumulated precipiation over the Grandes-Rousses domain (can
# be easily adapted to plot over different domain or the whole input domain)
# It interpolates a DEM grid on the grided precipitation data to plot 3D
# precipitation fields

print('USAGE : plot_radar_cumuls.py inputfile')
datadir = '/home/vernaym/These/DATA'
savedir = '/home/vernaym/These/figures'

filename = sys.argv[1]

datebegin = filename.split('.')[0].split('_')[-2]
dateend   = filename.split('.')[0].split('_')[-1]


# Domaine des Grandes Rousses
extract_dom = dict(
    latmax = 45.240,
    latmin = 44.990,
    lonmin = 6.010,
    lonmax = 6.490,
)

outProj = Proj(init='epsg:4326')
inProj = Proj(init='epsg:2154')

norm = plt.Normalize()

if not os.path.isfile(filename):
    print(f'WARNING : no such file or directory {filename}')
    print(f'lookink for the file under {datadir}')
    filename = os.path.join(datadir, filename)

if 'PANTHERE' in filename:

    if not os.path.exists(filename):
        ## First convert all csv files in netcdf file in order to read the with xarray
        date = datetime.strptime(datebegin, '%Y%m%d%H%M')
        while date <= datetime.strptime(dateend, '%Y%m%d%H%M'):
            print(date)
            csvfic = '{0:s}_010000_DATA.text'.format(date.strftime('%Y%m%d%H%M')) # ex: 201904030600_010000_DATA.text
            if os.path.exists(csvfic):
                tmp = pd.read_csv(csvfic, names=['y', 'x', 'rr'], header=None, dtype={'x':float, 'y':float, 'rr':float}, sep=' ', comment='#', index_col=['x', 'y'])
                tmp = tmp.query('x>6.0 & x<6.5 & y<45.25 & y>44.9')
                if 'df' in locals():
                    df = df + tmp
                else:
                    df = tmp
            date = date + timedelta(days=1)
        #df = df.rename(index=lambda val: round(val, 2))
        df.to_csv(filename)
    else:
        df = pd.read_csv(filename)

    mnt = xr.open_dataset(os.path.join(datadir, "MNT_GrandesRousses.nc"))  # MNT uniquement sur les grandes Rousses
    #mnt = xr.open_dataset("scriptMNTLouis.nc")  # MNT uniquement sur les grandes Rousses
    df = pd.read_csv(filename)
    product = 'PANTHERE'
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


    #plot=radar.hvplot(x='x',y='y')
    #hvplot.show(plot)
    ##hvplot.save(plot, 'CUMUL_PANTHERE.png')
    x, y = np.meshgrid(mnt['x'], mnt['y'])
    X, Y = transform(inProj, outProj, x, y)
    Z = mnt['ZS']

    radar = xr.DataArray(
            data=np.transpose(rr),
            name='rr',
            dims=["X", "Y"],
            coords=dict(longitude=(["X", "Y"], X), latitude=(["X", "Y"], Y)),
            attrs=dict(description="Total precipitation",units="mm"),
        )

    radar.to_netcdf(os.path.join(datadir, os.path.basename(filename).replace('csv', 'nc')))

    #colors = plt.cm.coolwarm(norm(np.transpose(radar.values)))
    colors = plt.cm.coolwarm(norm(radar.values))

else:

    ds = xr.open_dataset(filename)
    #mnt = xr.open_dataset("scriptMNTLouis.nc")  # MNT uniquement sur les grandes Rousses
    #mnt = xr.open_dataset(os.path.join(datadir, "MNT_GrandesRousses.nc"))  # MNT uniquement sur les grandes Rousses
    mnt = xr.open_dataset('/home/vernaym/QGIS/MNT/DEM_ALPES_WGS84_250m_bilinear.nc')  # Pour tracer sur toutes les Alpes
    if 'ANTILOPE' in filename:
        product = 'ANTILOPE'
        #tmp = ds.interp(X=mnt.x, Y=mnt.x, method='nearest') # Pour interpoller la grille ANTILOPE sur le MNT GrandesRousses
        tmp = mnt.interp(lon=ds.longitude, lat=ds.latitude, method='nearest')  # Pour interpoller le MNT sur la grille ANTILOPE
    elif 'krigeage' in filename:
        product = 'KRIGING'
        #tmp = ds.interp(X=mnt.x, Y=mnt.y, method='nearest') # Pour interpoller la grille ANTILOPE sur le MNT
        tmp = mnt.interp(lon=ds.longitude, lat=ds.latitude, method='nearest')  # Pour interpoller le MNT sur la grille ANTILOPE

    tmp['Z'] = tmp['Band1']
    tmp['Z'].values = np.nan_to_num(tmp['Z'].values)

    latmin = extract_dom['latmin']
    lonmin = extract_dom['lonmin']
    latmax = extract_dom['latmax']
    lonmax = extract_dom['lonmax']

    # On selectionne le sous domaine d'intéret avant de plotter...
    tmp = tmp.where((tmp.lon>=lonmin) & (tmp.lon<=lonmax) & (tmp.lat>=latmin) & (tmp.lat<=latmax), drop=True)
    X = tmp['lon'].values
    Y = tmp['lat'].values
    Z = tmp['Z']

    # define pixel colors
    radar = ds.where((ds.longitude>=lonmin) & (ds.longitude<=lonmax) & (ds.latitude>=latmin) & (ds.latitude<=latmax), drop=True).rr_cumul
    colors = plt.cm.coolwarm(norm(np.nan_to_num(radar.values)))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.view_init(elev=60., azim=135)  # Set point of view
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
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.cm.coolwarm), ax=ax, shrink=0.75, aspect=8, label=f'{product} cumulated precipitation \n between {datebegin} and {dateend} (mm)')
#plt.show()
plt.savefig(f'{savedir}/CUMUL3D_{product}_{datebegin}_{dateend}.pdf', format='pdf')

