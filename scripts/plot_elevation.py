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

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

#plt.rcParams["figure.figsize"] = [7.50, 3.50]
#plt.rcParams["axes.grid"] = False
plt.rcParams["figure.autolayout"] = True


##############################################################################################
##############################################################################################

domain = sys.argv[1]

datadir = '/home/vernaym/QGIS/MNT'
savedir = '/home/vernaym/These/figures'

# Domaine des Grandes Rousses
domain_coords = dict(
        GrandesRousses = dict(latmax=45.240, latmin=44.990, lonmin=6.010, lonmax = 6.490),
        NorthernAlps   = dict(lonmin=6.0, lonmax=6.9, latmin=45.6, latmax=46.35),
        CentralAlps    = dict(lonmin=5.6, lonmax=7.0, latmin=45.0, latmax=45.6),
        SouthernAlps   = dict(lonmin=5.7, lonmax=7.0, latmin=44.2, latmax=45.0),
        HauteSavoie    = dict(lonmin=6.45, lonmax=6.95, latmin=45.67, latmax=46.35),
        Savoie         = dict(lonmin=6.06, lonmax=7.06, latmin=45.15, latmax=45.65),
        Isere          = dict(lonmin=5.54, lonmax=6.19, latmin=44.89, latmax=45.16),
        Brianconnais   = dict(lonmin=6.48, lonmax=6.95, latmin=44.67, latmax=44.95),
        HautesAlpes    = dict(lonmin=5.90, lonmax=6.36, latmin=44.58, latmax=44.81),
        AlpesSud       = dict(lonmin=6.56, lonmax=6.92, latmin=44.18, latmax=40.49),
        alp            = dict(latmax=46.450, latmin=44.100, lonmin=5.400, lonmax=7.200),
)
latmin = domain_coords[domain]['latmin']
latmax = domain_coords[domain]['latmax']
lonmin = domain_coords[domain]['lonmin']
lonmax = domain_coords[domain]['lonmax']

d0 = 0.08
max_dist = d0*3


blacklist = [5063407, 5063410, 38191408]  # La Meije, LA GRAVE, Huez 2350

landmarks = {
        "Alpe d'Huez" : dict(lon=6.070, lat=45.092, alt=1800, marker='o'),
        "Les 2 Alpes" : dict(lon=6.127, lat=45.013, alt=1800, marker='o'),
        "Lautaret"    : dict(lon=6.408, lat=45.038, alt=2058, marker='X'),
        "La Meije"    : dict(lon=6.311, lat=45.008, alt=3500, marker='^'),  # real alt = 3984
        "Pic Blanc"   : dict(lon=6.131, lat=45.128, alt=3000, marker='^'),  # real alt = 3333
    }



def goto(path):
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)

def proj_mnt(mnt):
    outProj = Proj(init='epsg:4326')
    inProj = Proj(init='epsg:2154')
    x, y = np.meshgrid(mnt['x'], mnt['y'])
    X, Y = transform(inProj, outProj, x, y)
    Z = mnt['ZS']
    mnt_proj = xr.DataArray(
        data=Z,
        name='elevation',
        dims=["lat", "lon"],
        coords=dict(lon=X[0], lat=Y[:,0]),
        attrs=dict(description="Elevation",units="m"),
    )
    return mnt_proj

def add_landmarks(ax):
    # Add landmarks
    for landmark, infos in landmarks.items():
        ax.plot(infos['lon'], infos['lat'], marker=infos['marker'], color='red', markersize=5)
        ax.annotate(landmark, (infos['lon']+0.003, infos['lat']+0.003), color='red', fontsize=12)

def add_boundaries(ax):
    shapefile_name = os.path.join("/home/vernaym/QGIS/FondDeCarte/", "world-administrative-boundaries.shp")
    borders = shapefile.Reader(shapefile_name)
    for shape in borders.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x,y, color='k')

    massifs = shapefile.Reader("/home/vernaym/safran/ctes/shapefiles/massifs_safran.shp")
    for shape in massifs.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        ax.plot(x,y,color='k')

def add_scores(ax):
    fic_score = os.path.join(datadir, 'scores_2021080106_2022080106_alp.csv')
    scores = pd.read_csv(fic_score, sep=';')
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "blue", "green", "orange", "red"], 5)
    thresholds = [0., 0.5, 0.80, 1.2, 1.5, 10]
    norm = matplotlib.colors.BoundaryNorm(thresholds, cmap.N)

    def set_marker(row):
        if row['ratio'] <= 0.8:
            return 'v'
        elif row['ratio'] > 0.8 and row['ratio'] < 1.2:
            return 'o'
        else:
            return '^'
    scores["marker"] = scores.apply(set_marker, axis=1)  # axis=1 makes sure that function is applied to each row
    for marker, info in scores.groupby('marker'):
        #sc = ax.scatter(info['lons'], info['lats'], c=info['ratio'], cmap=cmap, norm=norm, marker=marker, s=150, edgecolors='black')
        sc = ax.scatter(info['lons'], info['lats'], c=info['ratio'], cmap=cmap, norm=norm, marker=marker, s=50, edgecolors='black')
#        labels = [str(num_poste) for num_poste in info['num_poste']]
#        for idx, label in enumerate(labels):
#            txt = ax.text(info['lons'].data[idx], info['lats'].data[idx], label)

def add_postes_nivometeo(ax):
    fic_score = os.path.join('/home/vernaym/These/DATA', 'postes_nivometeo.csv')
    scores = pd.read_csv(fic_score, sep=';')

    sc = ax.scatter(scores['poste_nivo.lon_dg'], scores['poste_nivo.lat_dg'], marker='o', s=150, color='black', edgecolors='black')

def add_radar_positions(ax):
    radars = dict(
        moucherotte = dict(lat=45.14776, lon=5.63933, alt=1920,name='Moucherotte'),
        colombis    = dict(lat=44.49664, lon=6.21729, alt=1742, name='Colombis'),
        ladole      = dict(lat=46.42565, lon=6.10001, alt=1677, name='La Dole'),
    )
    def getImage(path):
       return OffsetImage(plt.imread(path, format="png"), zoom=.1)

    symbole_radar = '/home/vernaym/These/figures/symbole_radar.png'
    for radar, infos in radars.items():
       ab = AnnotationBbox(getImage(symbole_radar), (infos['lon'], infos['lat']), frameon=False)
       ax.add_artist(ab)

def codistances(coords):
    """
    Solution pour le calcul des inter-distances trouvée sur : https://stackoverflow.com/questions/35296935/python-calculate-lots-of-distances-quickly
    """
    tree = cKDTree(coords)
    dist = tree.sparse_distance_matrix(tree, max_distance=max_dist, p=2, output_type='coo_matrix')
    dist = csr_matrix(dist)
    dist[dist.nonzero()] = dist[dist.nonzero()]/d0
    #np.exp(-dist.data**2, out=dist.data)
    np.exp(-dist.data, out=dist.data)
    return dist

def plot_correlation(ax, mnt):
    filename = os.path.join('/home/vernaym/These/DATA', f'codistance_alp_{d0}.npz')
    if not os.path.exists(filename):
        # Compute inter-distances
        coords=[(lon,lat) for lat in mnt.lat.data for lon in mnt.lon.data]
        codist = codistances(coords)
        scipy.sparse.save_npz(filename, codist, compressed=False)
    else:
        codist = scipy.sparse.load_npz(filename)

    fig2,ax2 = plt.subplots(figsize=(20,20))
    ax2.spy(codist, precision=0.5)
    fig2.savefig(os.path.join(savedir, f'codistance_matrix_correlation{d0}.png'), format='png')

    corr = codist.getrow(9882).toarray()[0].reshape((len(mnt.lat), len(mnt.lon)))
    corr = to_xarray(corr, mnt, varname='correlation')
    corr = corr.where(corr>0)
    cml = corr.plot(ax=ax, cmap=plt.cm.Greys, add_colorbar=False, alpha=0.5)
    circle = plt.Circle((5.685, 44.365), max_dist, color='k', fill=False, linewidth=2)
    ax.add_artist(circle)

def to_xarray(array, field, varname=''):
    output = xr.DataArray(
    name   = varname,
    data   = array,
    dims   = ["lat", "lon"],
    coords = dict(lon=field.lon, lat=field.lat),
    #attrs  = dict(description="Difference between each pixel cumul and the max of its neighbours"),
    )
    return output


mnt = xr.open_dataset(os.path.join(datadir, "DEM_ALPES_WGS84_250m_bilinear.nc"))
mnt=mnt.where((mnt['lat']>=latmin) & (mnt['lat']<=latmax) & (mnt['lon']>=lonmin) & (mnt['lon']<=lonmax), drop=True)
#filename = os.path.join(savedir, f'ReliefAlpes_correlation{d0}.pdf')
filename = os.path.join(savedir, f'ReliefAlpes_with_nivometeo.pdf')

# Plot elevation
#if not os.path.exists(filename):
fig,ax = plt.subplots(figsize=(14,16))
ax.set_frame_on(False)
#https://discourse.holoviz.org/t/cannot-remove-grid-for-hv-quadmesh/2211/8
#im = mnt.Band1.plot(ax=ax, cmap=plt.cm.terrain, subplot_kws={'frame_on':False}, linewidth=0, label='Elevation (m)', add_colorbar=False)
im = mnt.Band1.plot(ax=ax, cmap=plt.cm.terrain, linewidth=0, label='Elevation (m)', add_colorbar=False)

# Add optional features
add_boundaries(ax)
add_postes_nivometeo(ax)
#add_radar_positions(ax)

ax.set_frame_on(False)
#plot_correlation(ax, mnt)  # To add correlation area
cb = fig.colorbar(im)
cb.ax.tick_params(labelsize=20)
cb.set_label('Elevation (m)', size=24)
ax.set_xlabel(None)
ax.set_ylabel(None)
ax.tick_params(axis='both', which='major', labelsize=18)
fig.savefig(filename, format='pdf')

# Plot elevation difference
#filename = os.path.join(savedir, 'Elevation_diff.pdf')
##if not os.path.exists(filename):
#mnt['delta'] = np.abs(mnt['Band1']-1000)
#fig,ax = plt.subplots(figsize=(14,16))
#mnt.delta.plot(ax=ax)
#fig.savefig(os.path.join(savedir, 'Elevation_diff.pdf'))


