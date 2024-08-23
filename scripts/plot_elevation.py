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
from scipy.sparse import csr_matrix, csc_matrix, diags
from scipy.spatial import distance_matrix
import shapefile
#from metpy.interpolate import cross_section
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pyproj import Proj, transform

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as patches
from matplotlib_scalebar.scalebar import ScaleBar

from osgeo import gdal

from These.scripts import tools
from These.radar import Preprocessing_ANTILOPE

from snowtools.plots.maps import plot2D

import richdem as rd

#plt.rcParams["figure.figsize"] = [7.50, 3.50]
#plt.rcParams["axes.grid"] = False
plt.rcParams["figure.autolayout"] = True


##############################################################################################
##############################################################################################

if len(sys.argv) > 1:
    domain = sys.argv[1]
else:
    domain = 'alp'

datadir = '/home/vernaym/QGIS/MNT'
savedir = '/home/vernaym/These/figures'

# Domaine des Grandes Rousses
domain_coords = dict(
        GrandesRousses = dict(latmax=45.240, latmin=44.990, lonmin=6.010, lonmax = 6.490),
        NorthernAlps   = dict(lonmin=6.0, lonmax=6.9, latmin=45.6, latmax=46.35),
        CentralAlps    = dict(lonmin=5.6, lonmax=7.0, latmin=45.0, latmax=45.6),
        SouthernAlps   = dict(lonmin=5.7, lonmax=7.0, latmin=44.2, latmax=45.0),
        HauteSavoie    = dict(lonmin=5.82, lonmax=7.05, latmin=45.5, latmax=46.29),
        MontBlanc      = dict(lonmin=6.45, lonmax=7.1, latmin=45.65, latmax=46.1),
        Savoie         = dict(lonmin=6.06, lonmax=7.06, latmin=45.15, latmax=45.65),
        Isere          = dict(lonmin=5.54, lonmax=6.19, latmin=44.89, latmax=45.16),
        Brianconnais   = dict(lonmin=6.48, lonmax=6.95, latmin=44.67, latmax=44.95),
        HautesAlpes    = dict(lonmin=5.90, lonmax=6.36, latmin=44.58, latmax=44.81),
        AlpesSud       = dict(lonmin=6.56, lonmax=6.92, latmin=44.18, latmax=40.49),
        alp            = dict(latmax=46.450, latmin=44.100, lonmin=5.400, lonmax=7.200),
)
delta = 0
latmin = domain_coords[domain]['latmin'] - delta / 2
latmax = domain_coords[domain]['latmax'] + delta / 2
lonmin = domain_coords[domain]['lonmin'] - delta
lonmax = domain_coords[domain]['lonmax'] + delta / 2

figsize = dict(
        alp            = (15.1,16),
        GrandesRousses = (15,7),
        HauteSavoie    = (12,12),
        HautesAlpes    = (16,10),
        MontBlanc      = (15,10),
        Savoie         = (16,8),
        Isere          = (16,8),
)

d0 = 0.08
max_dist = d0*3


blacklist = [5063407, 5063410, 38191408]  # La Meije, LA GRAVE, Huez 2350

landmarks = {
        #"Alpe d'Huez" : dict(lon=6.070, lat=45.092, alt=1800, marker='o'),
        #"Les 2 Alpes" : dict(lon=6.127, lat=45.013, alt=1800, marker='o'),
        #"Lautaret"    : dict(lon=6.408, lat=45.038, alt=2058, marker='X'),
        "La Meije"    : dict(lon=6.311, lat=45.008, alt=3500, marker='^'),  # real alt = 3984
        "Pic Blanc"   : dict(lon=6.131, lat=45.128, alt=3000, marker='^'),  # real alt = 3333
        "Mont-Blanc"  : dict(lon=6.87, lat=45.84, alt=4807, marker='^'),
    }



def goto(path):
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)

def proj_mnt(mnt):
    outProj = Proj(init='epsg:4326')
    inProj = Proj(init='epsg:2154')
    x, y = np.meshgrid(mnt['xx'], mnt['yy'])
    X, Y = transform(inProj, outProj, x, y)
    #Z = mnt['ZS']
    mnt_proj = xr.DataArray(
        #data=Z,
        data=mnt.data,
        name='elevation',
        dims=["lat", "lon"],
        coords=dict(lon=X[0], lat=Y[:,0]),
        attrs=dict(description="Elevation",units="m"),
    )
    return mnt_proj

def add_landmarks(ax):
    # Add landmarks
    for landmark, infos in landmarks.items():
        ax.plot(infos['lon'], infos['lat'], marker=infos['marker'], color='white', markersize=20, transform=ccrs.PlateCarree())
        if landmark == 'Pic Blanc':
            ab = ax.annotate(landmark, (infos['lon']+0.03, infos['lat']+0.01), weight='bold', color='k', fontsize=22, transform=ccrs.PlateCarree())
        else:
            ab = ax.annotate(landmark, (infos['lon']-0.1, infos['lat']-0.07), weight='bold', color='k', fontsize=22, transform=ccrs.PlateCarree())
        ab.set_bbox(dict(facecolor='white', alpha=0.3, edgecolor='white'))

def add_rectangle(ax):
    # Create a Rectangle patch
    #for domain in ['MontBlanc', 'GrandesRousses']:
    for domain in ['GrandesRousses']:
        x0 = domain_coords[domain]['lonmin']
        y0 = domain_coords[domain]['latmin']
        dx = domain_coords[domain]['lonmax'] - x0
        dy = domain_coords[domain]['latmax'] - y0
        if domain =='MontBlanc':
            rect = patches.Rectangle((x0, y0), dx, dy, linewidth=4, edgecolor='k', facecolor='none', label='Mont-Blanc')  #  https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html
        elif domain == 'GrandesRousses':
            rect = patches.Rectangle((x0, y0), dx, dy, linewidth=4, edgecolor='r', facecolor='none', label='Grandes Rousses')  #  https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html

        # Add the patch to the Axes
        ax.add_patch(rect)

def add_boundaries(ax):
    shapefile_name = os.path.join("/home/vernaym/QGIS/FondDeCarte/", "world-administrative-boundaries.shp")
    borders = shapefile.Reader(shapefile_name)
    for shape in borders.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x,y, color='k', linestyle=':', alpha=0.8, transform=ccrs.PlateCarree())

def add_massifs(ax):
    massifs = shapefile.Reader("/home/vernaym/safran/ctes/shapefiles/massifs_safran.shp")
    for shape in massifs.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        ax.plot(x,y,color='k', transform=ccrs.PlateCarree())

def add_scores(ax):
    fic_score = os.path.join(datadir, 'scores_2021080106_2022080106_alp.csv')
    scores = pd.read_csv(fic_score, sep=';')
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "blue", "green", "orange", "red"], 5)
    thresholds = [0., 0.5, 0.80, 1.2, 1.5, 10]
    norm = matplotlib.colors.BoundaryNorm(thresholds, cmap.N, transform=ccrs.PlateCarree())

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

def add_postes(ax, type_poste='nivometeo'):
    postdir = '/home/vernaym/These/DATA'
    if type_poste.startswith('nivometeo'):
        #fic_postes = os.path.join(postdir, f'scores_2021110106_2022043006_alp.csv')
        fic_postes = os.path.join(postdir, f'postes_nivometeo.csv')
        marker = '*'
        color  = 'red'
        label  = 'Ski-resort station'
    else:
        fic_postes = os.path.join(postdir, f'scores_2021110106_2022043006_alpes_obs_auto.csv')
        marker = 'v'
        color  = 'k'
        label  = 'Automatic station'
    #fic_score = os.path.join('/home/vernaym/These/DATA', 'postes_nivometeo.csv')
    #scores = pd.read_csv(fic_score, sep=';')

    postes = pd.read_csv(fic_postes, sep=';')
    postes = postes.rename(columns={'poste_nivo.lat_dg':'lats', 'poste_nivo.lon_dg':'lons'})
    postes_domain = postes.loc[(postes.lons>=lonmin) & (postes.lons<=lonmax) & (postes.lats<=latmax) & (postes.lats>=latmin)]
    lons = postes_domain['lons']
    lats = postes_domain['lats']


    #sc = ax.scatter(scores['poste_nivo.lon_dg'], scores['poste_nivo.lat_dg'], c=scores['poste_nivo.alti'],  marker='^', s=300)
    #sc = ax.scatter(scores['poste_nivo.lon_dg'], scores['poste_nivo.lat_dg'],  marker='^', s=450, color='k')
    sc = ax.scatter(lons, lats,  marker=marker, s=250, color=color, label=label, transform=ccrs.PlateCarree())

def add_radar_positions(ax):
    radars = dict(
        moucherotte = dict(lat=45.14776, lon=5.63933, alt=1920,name='Moucherotte'),
        colombis    = dict(lat=44.49664, lon=6.21729, alt=1742, name='Colombis'),
        ladole      = dict(lat=46.42565, lon=6.10001, alt=1677, name='La Dôle'),
    )
    def getImage(path):
        return OffsetImage(plt.imread(path, format="png"), zoom=.05)

    symbole_radar = '/home/vernaym/These/figures/symbole_radar_violet.png'
    for radar, infos in radars.items():
        ab = AnnotationBbox(getImage(symbole_radar), (infos['lon'], infos['lat']), frameon=False, label='radar')
        ax.add_artist(ab)
        ab = ax.annotate(f'{infos["name"]}\n{infos["alt"]}m', (infos['lon']-0.05, infos['lat']-0.22), weight='bold', color='darkviolet', fontsize=24, transform=ccrs.PlateCarree())
        ab.set_bbox(dict(facecolor='white', alpha=0.3, edgecolor='white'))

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
    corr = tools.to_xarray(corr, mnt, varname='correlation')
    corr = corr.where(corr>0)
    cml = corr.plot(ax=ax, cmap=plt.cm.Greys, add_colorbar=False, alpha=0.5)
    circle = plt.Circle((5.685, 44.365), max_dist, color='k', fill=False, linewidth=2)
    ax.add_artist(circle)

def extract_cross_section(field, varname='elevation'):
    """
    Using MetPy : https://unidata.github.io/MetPy/latest/examples/cross_section.html
    """
    # Definition of the cross section coordinates :
    #start = (45.14776, 5.63933)  # Radar Moucherotte
    #start = (45.14776, 5.635)  # Radar Moucherotte
    #start = (46.42572, 6.10032)  # Radar La Dole
    #start = (46.02947300021354, 7.1429769396152825)  # Orsières (Suisse)
    start  = (45.93, 6.86)  # Chamonix
    #end   = (45.13761, 6.21261)
    #end   = (45.12142, 6.21189)  # Passe par le Pic Blanc : 45 km
    #end   = (45.11872, 6.27540)  # Passe par le Pic Blanc : 50 km
    #end   = (45.750494, 6.9680)  # From La Dole : passe par le Mont Blanc (100 km)
    #end   = (45.70184, 6.676548)  # Depuis Orsières : traverse le Mont-Blanc  (50 km)
    end    = (45.82, 6.864)  # Mont Blanc

    field = field.metpy.parse_cf(varname=varname).squeeze()
    cross = cross_section(field, start, end)

    return cross

def plot_vertical_cross_section(cross):
    #x = np.linspace(0, 50, len(cross.data))
    x = np.linspace(0, 12, len(cross.data))
    #x = np.linspace(0, 100, len(cross.data))
    y = cross.data + 250
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.fill_between(x, y, color='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.set_xlabel('Distance to the radar (km)', fontsize=26)
    ax.set_xlabel('Distance (km)', fontsize=16)
    ax.set_ylabel('Elevation (m)', fontsize=16)
    #plt.show()
    fig.savefig(os.path.join(savedir, 'Vertical_cross_section.pdf'), format='pdf')
    plt.close()




if domain == 'GrandesRousses':
    mnt = xr.open_dataset(os.path.join("/home/vernaym/.vortexrc/hack/uget/vernaym/data", "DEM_GrandesRousses25m_L93.tif"))
else:
    mnt = xr.open_dataset(os.path.join(datadir, "DEM_ALPES_WGS84_250m_bilinear.nc"))
    #mnt=mnt.where((mnt['lat']>=latmin) & (mnt['lat']<=latmax) & (mnt['lon']>=lonmin) & (mnt['lon']<=lonmax), drop=True)

#mnt = gdal.Open(os.path.join(datadir, "DEM_ALPES_WGS84_250m_bilinear.tif"))
#tmp = np.transpose(mnt.ReadAsArray().astype(np.float), axis=1)
#plt.contour(tmp, cmap = "viridis", levels = list(range(0, 5000, 100)))

if 'elevation'  in mnt.keys():
    mnt = mnt.elevation
elif 'band1' in mnt.keys():
    mnt = mnt.band1
elif 'Band1' in mnt.keys():
    mnt = mnt.Band1
elif 'band_data' in mnt.keys():
    mnt = mnt.band_data

# TMP : To produce 1km MNT over the Frenc Alps domain
#antilope = xr.open_dataset('/home/vernaym/These/DATA/CUMUL_ANTILOPEH_alp_2021103000_2022060200.nc')
#mnt1km = mnt.interp(lat=antilope.lat, lon=antilope.lon, method='linear')
#mnt1km.to_netcdf('/home/vernaym/These/DATA/DEM_ALPESFR_WGS84_1km.nc')

if domain == 'GrandesRousses':
    mnt = proj_mnt(mnt)

crossection = False
if crossection:

    cross = extract_cross_section(mnt)
    plot_vertical_cross_section(cross)

    sys.exit()

    ratio = xr.open_dataset(os.path.join('/home/vernaym/workdir/ASSIMILATION/mask/alp', 'Estimated_ratio.nc'))
    cross = extract_cross_section(ratio, varname='ratio')
    plt.imshow(np.atleast_2d(cross), cmap=plt.get_cmap('RdBu_r'), extent=(0, 50, 0, 1))

    import pdb
    pdb.set_trace()

    error = xr.open_dataset(os.path.join('/home/vernaym/workdir/ASSIMILATION/mask/alp', 'Observation_error.nc'))
    cross = extract_cross_section(error, varname='error')
    #plt.imshow(np.atleast_2d(cross), cmap=plt.get_cmap('Reds'), extent=(0, 50, 0, 1))
    plt.imshow(np.atleast_2d(cross), cmap=plt.get_cmap('YlOrBr'), extent=(0, 50, 0, 1))

    import pdb
    pdb.set_trace()

    #cumul = xr.open_dataset('/home/vernaym/These/DATA/./CUMUL_ANTILOPEH_alp_2021103000_2022060200.nc')
    #cross = extract_cross_section(cumul, varname='rr_cumul')
    #plt.imshow(np.atleast_2d(cross), cmap=plt.get_cmap('YlGnBu'), extent=(0, 50, 0, 1))

else:

    #antilope = xr.open_dataset('/home/vernaym/These/DATA/CUMUL_ANTILOPEH_alp_2021103000_2022060200.nc')

#    mnt1km = xr.open_dataset('/home/vernaym/These/DATA/DEM_ALPESFR_WGS84_1km.nc')
#    # Projection of DEM on the 1-km ANTILOPE grid
#    #data = rd.rdarray(mnt.interp(lat=antilope.lat, lon=antilope.lon).sortby('lat', ascending=False).elevation.data, no_data=-9999)
#    data = rd.rdarray(mnt1km.sortby('lat', ascending=False).elevation.data, no_data=-9999)
#
#    # 1. Elevation inter-distance
#    Z = mnt1km.elevation.data.flatten()
#    Z=diags(Z, 0)
#    coords=[(lon,lat) for lat in mnt1km.lat.data for lon in mnt1km.lon.data]
#    pond = Preprocessing_ANTILOPE.codistances(coords)
#    pond[pond.nonzero()] = 1
#    dZ = pond.dot(Z) - Z.dot(pond)  # Compute elevation inter-distance
#    dZ=np.abs(dZ)
#    scipy.sparse.save_npz('/home/vernaym/These/DATA/elevation_codistance_alp.npz', dZ, compressed=False)
#
#    # 2. Slope inter-distance
#    slope = rd.TerrainAttribute(data, attrib='slope_percentage')
#    #slope = rd.TerrainAttribute(data, attrib='slope_degrees')
#    rd.rdShow(slope, axes=False, cmap='magma', figsize=(8, 5.5))
#    plt.show()
#
#    # 3. Aspect inter-distance
#    aspect = rd.TerrainAttribute(data, attrib='aspect')
#    rd.rdShow(aspect, axes=False, cmap='jet', figsize=(8, 5.5))
#    plt.show()

    # 3. Plot elevation
    #filename = os.path.join(savedir, f'ReliefAlpes_correlation{d0}.pdf')
    filename = os.path.join(savedir, f'Relief_{domain}_with_nivometeo.pdf')

    # Plot elevation
    #if not os.path.exists(filename):
    fig,ax = plt.subplots(figsize=figsize[domain], subplot_kw=dict(projection=ccrs.PlateCarree()))
    #fig,ax = plt.subplots(figsize=(16, 8), subplot_kw=dict(projection=ccrs.PlateCarree()))
    #ax = ax.ravel()
    xmin = lonmin
    xmax = lonmax
    ymin = latmin
    ymax = latmax
    ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())
    ax.set_frame_on(False)
    #https://discourse.holoviz.org/t/cannot-remove-grid-for-hv-quadmesh/2211/8
    #im = mnt.elevation.plot(ax=ax, cmap=plt.cm.terrain, subplot_kws={'frame_on':False}, linewidth=0, label='Elevation (m)', add_colorbar=False)
    #im = mnt.plot(ax=ax, cmap=plt.cm.terrain, linewidth=0, label='Elevation (m)', add_colorbar=False, transform=ccrs.PlateCarree())
    lons, lats = np.meshgrid(mnt.lon.data, mnt.lat.data)
    #im = plot2D.plot_field(mnt.where(mnt>3000), ax=ax, transform=ccrs.PlateCarree(), cmap=plt.cm.terrain, add_colorbar=False, vmin=0)
    im = ax.contourf(lons, lats, mnt, cmap=plt.cm.terrain, levels=50, transform=ccrs.PlateCarree(), alpha=1, antialiased=True)  # https://www.earthdatascience.org/tutorials/visualize-digital-elevation-model-contours-matplotlib/
    # This is the fix for the white lines between contour levels
    for c in im.collections:
        c.set_edgecolor("face")
    c = ax.contour(lons, lats, mnt.data, colors='grey', levels=[1000, 2500], transform=ccrs.PlateCarree())  # https://www.earthdatascience.org/tutorials/visualize-digital-elevation-model-contours-matplotlib/
    plt.clabel(c, inline=1, fontsize=10)

    fullfeatures = True
    if fullfeatures:
        # Add optional features
        add_boundaries(ax)
        add_landmarks(ax)
        add_rectangle(ax)
        add_radar_positions(ax)
        add_postes(ax, type_poste='automatic stations')
        add_postes(ax, type_poste='nivometeo stations')
        # Add cross section line
        start = (45.14776, 5.63933)  # Radar Moucherotte
        end   = (45.11872, 6.27540)  # Passe par le Pic Blanc : 50 km
        #ax.plot([start[1], end[1]], [start[0], end[0]], color='grey', linestyle='-', linewidth=5, transform=ccrs.PlateCarree())

        ax.set_frame_on(False)
        # Add scalebar (https://stackoverflow.com/questions/39786714/how-to-insert-scale-bar-in-a-map-in-matplotlib)
        scalebar = ScaleBar(
                100000,  # 1 pixel = 1km
                length_fraction=0.4,
                location='lower right',
                #frameon=False,  # Switch on/off scale background
                box_alpha=0.8,  #Transparency of the scale background
                border_pad = 1,  # Pad between the scale anbd the border of the plot
                font_properties=dict(size=18),
                scale_loc='top',
            )
        #ax.add_artist(scalebar)
#        xticks = np.arange(xmin, xmax, 0.5)
#        yticks = np.arange(ymin, ymax, 0.5)
#        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
#        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
#        lon_formatter = LongitudeFormatter(zero_direction_label=True)
#        lat_formatter = LatitudeFormatter()
#        ax.xaxis.set_major_formatter(lon_formatter)
#        ax.yaxis.set_major_formatter(lat_formatter)
#        ax.tick_params(axis='both', which='major', labelsize=18)
#        ax.set_xlabel('longitude', fontsize=22)
#        ax.set_ylabel('latitude', fontsize=22)

    # Add colorbar
    #ax.legend(fontsize=20, loc=2)  # loc=2 --> upper-left
    ax.legend(fontsize=20, loc=(0.0, 0.85))  # loc=2 --> upper-left
    #plot_correlation(ax, mnt)  # To add correlation area
    # Force colorbar size
    cb = fig.colorbar(im, fraction=0.058, pad=0.04)  # From https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    cb.ax.tick_params(labelsize=20)
    cb.set_label('Elevation (m)', size=24)

    fig.savefig(filename, format='pdf')

    # Plot elevation difference
    #filename = os.path.join(savedir, 'Elevation_diff.pdf')
    ##if not os.path.exists(filename):
    #mnt['delta'] = np.abs(mnt['Band1']-1000)
    #fig,ax = plt.subplots(figsize=(14,16))
    #mnt.delta.plot(ax=ax)
    #fig.savefig(os.path.join(savedir, 'Elevation_diff.pdf'))


