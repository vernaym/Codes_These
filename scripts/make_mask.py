#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os, sys
from datetime import datetime,timedelta
import time
import numpy as np
from scipy.ndimage import uniform_filter
import xarray as xr
import pandas as pd
# To avoid pandas warning when modifying a copy of a dataframe :
pd.options.mode.chained_assignment = None  # default='warn'

import scipy
from scipy.sparse import csr_matrix, diags
from scipy.spatial import cKDTree

import shapefile

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.animation as animation

#plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

from pykrige.uk import UniversalKriging

##############################################################################################
##############################################################################################

domain = sys.argv[1]

datadir = '/home/vernaym/These/DATA'
#savedir = '/home/vernaym/workdir/ASSIMILATION/mask'
#savedir = '/home/vernaym/workdir/ASSIMILATION/mask/illustration_methode'
savedir = '/home/vernaym/workdir/ASSIMILATION/mask/KalmanFilter'

onlypostes = [5001400, 5085403]
onlypostes = [5001400, 5085403, 5133400]
onlypostes = [74056416, 73176400, 73257400, 73123402, 38548400, 73194401]
onlypostes = [38191400]
onlypostes = [5001400]
onlypostes = [38375400]
onlypostes = [38253400]
onlypostes = [73194401]
onlypostes = [73257400]
onlypostes = [74056416, 5001400,38548400]
onlypostes = [74056416, 73132400, 73176400, 73257400, 73194401]
onlypostes = [73257400]
onlypostes = [73306403]

d0 = 0.2
c0 = 2
max_dist = 0.5

# TODO ajouter les postes clim non utilisés par ANTILOPE temps réel
fic_score = os.path.join(datadir, 'scores_2021110106_2022043006_alpes.csv')
#fic_score = os.path.join(datadir, 'scores_2018110106_2019043006_alpes_10.csv')
#fic_score = os.path.join(datadir, 'scores_2021110106_2022043006_alpes_10.csv')  # WARNING : scores valid for precipitation >10mm

landmarks = {
        "Alpe d'Huez" : dict(lon=6.070, lat=45.092, alt=1800, marker='o'),
        "Les 2 Alpes" : dict(lon=6.127, lat=45.013, alt=1800, marker='o'),
        "Lautaret"    : dict(lon=6.408, lat=45.038, alt=2058, marker='X'),
        "La Meije"    : dict(lon=6.311, lat=45.008, alt=3500, marker='^'),  # real alt = 3984
        "Pic Blanc"   : dict(lon=6.131, lat=45.128, alt=3000, marker='^'),  # real alt = 3333
    }

# Domaine des Grandes Rousses
extract_dom = dict(
        GrandesRousses = dict(
            latmax = 45.240,
            latmin = 44.990,
            lonmin = 6.010,
            lonmax = 6.490,
        ),
        Savoie = dict(
            latmax = 45.6,
            latmin = 45.0,
            lonmin = 6.2,
            lonmax = 7.2,
        ),
        HautesAlpes = dict(
            latmax = 45.2,
            latmin = 44.25,
            lonmin = 6.2,
            lonmax = 7.1,
        ),
        MontBlanc = dict(
            latmax = 46.25,
            latmin = 45.5,
            lonmin = 6.5,
            lonmax = 7.1,
        ),
        alp = dict(
            latmax = 46.45,
            latmin = 44.1,
            lonmin = 5.4,
            lonmax = 7.2,
        ),
    )

latmin = extract_dom[domain]['latmin']
lonmin = extract_dom[domain]['lonmin']
latmax = extract_dom[domain]['latmax']
lonmax = extract_dom[domain]['lonmax']

def get_date(a_string):
    try:
        date = datetime.strptime(a_string, '%Y%m%d').replace(hour=6, minute=0, second=0, microsecond=0)
    except ValueError:
        try:
            date = datetime.strptime(a_string, '%y%m%d').replace(hour=6, minute=0, second=0, microsecond=0)
        except ValueError:
            try:
                date = datetime.strptime(a_string, '%Y%m%d%H').replace(minute=0, second=0, microsecond=0)
            except ValueError:
                try:
                    date = datetime.strptime(a_string, '%y%m%d%H').replace(minute=0, second=0, microsecond=0)
                except ValueError:
                    print('The start date provided is not in a good format (YYYYMMDDHH or YYMMDDHH or YYYYMMDDHHMM or YYMMDD or YYYYMMDD)')
                    raise
    finally:
        return date

def date_range(start, end, dt=24):
    start = start.replace(hour=6)
    dates = list()
    while start <= end:
        dates.append(start)
        start += timedelta(hours=dt)

    return dates

def goto(path):
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)

def add_obs_nivometeo():
    nivometeo = pd.read_csv(os.path.join(datadir, 'obs_nivometeo_daily_RR_20211201_20220430.csv'), sep=';', parse_dates=['Q.dat'],
        dtype={'Q.num_poste':int, 'poste_nivo.nom_usuel':str, 'poste_nivo.alti':int, 'poste_nivo.lat_dg':float, 'poste_nivo.lon_dg':float, 'poste_nivo.massif_nivo':int, 'Q.rr':float})

    dist = lambda dx,dy: np.sqrt(dx**2+dy**2)

    nivometeo = nivometeo.loc[(nivometeo['poste_nivo.lat_dg']>=latmin) & (nivometeo['poste_nivo.lat_dg']<=latmax) & (nivometeo['poste_nivo.lon_dg']>=lonmin) & (nivometeo['poste_nivo.lon_dg']<=lonmax)]
    keep = list()
    biais_antilope = list()
    lat = list()
    lon = list()
    for num_poste in nivometeo['Q.num_poste'].unique():
        tmp = nivometeo.loc[nivometeo['Q.num_poste']==num_poste]
        ndays = (tmp['Q.dat'].max()-tmp['Q.dat'].min()).days + 1
        if ndays < 100:
            print(f'Station {num_poste} is droped (not enough observations)')
        else:
            missing_days = ndays - len(tmp)
            if missing_days == 0:
                if 'first_day' not in locals():
                    first_day = tmp['Q.dat'].min()
                else:
                    first_day = max(first_day, tmp['Q.dat'].min())
                if 'last_day' not in locals():
                    last_day = tmp['Q.dat'].max()
                else:
                    last_day = min(last_day, tmp['Q.dat'].max())
                keep.append(num_poste)
                name = tmp['poste_nivo.nom_usuel'].values[0]
                print(f'keeping station {num_poste} ({name})')
                cumul = tmp['Q.rr'].sum()
                latitude = tmp['poste_nivo.lat_dg'].values[0]
                longitude = tmp['poste_nivo.lon_dg'].values[0]
                lat.append(latitude)
                lon.append(longitude)
                darr = dist(antilope['lat']-latitude, antilope['lon']-longitude)
                idx=np.where(darr == np.amin(darr))
                biais_antilope.append(antilope.rr_cumul[idx].data[0][0]-cumul)
            else:
                print(f'Station {num_poste} is droped (too many missing obs : {missing_days:d})')

    return biais_antilope, lon, lat

def add_scores(scores, ax, mycmap=None, vmin=None, vmax=None):
    if mycmap is None:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "darkviolet", "green", "orange", "red"], 5)
        thresholds = [0., 0.5, 0.80, 1.2, 1.5, 10]  # TODO : vérier la coéhrence des seuils entre les figures
        #thresholds = [0., 0.5, 0.90, 1.1, 1.5, 10]
        norm = matplotlib.colors.BoundaryNorm(thresholds, cmap.N)
    else:
        cmap = mycmap

    scores_domain = scores.loc[(scores.lons>=lonmin) & (scores.lons<=lonmax) & (scores.lats<=latmax) & (scores.lats>=latmin)]

    def set_marker(row):
        if row['ratio'] <= 0.8:
        #if row['ratio'] <= 0.9:  # TODO : vérier la coéhrence du seuil entre les figures
            return 'v'
        elif row['ratio'] > 0.8 and row['ratio'] < 1.2:
        #elif row['ratio'] > 0.9 and row['ratio'] < 1.1:  # TODO : vérier la coéhrence du seuil entre les figures
            return 'o'
        else:
            return '^'
    scores_domain["marker"] = scores_domain.apply(set_marker, axis=1)  # axis=1 makes sure that function is applied to each row
    for marker, info in scores_domain.groupby('marker'):
        #onlypostes = set(info.index) - set(blacklist)
        onlypostes = info.index
        info = info[info.index.isin(onlypostes)]
        if mycmap is None:
            #sc = plt.scatter(info['lons'], info['lats'], c=info['ratio'], cmap=cmap, norm=norm, marker=marker, s=450, edgecolors='black', linewidth=3, alpha=1)
            #sc = plt.scatter(info['lons'], info['lats'], c=info['ratio'], cmap=cmap, norm=norm, marker=marker, s=150, edgecolors='black', alpha=0.5)
            sc = plt.scatter(info['lons'], info['lats'], c=info['ratio'], cmap=cmap, norm=norm, marker=marker, s=300, edgecolors='black', alpha=1)
        else:
            #sc = plt.scatter(info['lons'], info['lats'], c=info['ratio'], cmap=cmap, vmin=vmin, vmax=vmax, marker=marker, s=300, edgecolors='black', alpha=1)
            sc = plt.scatter(info['lons'], info['lats'], c=info['ratio'], cmap=cmap, vmin=vmin, vmax=vmax, marker=marker, s=200, edgecolors='black', alpha=0.1)

        #labels = [str(num_poste) for num_poste in info['num_poste']]
        labels = [str(np.around(ratio, decimals=2)) for ratio in info['ratio']]
        # TODO : add score value
        #for idx, label in enumerate(labels):
        for idx in info.index:
            txt = plt.text(info['lons'][idx], info['lats'][idx], np.around(info['ratio'][idx], decimals=2), fontsize=16)
            #txt = plt.text(info['lons'][idx], info['lats'][idx], info['num_poste'][idx])
    #cb = ax.colorbar(sc, label=legend)
    #ax.colorbar(sc, label=legend, shrink=shrink, anchor=anchor)
    return sc

def add_landmarks(ax):
    # Add landmarks
    for landmark, infos in landmarks.items():
        ax.plot(infos['lon'], infos['lat'], marker=infos['marker'], color='red', markersize=5)
        ax.annotate(landmark, (infos['lon']+0.003, infos['lat']+0.003), color='red', fontsize=20)

def add_radar_positions(ax):
    radars = dict(
        moucherotte = dict(lat=45.14776, lon=5.63933, alt=1920,name='Moucherotte'),
        colombis    = dict(lat=44.49664, lon=6.21729, alt=1742, name='Colombis'),
        ladole      = dict(lat=46.42565, lon=6.10001, alt=1677, name='La Dole'),
    )
    def getImage(path):
       return OffsetImage(plt.imread(path, format="png"), zoom=0.03)

    symbole_radar = '/home/vernaym/These/figures/symbole_radar.png'
    for radar, infos in radars.items():
       ab = AnnotationBbox(getImage(symbole_radar), (infos['lon'], infos['lat']), frameon=False)
       ax.add_artist(ab)

def add_boundaries():

    shapefile_name = os.path.join("/home/vernaym/QGIS/FondDeCarte/", "world-administrative-boundaries.shp")
    borders = shapefile.Reader(shapefile_name)
    for shape in borders.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x,y, color='k')

def add_massifs():
#    shapefile_name = os.path.join("/home/vernaym/QGIS/FondDeCarte/", "world-administrative-boundaries.shp")
#    borders = shapefile.Reader(shapefile_name)
#    for shape in borders.shapeRecords():
#        x = [i[0] for i in shape.shape.points[:]]
#        y = [i[1] for i in shape.shape.points[:]]
#        plt.plot(x,y, color='k')

    massifs = shapefile.Reader("/home/vernaym/safran/ctes/shapefiles/massifs_safran.shp")
    for shape in massifs.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x, y,color='grey', alpha=0.5)

def add_cities(latmin, latmax, lonmin, lonmax):
    cities = pd.read_csv(os.path.join('/home/vernaym/safran/monitoring/', 'cities.csv'), sep=',')
    tmp = cities[(cities.population>25000) & (cities.lat>=latmin) & (cities.lat<=latmax) & (cities.lng>=lonmin) & (cities.lng<=lonmax)]
    plt.plot(tmp.lng, tmp.lat, marker='.', linestyle='')
    for idx in tmp.index:
        plt.text(tmp.lng[idx], tmp.lat[idx], tmp.city[idx], alpha=0.5)

def plot(antilope, datebegin, dateend, categories=True, biascorrection=False):

    #if not os.path.exists(os.path.join(savedir, f'CUMUL_ANTILOPE_2021080106_2022070106_{domain}.pdf')):

    if domain == 'alp':
        fig, ax = plt.subplots(figsize=(16,16))
    elif domain == 'GrandesRousses':
        fig, ax = plt.subplots(figsize=(19,8))
    elif domain == 'HautesAlpes':
        fig, ax = plt.subplots(figsize=(14,7))
    else:
        fig, ax = plt.subplots()

    latmin = np.min(antilope.lat.data)
    latmax = np.max(antilope.lat.data)
    lonmin = np.min(antilope.lon.data)
    lonmax = np.max(antilope.lon.data)

    if biascorrection:
        filename = os.path.join('/home/vernaym/These/DATA/mask', 'Estimated_ratio.nc')
        ratio = xr.open_dataset(filename)
        #ratio.lat.data = ratio.lat.data+0.005  # TODO : comprendre et resoudre le probleme de decallage des coordonnees
        ratio = ratio.where((ratio.lon>=lonmin) & (ratio.lon<=lonmax) & (ratio.lat<=latmax) & (ratio.lat>=latmin), drop=True)
        antilope.rr_cumul.data = antilope.rr_cumul.data / ratio.ratio.data

    cmap = plt.cm.YlGnBu

    if categories :
        # To group by range of data
        cmaplist = [cmap(i) for i in range(20, cmap.N+1)]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N-20)
        # define the bins and normalize
        bounds = np.arange(200, 1300, 100)
        #norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        norm = matplotlib.colors.BoundaryNorm(bounds, len(bounds)-2)
        cml = antilope.rr_cumul.plot.pcolormesh(ax=ax, cmap=cmap, norm=norm, add_colorbar=False)

        #cml = antilope.rr_cumul.plot(ax=ax, cmap=cmap, norm=norm, add_colorbar=False)
        #antilope.rr_cumul.plot(ax=ax, cbar_kwargs={"label":'Total precipitation between 2021080106 and 2022070106 (mm)'}, cmap=plt.cm.coolwarm)
        #cml = antilope.rr_cumul.plot(ax=ax, cmap=plt.cm.YlGnBu, add_colorbar=False)
        #cml = plt.contourf(antilope.lon, antilope.lat, antilope.rr_cumul, cmap=cmap, norm=norm, add_colorbar=False)
        #cml = antilope.rr_cumul.plot(ax=ax, cmap=plt.cm.YlGnBu, add_colorbar=False, alpha=0.5)
    else:
        cml = antilope.rr_cumul.plot(ax=ax, vmin=100, vmax=1200, cmap=plt.cm.YlGnBu, add_colorbar=False)
        #cml = antilope.rr_cumul.plot(ax=ax, cmap=plt.cm.YlGnBu, add_colorbar=False)
        #cml = antilope.rr_cumul.plot(ax=ax, vmin=150, vmax=500, cmap=plt.cm.YlGnBu, add_colorbar=False)

    #add_landmarks(ax)
    add_radar_positions(ax)
    scores = pd.read_csv(fic_score, sep=';')
    sc = add_scores(scores, ax)
    add_boundaries()
    add_cities(latmin, latmax, lonmin, lonmax)
    cb = fig.colorbar(sc)
    #cb.set_label(label='Mean ANTILOPE / rain-gauges ratio', fontsize=22, weight='bold')
    cb.set_label(label='ANTILOPE / rain-gauges ratio', fontsize=22)
    #cb.set_label(label='ANTILOPE / rain-gauges ratio', fontsize=14)
    cb.ax.tick_params(labelsize=16)
    cb2 = fig.colorbar(cml, extend='both')
    cb2.set_label(label=f'Total precipitation between \n {datebegin} and {dateend} (mm)', fontsize=22)
    #cb2.set_label(label=f'Total precipitation between \n {datebegin} and {dateend} (mm)', fontsize=14)
    cb2.ax.tick_params(labelsize=16)
    #cb2.ax.tick_params(labelsize=12)
    ax.grid(False)  # Remove grid lines (does not work !)
    #fig.legend()
    #fig.tight_layout()
    fig.savefig(os.path.join(savedir, f'CUMUL_ANTILOPE_{datebegin}_{dateend}_{domain}.pdf'), layout='tight')
    #sys.exit()

def nearest(array, value):
    """ Find the closest element of 'array' to 'value'. """
    return float(array[np.abs(array - value).argmin()].data)

def to_xarray(array, field, varname='rr'):
    output = xr.DataArray(
    name   = varname,
    data   = array,
    dims   = ["lat", "lon"],
    coords = dict(lon=field.lon, lat=field.lat),
    #attrs  = dict(description="Difference between each pixel cumul and the max of its neighbours"),
    )
    return output

def plot_and_save(field, name, cmap=plt.cm.Greys, vmin=None, vmax=None, scores=None):
    if domain == 'GrandesRousses':
        fig, ax = plt.subplots(figsize=(12,6))
    elif domain == 'HautesAlpes':
        fig, ax = plt.subplots(figsize=(14,12))
    elif domain == 'Savoie':
        fig, ax = plt.subplots(figsize=(12,6))
    elif domain == 'MontBlanc':
        fig, ax = plt.subplots(figsize=(12,11))
    elif domain == 'alp':
        fig, ax = plt.subplots(figsize=(16,16))
        #fig, ax = plt.subplots(figsize=(14,16))
    else:
        fig, ax = plt.subplots()
    ax = plot_field(fig, ax, field, cmap=cmap, vmin=vmin, vmax=vmax, scores=scores)
    fig.savefig(os.path.join(savedir, f'{name}.pdf'), format='pdf', layout='tight')
    field.to_netcdf(os.path.join(savedir, f'{name}.nc'))


def plot_field(fig, ax, field, cmap=plt.cm.Greys, vmin=None, vmax=None, scores=None, colorbar=True):

    if vmin is None:
        vmin = np.nanmin(field)
    if vmax is None:
        vmax = np.nanmax(field)
    if cmap == 'custom':
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "darkviolet", "green", "orange", "red"], 5)
        thresholds = [0., 0.5, 0.80, 1.2, 1.5, 10]  # TODO : vérier la coéhrence des seuils entre les figures
        #thresholds = [0., 0.5, 0.90, 1.1, 1.5, 10]
        norm = matplotlib.colors.BoundaryNorm(thresholds, cmap.N)
        cml = field.plot(ax=ax, cmap=cmap, norm=norm, add_colorbar=False)
    else:
        cml = field.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)

    if scores is not None:
        if field.name == 'ratio':
            # Plot scores with same cmap since it is the same information
            sc = add_scores(scores, ax, mycmap=cmap, vmin=vmin, vmax=vmax)
        else:
            sc = add_scores(scores, ax)

#    plt.plot(6.82, 45.85, marker='+', color='red')

    add_radar_positions(ax)
    add_boundaries()
    add_massifs()
    #add_cities(latmin, latmax, lonmin, lonmax)

    if colorbar:
        cb = fig.colorbar(cml)
        cb.set_label(field.name, fontsize=24)
        cb.ax.tick_params(labelsize=20)

    return ax

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


def KalmanFilter(field, moving_window=40):
    #rawdata = field.rr_cumul.data/334  # 334 is the number of days over wich the field cumul is made : we want a mean daily (24h) error
    rawdata = field.rr_cumul.data  # 334 is the number of days over wich the field cumul is made : we want a mean daily (24h) error
    tmp = uniform_filter(rawdata, size=moving_window)
    smoothed = to_xarray(tmp, field)
    smoothratio = rawdata/smoothed

    smoothdiff = rawdata-smoothed
    plot_and_save(smoothdiff, f'Observation_error_smoothingsize{moving_window}_{domain}', cmap=plt.cm.coolwarm)

    lons, lats = np.meshgrid(field.lon.data, field.lat.data)

    filename = os.path.join('/home/vernaym/These/DATA', f'codistance_{max_dist}_{c0}_alp.npz')
    if not os.path.exists(filename):
        # Compute inter-distances
        coords=[(lon,lat) for lat in field.lat.data for lon in field.lon.data]
        codist = codistances(coords)
        scipy.sparse.save_npz(filename, codist, compressed=False)
    else:
        codist = scipy.sparse.load_npz(filename)

#    tmp = codist.getrow(10000).toarray()[0].reshape((len(field.lat), len(field.lon)))
#    tmp = to_xarray(tmp, field)
#    plot_and_save(tmp, f'correlation', cmap=plt.cm.Greys)

    smoothdiff = np.where(np.isnan(smoothdiff.data), 0, smoothdiff.data)
    smoothdiff = smoothdiff / np.max(np.abs(smoothdiff))
    diff = diags(smoothdiff.flatten(), 0)
    #diff = diags(np.ones(np.shape(smoothdiff)).flatten())
    P = diff.dot(codist.dot(diff))
    P = csr_matrix(P)

    #P = np.diag([1]*42716)

    scores = pd.read_csv(fic_score, sep=';')
    scores = scores.set_index('num_poste')
    scores = scores.sort_values('lats')

    X = rawdata.flatten()
    X = np.where(np.isnan(X), 0, X)
    Y = list()
    H = list()
    for i,poste in enumerate(scores.index):
    #for i,poste in enumerate(scores.index[[55, -15,-20]]):
        if poste == 74056416:
            print(i)
            P0 = to_xarray(P[idx*len(field.lon)+idy].reshape((len(field.lat), len(field.lon))).todense(), field)
            plot_and_save(P0, f'P0', cmap=plt.cm.Greys)

        ratio = scores.loc[poste, 'ratio']
        dist = np.sqrt((lats-scores.loc[poste,'lats'])**2+(lons-scores.loc[poste, 'lons'])**2)  # Euclidian horizontal distance
        idx, idy = np.where(dist==np.min(dist))
        Y.append(field.rr_cumul.data[idx[0],idy[0]]/ratio)  # Construction of the "observation vector"

        tmp = np.zeros(len(field.lat)*len(field.lon))
        tmp[idx*len(field.lon)+idy] = 1
        H.append(tmp)


    R = R = np.diag(np.array(Y)/100000)
    H = np.array(H)
    HX = np.dot(H, X)
    Ht = np.transpose(H)
    H = csr_matrix(H)
    #HP = np.dot(H, P)
    HP = H.dot(P)
    #HP=HP.todense()
    #HPHt = HP.dot(Ht).todense()
    HPHt = HP.dot(Ht)
    K = P.dot(np.dot(Ht, np.linalg.inv(HPHt+R)))
    A = X+np.dot(K, (Y-HX))
    A = A.reshape((len(field.lat), len(field.lon)))

    tmp = to_xarray(np.dot(Ht, np.linalg.inv(HPHt+R))[:,0].reshape((len(field.lat), len(field.lon))), field)
    plot_and_save(tmp, f'Ht(HPHt+R)^-1', cmap=plt.cm.coolwarm)

    tmp = to_xarray(np.dot(K, (Y-HX)).reshape((len(field.lat), len(field.lon))), field)
    plot_and_save(tmp, f'Inovation', cmap=plt.cm.coolwarm)

    tmp = to_xarray(K[:,0].reshape((len(field.lat), len(field.lon))), field)
    plot_and_save(tmp, f'Kalman_Gain0', cmap=plt.cm.coolwarm)

    tmp = to_xarray(K[:,1].reshape((len(field.lat), len(field.lon))), field)
    plot_and_save(tmp, f'Kalman_Gain1', cmap=plt.cm.coolwarm)

    A = to_xarray(A, field, varname='ratio')
    plot_and_save(A, 'TMP', cmap=plt.cm.YlGnBu)

    import pdb
    pdb.set_trace()


def ratio_estimation(field, moving_window=25):
    """
    Two steps :
    1. filter accumulation field to produce a map of deviation to the
    smoothed field (parameter : moving window size set to 15)
    2. Use evaluation data to apply the ratio to neighboring points with a
    ponderation depending on the ratio estimated at step 1 and the distance
    between each pixel and the scores (parameter : correlation lenght set to 0.15)

    This provides both a debiaising mask to apply to each new observation
    before assimilation and an observation error field.

    When this debiasing method is applied to ANTILOPE accumulations, the
    field is smoother and visible artifical patterns are attenuated.
    """
#    latmin = field.lat.data.min()
#    latmax = field.lat.data.max()
#    lonmin = field.lon.data.min()
#    lonmax = field.lon.data.max()
#    if domain == 'custom':
#        latmin = 45
#        latmax = 45.5
#        lonmin = 6.25
#        lonmax = 6.75
#        lats = [np.round(lat,2) for lat in np.arange(latmin,latmax,0.01)]
#        lons = [np.round(lon,2) for lon in np.arange(lonmin,lonmax,0.01)]
#        field = field.sel({'lat':lats, 'lon':lons})

    rawdata = field.rr_cumul.data/334  # 334 is the number of days over wich the field cumul is made : we want a mean daily (24h) error
    tmp = uniform_filter(rawdata, size=moving_window)
    smoothed = to_xarray(tmp, field)
    smoothratio = rawdata/smoothed

    #diff = (rawdata-smoothed) * rawdata
    diff = rawdata-smoothed

    scores = pd.read_csv(fic_score, sep=';')
    scores = scores.set_index('num_poste')
    scores = scores.sort_values('lats')

    lons, lats = np.meshgrid(field.lon.data, field.lat.data)
    #estimated_ratio = np.ones(np.shape(field.rr_cumul.data))*scores.ratio.mean()
    estimated_ratio = np.ones(np.shape(field.rr_cumul.data))
    ##estimated_ratio = smoothratio
    inov = np.zeros(np.shape(field.rr_cumul.data))
    #inov = np.ones(np.shape(field.rr_cumul.data))
    #weight  = np.ones(np.shape(field.rr_cumul.data))
    weight = np.zeros(np.shape(field.rr_cumul.data))
    #inov = smoothratio
    #wsum  = np.ones(np.shape(field.rr_cumul.data))
    scores = scores
    used_scores = []

    # Mont-Blanc
#    xx = np.where(field.lon==6.82)[0][0]
#    yy = np.where(field.lat==45.85)[0][0]
    # poste n° 73176400
    xx = np.where(field.lon==6.85)[0][0]
    yy = np.where(field.lat==45.63)[0][0]
#    xx = 0
#    yy = 0

    # TODO : trouver un moyen de rendre l'estimation indépendante de l'ordre de traitement
    #onlypostes = set(scores.index) - set(blacklist)
    onlypostes = scores.index
    rr = list()
    ww = list()
    weights = list()
    ratios  = list()
    for i,poste in enumerate(scores.index):
    #for i,poste in enumerate(reversed(scores.index)):
        if poste in onlypostes:
            used_scores.append(poste)
            #print(scores.loc[poste])
            ratio = scores.loc[poste, 'ratio']
            dist = np.sqrt((lats-scores.loc[poste,'lats'])**2+(lons-scores.loc[poste, 'lons'])**2)  # Euclidian horizontal distance
            idx, idy = np.where(dist==np.min(dist))
            ref_cumul = field.rr_cumul.data[idx[0],idy[0]]
            if poste == 74056416:
                rcc = ref_cumul.copy()
                rr0 = ratio.copy()
            cumul_dist = field.rr_cumul.data-ref_cumul
            cumul_ratio = field.rr_cumul.data/ref_cumul
            #estimated_ratio = estimated_ratio+(cumul_ratio-estimated_ratio)*np.exp(-np.abs(cumul_dist)/(ref_cumul/5))*np.exp(-dist/0.3)  # Marche bien mais n'utilise pas le score !
            #estimated_ratio = estimated_ratio+((cumul_ratio-estimated_ratio)*np.exp(-np.abs(cumul_dist)/(ref_cumul))-estimated_ratio)*np.exp(-dist/0.2)  # DERIVE
            #pond = np.exp(-dist/0.3)
            #pond[np.where(dist>0.5)]=0
            #estimated_ratio = estimated_ratio+(ratio-estimated_ratio)*np.exp(-np.abs(cumul_dist)/(ref_cumul/10))*np.exp(-dist/1)  # PAS MAL
            #estimated_ratio = estimated_ratio+(ratio*cumul_ratio-estimated_ratio)*np.exp(-np.abs(cumul_dist)/(ref_cumul/c0))*np.exp(-dist/d0)  # Marche bien avec d0=0.4 et c0=5
            #estimated_ratio = estimated_ratio+(ratio-estimated_ratio)*np.exp(-np.abs(cumul_dist)/(ref_cumul/c0))*np.exp(-dist/d0)  # TEST
            #estimated_ratio = estimated_ratio+(ratio*cumul_ratio-estimated_ratio)*np.exp(-dist/d0)  # TEST
            #estimated_ratio = estimated_ratio+(ratio*smoothratio-estimated_ratio)*np.exp(-dist/d0)*np.exp(-np.abs(cumul_dist)/(ref_cumul/c0)) # TEST
            #estimated_ratio = estimated_ratio+(ratio*cumul_ratio-estimated_ratio)*np.exp(-dist/d0)*np.exp(-np.abs(cumul_dist)/(ref_cumul/c0))
            #estimated_ratio = estimated_ratio+(ratio*cumul_ratio-1)*np.exp(-(dist/d0)**2)*np.exp(-(np.abs(cumul_dist)/(ref_cumul/c0))**2)

            #w = np.exp(-(dist/d0)**2)*np.exp(-(np.abs(cumul_dist)/(ref_cumul/c0))**2)  # PROBLEME : ref_cumul/c0 donne plus de poids aux bias >0 !!

            # TODO : on veut que dans le cercle de corrélation, un gros écart de cumul entraine une forte correction du biais (et non pas une décroissance indépendante)

            #w = np.exp(-1/2*(dist/d0)**2)*np.exp(-1/2*(np.abs(cumul_dist)/(ref_cumul/(ratio*c0))))
            #w = np.exp(-(dist/d0)**2*np.abs(cumul_dist)/(ref_cumul/(ratio*c0)))
            #w = np.exp(-(dist/d0))*np.exp(-np.abs(cumul_dist)/(ref_cumul/(ratio*c0)))
            #w = np.exp(-(dist/d0)**2)*np.exp(-np.abs(cumul_dist)/(ref_cumul/(ratio*c0)))

            #w = np.exp(-(dist/d0))*np.exp(-np.abs(cumul_dist)/(ref_cumul/(ratio*c0)))
            #w = np.exp(-(dist/d0)**2)*np.exp(-np.abs(cumul_dist)/(ref_cumul/(ratio*c0))**2)
            w = np.exp(-(dist/d0))*np.exp(-np.abs(cumul_dist)/(ref_cumul/(ratio*c0)))
            weights.append(w)
            ratios.append(ratio*cumul_ratio)

            #w = np.exp(-(dist/d0)**2)
            #w = np.exp(-(dist/d0))
            #w = np.exp(-(dist/d0))*np.exp(-(np.abs(cumul_dist)/(ref_cumul/(ratio*c0))))
            #w = np.exp(-(dist/d0)**2)*np.exp(-(np.abs(cumul_dist)/1000)**2)
            #w = np.exp(-(dist/d0))*np.exp(-(np.abs(cumul_dist)/800))
            #w = np.exp(-(dist/d0))*np.exp(-(np.abs(cumul_dist)/(ref_cumul/c0)))
            #inov = inov + (ratio*cumul_ratio-estimated_ratio)*w
            #toto = (1+(ratio*cumul_ratio-1)*w)*w
            guess = ratio*cumul_ratio
            #print(poste)
            #print(toto[xx,yy], w[xx,yy])
            rr.append(guess[yy,xx])
            ww.append(w[yy,xx])
            #inov = inov + (1+(ratio*cumul_ratio-1)*w)*w
            inov = inov + ratio*cumul_ratio*w
            wmax = np.maximum(weight, w)
            weight = weight + w

    #estimated_ratio =  estimated_ratio + (inov/weight-estimated_ratio) * ?
    rr = np.array(rr)
    ww = np.array(ww)
    w0 = max(0, 1-np.sum(ww))
    ee = (1*w0+np.sum(rr*ww))/(w0+np.sum(ww))
    mask = np.flip(np.argsort(ww))
    wws = ww[mask]
    rrs = rr[mask]
    cc = field.rr_cumul.data[yy,xx]


    mask = np.flip(np.argsort(weights, axis=0), axis=0)
    sweights = np.take_along_axis(np.array(weights), mask, axis=0)
    sratios  = np.take_along_axis(np.array(ratios), mask, axis=0)
    tmp = np.cumsum(sweights,axis=0)  # ex [0.8, 1.1, 1.3, 1.4]
    tmp[tmp>1] = 1  # Filter values > 1 : [0.8, 1, 1, 1]
    tmp = np.diff(tmp, axis=0, prepend=0)  # "Un-cumsum" : [0.8, 1, 0, 0]
    totalweight = np.sum(tmp, axis=0)  # =1 if enough info else <1
    w0 = 1 - totalweight
    estimated_ratio = 1*w0 + np.sum(tmp*sratios, axis=0)
    #tmp.apply_along_axis(
    #np.take_along_axis(
    #(np.cumsum(sweights,axis=0)<=1).argmin()  #TODO

#    w0 = 1 - weight
#    w0[w0<0] = 0
#    estimated_ratio =  (1*w0+inov)/(w0+weight)
    #estimated_ratio = estimated_ratio + (inov-estimated_ratio)*weight/(1+weight)
    #estimated_ratio = estimated_ratio + (inov-estimated_ratio)*weight/(len(onlypostes))

    #weight = np.ones(np.shape(field.rr_cumul.data))
    #estimated_ratio = estimated_ratio*(1+inov*(weight/(N+weight)))

    #TODO : TMP
    #mean_ratio = uniform_filter(estimated_ratio, size=moving_window)
    #estimated_ratio = estimated_ratio - mean_ratio
    ratio_field = to_xarray(estimated_ratio, field, varname='ratio')
    #ratio_field = ratio_field.rename('ratio')
    #observation_error = ((np.abs(ratio_field-1) + np.abs(diff))**2)*10

    #observation_error = (np.abs(ratio_field-1)*5 + 5*np.abs(diff))**2
    #observation_error = ratio_field-1  # r=06 ==> err = -1.4
    # To take into account spatial correlation we must keep the sign of the observtaion error
    observation_error = ratio_field - 1
    neg = np.where(observation_error.data<0)
    observation_error.data[neg] = 4*observation_error.data[neg]  # r=0.5 ==> err = -2
    pos= np.where(observation_error.data>=0)
    observation_error.data[pos] = 2*observation_error.data[pos]  # r=1.5 ==> err = 1
    #observation_error = 1+np.abs(smoothratio-1)

    #observation_error = np.abs(ratio_field-1)
    #observation_error = np.abs(ratio_field**2-1)*50
    #observation_error = np.exp((observation_error-0.3))
#    plt.hist(observation_error, bins = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55])
#    plt.title("histogram")
#    plt.show()

    # PLots
    #######
    scores = scores.loc[(scores.lats>=latmin) & (scores.lats<=latmax) & (scores.lons>=lonmin) & (scores.lons<=lonmax)]
    scores = scores.loc[used_scores]
    #plot_and_save(ratio_field, f'Estimated_ratio_{domain}_{d0}_{moving_window}', vmin=0.5, vmax=1.5, cmap=plt.cm.coolwarm, scores=scores)
    #plot_and_save(ratio_field, f'Estimated_ratio_{domain}_{d0}', vmin=0.5, vmax=1.5, cmap=plt.cm.coolwarm, scores=scores)
    #plot_and_save(ratio_field, f'Estimated_ratio_{domain}_{moving_window}_{d0}_{c0}', vmin=0.4, vmax=1.6, cmap=plt.cm.coolwarm, scores=scores)  # To add scores
    #plot_and_save(ratio_field, f'Estimated_ratio_{domain}_{moving_window}_{d0}_{c0}', vmin=0.4, vmax=1.6, cmap=plt.cm.coolwarm, scores=scores)
    plot_and_save(ratio_field, f'Estimated_ratio_{domain}_{d0}_{c0}', vmin=0.4, vmax=1.6, cmap=plt.cm.coolwarm, scores=scores)
#    plot_and_save(ratio_field, f'Estimated_ratio_{domain}_{d0}_{c0}', cmap=plt.cm.coolwarm, scores=scores)


#    estimated_ratio = np.ones(np.shape(field.rr_cumul.data))
#    #for i,poste in enumerate(scores.index):
#    for i,poste in enumerate(reversed(scores.index)):
#        if poste in onlypostes:
#            ratio = scores.loc[poste, 'ratio']
#            dist = np.sqrt((lats-scores.loc[poste,'lats'])**2+(lons-scores.loc[poste, 'lons'])**2)  # Euclidian horizontal distance
#            idx, idy = np.where(dist==np.min(dist))
#            ref_cumul = field.rr_cumul.data[idx[0],idy[0]]
#            cumul_dist = field.rr_cumul.data-ref_cumul
#            cumul_ratio = field.rr_cumul.data/ref_cumul
#            w = np.exp(-(dist/d0)**2)*np.exp(-np.abs(cumul_dist)/(ref_cumul/(ratio*c0))**2)
#            estimated_ratio = estimated_ratio+(ratio*cumul_ratio-estimated_ratio)*w
#    ratio_field = to_xarray(estimated_ratio, field, varname='ratio')
#    plot_and_save(ratio_field, f'Estimated_ratio_{domain}_{d0}_{c0}_reverse', cmap=plt.cm.coolwarm, scores=scores)

    #plot_and_save(smoothed, f'Smoothed_field_{moving_window}_{domain}', cmap=plt.cm.YlGnBu)
    #plot_and_save(diff, f'Observation_error_smoothingsize{moving_window}_{domain}', cmap=plt.cm.coolwarm)
    #plot_and_save(np.abs(diff), f'Observation_error_absolute_value_smoothingsize{moving_window}_{domain}', cmap=plt.cm.Greys, scores=scores)
    #plot_and_save(smoothratio, f'Observation_error_ratio_smoothingsize{moving_window}_{domain}', vmin=0.6, vmax=1.4, cmap=plt.cm.coolwarm, scores=scores)
    #plot_and_save(observation_error, f'Observation_error_{moving_window}_{d0}_{c0}_{domain}', cmap=plt.cm.coolwarm, scores=scores)
    plot_and_save(observation_error, f'Observation_error_{d0}_{c0}_{domain}', vmin=-2, vmax=2, cmap=plt.cm.coolwarm, scores=scores)
#    plot_and_save(observation_error, f'Observation_error_{d0}_{c0}_{domain}', cmap=plt.cm.coolwarm, scores=scores)

def animation_mask(field):

    scores = pd.read_csv(fic_score, sep=';')
    scores = scores.loc[(scores.lons>=lonmin) & (scores.lons<=lonmax) & (scores.lats<=latmax) & (scores.lats>=latmin)]
    scores = scores.set_index('num_poste')
    scores = scores.sort_values('lats')

    lons, lats = np.meshgrid(field.lon.data, field.lat.data)
    estimated_ratio = np.ones(np.shape(field.rr_cumul.data))*scores.ratio.mean()
    #estimated_ratio = smoothratio
    def animate_func(num):
        """
        From : https://towardsdatascience.com/how-to-animate-plots-in-python-2512327c8263
        """
        ax = plt.axes()
        ax.clear()
        estimated_ratio = estimated_ratio = np.ones(np.shape(field.rr_cumul.data))*scores.ratio.mean()
        used_scores = scores.index[:num]
        if num == 0 :
            ratio_field = to_xarray(estimated_ratio, field)
            plot_field(fig, ax, ratio_field, cmap=plt.cm.coolwarm, vmin=0.4, vmax=1.6, colorbar=True)
        else:
            for poste in used_scores:
                ratio = scores.loc[poste, 'ratio']
                dist = np.sqrt((lats-scores.loc[poste,'lats'])**2+(lons-scores.loc[poste, 'lons'])**2)  # Euclidian horizontal distance
                idx, idy = np.where(dist==np.min(dist))
                ref_cumul = field.rr_cumul.data[idx[0],idy[0]]
                cumul_dist = field.rr_cumul.data-ref_cumul
                cumul_ratio = field.rr_cumul.data/ref_cumul
                estimated_ratio = estimated_ratio+(ratio*cumul_ratio-estimated_ratio)*np.exp(-dist/d0)*np.exp(-np.abs(cumul_dist)/(ref_cumul/c0))
            ratio_field = to_xarray(estimated_ratio, field)
            plot_field(fig, ax, ratio_field, cmap=plt.cm.coolwarm, vmin=0.4, vmax=1.6, scores=scores.loc[used_scores], colorbar=False)

    #fig, ax = plt.subplots(figsize=(14,16))
    if domain == 'alp':
        fig = plt.figure(figsize=(14,16))
    elif domain == 'GrandesRousses':
        fig = plt.figure(figsize=(12,6))

#    ax = plt.axes()
    line_ani = animation.FuncAnimation(fig, animate_func, interval=750, frames=len(scores.index))
    # Saving the Animation
    f = os.path.join('/home/vernaym/These/figures/mask', f'animation_mask_{domain}.gif')
    writergif = animation.PillowWriter(fps=1)
    line_ani.save(f, writer=writergif)

def krigeage_scores(field):
    variogram  = 'exponential'  # The same as for ANTILOPE without RADAR data

    scores = pd.read_csv(fic_score, sep=';')

    kriging = UniversalKriging(scores.lons.values, scores.lats.values, scores.ratio.values, variogram_model=variogram)
    score, ss = kriging.execute('grid', field.lon, field.lat)

    ratio = xr.DataArray(
        name   = 'ratio',
        data   = score.data,
        dims   = ["lat", "lon"],
        coords = dict(lon=field.lon, lat=field.lat),
    )

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "blue", "green", "orange", "red"], 5)
    thresholds = [0., 0.5, 0.80, 1.2, 1.5, 10]
    norm = matplotlib.colors.BoundaryNorm(thresholds, cmap.N)
    fig, ax = plt.subplots(figsize=(14,16))
    ratio.plot(ax=ax, cmap=cmap, norm=norm)
    add_scores(scores, ax)
    fig.savefig(os.path.join(savedir, f'kriging_ratio.pdf'), format='pdf', layout='tight')

#def plot_field(field, scores):
#    lt.subplots(figsize=(14,16))
#    field.rr_cumul.plot(ax=ax, cmap=plt.cm.coolwarm)
#    add_scores(scores)
#    fig.savefig(os.path.join(datadir, filename.replace('.nc', '.pdf')), format='pdf', layout='tight')

if __name__ == "__main__":
    if domain == 'GrandesRousses':
        filename = 'CUMUL_ANTILOPEH_GrandesRousses_2021073106_2022070106.nc'
    else:
        filename ='CUMUL_ANTILOPEH_alp_2021103000_2022060200.nc'
        #filename ='CUMUL_ANTILOPEQ_alp_2018080106_2019043006.nc'
    datebegin = filename.split('.')[0].split('_')[-2]
    dateend = filename.split('.')[0].split('_')[-1]
    antilope = xr.open_dataset(os.path.join(datadir, filename))
#    antilope.lat.data = antilope.lat.data+0.005  # TODO : comprendre et resoudre le probleme de decallage des coordonnees
    antilope = antilope.where((antilope.lon>=lonmin) & (antilope.lon<=lonmax) & (antilope.lat<=latmax) & (antilope.lat>=latmin), drop=True)

#    plot(antilope, datebegin, dateend, categories=True, biascorrection=True)
    plot(antilope, datebegin, dateend, categories=False, biascorrection=True)
#    plot(antilope, datebegin, dateend, categories=True)

#    ratio_estimation(antilope)
#    KalmanFilter(antilope)
#    animation_mask(antilope)

#    krigeage_scores(antilope)
#    make_mask(antilope)
#    ratio_estimation(antilope)
    #moving_average(antilope)





