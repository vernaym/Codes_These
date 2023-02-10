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

import shapefile

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

#plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

from pykrige.uk import UniversalKriging

##############################################################################################
##############################################################################################

domain = sys.argv[1]

datadir = '/home/vernaym/These/DATA'
savedir = '/home/vernaym/workdir/ASSIMILATION/mask'

# TODO ajouter les postes clim non utilisés par ANTILOPE temps réel
fic_score = os.path.join(datadir, 'scores_2021110106_2022043006_alpes.csv')
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
            latmax = 45.5,
            latmin = 45.0,
            lonmin = 6.25,
            lonmax = 6.75,
        ),
        HautesAlpes = dict(
            latmax = 45.0,
            latmin = 44.7,
            lonmin = 6.4,
            lonmax = 7.0,
        ),
    )

if not domain == 'alp':  # Use all domain for the Alps
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

    if domain == 'alp':
        scores_domain = scores
    else:
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
        if mycmap is None:
            #sc = plt.scatter(info['lons'], info['lats'], c=info['ratio'], cmap=cmap, norm=norm, marker=marker, s=450, edgecolors='black', linewidth=3, alpha=1)
            sc = plt.scatter(info['lons'], info['lats'], c=info['ratio'], cmap=cmap, norm=norm, marker=marker, s=150, edgecolors='black', alpha=0.5)
            #sc = plt.scatter(info['lons'], info['lats'], c=info['ratio'], cmap=cmap, norm=norm, marker=marker, s=350, edgecolors='black', alpha=1)
        else:
            sc = plt.scatter(info['lons'], info['lats'], c=info['ratio'], cmap=cmap, vmin=vmin, vmax=vmax, marker=marker, s=300, edgecolors='black', alpha=1)

        #labels = [str(num_poste) for num_poste in info['num_poste']]
        labels = [str(np.around(ratio, decimals=2)) for ratio in info['ratio']]
        # TODO : add score value
        #for idx, label in enumerate(labels):
        for idx in info.index:
            txt = plt.text(info['lons'][idx], info['lats'][idx], np.around(info['ratio'][idx], decimals=2), fontsize=16)
            #txt = plt.text(info['lons'][idx], info['lats'][idx], info['num_poste'][idx])
    #ax.colorbar(sc, label=legend)
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
        plt.plot(x,y)

def add_cities(latmin, latmax, lonmin, lonmax):
    cities = pd.read_csv(os.path.join('/home/vernaym/safran/monitoring/', 'cities.csv'), sep=',')
    tmp = cities[(cities.population>25000) & (cities.lat>=latmin) & (cities.lat<=latmax) & (cities.lng>=lonmin) & (cities.lng<=lonmax)]
    plt.plot(tmp.lng, tmp.lat, marker='.', linestyle='')
    for idx in tmp.index:
        plt.text(tmp.lng[idx], tmp.lat[idx], tmp.city[idx], alpha=0.5)

def plot(antilope, categories=True, baiscorrection=False):

    #if not os.path.exists(os.path.join(savedir, f'CUMUL_ANTILOPE_2021080106_2022070106_{domain}.pdf')):
    if True:

        if domain == 'alp':
            fig, ax = plt.subplots(figsize=(16,16))
        elif domain == 'GrandesRousses':
            fig, ax = plt.subplots(figsize=(15,7))
        elif domain == 'HautesAlpes':
            fig, ax = plt.subplots(figsize=(14,7))
        else:
            fig, ax = plt.subplots()

        latmin = np.min(antilope.lat.data)
        latmax = np.max(antilope.lat.data)
        lonmin = np.min(antilope.lon.data)
        lonmax = np.max(antilope.lon.data)

        if baiscorrection:
            filename ='Estimated_ratio_alp.nc'
            ratio = xr.open_dataset(filename)
            if not domain == 'alp':
                ratio = ratio.where((ratio.lon>=lonmin) & (ratio.lon<=lonmax) & (ratio.lat<=latmax) & (ratio.lat>=latmin), drop=True)
            antilope.rr_cumul.data = antilope.rr_cumul.data / ratio.rr.data

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
            #cml = antilope.rr_cumul.plot(ax=ax, vmin=100, vmax=1200, cmap=plt.cm.YlGnBu, add_colorbar=False)
            cml = antilope.rr_cumul.plot(ax=ax, vmin=150, vmax=500, cmap=plt.cm.YlGnBu, add_colorbar=False)

        #add_landmarks(ax)
        add_radar_positions(ax)
        scores = pd.read_csv(fic_score, sep=';')
        sc = add_scores(scores, ax)
        add_boundaries()
        add_cities(latmin, latmax, lonmin, lonmax)
        cb = fig.colorbar(sc)
        #cb.set_label(label='Mean ANTILOPE / rain-gauges ratio', fontsize=22, weight='bold')
        #cb.set_label(label='ANTILOPE / rain-gauges ratio', fontsize=22)
        cb.set_label(label='ANTILOPE / rain-gauges ratio', fontsize=14)
        cb.ax.tick_params(labelsize=16)
        cb2 = fig.colorbar(cml, extend='both')
        #cb2.set_label(label='Total precipitation between \n 2021080106 and 2022070106 (mm)', fontsize=22)
        cb2.set_label(label='Total precipitation between \n 2021080106 and 2022070106 (mm)', fontsize=14)
        #cb2.ax.tick_params(labelsize=16)
        cb2.ax.tick_params(labelsize=12)
        ax.grid(False)  # Remove grid lines (does not work !)
        #fig.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(savedir, f'CUMUL_ANTILOPE_2021080106_2022070106_{domain}.pdf'), layout='tight')
        #sys.exit()

def nearest(array, value):
    """ Find the closest element of 'array' to 'value'. """
    return float(array[np.abs(array - value).argmin()].data)

def make_mask(field):
    """
    Generates a imultiplicative mask and a ratio estimation field based on local homogeneity of precipitation accumulation

    mask1:
    ------
    Binary mask (*1 or *2) based on the difference (pixel_cumul - max_cumul_in_surroundings) / pixel_cumul with a threshold

    mask2:
    ------
    Directionnal mask looking for the number of "anomalies" in each of the 4 directions (N, E, S, W) to produce a 5-level
    mask (0, 1, 2, 3, 4, 5 anomalies).

    mask3:
    ------
    mask3 = |variability * anomaly| * 10
    where variability = (max - min) / mean and anomalie = (value - mean) / value
    Produces a continuous mask (need to clip max values)
    While variability seems a good criteria, anomalie doesn't identify any interesting pattern

    mask4/bias_estimation:
    ------
    Same as mask3 but uses score information to locally increase/decrease the mask
    bias_estimation is a kriging-like method based on a linear interpolation of all available scores
    weighted by the distance between the pixel and the scores


    mask5/bias_estimation2:
    ------
    Scores are projected in space according to the proximity of
    precipitation accumulation under the assumption that 2 similar accumulation
    not too far away have similar biases.
    ==> This bias estimation is used for both debiasing and increasing the
    observation error.depending on the precipitaion value

    Similarly a homogeneity analysis is used for a static increase of the observation
    error where the heterogeneity of precipitation accumulation is important.

    """

    latmin = field.lat.data.min()
    latmax = field.lat.data.max()
    lonmin = field.lon.data.min()
    lonmax = field.lon.data.max()

    scores = pd.read_csv(fic_score, sep=';')
    scores = scores.loc[(scores.lats>=latmin) & (scores.lats<=latmax) & (scores.lons>=lonmin) & (scores.lons<=lonmax)]


    # loc doit être faible (~10) pour avoir de bons résultats pour l'homogénéité, mais suffisament élevé (optimum vers 25, mauvais à 50) pour ne pas avoir un champs de biais estimé
    # trop bruité (ruptures brutales,...)
    # TODO : régler la localisation de façon dynamique en fonction de l'homogénéité
    loc = 10
    # TODO : WARNING l'homogenetite est fausse sur les bords du domaine (pas le même nombre de pisxels considérés)
    # TODO : il faut normaliser par le nombre de pixels !
    seuil = 0.6
    seuil_homogeneite = 0.1
    null = np.empty((len(field.lat), len(field.lon)))
    empty_field = xr.DataArray(
        name   = 'empty',
        data   = null,
        dims   = ["lat", "lon"],
        coords = dict(lon=field.lon, lat=field.lat),
        #attrs  = dict(description="Difference between each pixel cumul and the max of its neighbours"),
        )
    null[:] = np.nan
    maxdiff  = np.zeros((len(field.lat), len(field.lon)))
    weight = np.zeros((len(field.lat), len(field.lon)))
    weight2 = np.zeros((len(field.lat), len(field.lon)))
    directional_diff = np.zeros((len(field.lat), len(field.lon)))
    estimated_bias = np.zeros((len(field.lat), len(field.lon)))
    estimated_rmse = np.zeros((len(field.lat), len(field.lon)))
    estimated_ratio = np.zeros((len(field.lat), len(field.lon)))
    estimated_ratio2 = null.copy()
    variability = np.zeros((len(field.lat), len(field.lon)))
    variability2 = np.zeros((len(field.lat), len(field.lon)))
    homogeneous_size = null.copy()
    mask5 = null.copy()
    anomaly = np.zeros((len(field.lat), len(field.lon)))
    #for idx,lon in enumerate(field.lon.data[::-1]):
    t1 = time.time()
    t2 = time.time()
    t3 = time.time()
    t4 = time.time()
    for idx,lon in enumerate(field.lon.data):
        print(f'Lon {idx+1}/{len(field.lon)}')
        print(f'reading neighbour values took {(t2-t1)*1000.}ms')
        print(f'reading neighbour scores took {(t3-t2)*1000.}ms')
        print(f'End of the loop took {(t4-t3)*1000.}ms')
#        if lon == 6.2:
        for idy,lat in enumerate(field.lat.data):

            #if lat == 45.10:
            # TODO : si zone homogène mais score mauvais, trouver un moyen d'augmenter le poid
            # TODO : agrandir itérativement la zone de localisation tant que la variabilité reste faible et appliquer
            # le score le plus proche à la plus grande zone homogène possible
            # TODO : Utiliser SPAZM pour estimer le champs de biais.
#            # TODO : essayer un krigeage des scores plutot
#            # On veut estimer le biais et l'erreur du pixel en fonction des 3 points connus les plus proches :
#            #  - Si on est sur un point connu, on prend son bias et son erreur
#            #  - Si on est trop loin (>0.1°) on prend la valeur moyenne
#            #  - sinon on pondère chaque score avec la distance au point
            scores['dist'] = np.sqrt((lat-scores.lats)**2+(lon-scores.lons)**2)
#            if scores.dist.min() <= 0.01:  # On est sur un pixel connu --> on prend ses scores
#                if int(scores[scores['dist']==scores.dist.min()].num_poste) == 5001400:
#                    import pdb
#                    pdb.set_trace()
#            if lon == 6.7 and lat == 45.75:
#                import pdb
#                pdb.set_trace()
#            correlation_dist = seuil / 10.
            correlation_dist = loc / 100. - 0.01  # securité
#            if variability <=0.5:
#                correlation_dist = 0.3  # Si la variabilité locale est faible, on considère que les scores proches sont représentatifs
#            else:
#                correlation_dist = 1  # Sinon on lisse au maximum pour que chaque score n'influe réellement que son voisinage immédiat
            tmp = scores[scores['dist']<correlation_dist]  #select nearest scores (TODO : choix de la distance à valider)
            tmp['inv_dist'] = 1/tmp['dist']
            tmp['lat']=np.round(tmp['lats'], 2)  # TODO : comprendre les WARNINGS
            tmp['lon']=np.round(tmp['lons'], 2)
            tmp['ponderation'] = (tmp.ratio*tmp.inv_dist)/tmp.inv_dist.sum()

            east = np.array(
                    [field.lon.data[idx+dlon] if idx+dlon<len(field.lon.data) else np.nan
                    #[field.lon.data[idx+dlon] if dlon<=emax else np.nan
                        #for dlon in range(1, loc+1)]  # On ne veut pas "lon" dans localisation_lon, ==> en fait si sinon on élimine une colone entière !!
                        for dlon in range(loc)]  # ==> en fait si sinon on élimine une colone entière !!
                )
            east = east[~np.isnan(east)]
            west = np.array(
                    [field.lon.data[idx-dlon] if idx-dlon>=0 else np.nan
                    #[field.lon.data[idx-dlon] if dlon<=wmax else np.nan
                        #for dlon in range(1,loc+1)]  # On ne veut pas "lon" dans localisation_lon
                        for dlon in range(loc)]  # ==> en fait si sinon on élimine une colone entière !!
                )
            west = west[~np.isnan(west)]
            localisation_lon = np.unique(np.concatenate([west, east]))
            #neighbours_lon = field.sel({'lon':localisation_lon})
            north = np.array(
                [field.lat.data[idy+dlat] if idy+dlat<len(field.lat.data) else np.nan
                #[field.lat.data[idy+dlat] if dlat<=nmax else np.nan
                    #for dlat in range(1,loc+1)]  # On ne veut pas "lat" dans localisation_lat
                    for dlat in range(loc)]  # ==> en fait si, sinon on élimine une colone entière !!
            )
            north = north[~np.isnan(north)]
            south = np.array(
                [field.lat.data[idy-dlat] if idy-dlat>=0 else np.nan
                #[field.lat.data[idy-dlat] if dlat<=smax else np.nan
                    #for dlat in range(1, loc+1)]  # On ne veut pas "lat" dans localisation_lat
                    for dlat in range(loc)]  # ==> en fait si, sinon on élimine une colone entière !!
            )
            south = south[~np.isnan(south)]
            localisation_lat = np.unique(np.concatenate([south, north]))
            pixel_value = field.sel({'lon':lon, 'lat':lat}).rr_cumul.data

            if not np.isnan(pixel_value):

                t1 = time.time()
                #neighbours  = field.sel({'lon':localisation_lon, 'lat':localisation_lat}).rr_cumul.data
                neighbours  = field.sel({'lat':localisation_lat, 'lon':localisation_lon})

                # TODO : plot selected area to see if it looks good
                # TODO : calibrer la zone selectionnée pour le point très biaisé des hautes Alpes
                selec = neighbours.where(((neighbours>=pixel_value*(1-seuil_homogeneite)) & (neighbours<=pixel_value*(1+seuil_homogeneite))))  # WARNING : renvoie 0 si pixel_value=np.nan
                #homogeneous_size[idy,idx] = np.count_nonzero(~np.isnan(neighbours.rr_cumul))
                # sécurité pour éviter de mettre des 0 là où il n'y a pas de donnée
                t2 = time.time()
                if not np.isnan(pixel_value):
                    #homogeneous_size[idy,idx] = np.count_nonzero(~np.isnan(selec.rr_cumul)) / np.count_nonzero(~np.isnan(neighbours.rr_cumul))
                    homogeneous_size[idy,idx] = np.count_nonzero(~np.isnan(selec.rr_cumul))
                    mask5[idy,idx] = np.count_nonzero(~np.isnan(neighbours.rr_cumul)) / np.count_nonzero(~np.isnan(selec.rr_cumul))
                    # TODO : selectionner les scores dans la zone d'homogeneite
#                coords =[(ln.lon.data, lt.lat.data) if not np.isnan(selec.sel({'lat':lt, 'lon':ln}).rr_cumul) else np.nan for lt in selec.lat for ln in selec.lon]  # Beaucoup trop long !
#                #if lon == 5.55 and lat == 44.90:
#                print('DBUG1')
#                if len(tmp)>0:
#                    import pdb
#                    pdb.set_trace()
#                    local_scores = tmp[(tmp['lat'], tmp['lon']) in coords]
#                #TODO : continue
                    local_ratio = np.array([])
                    inv_dist = np.array([])
                    tmp['cumul'] = None
                    for ind in tmp.index:
#                        if lat == 45.37 and lon == 5.45 and tmp.lat[ind]==45.46 and tmp.lon[ind]==6.45:
#                            print(lat,lon)
#                            print(tmp.lat[ind], tmp.lon[ind])
#                            import pdb
#                            pdb.set_trace()
                        tmp['cumul'][ind] = neighbours.sel({'lat':tmp.lat[ind], 'lon':tmp.lon[ind]}).rr_cumul.data
                        # On selectionne les scores dont le cumul annuel est proche (+/- i'seuil homogeneite) de celui du pixel traité (hypothèse : le biais peut être considéré comme semblable)
                        if tmp.lat[ind] in selec.lat and tmp.lon[ind] in selec.lon:
                            if not np.isnan(selec.sel({'lat':tmp.lat[ind], 'lon':tmp.lon[ind]}).rr_cumul.data):
                                local_ratio = np.append(local_ratio, tmp.ratio[ind])
                                inv_dist = np.append(inv_dist, tmp.inv_dist[ind])

                    # TODO : revoir l'estimation du ratio pour ne pas introduire de structure spatiale
                    if len(local_ratio)>0:
                        # Si on a des scores dont le cumuls sont proches, on prend la moyenne des scores sans pondération de distance
                        # (l'hypothèse est que sur une zone homogène les scores sont équiprobables)
                        # TODO : ajouter quand même un effet "lissant" pour éviter de faire apparaitre des cercles autour des points avec scores
                        # si la distance de localisation est trop faible (<50)
                        #estimated_ratio2[idy, idx] = np.sum(local_ratio * inv_dist / np.sum(inv_dist))
                        estimated_ratio2[idy, idx] = np.mean(local_ratio)
                    elif len(tmp)>0:
#                    if len(tmp)>0:
                        # Si on a aucun score dont le cumul est similaire, on prend le score dont le cumul
                        # est le plus proche et on pondère par le ratio estre les 2 cumuls.
                        # HYPOTHESE FORTE : le cumul "reel" là où on a pas d'observation indépendante est identique au cumul "réel" le plus proche
                        # - sur le point avec obs : O1=cumul réel, X1=cumul ANTILOPE ==> ratio=X1/O1
                        # - sur le point sans obs : O2=inconnu, X2=cumul ANTILOPE ==> ratio_estimé = X1/O1 * X2/X1 = X2/O1
                        # On sélectionne le score avec le cumul le plus proche
                        tmp['diff_cumul'] = np.abs(tmp['cumul'] - pixel_value)
                        tmp['ratio_cumul'] = pixel_value/tmp['cumul']
                        nearest_cumul = tmp[tmp['diff_cumul']==np.min(tmp['diff_cumul'])]
                        # Le ratio estimé est le produit entre la ratio connu sur le pixel dont le cumul est le plus proche
                        # et le ratio des cumuls
                        if len(nearest_cumul)>1:  # Securité
                            nearest_cumul = nearest_cumul[nearest_cumul['dist']==np.min(nearest_cumul['dist'])]
                        estimated_ratio2[idy, idx] = nearest_cumul.ratio_cumul * nearest_cumul.ratio
                    else:
                        # Si on a aucune information assez proche, on met le ratio moyen
                        estimated_ratio2[idy, idx] = scores.ratio.mean()

                else:
                    homogeneous_size[idy,idx] = np.nan
                    mask5[idy,idx] = np.nan

                neighbours = neighbours.where(~((neighbours.lat==lat)&(neighbours.lon==lon)))  # remove pixel value

                #if (lon == 6.14 and lat == 45.13) or (lon == 6.2 and lat == 45.13):
                if (lon == 6.1 and lat == 45.11) or (lon == 6.2 and lat == 45.13):
                    print(homogeneous_size[idy,idx])
                    fig, ax = plt.subplots(figsize=(12,6))
                    vmin = np.nanmin(field.rr_cumul)
                    vmax = np.nanmax(field.rr_cumul)
                    empty_field.loc[{'lat':selec.lat, 'lon':selec.lon}]=selec.rr_cumul
                    cml = empty_field.plot(ax=ax, cmap=plt.cm.YlGnBu, vmin=vmin, vmax=vmax)
                    add_scores(scores, ax)
                    fig.savefig(os.path.join(savedir, f'zone_homogene_{lon:.2f}_{lat:.2f}.pdf'), format='pdf', layout='tight')


#                maxloc = np.nanmax(neighbours.rr_cumul)
#                minloc = np.nanmin(neighbours.rr_cumul)
#                meanloc = np.nanmean(neighbours.rr_cumul)
#                # TODO : comprendre pourquoi le "trou" en haut à doite augmente...
#                #variability[idy,idx] = (maxloc-minloc)/meanloc  # Measures the local variability in the neighboring
#                #variability2[idy,idx] = maxloc-minloc  # Measures the local variability in the neighboring
#                anomaly[idy,idx] = (pixel_value-meanloc)/pixel_value  # Measures the "anlomaly" on the pixel against its neighbors. WARNING : donne plus de poid aux anomalies <0 !

#                ############################################################################################################
#                # METHDOE D'IDENTIFICATION ITERATIVE
#                ############################################################################################################
#                    totalvar = 0
#                    emax = len(field.lon.data)-1-idx
#                    wmax = idx
#                    nmax = len(field.lat.data)-1-idy
#                    smax = idy
#            maxvar = 1
#            minvar = 1
#            niter = 0
#            # WARNING : very expensive loop !
#            threshold = 0.2
#            while (maxvar>=threshold and minvar>=threshold) or (niter==(loc*loc)/2):  # 1250 = (loc*loc)/2
#            #while totalvar<0.5:
#                maxloc = np.nanmax(neighbours.rr_cumul)
#                minloc = np.nanmin(neighbours.rr_cumul)
#                meanloc = np.nanmean(neighbours.rr_cumul)
#
#                # define creterion for max/min variablity
#                maxvar = (maxloc-meanloc)/meanloc
#                minvar = (meanloc-minloc)/meanloc
#                totalvar = (maxloc-minloc)/meanloc
#                #print(totalvar)
#                #if totalvar>=0.5:
#
#                if maxvar>=threshold:
##                            import pdb
##                            pdb.set_trace()
##                            np.where(neighbours.rr_cumul==maxloc)
##                            neighbours.rr_cumul[np.where(neighbours.rr_cumul==maxloc)]
#                    neighbours = neighbours.where(~(neighbours==maxloc))
##                            maxlat = neighbours.lat[np.where(neighbours.rr_cumul==maxloc)[0]]
##                            maxlon = neighbours.lon[np.where(neighbours.rr_cumul==maxloc)[1]]
##                            import pdb
##                            pdb.set_trace()
##                            neighbours = neighbours.where(~((neighbours.lat==maxlat)&(neighbours.lon==maxlon)))
#
#                if minvar>=threshold:
#                    neighbours = neighbours.where(~(neighbours==minloc))
#
#                niter += 1
#
#            print(niter)
#               ############################################################################################################
#               # METHDOE D'IDENTIFICATION ITERATIVE
#               ############################################################################################################

#                t3 = time.time()
#                if len(tmp)==0:  # Aucune info proche --> on prend les valeurs moyennes
#                    estimated_bias[idy,idx] = scores.biais.mean()
#                    estimated_rmse[idy, idx] = scores.rmse.mean()
#                    estimated_ratio[idy,idx] = scores.ratio.mean()
#                elif tmp.dist.min() <= 0.01:  # On est sur un pixel connu --> on prend ses scores
#                    #print(tmp[tmp['dist']==tmp.dist.min()].num_poste)
##                if int(tmp[tmp['dist']==tmp.dist.min()].num_poste) == 73173400:
##                    import pdb
##                    pdb.set_trace()
#                    estimated_bias[idy,idx] = float(tmp[tmp['dist']==tmp.dist.min()].biais)
#                    estimated_rmse[idy, idx] = float(tmp[tmp['dist']==tmp.dist.min()].rmse)
#                    estimated_ratio[idy,idx] = float(tmp[tmp['dist']==tmp.dist.min()].ratio)
#                else:  # On fait la moyenne des scores pondérée par la distance
#                    #tmp['inv_dist'] = 1/tmp.loc[:,'dist']  # equivalent
#                    tmp['ponderation'] = (tmp.biais*tmp.inv_dist)/tmp.inv_dist.sum()
#                    estimated_bias[idy,idx] = tmp.ponderation.sum()
#                    tmp['ponderation'] = (tmp.rmse*tmp.inv_dist)/tmp.inv_dist.sum()
#                    estimated_rmse[idy,idx] = tmp.ponderation.sum()
#                    tmp['ponderation'] = (tmp.ratio*tmp.inv_dist)/tmp.inv_dist.sum()
#                    estimated_ratio[idy,idx] = tmp.ponderation.sum()
#
#                # TODO : trouver un moyen pour que "weight"/"weight2" soit normalisé entre 1 et 10 par exemple
#                # TODO : faire un porduit matricielle en dehors de la boucle maintenant que variability et anomaly sont des matrices
#                #weight[idy,idx] = variability[idy,idx] * anomaly[idy, idx]
#                #weight[idy,idx] = anomaly[idy, idx] / homogeneous_size[idy,idx]
#                weight[idy,idx] = anomaly[idy, idx] * homogeneous_size[idy,idx]
#                if estimated_ratio[idy,idx] >= 1:
#                    weight2[idy,idx] = variability[idy,idx] * anomaly[idy,idx] * estimated_ratio[idy,idx]
#                else:
#                    weight2[idy,idx] = variability[idy,idx] * anomaly[idy,idx] / estimated_ratio[idy,idx]
#                #weight[idy,idx] = variability*np.abs(maxloc-pixel_value)*np.abs(pixel_value-minloc)  # TODO : add a pondertion according to neigboring ratios ?
#
##            if tmp.dist.min() <= 0.01:  # On est sur un pixel connu --> on prend ses scores
##                print(tmp[tmp['dist']==tmp.dist.min()].num_poste)
##                if int(tmp[tmp['dist']==tmp.dist.min()].num_poste) == 5001400:
##                    import pdb
##                    pdb.set_trace()
##            # To see the result around Alpe d'Huez
##            if lon == 6.1 and lat == 45.11:
#                import pdb
##                pdb.set_trace()
#
#                #meandiff[idx,idy] = pixel_value-np.mean(neighbours)
#                maxdiff[idy,idx] = (pixel_value-maxloc)/pixel_value
#
#                # TODO : considérer 4 max (1 par cadran par exemple) pour éviter de fausser les résultats autour d'un pixel isolé très arrosé
#                # et durcir le seuil (-0.4 par exemple)
##            westloc = field.sel({'lon':west, 'lat':lat}).rr_cumul.data if len(west)>0 else np.array([])
##            eastloc = field.sel({'lon':east, 'lat':lat}).rr_cumul.data if len(east)>0 else np.array([])
##            northloc = field.sel({'lon':lon, 'lat':north}).rr_cumul.data if len(north)>0 else np.array([])
##            southloc = field.sel({'lon':lon, 'lat':south}).rr_cumul.data if len(south)>0 else np.array([])
##            westmax = (pixel_value-np.max(westloc))/pixel_value if len(west)>0 else np.nan
##            eastmax = (pixel_value-np.max(eastloc))/pixel_value if len(east)>0 else np.nan
##            northmax = (pixel_value-np.max(northloc))/pixel_value if len(north)>0 else np.nan
##            southmax = (pixel_value-np.max(southloc))/pixel_value if len(south)>0 else np.nan
##            tmp = np.array([westmax,eastmax,northmax,southmax])
##            tmp = tmp[~np.isnan(tmp)]
##            directional_diff[idy,idx] = np.count_nonzero(tmp<-0.1) + 1
#                #print(f'End of the loop took {(t4-t3)*1000.}ms')
#
#                # To see the result for the max of the field
##            if lon == 6.04 and lat == 45.20:
##                import pdb
##                pdb.set_trace()
#
#                t4 = time.time()


#    maskarray = np.where(maxdiff>-seuil, 1, 2)
#    mask1 = to_xarray(maskarray, 'mask')
#    plot_and_save(mask1, f'mask1_loc{loc}_seuil{seuil}_{domain}')
#
#    mask2 = to_xarray(directional_diff, 'mask')
#    plot_and_save(mask2, f'mask2_loc{loc}_seuil{seuil}_{domain}')
#
##    weight = weight*10
#    weight_array = to_xarray(weight, 'weight')
#    plot_and_save(weight_array, f'weight_loc{loc}_{domain}', cmap=plt.cm.YlOrBr)
#
#    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#    # TODO : le mask doit être clippé à 1 (c'est un facteur !)
#    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#    #weight = np.clip(np.abs(weight).clip(0.1)*10, 1, 5)
#    #weight = np.clip(np.abs(weight), 1, 10)
#    weight = np.abs(weight)*10
#    mask3 = to_xarray(weight, 'mask')
#    plot_and_save(mask3, f'mask3_loc{loc}_{domain}')
#
#    weight2_array = to_xarray(weight2, 'weight')
#    plot_and_save(weight2_array, f'weight2_loc{loc}_{domain}', cmap=plt.cm.YlOrBr)
#
#    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#    # TODO : le mask doit être clippé à 1 (c'est un facteur !)
#    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#    weight2 = np.clip(np.abs(weight2).clip(0.1)*10, 1, 5)
#    mask4 = to_xarray(weight2, 'mask')
#    plot_and_save(mask4, f'mask4_loc{loc}_{domain}')
#
#    ratio = to_xarray(estimated_ratio, 'ratio')
#    plot_and_save(ratio, f'estimated_ratio_{domain}', cmap=plt.cm.coolwarm)
#
#    ratio2 = to_xarray(estimated_ratio2, 'ratio')
#    vmin = np.nanmin(ratio2)
#    vmax = 2 - vmin
#    plot_and_save(ratio2, f'estimated_ratio2_loc{loc}_seuil_{seuil_homogeneite}_{domain}', cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax)
#    #plot_and_save(ratio2, f'estimated_ratio2_loc{loc}_seuil_{seuil_homogeneite}_{domain}', cmap='custom', vmin=vmin, vmax=vmax)

#    variability = to_xarray(variability, 'variability')
#    plot_and_save(variability, f'variability_loc{loc}_{domain}', cmap=plt.cm.coolwarm)
#
#    variability2 = to_xarray(variability2, 'variability')
#    plot_and_save(variability2, f'variability2_loc{loc}_{domain}', cmap=plt.cm.coolwarm)
#
#    anomaly = to_xarray(anomaly, 'anomaly')
#    plot_and_save(anomaly, f'anomaly_loc{loc}_{domain}', cmap=plt.cm.coolwarm)

    homegeneity = to_xarray(homogeneous_size, field, 'homogeneity')
    #plot_and_save(homegeneity, f'homogeneity_loc{loc}_seuil{seuil_homogeneite}_{domain}', cmap=plt.cm.PuOr)  # TODO : change colorbar (not diverging)
    plot_and_save(homegeneity, f'homogeneity_loc{loc}_seuil{seuil_homogeneite}_{domain}', cmap=plt.cm.YlOrBr)  # TODO : change colorbar (not diverging)

    #mask5 = to_xarray(1+mask5*20/np.nanmax(mask5))  # On "normalise" entre 1 et 20 pour avoir une erreur d'observation comprise entre 0.261mm et 0.261*20~5mm
    # TODO : augmenter l'erreur d'observation
    mask5 = to_xarray(np.clip(mask5, np.nanmin(mask5), 20), field, 'mask')  # On plafonne (arbitrairement) le masque à 20 pour avoir une erreur d'observation comprise entre 0.261mm et 0.261*20~5mm
    # lorsque ANTILOPE observe 0mm
    plot_and_save(mask5, f'mask5_loc{loc}_seuil{seuil_homogeneite}_{domain}', cmap=plt.cm.Greys)  # TODO : change colorbar (not diverging)

    # TODO : USe a PuOr colorbar for creterion fields and Grey colorbar for mask fields

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
    elif domain == 'alp':
        fig, ax = plt.subplots(figsize=(14,16))
    else:
        fig, ax = plt.subplots()

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
            add_scores(scores, ax, mycmap=cmap, vmin=vmin, vmax=vmax)
        else:
            add_scores(scores, ax)
    add_radar_positions(ax)
    add_boundaries()
    #add_cities(latmin, latmax, lonmin, lonmax)
    cb = fig.colorbar(cml)
    cb.set_label(field.name, fontsize=24)
    cb.ax.tick_params(labelsize=20)
    fig.savefig(os.path.join(savedir, f'{name}.pdf'), format='pdf', layout='tight')
    field.to_netcdf(os.path.join(savedir, f'{name}.nc'))

def ratio_estimation(field, moving_window=15):
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

    lons, lats = np.meshgrid(field.lon.data, field.lat.data)
    estimated_ratio = np.ones(np.shape(field.rr_cumul.data))*scores.ratio.mean()
    for poste in scores.index:
        #print(scores.loc[poste])
        ratio = scores.loc[poste, 'ratio']
        dist = np.sqrt((lats-scores.loc[poste,'lats'])**2+(lons-scores.loc[poste, 'lons'])**2)  # Euclidian horizontal distance
        idx, idy = np.where(dist==np.min(dist))
        ref_cumul = field.rr_cumul.data[idx[0],idy[0]]
        cumul_dist = field.rr_cumul.data-ref_cumul
        cumul_ratio = field.rr_cumul.data/ref_cumul
        #estimated_ratio = estimated_ratio+(cumul_ratio-estimated_ratio)*np.exp(-np.abs(cumul_dist)/(ref_cumul/5))*np.exp(-dist/0.3)  # Marche bien mais n'utilise pas le score !
        #estimated_ratio = estimated_ratio+((cumul_ratio-estimated_ratio)*np.exp(-np.abs(cumul_dist)/(ref_cumul))-estimated_ratio)*np.exp(-dist/0.2)  # DERIVE
        #pond = np.exp(-dist/0.3)
        #pond[np.where(dist>0.5)]=0
        #estimated_ratio = estimated_ratio+(ratio-estimated_ratio)*np.exp(-np.abs(cumul_dist)/(ref_cumul/10))*np.exp(-dist/1)  # PAS MAL
        d0 = 0.25
        c0 = 1
        #estimated_ratio = estimated_ratio+(ratio*cumul_ratio-estimated_ratio)*np.exp(-np.abs(cumul_dist)/(ref_cumul/c0))*np.exp(-dist/d0)  # Marche bien avec d0=0.4 et c0=5
        #estimated_ratio = estimated_ratio+(ratio-estimated_ratio)*np.exp(-np.abs(cumul_dist)/(ref_cumul/c0))*np.exp(-dist/d0)  # TEST
        #estimated_ratio = estimated_ratio+(ratio*cumul_ratio-estimated_ratio)*np.exp(-dist/d0)  # TEST
        #estimated_ratio = estimated_ratio+(ratio*smoothratio-estimated_ratio)*np.exp(-dist/d0)*np.exp(-np.abs(cumul_dist)/(ref_cumul/c0)) # TEST
        estimated_ratio = estimated_ratio+(ratio*cumul_ratio-estimated_ratio)*np.exp(-dist/d0)*np.exp(-np.abs(cumul_dist)/(ref_cumul/c0))

    #TODO : TMP
    #mean_ratio = uniform_filter(estimated_ratio, size=moving_window)
    #estimated_ratio = estimated_ratio - mean_ratio
    ratio_field = to_xarray(estimated_ratio, field)
    #observation_error = ((np.abs(ratio_field-1) + np.abs(diff))**2)*10

    #observation_error = (np.abs(ratio_field-1)*5 + 5*np.abs(diff))**2
    observation_error = 1+np.abs(ratio_field-1)
    #observation_error = 1+np.abs(smoothratio-1)

    #observation_error = np.abs(ratio_field-1)
    #observation_error = np.abs(ratio_field**2-1)*50
    #observation_error = np.exp((observation_error-0.3))
#    plt.hist(observation_error, bins = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55])
#    plt.title("histogram")
#    plt.show()

    # PLots
    #######
    if not domain == 'alp':
        scores = scores.loc[(scores.lats>=latmin) & (scores.lats<=latmax) & (scores.lons>=lonmin) & (scores.lons<=lonmax)]
    #plot_and_save(ratio_field, f'Estimated_ratio_{domain}_{d0}_{moving_window}', vmin=0.5, vmax=1.5, cmap=plt.cm.coolwarm, scores=scores)
    #plot_and_save(ratio_field, f'Estimated_ratio_{domain}_{d0}', vmin=0.5, vmax=1.5, cmap=plt.cm.coolwarm, scores=scores)
    plot_and_save(ratio_field, f'Estimated_ratio_{domain}_{d0}_{c0}', vmin=0.5, vmax=1.5, cmap=plt.cm.coolwarm, scores=scores)

    plot_and_save(smoothed, f'Smoothed_field_{moving_window}_{domain}', cmap=plt.cm.YlGnBu)
    plot_and_save(diff, f'Observation_error_smoothingsize{moving_window}_{domain}', cmap=plt.cm.coolwarm)
    plot_and_save(np.abs(diff), f'Observation_error_absolute_value_smoothingsize{moving_window}_{domain}', cmap=plt.cm.Greys, scores=scores)
    plot_and_save(smoothratio, f'Observation_error_ratio_smoothingsize{moving_window}_{domain}', vmin=0.6, vmax=1.4, cmap=plt.cm.coolwarm, scores=scores)
    plot_and_save(observation_error, f'Observation_error_{d0}_{c0}_{domain}', cmap=plt.cm.Greys, scores=scores)
    #plot_and_save(observation_error, f'Observation_error_{moving_window}_{d0}_{domain}', vmin=0, vmax=30, cmap=plt.cm.Greys, scores=scores)


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
    add_scores(scores)
    fig.savefig(os.path.join(savedir, f'kriging_ratio.pdf'), format='pdf', layout='tight')

def plot_field(field, scores):
    lt.subplots(figsize=(14,16))
    field.rr_cumul.plot(ax=ax, cmap=plt.cm.coolwarm)
    add_scores(scores)
    fig.savefig(os.path.join(datadir, filename.replace('.nc', '.pdf')), format='pdf', layout='tight')

if __name__ == "__main__":
    if domain == 'GrandesRousses':
        filename = 'CUMUL_ANTILOPEH_GrandesRousses_2021073106_2022070106.nc'
    else:
        filename ='CUMUL_ANTILOPEH_alp_2021103000_2022060200.nc'
    antilope = xr.open_dataset(os.path.join(datadir, filename))
    if not domain == 'alp':
        antilope = antilope.where((antilope.lon>=lonmin) & (antilope.lon<=lonmax) & (antilope.lat<=latmax) & (antilope.lat>=latmin), drop=True)

    #plot(antilope, categories=False, baiscorrection=True)
    #plot(antilope, categories=False)

    ratio_estimation(antilope)

#    krigeage_scores(antilope)
#    make_mask(antilope)
#    ratio_estimation(antilope)
    #moving_average(antilope)





