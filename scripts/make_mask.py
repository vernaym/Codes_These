#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os, sys
from datetime import datetime,timedelta
import numpy as np
import xarray as xr
import pandas as pd

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

landmarks = {
        "Alpe d'Huez" : dict(lon=6.070, lat=45.092, alt=1800, marker='o'),
        "Les 2 Alpes" : dict(lon=6.127, lat=45.013, alt=1800, marker='o'),
        "Lautaret"    : dict(lon=6.408, lat=45.038, alt=2058, marker='X'),
        "La Meije"    : dict(lon=6.311, lat=45.008, alt=3500, marker='^'),  # real alt = 3984
        "Pic Blanc"   : dict(lon=6.131, lat=45.128, alt=3000, marker='^'),  # real alt = 3333
    }

# Domaine des Grandes Rousses
extract_dom = dict(
    latmax = 45.240,
    latmin = 44.990,
    lonmin = 6.010,
    lonmax = 6.490,
)
latmin = extract_dom['latmin']
lonmin = extract_dom['lonmin']
latmax = extract_dom['latmax']
lonmax = extract_dom['lonmax']

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

def add_scores(scores):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "blue", "green", "orange", "red"], 5)
    thresholds = [0., 0.5, 0.80, 1.2, 1.5, 10]  # TODO : vérier la coéhrence des seuils entre les figures
    #thresholds = [0., 0.5, 0.90, 1.1, 1.5, 10]
    norm = matplotlib.colors.BoundaryNorm(thresholds, cmap.N)

    def set_marker(row):
        if row['ratio'] <= 0.8:
        #if row['ratio'] <= 0.9:  # TODO : vérier la coéhrence du seuil entre les figures
            return 'v'
        elif row['ratio'] > 0.8 and row['ratio'] < 1.2:
        #elif row['ratio'] > 0.9 and row['ratio'] < 1.1:  # TODO : vérier la coéhrence du seuil entre les figures
            return 'o'
        else:
            return '^'
    scores["marker"] = scores.apply(set_marker, axis=1)  # axis=1 makes sure that function is applied to each row
    for marker, info in scores.groupby('marker'):
        sc = plt.scatter(info['lons'], info['lats'], c=info['ratio'], cmap=cmap, norm=norm, marker=marker, s=150, edgecolors='black')
        labels = [str(num_poste) for num_poste in info['num_poste']]
        for idx, label in enumerate(labels):
            txt = plt.text(info['lons'].data[idx], info['lats'].data[idx], label)

def add_landmarks(ax):
    # Add landmarks
    for landmark, infos in landmarks.items():
        ax.plot(infos['lon'], infos['lat'], marker=infos['marker'], color='red', markersize=5)
        ax.annotate(landmark, (infos['lon']+0.003, infos['lat']+0.003), color='red', fontsize=12)

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

def plot(antilope):

    if not os.path.exists(os.path.join(savedir, f'CUMUL_ANTILOPE_2021080106_2022070106_{domain}.pdf')):

        if domain == 'alp':
            fig, ax = plt.subplots(figsize=(14,16))
        elif domain == 'GrandesRousses':
            fig, ax = plt.subplots(figsize=(12,6))

        antilope.rr_cumul.plot(ax=ax, cbar_kwargs={"label":'Total precipitation between 2021080106 and 2022070106 (mm)'}, cmap=plt.cm.coolwarm)
        add_landmarks(ax)
        add_radar_positions(ax)
        fic_score = os.path.join(datadir, 'scores_2021110106_2022043006_alpes_10.csv')
        scores = pd.read_csv(fic_score, sep=';')
        add_scores(scores)
        fig.tight_layout()
        fig.savefig(os.path.join(savedir, f'CUMUL_ANTILOPE_2021080106_2022070106_{domain}.pdf'))

def nearest(array, value):
    """ Find the closest element of 'array' to 'value'. """
    return float(array[np.abs(array - value).argmin()].data)

def make_mask(field):

    fic_score = os.path.join(datadir, 'scores_2021110106_2022043006_alpes_10.csv')
    scores = pd.read_csv(fic_score, sep=';')

    loc = 10
    seuil = 0.6
    #null = np.empty((len(field.lat), len(field.lon)))
    maxdiff  = np.zeros((len(field.lat), len(field.lon)))
    weight = np.zeros((len(field.lat), len(field.lon)))
    weight2 = np.zeros((len(field.lat), len(field.lon)))
    directional_diff = np.zeros((len(field.lat), len(field.lon)))
    estimated_bias = np.zeros((len(field.lat), len(field.lon)))
    estimated_rmse = np.zeros((len(field.lat), len(field.lon)))
    estimated_ratio = np.zeros((len(field.lat), len(field.lon)))
    #for idx,lon in enumerate(field.lon.data[::-1]):
    for idx,lon in enumerate(field.lon.data):
        print(f'Lon {idx+1}/{len(field.lon)}')
        east = np.array(
                [field.lon.data[idx+dlon] if idx+dlon<len(field.lon.data) else np.nan
                    for dlon in range(1, loc+1)]  # On ne veut pas "lon" dans localisation_lon
            )
        east = east[~np.isnan(east)]
        west = np.array(
                [field.lon.data[idx-dlon] if idx-dlon>=0 else np.nan
                    for dlon in range(1, loc+1)]  # On ne veut pas "lon" dans localisation_lon
            )
        west = west[~np.isnan(west)]
        localisation_lon = np.concatenate([west, east])
        for idy,lat in enumerate(field.lat.data):

#            # TODO : essayer un krigeage des scores plutot
#            # On veut estimer le biais et l'erreur du pixel en fonction des 3 points connus les plus proches :
#            #  - Si on est sur un point connu, on prend son bias et son erreur
#            #  - Si on est trop loin (>0.1°) on prend la valeur moyenne
#            #  - sinon on pondère chaque score avec la distance au point
            scores['dist'] = np.sqrt((lat-scores.lats)**2+(lon-scores.lons)**2)
            tmp = scores[scores['dist']<=1]  #select nearest scores (TODO : choix de la distance à valider)
            if len(tmp)==0:  # Aucune info proche --> on prend les valeurs moyennes
                estimated_bias[idy,idx] = scores.biais.mean()
                estimated_rmse[idy, idx] = scores.rmse.mean()
                estimated_ratio[idy,idx] = scores.ratio.mean()
            elif tmp.dist.min() <= 0.01:  # On est sur un pixel connu --> on prend ses scores
                print(tmp[tmp['dist']==tmp.dist.min()].num_poste)
#                if int(tmp[tmp['dist']==tmp.dist.min()].num_poste) == 73173400:
#                    import pdb
#                    pdb.set_trace()
                estimated_bias[idy,idx] = float(tmp[tmp['dist']==tmp.dist.min()].biais)
                estimated_rmse[idy, idx] = float(tmp[tmp['dist']==tmp.dist.min()].rmse)
                estimated_ratio[idy,idx] = float(tmp[tmp['dist']==tmp.dist.min()].ratio)
            else:  # On fait la moyenne des scores pondérée par la distance
                tmp['inv_dist'] = 1/tmp['dist']
                tmp['ponderation'] = (tmp.biais*tmp.inv_dist)/tmp.inv_dist.sum()
                estimated_bias[idy,idx] = tmp.ponderation.sum()
                tmp['ponderation'] = (tmp.rmse*tmp.inv_dist)/tmp.inv_dist.sum()
                estimated_rmse[idy,idx] = tmp.ponderation.sum()
                tmp['ponderation'] = (tmp.ratio*tmp.inv_dist)/tmp.inv_dist.sum()
                estimated_ratio[idy,idx] = tmp.ponderation.sum()
##            if lon == 6.14 and lat == 45.13:
##                import pdb
##                pdb.set_trace()
            north = np.array(
                [field.lat.data[idy+dlat] if idy+dlat<len(field.lat.data) else np.nan
                    for dlat in range(1, loc+1)]  # On ne veut pas "lat" dans localisation_lat
            )
            north = north[~np.isnan(north)]
            south = np.array(
                [field.lat.data[idy-dlat] if idy-dlat>=0 else np.nan
                    for dlat in range(1, loc+1)]  # On ne veut pas "lat" dans localisation_lat
            )
            south = south[~np.isnan(south)]
            localisation_lat = np.concatenate([south, north])
#            localisation_lat = np.array(
#                [field.lat.data[idy+dlat] if idy+dlat>=0 and idy+dlat<len(field.lat.data) and dlat!=0 else np.nan  # On ne veut pas "lon" dans localisation_lat
#                    for dlat in range(-loc,loc+1)]
#            )
            pixel_value = field.sel({'lon':lon, 'lat':lat}).rr_cumul.data
            neighbours  = field.sel({'lon':localisation_lon, 'lat':localisation_lat}).rr_cumul.data
            maxloc = np.max(neighbours)
            minloc = np.min(neighbours)
            meanloc = np.mean(neighbours)
            variability = (maxloc-minloc)/meanloc  # Measures the local variability in the neighboring
            anomaly = (pixel_value-meanloc)/pixel_value  # Measures the "anlomaly" on the pixel against its neighbors
            weight[idy,idx] = variability * anomaly
            if estimated_ratio[idy,idx] >= 1:
                weight2[idy,idx] = variability * anomaly * estimated_ratio[idy,idx]
            else:
                weight2[idy,idx] = variability * anomaly / estimated_ratio[idy,idx]
            #weight[idy,idx] = variability*np.abs(maxloc-pixel_value)*np.abs(pixel_value-minloc)  # TODO : add a pondertion according to neigboring ratios ?

#            if tmp.dist.min() <= 0.01:  # On est sur un pixel connu --> on prend ses scores
#                print(tmp[tmp['dist']==tmp.dist.min()].num_poste)
#                if int(tmp[tmp['dist']==tmp.dist.min()].num_poste) == 5001400:
#                    import pdb
#                    pdb.set_trace()
#            # To see the result around Alpe d'Huez
#            if lon == 6.1 and lat == 45.11:
#                import pdb
#                pdb.set_trace()

            #meandiff[idx,idy] = pixel_value-np.mean(neighbours)
            maxdiff[idy,idx] = (pixel_value-maxloc)/pixel_value


            # TODO : considérer 4 max (1 par cadran par exemple) pour éviter de fausser les résultats autour d'un pixel isolé très arrosé
            # et durcir le seuil (-0.4 par exemple)
            westloc = field.sel({'lon':west, 'lat':lat}).rr_cumul.data if len(west)>0 else np.array([])
            eastloc = field.sel({'lon':east, 'lat':lat}).rr_cumul.data if len(east)>0 else np.array([])
            northloc = field.sel({'lon':lon, 'lat':north}).rr_cumul.data if len(north)>0 else np.array([])
            southloc = field.sel({'lon':lon, 'lat':south}).rr_cumul.data if len(south)>0 else np.array([])
            westmax = (pixel_value-np.max(westloc))/pixel_value if len(west)>0 else np.nan
            eastmax = (pixel_value-np.max(eastloc))/pixel_value if len(east)>0 else np.nan
            northmax = (pixel_value-np.max(northloc))/pixel_value if len(north)>0 else np.nan
            southmax = (pixel_value-np.max(southloc))/pixel_value if len(south)>0 else np.nan
            tmp = np.array([westmax,eastmax,northmax,southmax])
            tmp = tmp[~np.isnan(tmp)]
            directional_diff[idy,idx] = np.count_nonzero(tmp<-0.1)

            # To see the result for the max of the field
#            if lon == 6.04 and lat == 45.20:
#                import pdb
#                pdb.set_trace()


    output = xr.DataArray(
        name   = 'maxdiff',
        data   = maxdiff,
        dims   = ["lat", "lon"],
        coords = dict(lon=field.lon, lat=field.lat),
        attrs  = dict(description="Difference between each pixel cumul and the max of its neighbours"),
    )
    #output.plot()
    maskarray = np.where(maxdiff>-seuil, 1, 2)
    mask1 = xr.DataArray(
        name   = 'mask',
        data   = maskarray,
        dims   = ["lat", "lon"],
        coords = dict(lon=field.lon, lat=field.lat),
    )
    if domain == 'alp':
        fig, ax = plt.subplots(figsize=(14,16))
    elif domain == 'GrandesRousses':
        fig, ax = plt.subplots(figsize=(12,6))
    mask1.plot(ax=ax, cmap=plt.cm.Greys)
    add_scores(scores)
    add_radar_positions(ax)
    name = f'mask1_loc{loc}_seuil{seuil}_{domain}'
    fig.savefig(os.path.join(savedir, f'{name}.pdf'), format='pdf', layout='tight')
    mask1.to_netcdf(os.path.join(savedir, f'{name}.nc'))

    mask2 = xr.DataArray(
        name   = 'mask',
        data   = directional_diff,
        dims   = ["lat", "lon"],
        coords = dict(lon=field.lon, lat=field.lat),
    )
    if domain == 'alp':
        fig, ax = plt.subplots(figsize=(14,16))
    elif domain == 'GrandesRousses':
        fig, ax = plt.subplots(figsize=(12,6))
    mask2.plot(ax=ax, cmap=plt.cm.Greys)
    add_scores(scores)
    add_radar_positions(ax)
    name = f'mask2_loc{loc}_seuil{seuil}_{domain}'
    fig.savefig(os.path.join(savedir, f'{name}.pdf'), format='pdf', layout='tight')
    mask2.to_netcdf(os.path.join(savedir, f'{name}.nc'))

    weight = xr.DataArray(
        name   = 'mask',
        data   = weight,
        dims   = ["lat", "lon"],
        coords = dict(lon=field.lon, lat=field.lat),
    )
    if domain == 'alp':
        fig, ax = plt.subplots(figsize=(14,16))
    elif domain == 'GrandesRousses':
        fig, ax = plt.subplots(figsize=(12,6))
    weight.plot(ax=ax, cmap=plt.cm.PuOr)
    add_scores(scores)
    add_radar_positions(ax)
    name = f'weight_loc{loc}_{domain}'
    fig.savefig(os.path.join(savedir, f'{name}.pdf'), format='pdf', layout='tight')
    weight.to_netcdf(os.path.join(savedir, f'{name}.nc'))

    # Transform weight into a mask factor
#    weight = np.where(np.abs(weight)>=0.3, 4, weight)
#    weight = np.where((np.abs(weight)>=0.2) & (np.abs(weight)<0.3), 3, weight)
#    weight = np.where((np.abs(weight)>=0.1) & (np.abs(weight)<0.2), 2, weight)
#    weight = np.where(np.abs(weight)<0.1, 1, weight)
    weight = np.clip(np.abs(weight)*10, 0, 10)
    mask3 = xr.DataArray(
        name   = 'mask',
        data   = weight,
        dims   = ["lat", "lon"],
        coords = dict(lon=field.lon, lat=field.lat),
    )
    if domain == 'alp':
        fig, ax = plt.subplots(figsize=(14,16))
    elif domain == 'GrandesRousses':
        fig, ax = plt.subplots(figsize=(12,6))
    #mask3.plot(ax=ax, cmap=plt.cm.YlOrBr)
    mask3.plot(ax=ax, cmap=plt.cm.Greys)
    add_scores(scores)
    add_radar_positions(ax)
    name = f'mask3_loc{loc}_{domain}'
    fig.savefig(os.path.join(savedir, f'{name}.pdf'), format='pdf', layout='tight')
    mask3.to_netcdf(os.path.join(savedir, f'{name}.nc'))

    ratio = xr.DataArray(
        name   = 'ratio',
        data   = estimated_ratio,
        dims   = ["lat", "lon"],
        coords = dict(lon=field.lon, lat=field.lat),
    )
    if domain == 'alp':
        fig, ax = plt.subplots(figsize=(14,16))
    elif domain == 'GrandesRousses':
        fig, ax = plt.subplots(figsize=(12,6))
    ratio.plot(ax=ax, cmap=plt.cm.coolwarm)
    add_scores(scores)
    fig.savefig(os.path.join(savedir, f'estimated_ratio_{domain}.pdf'), format='pdf', layout='tight')

#    estimated_ratio = np.where(estimated_ratio>=1.5, 4, estimated_ratio)
#    estimated_ratio = np.where((estimated_ratio>1.1) & (estimated_ratio<1.5), 3, estimated_ratio)
#    estimated_ratio = np.where((estimated_ratio>=0.90) & (estimated_ratio<=1.1), 2, estimated_ratio)
#    estimated_ratio = np.where((estimated_ratio>0.5) & (estimated_ratio<0.9), 1, estimated_ratio)
#    estimated_ratio = np.where(estimated_ratio<=0.5, 0, estimated_ratio)

    ratio = xr.DataArray(
        name   = 'ratio_category',
        data   = estimated_ratio,
        dims   = ["lat", "lon"],
        coords = dict(lon=field.lon, lat=field.lat),
    )
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "blue", "green", "orange", "red"], 5)
    thresholds = [0., 0.5, 0.80, 1.2, 1.5, 10]
    norm = matplotlib.colors.BoundaryNorm(thresholds, cmap.N)
    if domain == 'alp':
        fig, ax = plt.subplots(figsize=(14,16))
    elif domain == 'GrandesRousses':
        fig, ax = plt.subplots(figsize=(12,6))
    ratio.plot(ax=ax, cmap=cmap, norm=norm)
    add_scores(scores)
    fig.savefig(os.path.join(savedir, f'estimated_ratio_categories_{domain}.pdf'), format='pdf', layout='tight')

def krigeage_scores(field):
    variogram  = 'exponential'  # The same as for ANTILOPE without RADAR data

    fic_score = os.path.join(datadir, 'scores_2021110106_2022043006_alpes_10.csv')
    scores = pd.read_csv(fic_score, sep=';')
    data = np.array(
    [
        [0.3, 1.2, 0.47],
        [1.9, 0.6, 0.56],
        [1.1, 3.2, 0.74],
        [3.3, 4.4, 1.47],
        [4.7, 3.8, 1.74],
    ])

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
    if domain == 'alp':
        filename ='CUMUL_ANTILOPEH_alp_2021103000_2022060200.nc'
    elif domain == 'GrandesRousses':
        filename = 'CUMUL_ANTILOPEH_GrandesRousses_2021073106_2022070106.nc'
    antilope = xr.open_dataset(os.path.join(datadir, filename))
    if domain == 'GrandesRousses':
        antilope = antilope.where((antilope.lon>=lonmin) & (antilope.lon<=lonmax) & (antilope.lat<=latmax) & (antilope.lat>=latmin), drop=True)

    plot(antilope)
#    krigeage_scores(antilope)
    make_mask(antilope)





