#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os
import datetime
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import argparse

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

from snowtools.plots.maps import cartopy


##############################################################################################
# Ce script sert permet de comparer les données ANTILOPE extraites de la BDAP par le script
# "Extraction_ANTILOPE_bdap.py aux observations des postes du réseau nivométéo correspondants 
# extraites par le script "extract_obs_nivometeo.py" de snowtools dans le but d'évaluer le 
# biais d'ANTILOPE avec l'altitude.
##############################################################################################

map_massifs = dict(
    alpes = [*range(1, 24)],
    pyrenees = [*range(64, 75), *range(80, 92)],
    corse = [40, 41],
    )

def parse_command_line():
    description = "BDAP extraction of ANTILOPE data."
    parser = argparse.ArgumentParser(description=description)
    #parser.add_argument('-d', '--rundate', help='Rundate for operational executions, format YYMMDDHH')
    parser.add_argument('-b', '--datebegin', help='Begining date of extraction, format YYYYMMDDHH or YYMMDDHH', required=True)
    parser.add_argument('-e', '--dateend', help = 'Final date of extraction (default=datebegin)')
    parser.add_argument('-w', '--workdir', help='Runing directory (default for guppy)', default='/home/mrns/vernaym/workdir/extraction_antilope')
    parser.add_argument('-m', '--massif', help='PLot for a specific massif', default=None, type=int)

    args = parser.parse_args()

    args.datebegin = get_date(args.datebegin)
    if args.dateend:
        args.dateend = get_date(args.dateend)
    else:
        args.dateend = args.datebegin + timedelta(hours=24)

    return args

def nivologyseason(date):
    """Return the nivology season of a current date"""
    if date.month < 8:
        season_begin = datetime(date.year - 1, 8, 1)
        season_end   = datetime(date.year, 7, 31)
    else:
        season_begin = datetime(date.year, 8, 1)
        season_end   = datetime(date.year + 1, 7, 31)

    return season_begin.strftime('%y') + season_end.strftime('%y')

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

def predict(x):
   return slope * x + intercept       

def raw_scatterplot(rr_nivometeo, rr_antilope, elevations, datebegin, dateend):
    nbpoint = len(rr_nivometeo)
    fig = plt.figure(figsize=(10,6))
    #plt.scatter(rr_nivometeo.to_numpy(), rr_antilope.to_numpy(), s=nb_values.to_numpy(), marker='o')
    sc = plt.scatter(rr_nivometeo, rr_antilope, marker='D', s=10, c=elevations)
    plt.text(1000, 6000, f'{nbpoint} stations', fontsize=18, color='black')
    plt.colorbar(sc, label='Elevation')
    xpoints = ypoints = plt.xlim()
    plt.grid(visible=True, linestyle=':', linewidth=0.5)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.plot(xpoints, ypoints, linestyle='--', color='grey', lw=1, scalex=False, scaley=False)
    plt.ylabel('Total ANTILOPE precipitation estimate (mm)', fontsize=12)
    plt.xlabel('Total rain-gauges observed precipitation (mm)', fontsize=12)

    # Regression linéaire
    reg = LinearRegression().fit(rr_nivometeo.reshape((-1, 1)), rr_antilope)
    y = reg.predict(rr_nivometeo.reshape((-1,1)))
    r2 = r2_score(rr_nivometeo.reshape((-1, 1)), rr_antilope)
    plt.plot(rr_nivometeo, y, color="red", linewidth=1)
    plt.text(3500, 2100, f'R²={r2:.4}', fontsize=18, color='red')

    plt.tight_layout()
    #fig.savefig('raw_scatterplot_{0:s}_{1:s}.pdf'.format(datebegin.strftime('%Y%m%d'), dateend.strftime('%Y%m%d')), bbox_inches='tight', format='pdf')
    fig.savefig('raw_scatterplot_{0:s}_{1:s}.png'.format(datebegin.strftime('%Y%m%d'), dateend.strftime('%Y%m%d')), bbox_inches='tight', format='png')

def linear_regression(x, y):
    reg = LinearRegression().fit(x.reshape((-1, 1)), y)
    model = reg.predict(x.reshape((-1,1)))
    r2 = reg.score(x.reshape((-1, 1)), y)
    det = " R²={0:.4f}".format(r2)
    return model, r2, det 

def RANSAC(x, y):
    reg = RANSACRegressor(random_state=0).fit(x, y)
    model = reg.predict(x.reshape((-1,1)))
    score = reg.score(x, y)
    return score, model

def massif_scatterplot(workdf, datebegin, dateend, massif):
    fig, ax = plt.subplots(figsize=(12,9))
    stations = np.unique(workdf['num_poste'])
    if len(stations) > 10:
        cm = plt.cm.get_cmap('tab20')
    else:
        cm = plt.cm.get_cmap('tab10')

    elevations = np.unique(workdf['alti'])
    elevations = workdf.groupby(['num_poste']).alti.first().to_numpy()
    names = workdf.groupby(['num_poste']).name.first().to_numpy()
    for i,station in enumerate(stations):
        tmp = workdf.loc[workdf['num_poste']==station]
        ndays = len(tmp)
        x = tmp['rr_nivometeo'].to_numpy()
        y = tmp['rr_antilope'].to_numpy()
        model, r2, det = linear_regression(x.reshape((-1, 1)), y)
        ax.scatter(x, y, marker='D', s=10, c=cm.colors[i], label=f'{names[i]}, {elevations[i]} m, {ndays} obs, {det}')

    ax.legend(fontsize=14)
    ax.grid(visible=True, linestyle=':', linewidth=0.5)
    xpoints = ypoints = ax.get_xlim()
    ax.plot(xpoints, ypoints, linestyle='--', color='grey', lw=1, scalex=False, scaley=False)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Total ANTILOPE precipitation estimate (mm)', fontsize=12)
    ax.set_xlabel('Total rain-gauges observed precipitation (mm)', fontsize=12)
    fig.savefig(f'scatterplot_massif_{massif}.svg', bbox_inches='tight', format='svg')

def elevation_scatterplot(workdf, datebegin, dateend):

    fig1, ax1 = plt.subplots(figsize=(12,9))
    fig2, ax2 = plt.subplots(figsize=(12,9))

    workdf = workdf.sort_values('elevation')
    workdf['ratio'] = workdf['rr_antilope'] / workdf['rr_nivometeo']
    workdf.replace([np.inf, -np.inf], np.nan, inplace=True)
    workdf = workdf.loc[~workdf['ratio'].isna()]
    nbpoint = len(workdf['elevation'].to_numpy())

    elevation_thresholds = [1450, 1725, 2000, 4000] # Based on the main "jumps"
    elevation_thresholds = [1530, 1739, 1967, 4000] # Based on the quantiles
    elevation_thresholds = [np.quantile(workdf['elevation'], q) for q in [0.25, 0.5, 0.75, 1]]
    colors = ['darkblue', 'cyan', 'gold', 'red']

    reg = LinearRegression().fit(workdf['elevation'].to_numpy().reshape((-1, 1)), workdf['ratio'].to_numpy())
    model = reg.predict(workdf['elevation'].to_numpy().reshape((-1,1)))
    r2 = reg.score(workdf['elevation'].to_numpy().reshape((-1, 1)),workdf['ratio'].to_numpy())
    ax2.plot(workdf['elevation'].to_numpy(), model, color='black', linewidth=2)
    ax2.text(1000, 2, f'R²={r2:.4}', fontsize=18, color='black')
    ax2.text(1000, 2.2, f'{nbpoint} stations', fontsize=18, color='black')
    ax1.text(500, 5500, f'{nbpoint} stations', fontsize=18, color='black')
#    score, ransac = RANSAC(workdf['elevation'].to_numpy().reshape((-1, 1)), workdf['ratio'].to_numpy())
#    ax2.plot(workdf['elevation'].to_numpy().reshape((-1, 1)), ransac, color = 'red', linewidth=2)
#    ax2.text(1000, 2.2, f'RANSAC score={score}', fontsize=18, color='red')

    for i, threshold in enumerate(elevation_thresholds):
        index_names = workdf[workdf['elevation'] < threshold].index
        tmp = workdf.loc[workdf.index.isin(index_names)]
        workdf.drop(index_names, inplace = True)

        def plot(x, y, ax, regression=True):
            if regression:
                model, r2, det = linear_regression(x.reshape((-1, 1)), y)
            else:
                det = ""
            if i == 0:
                label = ','.join(['< {0:d} m'.format(int(threshold)), det])
            elif i == 3:
                label = ','.join(['> {0:d} m'.format(int(elevation_thresholds[i-1])), det])
            else:
                label = ','.join(['[{0:d} m - {1:d} m]'.format(int(elevation_thresholds[i-1]), int(elevation_thresholds[i])), det])
            ax.scatter(x, y, marker='D', s=10, c=colors[i], label=label)
            if regression:
                ax.plot(x, model, color=colors[i], linewidth=1)

        plot(tmp['rr_nivometeo'].to_numpy(), tmp['rr_antilope'].to_numpy(), ax1, regression=True)
        plot(tmp['elevation'].to_numpy(), tmp['ratio'].to_numpy(), ax2, regression=False)

    def layout(ax):
        ax.legend(fontsize=14)
        ax.grid(visible=True, linestyle=':', linewidth=0.5)

    xpoints = ypoints = ax1.get_xlim()
    ax1.plot(xpoints, ypoints, linestyle='--', color='grey', lw=1, scalex=False, scaley=False)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    layout(ax1)
    layout(ax2)
    ax1.set_ylabel('Total ANTILOPE precipitation estimate (mm)', fontsize=12)
    ax1.set_xlabel('Total rain-gauges observed precipitation (mm)', fontsize=12)
    ax2.set_ylabel('Total ANTILOPE/rain-gauges ratio', fontsize=12)
    ax2.set_xlabel('Rain-gauge elevation (m)', fontsize=12)
    #plt.tight_layout()
    #fig.savefig('scatterplot_{0:s}_{1:s}.pdf'.format(datebegin.strftime('%Y%m%d'), dateend.strftime('%Y%m%d')), bbox_inches='tight', format='pdf')
    fig1.savefig('scatterplot_by_elevation_{0:s}_{1:s}.png'.format(datebegin.strftime('%Y%m%d'), dateend.strftime('%Y%m%d')), bbox_inches='tight', format='png')
    fig2.savefig('ratio_scatterplot_by_elevation_{0:s}_{1:s}.png'.format(datebegin.strftime('%Y%m%d'), dateend.strftime('%Y%m%d')), bbox_inches='tight', format='png')

def plot_massif(df, massif_number):
    #df = df.loc[~df['rr'].isna()].loc[~df['rr_antilope'].isna()]
    df['ratio'] = df['rr_antilope'] / df['rr_nivometeo']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.loc[~df['ratio'].isna()]

    #colorbar = sns.diverging_palette(240, 10, n=9)
    #fig = cartopy.Zoom_massif(massif_number, bgimage=True)
    fig = cartopy.Zoom_massif(massif_number)
    fig.init_massifs()
    #fig.highlight_massif(massif_number)

    cmap = ListedColormap(sns.diverging_palette(240, 10, n=9).as_hex())
    thresholds = [0.25, 0.5, 0.75, 0.95, 1.05, 1.5, 2, 4]
    norm = matplotlib.colors.BoundaryNorm(thresholds, cmap.N)

    mean_ratio = list()
    lats = list()
    lons = list()
    color = list()
    for station in np.unique(df['num_poste']):
        tmp = df[df['num_poste']==station]
        if len(tmp) > 100:
            ratio = tmp['ratio'].mean()
            mean_ratio.append(ratio)
            #color.append(colorbar[np.searchsorted(thresholds, ratio)])
            lats.append(tmp['lat'].mean())
            lons.append(tmp['lon'].mean())
            model, r2, det = linear_regression(tmp['rr_nivometeo'].to_numpy(), tmp['rr_antilope'].to_numpy())
            text = f'{ratio.round(2)} ; {r2.round(2)} ; {len(tmp)}'
            fig.map.annotate(text, # this is the text
                 (tmp['lon'].mean(), tmp['lat'].mean()), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center')
            #fig.map.text(tmp['lon'].mean(), tmp['lat'].mean(), tmp['ratio'].mean().round(3), horizontalalignment='right', verticalalignment='top', color='red')

    sc = fig.map.scatter(lons, lats, c=mean_ratio, cmap=cmap, norm=norm, marker="^", s=150)
    fig.fig.colorbar(sc, label='Radar/Rain-gauge ratio')
    #fig.addpoints(lons, lats, color=color, marker="^")
    fig.save(f'ratio_{massif_number}.svg', formatout='svg')
    fig.close()

def plot_obs(domain, lat, lon, alt):
    # Plot stations on the map
    class_ = getattr(cartopy, f'Map_{domain}')
    fig = class_()
    fig.init_massifs()
    fig.addpoints(lon, lat, marker='+')
    fig.save(f'obs_{domain}.svg', formatout='svg')
    fig.close()
    for massif in map_massifs[domain]:
        fig = cartopy.Zoom_massif(massif)
        fig.init_massifs()
        fig.addpoints(lon, lat, alt)
        fig.save(f'alti_obs_massif{massif}.svg', formatout='svg')
        fig.close()


def plot_full_domain(domain, lat, lon, df):
   
    fill_all_massifs(domain, lat, lon, df)
    point_stations_info(domain, lat, lon, df)

def point_stations_info(domain, lat, lon, df):
    pass

def fill_all_massifs(domain, lat, lon, df):

    class_ = getattr(cartopy, f'Map_{domain}')
    r2_by_massif = dict()
    bias = dict()
    nb_stations = dict()
    rmse = dict()
    ratio = dict()
    for massif in map_massifs[domain]:
        tmp = df.loc[df["massif_number"]==massif]
        nb_stations[massif] = len(tmp)
        if len(tmp) > 3:
            model,r2,det = linear_regression(tmp['rr_nivometeo'].to_numpy().reshape((-1,1)), tmp['rr_antilope'].to_numpy())
            r2_by_massif[massif] = r2
            model,r2,det = linear_regression((tmp['rr_antilope'] / tmp['rr_nivometeo']).to_numpy().reshape((-1,1)), tmp['elevation'].to_numpy())
            ratio[massif] = r2
        bias[massif] = ((tmp['rr_antilope'] - tmp['rr_nivometeo']) / tmp['ndays']).mean()
        rmse[massif] = np.sqrt((np.square((tmp['rr_antilope'] - tmp['rr_nivometeo']) / tmp['ndays'])).mean())

    massif_numbers = np.fromiter(nb_stations.keys(), dtype=int)
    attributes = dict(palette='YlGnBu', forcemin=0., forcemax=1., seuiltext=50., label="R² of the linear regression ANTILOPE / rain gauges")
    fig = class_()
    fig.draw_massifs(np.fromiter(r2_by_massif.keys(), dtype=int), np.fromiter(r2_by_massif.values(), dtype=float), **attributes)
    fig.plot_center_massif(massif_numbers, np.fromiter(nb_stations.values(), dtype=int), **attributes)
    fig.save(f'R2_by_massif_{domain}.svg', formatout='svg')
    fig.close()

    attributes = dict(palette='YlGnBu', forcemin=0., forcemax=np.max(np.fromiter(ratio.values(), dtype=float)), seuiltext=50., label="R² linear regression of the ratio ANTILOPE / rain gauges function of elevation")
    fig = class_()
    fig.draw_massifs(np.fromiter(ratio.keys(), dtype=int), np.fromiter(ratio.values(), dtype=float), **attributes)
    fig.plot_center_massif(massif_numbers, np.fromiter(nb_stations.values(), dtype=int), **attributes)
    fig.save(f'Ratio_by_massif_{domain}.svg', formatout='svg')
    fig.close()

    extreme_value = np.nanmax(np.abs(np.fromiter(bias.values(), dtype=float)))
    attributes = dict(palette='seismic', forcemin=-extreme_value, forcemax=extreme_value, seuiltext=50., label='Mean daily bias ANTILOPE / rain gauges (mm)')
    fig = class_()
    fig.draw_massifs(np.fromiter(bias.keys(), dtype=int), np.fromiter(bias.values(), dtype=float), **attributes)
    fig.plot_center_massif(massif_numbers, np.fromiter(nb_stations.values(), dtype=int), **attributes)
    fig.save(f'Bias_by_massif_{domain}.svg', formatout='svg')
    fig.close()

    attributes = dict(palette='YlGnBu', forcemin=0., forcemax=np.nanmax(np.fromiter(rmse.values(), dtype=float)), seuiltext=50., label='Mean daily RMSE ANTILOPE/rain gauges (mm)')
    fig = class_()
    fig.draw_massifs(np.fromiter(rmse.keys(), dtype=int), np.fromiter(rmse.values(), dtype=float), **attributes)
    fig.plot_center_massif(massif_numbers, np.fromiter(nb_stations.values(), dtype=int), **attributes)
    fig.save(f'RMSE_by_massif_{domain}.svg', formatout='svg')
    fig.close()

if __name__ == "__main__":
    args = parse_command_line()

    extract_period = date_range(args.datebegin, args.dateend)
    ANTILOPE_data = 'ANTILOPE_{0:s}_{1:s}.csv'.format(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'))
    nivometeo_data = 'obs_nivometeo_daily_RR_{0:s}_{1:s}.csv'.format(args.datebegin.strftime('%Y%m%d'), args.dateend.strftime('%Y%m%d'))

    # I- Lecture et mise en forme des données
    #########################################
    antilope = pd.read_csv(ANTILOPE_data, sep=';', parse_dates=['date'], dtype={'rr_antilope': float, 'num_poste': int}, na_values=['--'])
    nivometeo = pd.read_csv(nivometeo_data, sep=';', parse_dates=['dat'], 
            dtype={'Q.num_poste':int, 'poste_nivo.nom_usuel':str, 'poste_nivo.alti':int, 'poste_nivo.lat_dg':float, 'poste_nivo.lon_dg':float, 'rr':float, 'poste_nivo.massif_nivo':int})
    antilope['date'] = antilope['date'].dt.date
    # Renomage de certaines colonnes (pour le merge des DF et pour faciliter la manipulation)
    nivometeo['date'] = nivometeo['dat'].dt.date # Necessaire (changement de type)
    nivometeo = nivometeo.rename(columns={'Q.num_poste':'num_poste', 'poste_nivo.lat_dg':'lat', 'poste_nivo.lon_dg':'lon', 
        'poste_nivo.nom_usuel':'name', 'poste_nivo.massif_nivo':'massif_number', 'poste_nivo.alti':'alti', 'rr':'rr_nivometeo'}) # facultatif
    # merge des DF
    df = pd.merge(antilope, nivometeo, on=["date", "num_poste"])
    # Selection de la période 
    df = df.loc[df["date"]>=datetime.date(args.datebegin)].loc[df["date"]<=datetime.date(args.dateend)]
    # Retrait des données non exploitables
    df = df.loc[~df['rr_nivometeo'].isna()].loc[~df['rr_antilope'].isna()] # Remove lines with missing value
    df = df.loc[df['massif_number']<99] # Remove Stations not associated to 1 massif
    df = df.loc[~df['name'].str.contains('EDFNIVO')] # Remove EDFNIVO stations
    # Calcul des valeurs agrégées par station
    rr_nivometeo  = df.groupby(['num_poste']).rr_nivometeo.sum()
    rr_antilope   = df.groupby(['num_poste']).rr_antilope.sum()
    nb_values     = df.groupby(['num_poste']).date.count()
    elevations    = df.groupby(['num_poste']).alti.mean()
    lats          = df.groupby(['num_poste']).lat.mean()
    lons          = df.groupby(['num_poste']).lon.mean()
    massif_number = df.groupby(['num_poste']).massif_number.mean() 
    # Regroupement dans une nouvelle dataframe (il est surement possible d'extraire directement cette DF depuis 'df' pour simplifier le code)
    workdict = {'elevation':elevations, 'rr_nivometeo':rr_nivometeo, 'rr_antilope':rr_antilope, 'ndays':nb_values, 'massif_number':massif_number, 'lats':lats, 'lons':lons}
    workdf   = pd.DataFrame(workdict)
    workdf = workdf.loc[workdf['ndays']>100] # Consider only points with at least 100 observations

    # I- Visualisations des données
    ###############################
    if args.massif is not None:
        # Focus sur un unique massif (carte)
        workdf = df.loc[df['massif_number'] == args.massif]
        plot_massif(workdf, args.massif)
        massif_scatterplot(workdf, args.datebegin, args.dateend, args.massif)
        
    else:
        # 1. Raw sactter plot of all availbale stations
        raw_scatterplot(workdf['rr_nivometeo'].to_numpy(), workdf['rr_antilope'].to_numpy(), workdf['elevation'].to_numpy(), args.datebegin, args.dateend)

        # 2. Scatter plot with stations sorted by elevation range
        elevation_scatterplot(workdf, args.datebegin, args.dateend)

        # 3. Maps
        liste_postes = np.unique(df['num_poste']).astype(int)
        for domain in ['alpes', 'pyrenees', 'corse']:
            plot_obs(domain, lats.to_numpy(), lons.to_numpy(), elevations.to_numpy())
            plot_full_domain(domain, lats.to_numpy(), lons.to_numpy(), workdf)





