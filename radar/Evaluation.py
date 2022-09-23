#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os
from datetime import datetime, timedelta
import pandas as pd  # Version 0.25.3
import numpy as np
import copy

import argparse

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns

from sklearn.linear_model import LinearRegression, RANSACRegressor

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

nb_obs_min = 60

#subdomain_map = dict(
#    1 = dict(name='North-West Alps', massifs=[1, 2, 3, 4, 5, 7, 8]),
#    2 = dict(name='North-East Alps', massifs=[6, 9, 10, 11, 13]),
#    3 = dict(name='Central Alps', massifs=[12, 14, 15, 18, 19]),
#    4 = dict(name='Southern Alps', massifs=[13, 16, 17, 20, 21, 22, 23]),
#    5 = dict(name='Western Pyrenees', massifs=[64, 65, 66]),
#    6 = dict(name='Central Pyrenees', massifs=[67, 68, 69]),
#    7 = dict(name='Eastern Pyrenees', massifs=[70, 71, 72, 73, 74]),
#    )
subdomain_map = dict(
    NWA = [1, 2, 3, 4, 5, 7, 8],
    NEA = [6, 9, 10, 11, 13],
    CA  = [12, 14, 15, 18, 19],
    SA  = [13, 16, 17, 20, 21, 22, 23],
    WP  = [64, 65, 66],
    CP  = [67, 68, 69],
    EP  = [70, 71, 72, 73, 74],
)

def parse_command_line():
    description = "Evaluation of RADAR products (ANTILOPE or PANTHERE) using nivo-météo network observations"
    parser = argparse.ArgumentParser(description=description)
    #parser.add_argument('-d', '--rundate', help='Rundate for operational executions, format YYMMDDHH')
    parser.add_argument('-b', '--datebegin', help='Begining date of extraction, format YYYYMMDDHH or YYMMDDHH', required=True)
    parser.add_argument('-e', '--dateend', help = 'Final date of extraction (default=datebegin)')
    parser.add_argument('-w', '--workdir', help='Runing directory (default for guppy)', default='/home/mrns/vernaym/workdir/extraction_antilope')
    parser.add_argument('-d', '--datadir', help='Runing directory (default for guppy)', default='/home/mrns/vernaym/These/DATA')
    parser.add_argument('-m', '--massif', help='PLot for a specific massif', default=None, type=int)
    parser.add_argument('-s', '--subdomain', default=None, help='PLot for a specific subdomain, values=[NWA, NEA, CA, SA, WP, CP, EP]')
    parser.add_argument('-t', '--threshold', default=None, help='Threshold of precipitation (mm) to apply in the data to consider', type=int)
    parser.add_argument('-p', '--product', default='antilopejp1', help='Product to deal with', choices=['antilope', 'antilopejp1', 'panthere', 'kriging', 'arome'])
    parser.add_argument('-l', '--lpn', action='store_true', help='Take into account the rain-snow limit')

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

def add_radar_positions(ax):
    radars = dict(
    	moucherotte  = dict(lat=45.14776, lon=5.63933, alt=1920,name='Moucherotte'),
        colombis     = dict(lat=44.49664, lon=6.21729, alt=1742, name='Colombis'),
        ladole       = dict(lat=46.42565, lon=6.10001, alt=1677, name='La Dole'),
        #collobrieres = dict(lat=43.22, lon=6.37, alt=641, name='Collobrières'),
    )
    def getImage(path):
       return OffsetImage(plt.imread(path, format="png"), zoom=.03)

    symbole_radar = '/home/vernaym/These/figures/symbole_radar.png'
    for radar, infos in radars.items():
       ab = AnnotationBbox(getImage(symbole_radar), (infos['lon'], infos['lat']), frameon=False)
       ax.add_artist(ab)
       circle = plt.Circle((infos['lon'], infos['lat']), 1, color='grey', fill=False, linestyle='--')
       ax.add_patch(circle)

def raw_scatterplot(rr_nivometeo, rr_antilope, elevations, datebegin, dateend, suffix=None, **kw):
    nbpoint = len(rr_nivometeo)
    minval = min([min(rr_nivometeo), min(rr_antilope)]) - 0.3
    maxval = max([max(rr_nivometeo), max(rr_antilope)]) * 1.02
    fig = plt.figure(figsize=(10,6))
    #plt.scatter(rr_nivometeo.to_numpy(), rr_antilope.to_numpy(), s=nb_values.to_numpy(), marker='o')
    sc = plt.scatter(rr_nivometeo, rr_antilope, marker='D', s=10, c=elevations)
    plt.text(minval+1, maxval*0.8, f'{nbpoint} stations', fontsize=18, color='black')
    plt.colorbar(sc, label='Elevation')
    xpoints = ypoints = np.arange(0, maxval, 0.1)
    plt.grid(visible=True, linestyle=':', linewidth=0.5)
    plt.xlim(xmin=minval, xmax=maxval)
    plt.ylim(ymin=minval, ymax=maxval)
    plt.plot(xpoints, ypoints, linestyle='--', color='grey', lw=1, scalex=False, scaley=False)
    plt.ylabel(f'Mean daily {kw["product"]} precipitation estimate (mm/day)', fontsize=12)
    plt.xlabel('Mean daily rain-gauges observed precipitation (mm/day)', fontsize=12)

    # Regression linéaire
    reg = LinearRegression().fit(rr_nivometeo.reshape((-1, 1)), rr_antilope)
    y = reg.predict(rr_nivometeo.reshape((-1,1)))
    r2 = reg.score(rr_nivometeo.reshape((-1, 1)), rr_antilope)
    plt.plot(rr_nivometeo, y, color="red", linewidth=1)
    plt.text(minval+1, maxval*0.7, f'R²={r2:.4}', fontsize=18, color='red')

    #fig.savefig('raw_scatterplot_{0:s}_{1:s}.svg'.format(datebegin.strftime('%Y%m%d'), dateend.strftime('%Y%m%d')), bbox_inches='tight', format='svg')
    filename = 'raw_scatterplot_{0:s}_{1:s}'.format(datebegin.strftime('%Y%m%d'), dateend.strftime('%Y%m%d'))
    if suffix is not None:
        filename = f'{filename}_{suffix}'

    #plt.tight_layout()
    fig.savefig(f'{filename}.svg', format='svg', bbox_inches='tight')

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

def daily_scatterplot(workdf, datebegin, dateend, massif=None, subdomain=None, suffix=None, **kw):
    infostation = True
    if massif is not None:
        filename1 = f'daily_scatterplot_massif{massif}'
        filename2 = f'daily_scatterplot_alti_massif{massif}'
    elif subdomain is not None:
        filename1 = f'daily_scatterplot_subdomain_{subdomain}'
        filename2 = f'daily_scatterplot_alti_subdomain_{subdomain}'
    else:
        filename1 = f'daily_scatterplot'
        filename2 = f'daily_scatterplot_alti'
        infostation = False
    if suffix is not None:
        filename1 = f'{filename1}_{suffix}'
        filename2 = f'{filename2}_{suffix}'

    #workdf['nb_obs_per_day'] = workdf.date.map(workdf.date.value_counts())
    stations = np.unique(workdf['num_poste'])
    cm = copy.copy(plt.cm.get_cmap('tab10'))
#    elif len(stations) <= 20:
#        cm = plt.cm.get_cmap('tab20')

    #elevations = np.unique(workdf['elevation'])
    workdf.reset_index(drop = True, inplace = True)
    elevations = workdf.groupby(['num_poste']).elevation.first().to_numpy()
    names = workdf.groupby(['num_poste']).name.first().to_numpy()
    x = workdf['rr_nivometeo'].to_numpy()
    y = workdf[f'rr_{kw["product"]}'].to_numpy()
    minval = min([min(x), min(y)])-0.5
    maxval = max([max(x), max(y)])*1.05
    model, r2, det = linear_regression(x.reshape((-1, 1)), y)

    fig, ax = plt.subplots(figsize=(12,9))

    ax.plot(x, model, color='black', linewidth=2)
    ax.text(maxval*0.75, maxval*0.5, f'R²={r2:.4}', fontsize=18, color='black')
    mean_rr_nivometeo = dict()
    mean_rr_antilope = dict()
    if massif is None and subdomain is None:

        # The goal of the following is to add density on the scatter plot
        # 1st method using histograms (axis are missing when the number of poitn is too important ??)
        #--------------------------------------------------------------------------------------------
        #histogram definition
#        bins = [1000, 1000] # number of bins
#        # histogram the data
#        hh, locx, locy = np.histogram2d(x, y, bins=bins)
#        # Sort the points by density, so that the densest points are plotted last
#        z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
#        idx = z.argsort()
#        x2, y2, z2 = x[idx], y[idx], z[idx]
#        ax.scatter(x2, y2, c=z2, cmap='jet', marker='D', s=10)

        # 2nd method : computing the density with gaussian_kde (very long, doesn't work for huge number of points)
        #---------------------------------------------------------------------------------------------------------
#        t0 = datetime.now()
#        print('Number of points (x/y) = {0:d}, {1:d}'.format(len(x), len(y)))
#        xy = np.vstack((x, y))
#        z = gaussian_kde(xy)(xy)
#        # Sort the points by density, so that the densest points are plotted last
#        idx = z.argsort()
#        x, y, z = x[idx], y[idx], z[idx]
#        t1 = datetime.now()
#        print('time to plot density : ', (t1-t0))
#        ax.scatter(x, y, c=z, s=10, marker='D')

        ax.scatter(x, y, s=10, marker='D')

    for i,station in enumerate(stations):
        tmp = workdf.loc[workdf['num_poste']==station]
        ndays = len(tmp)
        x = tmp['rr_nivometeo'].to_numpy()
        y = tmp[f'rr_{kw["product"]}'].to_numpy()
        model, r2, det = linear_regression(x.reshape((-1, 1)), y)
        if len(stations) <= 10:
            color = cm.colors[i]
        else:
            color = 'blue'
        if massif is not None or subdomain is not None:
            ax.scatter(x, y, marker='D', s=10, color=color, label=f'{names[i]}, {elevations[i]} m, {ndays} obs, {det}')
        mean_rr_nivometeo[station] = x.mean()
        mean_rr_antilope[station] = y.mean()

    if infostation:
        ax.legend(fontsize=14)  # Add legend with station informations
    ax.grid(visible=True, linestyle=':', linewidth=0.5)
    xpoints = ypoints = np.arange(minval, maxval, 0.1)
    ax.plot(xpoints, ypoints, linestyle='--', color='grey', lw=1, scalex=False, scaley=False)
    ax.set_xlim(left=minval, right=maxval)
    ax.set_ylim(bottom=minval, top=maxval)
    ax.set_ylabel(f'Daily {kw["product"]} precipitation estimate (mm/day)', fontsize=12)
    ax.set_xlabel('Daily rain-gauges observed precipitation (mm/day)', fontsize=12)
    #plt.tight_layout()
    #fig.savefig(f'scatterplot_massif_{massif}.svg', bbox_inches='tight', format='svg')
    fig.savefig(f'{filename1}.svg', format='svg', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(12,9))
    #ax.set_xlim(left=700, right=np.max(elevations) * 1.1)
    # 1. Nivo-meteo RR values
    y1 = np.fromiter(mean_rr_nivometeo.values(), dtype=float)
    ax.scatter(elevations, y1, marker='D', s=10, color='blue', label='Rain gauges')
    reg1 = LinearRegression().fit(elevations.reshape((-1, 1)), y1)
    model1 = reg1.predict(elevations.reshape((-1,1)))
    r2 = reg1.score(elevations.reshape((-1, 1)), y1)
    ax.plot(elevations, model1, color='blue', linewidth=2)
    ax.text(1500, 1, f'R²={r2:.4}', fontsize=18, color='blue')

    # 2. ANTILOPE RR values
    y2 = np.fromiter(mean_rr_antilope.values(), dtype=float)
    ax.scatter(elevations, y2, marker='D', s=10, color='red', label=kw['product'])
    reg2 = LinearRegression().fit(elevations.reshape((-1, 1)), y2)
    model2 = reg2.predict(elevations.reshape((-1,1)))
    r2 = reg2.score(elevations.reshape((-1, 1)), y2)
    ax.plot(elevations, model2, color='red', linewidth=2)
    ax.text(1500, 3, f'R²={r2:.4}', fontsize=18, color='red')
    #fig.savefig(f'scatterplot_alti_massif_{massif}.svg', bbox_inches='tight', format='svg')
    ax.set_ylim(bottom=0, top=np.max([np.max(y1), np.max(y2)]) * 1.1)
    ax.set_ylabel('Mean daily precipitation (mm/day)', fontsize=12)
    ax.set_xlabel('Elevation (m)', fontsize=12)
    #plt.tight_layout()
    fig.savefig(f'{filename2}.svg', format='svg', bbox_inches='tight')

def elevation_scatterplot(workdf, datebegin, dateend, suffix=None, **kw):

    fig1, ax1 = plt.subplots(figsize=(12,9))
    fig2, ax2 = plt.subplots(figsize=(12,9))

    workdf = workdf.sort_values('elevation')
    workdf['ratio'] = workdf[f'rr_{kw["product"]}'] / workdf['rr_nivometeo']
    workdf.replace([np.inf, -np.inf], np.nan, inplace=True)
    workdf = workdf.loc[~workdf['ratio'].isna()]
    nbpoint = len(workdf['elevation'].to_numpy())

    elevation_thresholds = [1450, 1725, 2000, 4000]  # Based on the main "jumps"
    elevation_thresholds = [1530, 1739, 1967, 4000]  # Based on the quantiles
    elevation_thresholds = [np.quantile(workdf['elevation'], q) for q in [0.25, 0.5, 0.75, 1]]
    colors = ['darkblue', 'cyan', 'gold', 'red']

    reg = LinearRegression().fit(workdf['elevation'].to_numpy().reshape((-1, 1)), workdf['ratio'].to_numpy())
    model = reg.predict(workdf['elevation'].to_numpy().reshape((-1,1)))
    r2 = reg.score(workdf['elevation'].to_numpy().reshape((-1, 1)),workdf['ratio'].to_numpy())
    ax2.plot(workdf['elevation'].to_numpy(), model, color='black', linewidth=2)
    ax2.text(1300, max(workdf['ratio'].to_numpy()), f'R²={r2:.4}', fontsize=18, color='black')
    ax2.text(1300, max(workdf['ratio'].to_numpy()) * 0.95, f'{nbpoint} stations', fontsize=18, color='black')
#    score, ransac = RANSAC(workdf['elevation'].to_numpy().reshape((-1, 1)), workdf['ratio'].to_numpy())
#    ax2.plot(workdf['elevation'].to_numpy().reshape((-1, 1)), ransac, color = 'red', linewidth=2)
#    ax2.text(1000, 2.2, f'RANSAC score={score}', fontsize=18, color='red')
    minval = min([min(workdf[f'rr_nivometeo']), min(workdf[f'rr_{kw["product"]}'])]) - 0.3
    maxval = max([max(workdf[f'rr_nivometeo']), max(workdf[f'rr_{kw["product"]}'])]) * 1.02

    for i, threshold in enumerate(elevation_thresholds):
        index_names = workdf[workdf['elevation'] < threshold].index
        tmp = workdf.loc[workdf.index.isin(index_names)]
        workdf.drop(index_names, inplace = True)

        def plot(x, y, ax, regression=True):
            if regression and len(x) > 1:
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
            if regression and len(x) > 1:
                ax.plot(x, model, color=colors[i], linewidth=1)
        plot(tmp['rr_nivometeo'].to_numpy(), tmp[f'rr_{kw["product"]}'].to_numpy(), ax1, regression=True)
        plot(tmp['elevation'].to_numpy(), tmp['ratio'].to_numpy(), ax2, regression=False)

    def layout(ax):
        ax.legend(fontsize=14)
        ax.grid(visible=True, linestyle=':', linewidth=0.5)

    xpoints = ypoints = np.arange(0, maxval, 0.5)
    ax1.plot(xpoints, ypoints, linestyle='--', color='grey', lw=1, scalex=False, scaley=False)
    ax1.text(minval+1, maxval*0.7, f'{nbpoint} stations', fontsize=18, color='black')
    ax1.set_xlim(left=minval, right=maxval)
    ax1.set_ylim(bottom=minval, top=maxval)
    layout(ax1)
    layout(ax2)
    ax1.set_ylabel(f'Mean daily {kw["product"]} precipitation estimate (mm/day)', fontsize=12)
    ax1.set_xlabel('Mean daily rain-gauges observed precipitation (mm/day)', fontsize=12)
    plt.tight_layout()
    ax2.set_ylabel(f'{kw["product"]}/rain-gauges ratio', fontsize=12)
    ax2.set_xlabel('Rain-gauge elevation (m)', fontsize=12)
    plt.tight_layout()
    #fig1.savefig('scatterplot_by_elevation_{0:s}_{1:s}.svg'.format(datebegin.strftime('%Y%m%d'), dateend.strftime('%Y%m%d')), bbox_inches='tight', format='svg')
    filename1 = 'scatterplot_by_elevation_{0:s}_{1:s}'.format(datebegin.strftime('%Y%m%d'), dateend.strftime('%Y%m%d'))
    filename2 = 'ratio_scatterplot_by_elevation_{0:s}_{1:s}'.format(datebegin.strftime('%Y%m%d'), dateend.strftime('%Y%m%d'))
    if suffix is not None:
        filename1 = f'{filename1}_{suffix}'
        filename2 = f'{filename2}_{suffix}'
    fig1.savefig(f'{filename1}.svg', format='svg', bbox_inches='tight')
    #fig2.savefig('ratio_scatterplot_by_elevation_{0:s}_{1:s}.svg'.format(datebegin.strftime('%Y%m%d'), dateend.strftime('%Y%m%d')), bbox_inches='tight', format='svg')
    fig2.savefig(f'{filename2}.svg', format='svg', bbox_inches='tight')

def plot_massif(mydf, massif=None, subdomain=None, error=0.2, **kw):
    #mydf = mydf.loc[~mydf['rr'].isna()].loc[~mydf['rr_antilope'].isna()]
    mydf = mydf.loc[~mydf[f'rr_{kw["product"]}'].isna()].loc[~mydf['rr_nivometeo'].isna()]
    mydf['error'] = (mydf[f'rr_{kw["product"]}'] >= mydf['rr_nivometeo'] * (1-error)) & (mydf[f'rr_{kw["product"]}'] <= mydf['rr_nivometeo'] * (1+error))
    tmp = pd.DataFrame()
    tmp['lons'] = mydf.groupby(['num_poste']).lon.mean()
    tmp['lats'] = mydf.groupby(['num_poste']).lat.mean()
    tmp['rr_radar'] = mydf.groupby(['num_poste'])[f'rr_{kw["product"]}'].mean()
    tmp['rr_nivometeo'] = mydf.groupby(['num_poste'])['rr_nivometeo'].mean()
    tmp['freq_error'] = (mydf.groupby(['num_poste']).error.sum() / mydf.groupby(['num_poste']).error.count()) * 100
    tmp['biais'] = tmp['rr_radar'] - tmp['rr_nivometeo']
    mydf['diff'] = np.square(mydf[f'rr_{kw["product"]}'] - mydf['rr_nivometeo'])
    tmp['nb_days'] = len(mydf.groupby(['num_poste'])["diff"])
    #tmp['nb_days'] = mydf.groupby(['num_poste']).diff.count()
    #tmp['rmse']  = np.sqrt(mydf.groupby(['num_poste']).diff.mean())
    tmp['rmse']  = np.sqrt(mydf.groupby(['num_poste'])["diff"].mean())
    tmp['ratio'] = tmp['rr_radar'] / tmp['rr_nivometeo']
    tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
    tmp.to_csv('scores_{0:s}_{1:s}_{2:s}_{3:d}.csv'.format(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'), domain, args.threshold), sep=';')

    #====================================== Creation de la palette ===================================
    #cmap = ListedColormap(sns.diverging_palette(240, 10, n=9).as_hex())
    #cmap = copy.copy(ListedColormap(sns.diverging_palette(240, 10, n=9).as_hex())) # original one
    #cmap = copy.copy(ListedColormap(sns.color_palette('viridis', 12).as_hex()))
    #cmap = copy.copy(ListedColormap(sns.diverging_palette(240, 12, n=11, center='dark').as_hex()))
    #cmap = copy.copy(plt.cm.get_cmap('nipy_spectral'))

    # Methode permettant de créer une colormap 'tronquée'
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
        new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    #fig = cartopy.Zoom_massif(massif_number, bgimage=True)
    shrink = 1
    anchor = (0.0, 0.5)  # Default anchor value
    for score in ['biais', 'rmse', 'ratio', 'freq_error']:
        if subdomain is not None:
            if subdomain in ['NWA', 'NEA', 'CA', 'SA']:
                class_ = getattr(cartopy, 'Map_alpes')
            elif subdomain in ['WP', 'CP', 'EP']:
                class_ = getattr(cartopy, 'Map_pyrenees')
            fig = class_()
            filename = f'map_{score}_{subdomain}'
        elif massif is not None:
            mappos    = [0.06, 0.05, 0.96, 0.90]
            fig = cartopy.Zoom_massif(massif, mappos=mappos)
            filename = f'map_{score}_massif{massif}'
            anchor = (0.1, 0.95)
        else:
            class_ = getattr(cartopy, f'Map_{kw["domain"]}')
            filename = f'map_{score}_{kw["domain"]}'
            # Set colorbar size
            if kw["domain"] == 'pyrenees':
                shrink = 0.7
                mappos = [0.05, 0.05, 1, 0.98]
            else:
                shrink = 0.9
                mappos = [0.06, 0.06, 0.98, 0.92]
            fig = class_(mappos=mappos)

        if suffix is not None:
            filename = f'{filename}_{suffix}'
        fig.init_massifs()
        #fig.highlight_massif(massif_number)

        if score == 'ratio':
            #tmp = tmp.loc[~tmp['ratio'].isna()] # Remove Nan Values to avoid problems
            #cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["navy", "cornflowerblue", "green", "orange", "red"], 5)
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "blue", "green", "orange", "red"], 5)
            thresholds = [0., 0.5, 0.80, 1.2, 1.5, 10]
#            if suffix is not None:  # A threshold has been applied on precipitation values
#                #thresholds = [0.05, 0.1, 0.5, 0.80, 0.95, 1.05, 1.2, 2, 10, 20]
#                thresholds = [0., 0.5, 0.80, 1.2, 1.5, 10]
#            else:
#                cmap = copy.copy(plt.cm.get_cmap('nipy_spectral', 9))
#                thresholds = [0.2, 0.5, 0.6, 0.8, 0.95, 1.05, 1.2, 1.4, 2, 5]

            def set_marker(row):
                if row['ratio'] <= 0.8:
                    return 'v'
                elif row['ratio'] > 0.8 and row['ratio'] < 1.2:
                    return 'o'
                else:
                    return '^'
            tmp["marker"] = tmp.apply(set_marker, axis=1)
            # axis=1 makes sure that function is applied to each row
            legend = f'{kw["product"]}/rain-gauges {score}'
        elif score == 'biais':
            cmap = copy.copy(plt.cm.get_cmap('nipy_spectral', 9))
            if suffix is not None:  # A threshold has been applied on precipitation values
                thresholds = [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]
            else:
                thresholds = [-4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4]
            legend = f'{kw["product"]} {score} (mm/day)'
        elif score == 'freq_error':
            cmap = copy.copy(plt.cm.get_cmap('nipy_spectral_r'))  # _r reverse the colormap
            cmap = truncate_colormap(cmap, 0.05, 0.5, n=10)
            thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            #thresholds = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
            legend = f'Frequency of {kw["product"]} precipitation with an error lower than {int(error*100)}%'
            filename= f'{filename}_{error}'
        else:
            cmap = copy.copy(sns.color_palette('Reds', as_cmap=True))
            if suffix is not None:  # A threshold has been applied on precipitation values
                thresholds = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
            else:
                thresholds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            legend = f'{kw["product"]} {score} (mm/day)'

        norm = matplotlib.colors.BoundaryNorm(thresholds, cmap.N)

###################################################################################
# To plot the defined colormap
#        gradient = np.linspace(0, 1, 256)
#        gradient = np.vstack((gradient, gradient))
#        def plot_colormap(cmap_list):
#            # Create figure and adjust figure height to number of colormaps
#            nrows = len(cmap_list)
#            figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
#            fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
#            fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
#                                left=0.2, right=0.99)
#
#            for ax, name in zip(axs, cmap_list):
#                ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
#                ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
#                        transform=ax.transAxes)
#
#            # Turn off *all* ticks & spines, not just the ones with colormaps.
#            for ax in axs:
#                ax.set_axis_off()
#            plt.show()
#        plot_colormap([cmap])
###################################################################################

#        mean_score = list()
#        lats = list()
#        lons = list()
#        color = list()
#        for station in np.unique(mydf['num_poste']):
#            tmp = mydf[mydf['num_poste']==station]
#            if len(tmp) > 100:
#                value = tmp[f"{score}"].mean()
#                mean_score.append(value)
#                #color.append(colorbar[np.searchsorted(thresholds, ratio)])
#                lats.append(tmp['lat'].mean())
#                lons.append(tmp['lon'].mean())
#                model, r2, det = linear_regression(tmp['rr_nivometeo'].to_numpy(), tmp[f'rr_{kw["product"]}'].to_numpy())
#                text = f'{value.round(2)} ; {r2.round(2)} ; {len(tmp)}'
#                if massif is not None or subdomain is not None:
#                    fig.map.annotate(text, # this is the text
#                         (tmp['lon'].mean(), tmp['lat'].mean()), # these are the coordinates to position the label
#                         textcoords="offset points", # how to position the text
#                         xytext=(0,10), # distance from text to points (x,y)
#                         ha='center')
#                    #fig.map.text(tmp['lon'].mean(), tmp['lat'].mean(), tmp['ratio'].mean().round(3), horizontalalignment='right', verticalalignment='top', color='red')
#        sc = fig.map.scatter(lons, lats, c=mean_score, cmap=cmap, norm=norm, marker="^", s=150)

        ax = fig.fig.axes[0]
        add_radar_positions(ax)
        if score == 'ratio':
            for marker, d in tmp.groupby('marker'):
                sc = fig.map.scatter(d['lons'], d['lats'], c=d[f'{score}'], cmap=cmap, norm=norm, marker=marker, s=150, edgecolors='black')
        else:
            sc = fig.map.scatter(tmp['lons'], tmp['lats'], c=tmp[f'{score}'], cmap=cmap, norm=norm, marker="^", s=150, edgecolors='black')
        if massif is not None:
            for i, val in enumerate(tmp[f'{score}']):
                text = '{0:.2f} ({1:d})'.format(val, tmp['nb_days'].to_numpy()[i])
                fig.map.text(tmp['lons'].to_numpy()[i]-0.018, tmp['lats'].to_numpy()[i]-0.014, text, fontsize=8)
        #fig.fig.colorbar(sc, label='Radar/Rain-gauge ratio', shrink=shrink)
        #fig.fig.colorbar(sc, label='Radar/Rain-gauge ratio', ax=fig.fig.axes[0], shrink=shrink)
        fig.fig.colorbar(sc, label=legend, shrink=shrink, anchor=anchor)
        plt.tight_layout()
        fig.save(f'{filename}.svg', formatout='svg', bbox_inches='tight')
        fig.close()

def plot_obs(lat, lon, alt, datebegin, dateend):
    for domain in ['alpes', 'pyrenees']:
        print(domain)
        # Plot stations on the map
        class_ = getattr(cartopy, f'Map_{domain}')
        if domain == 'pyrenees':
            shrink = 0.7
            mappos = [0.05, 0.05, 1, 0.98]
        else:
            shrink = 1
            mappos = [0.06, 0.06, 0.98, 0.92]
        fig = class_(mappos=mappos)
        fig.init_massifs()
        #myplot = fig.addpoints(lon, lat, marker='D', color=alt)
        ax = fig.fig.axes[0]
        add_radar_positions(ax)
        sc = fig.map.scatter(lon, lat, c=alt, marker="^", s=150)
        plt.colorbar(sc, label='Elevation (m)', shrink=shrink)
        plt.tight_layout()
        fig.save(f'obs_{domain}_{datebegin}_{dateend}.svg', formatout='svg', bbox_inches='tight')
        fig.close()
#        for massif in map_massifs[domain]:
#            fig = cartopy.Zoom_massif(massif)
#            fig.init_massifs()
#            fig.addpoints(lon, lat, labels=alt)
#            plt.tight_layout()
#            fig.save(f'alti_obs_massif{massif}.svg', formatout='svg')
#            fig.close()


def plot_full_domain(domain, lat, lon, df, **kw):

    #fill_all_massifs(domain, lat, lon, df, **kw)
    point_stations_info(domain, lat, lon, df)

def point_stations_info(domain, lat, lon, df):
    pass

def fill_all_massifs(domain, lat, lon, df, suffix=None, **kw):

    class_ = getattr(cartopy, f'Map_{domain}')
    if domain == 'pyrenees':
        mappos = [0.05, 0.06, 0.8, 0.8]
        legendpos = [0.92, 0.1, 0.02, 0.7]
    else:
        mappos = [0.02, 0.06, 0.8, 0.9]
        legendpos = [0.85, 0.15, 0.03, 0.6]
    r2_by_massif = dict()
    bias = dict()
    nb_stations = dict()
    rmse = dict()
    ratio = dict()
    for massif in map_massifs[domain]:
        tmp = df.loc[df["massif_number"]==massif]
        nb_stations[massif] = len(tmp)
        if len(tmp) >= 3:
            model,r2,det = linear_regression(tmp['rr_nivometeo'].to_numpy().reshape((-1,1)), tmp[f'rr_{kw["product"]}'].to_numpy())
            r2_by_massif[massif] = r2
            model,r2,det = linear_regression((tmp[f'rr_{kw["product"]}'] / tmp['rr_nivometeo']).to_numpy().reshape((-1,1)), tmp['elevation'].to_numpy())
            ratio[massif] = r2
        bias[massif] = (tmp[f'rr_{kw["product"]}'] - tmp['rr_nivometeo']).mean()
        rmse[massif] = np.sqrt(np.square(tmp[f'rr_{kw["product"]}'] - tmp['rr_nivometeo']).mean())

    massif_numbers = np.fromiter(nb_stations.keys(), dtype=int)

#    attributes = dict(palette='YlGnBu', forcemin=0., forcemax=1., seuiltext=50., label=f'R² of the linear regression {kw["product"]} / rain gauges', transparency=np.fromiter(nb_stations.values(), dtype=int))
#    fig = class_()
#    fig.draw_massifs(np.fromiter(r2_by_massif.keys(), dtype=int), np.fromiter(r2_by_massif.values(), dtype=float), **attributes)
#    fig.plot_center_massif(massif_numbers, np.fromiter(nb_stations.values(), dtype=int), **attributes)
#    filename = f'R2_by_massif_{domain}'
#    if suffix is not None:
#        filename = f'{filename}_{suffix}'
#    plt.tight_layout()
#    fig.save(f'{filename}.svg', formatout='svg')
#    fig.close()
#    attributes = dict(palette='YlGnBu', forcemin=0., forcemax=np.max(np.fromiter(ratio.values(), dtype=float)), seuiltext=50., label=f'R² linear regression of the ratio {kw["product"]} / rain gauges function of elevation')
#    fig = class_()
#    fig.draw_massifs(np.fromiter(ratio.keys(), dtype=int), np.fromiter(ratio.values(), dtype=float), **attributes)
#    fig.plot_center_massif(massif_numbers, np.fromiter(nb_stations.values(), dtype=int), **attributes)
#    filename = f'Ratio_by_massif_{domain}'
#    if suffix is not None:
#        filename = f'{filename}_{suffix}'
#    plt.tight_layout()
#    fig.save(f'{filename}.svg', formatout='svg')
#    fig.close()

    #extreme_value = np.nanmax(np.abs(np.fromiter(bias.values(), dtype=float)))
    filename = f'Bias_by_massif_{domain}'
    if suffix is not None:
        extreme_value = 10  # The scale must be larger when considering precipitation above a 10mm threshold
        filename = f'{filename}_{suffix}'
    else:
        extreme_value = 3  # To have the same scale for all figures
    attributes = dict(palette='seismic', forcemin=-extreme_value, forcemax=extreme_value, label=f'Mean daily {kw["product"]} bias (mm/day)')
    fig = class_(mappos=mappos, legendpos=legendpos)
    fig.draw_massifs(np.fromiter(bias.keys(), dtype=int), np.fromiter(bias.values(), dtype=float), **attributes)
    fig.plot_center_massif(massif_numbers, np.fromiter(nb_stations.values(), dtype=int), **attributes)
    #plt.tight_layout()
    fig.save(f'{filename}.svg', formatout='svg', bbox_inches='tight')
    fig.close()

    filename = f'RMSE_by_massif_{domain}'
    if suffix is not None:
        extreme_value = 12  # The scale must be larger when considering precipitation above a 10mm threshold
        filename = f'{filename}_{suffix}'
    else:
        extreme_value = 3
    attributes = dict(palette='YlGnBu', forcemin=0., forcemax=extreme_value, label=f'Mean daily {kw["product"]} RMSE (mm/day)')
    fig = class_(mappos=mappos, legendpos=legendpos)
    fig.draw_massifs(np.fromiter(rmse.keys(), dtype=int), np.fromiter(rmse.values(), dtype=float), **attributes)
    fig.plot_center_massif(massif_numbers, np.fromiter(nb_stations.values(), dtype=int), **attributes)
    #plt.tight_layout()
    fig.save(f'{filename}.svg', formatout='svg', bbox_inches='tight')
    fig.close()

def error_vs_RR(df, datebegin, dateend, **kw):
    #import seaborn as sns
    df['error'] = np.sqrt(np.square(df[f'rr_{kw["product"]}'] - df['rr_nivometeo']))
    fig = plt.figure()
    #sns.regplot(df[f'rr_{kw["product"]}'], df['error'])
    plt.plot(df[f'rr_{kw["product"]}'], df['error'], linestyle='', marker='+')
    a, b = np.polyfit(df[f'rr_{kw["product"]}'], df['error'], deg=1)
    x = np.array([0, np.max(df[f'rr_{kw["product"]}'])])
    plt.plot(x, a*x+b, marker=None, color='k', label=f'Regression parameters : slope={a:.3f}, intercept={b:.3f}')
    plt.xlabel('ANTILOPE 24-hour precipitation (mm)')
    plt.ylabel('ANTILOPE root mean square deviation (mm)')
    plt.legend()
    plt.tight_layout()
    fig.savefig(f'ANTILOPE_rmsd_vs_ANTILOPE_RR_{datebegin.strftime("%Y%m%d")}_{dateend.strftime("%Y%m%d")}.pdf', format='pdf', bbox_inches='tight')

def read_nivometeo():

    # I- Lecture et mise en forme des données
    #########################################
    print(f'Reading the following radar data {RADAR_data} from product {args.product}')
    # I.1 Observations nivometeo
    #---------------------------
    #nivometeo_data = 'obs_nivometeo_daily_RR_{0:s}_{1:s}.csv'.format(args.datebegin.strftime('%Y%m%d'), args.dateend.strftime('%Y%m%d'))
    nivometeo_data = 'obs_nivometeo_daily_RR.csv'
    nivometeo = pd.read_csv(nivometeo_data, sep=';', parse_dates=['Q.dat'],
            dtype={'Q.num_poste':int, 'poste_nivo.nom_usuel':str, 'poste_nivo.alti':int, 'poste_nivo.lat_dg':float, 'poste_nivo.lon_dg':float, 'rr':float,
                'poste_nivo.massif_nivo':int, 'hist_reseau_poste.reseau_poste':int})
#    nivometeo_data = 'obs_nivometeo_hourly_RR.csv'
#    nivometeo = pd.read_csv(nivometeo_data, sep=';', parse_dates=['H.dat'],
#            dtype={'H.num_poste':int, 'poste_nivo.nom_usuel':str, 'poste_nivo.alti':int, 'poste_nivo.lat_dg':float, 'poste_nivo.lon_dg':float, 'H.rr1':float,
#                'poste_nivo.massif_nivo':int, 'hist_reseau_poste.reseau_poste':int})
    # Renomage de certaines colonnes (pour le merge des DF et pour faciliter la manipulation)
    nivometeo['date'] = nivometeo['Q.dat'].dt.date + pd.Timedelta("1d")  # Changement de type + matching dates with radar data (BDClim extraction
        # for date ymd is the observation from ymd6h to ym(d+1)6h )
    nivometeo = nivometeo.rename(columns={'Q.num_poste':'num_poste', 'poste_nivo.lat_dg':'lat', 'poste_nivo.lon_dg':'lon',
        'poste_nivo.nom_usuel':'name', 'poste_nivo.massif_nivo':'massif_number', 'poste_nivo.alti':'elevation', 'Q.rr':'rr_nivometeo'})  # facultatif

    return nivometeo

if __name__ == "__main__":
    args = parse_command_line()

    extract_period = date_range(args.datebegin, args.dateend)
    if args.product == 'antilope':
        RADAR_data = 'ANTILOPEQ_{0:s}_{1:s}.csv'.format(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'))
    elif args.product == 'antilopejp1':
        RADAR_data = 'ANTILOPEJP1Q_{0:s}_{1:s}.csv'.format(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'))
    elif args.product == 'kriging':
        RADAR_data = 'Kriging_{0:s}_{1:s}_{2:s}.csv'.format(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'), 'exponential')
    elif args.product == 'arome':
        RADAR_data = 'arome_{0:s}_{1:s}_GrandesRousses.nc'.format(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'))
    else:
        RADAR_data = 'PANTHERE_{0:s}_{1:s}.csv'.format(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'))

    nivometeo = read_nivometeo()

    # I.2 Produit radar
    #------------------
    import xarray as xr
    if args.product == 'arome':
        datadir = '/home/vernaym/These/DATA'
        xrdata = xr.open_dataset(os.path.join(datadir, RADAR_data))
        antilope = xrdata.to_dataframe().reset_index().rename(columns={'time':'date'})
        # TODO : Extraire les valeurs de la PEAROME correspondant aux point d'obs nivometeo
        import pdb
        pdb.set_trace()
    else:
        antilope = pd.read_csv(RADAR_data, sep=';', parse_dates=['date'], dtype={f'rr_{args.product}': float, 'num_poste': int}, na_values=['--'])
    antilope['date'] = antilope['date'].dt.date

    # II- Merge des DF et mise en forme des données
    ###############################################
    # TODO : merge DF
    df = pd.merge(antilope, nivometeo, on=["date", "num_poste"])
    if args.lpn:
        #lpn = pd.read_csv('LPN_nivometeo.csv', sep=';', parse_dates=['H_NIVO.DAT'], dtype={'H.num_poste':int, 'H_NIVO.ALTI_LPNX':int}, index_col=['H_NIVO.DAT'])
        lpn = pd.read_csv('LPN_nivometeo.csv', sep=';', parse_dates=['H_NIVO.DAT'], dtype={'H_NIVO.NUM_POSTE':int, 'H_NIVO.ALTI_LPNX':int})
        lpn.rename(columns={'H_NIVO.ALTI_LPNX':'LPNX', 'H_NIVO.NUM_POSTE':'num_poste'}, inplace=True)
        lpn = lpn[lpn['LPNX']>0]  # consider 0 values as missing observation
        lpn['date'] = lpn['H_NIVO.DAT'] + pd.Timedelta("12h")  # La LPN observée à 12:00 D concerne les précipitations entre D (6:00) et D+1 (6:00) que l'on veut identifier
        # par la date ym(D+1), on décale donc de 12h pour que la date de l'obs passe à D+1
        lpn.index = lpn['date']
        lpn = lpn.groupby(['num_poste']).resample('1D').max()  # When 2 observation (at 6:00 and 12:00) are available, set the daily LPN as the maximum
        lpn = lpn[~np.isnan(lpn['LPNX'])]['LPNX'].reset_index()
        lpn.date = lpn.date.dt.date
        df = pd.merge(df, lpn, on=["date", "num_poste"])
        # Pour ne prendre en compte que les situations de neige :
        df = df[df['elevation']>df['LPNX']]
        # On se met dans un répertoire spécifique pour ne pas mélanger les scores avec toutes précip confondues et seulement la neige
        goto('onlysnow')
        nb_obs_min = 10

    # Selection de la période
    df = df.loc[df["date"]>=datetime.date(args.datebegin)].loc[df["date"]<=datetime.date(args.dateend)]
    # Retrait des données non exploitables
    df = df.loc[~df['rr_nivometeo'].isna()].loc[~df[f'rr_{args.product}'].isna()]  # Remove lines with missing value
    df = df.loc[df['massif_number']<99]  # Remove Stations not associated to 1 massif
    df = df.loc[~df['name'].str.contains('EDFNIVO')]  # Remove EDFNIVO stations
    tmp = df.groupby(['num_poste']).date.count()
    tmp = tmp.loc[tmp>nb_obs_min]
    valid_stations = tmp.index.to_numpy()

    # 1. Scatter plot of the error as a function of the observed precipitation value
    #-------------------------------------------------------------------------------
    error_vs_RR(df, args.datebegin, args.dateend, product=args.product)

    df = df.loc[df['num_poste'].isin(valid_stations)]  # Consider only points with a minimum number of observations
    suffix = None
    if args.threshold is not None:
        df = df.loc[df['rr_nivometeo']>args.threshold]  # If a threshold is given, filter data above
        suffix = f'{args.threshold}mm'
        nb_obs_min = 20

    # Calcul des valeurs agrégées par station
    rr_nivometeo  = df.groupby(['num_poste']).rr_nivometeo.mean()
    rr_antilope   = df.groupby(['num_poste'])[f'rr_{args.product}'].mean()
#    if args.product in == 'antilope':
#        rr_antilope   = df.groupby(['num_poste']).rr_antilope.mean()
#    elif args.product in == 'antilopejp1':
#        rr_antilope   = df.groupby(['num_poste']).rr_antilopejp1.mean()
#    else:
#        rr_antilope   = df.groupby(['num_poste']).rr_panthere.mean()
    nb_values     = df.groupby(['num_poste']).date.count()
    elevations    = df.groupby(['num_poste']).elevation.mean()
    lats          = df.groupby(['num_poste']).lat.mean()
    lons          = df.groupby(['num_poste']).lon.mean()
    massif_number = df.groupby(['num_poste']).massif_number.mean()
    names         = df.groupby(['num_poste']).name.first()
    # Regroupement dans une nouvelle dataframe (il est surement possible d'extraire directement cette DF depuis 'df' pour simplifier le code)
    num_poste = df.groupby(['num_poste']).num_poste.mean()
    workdict = {'elevation':elevations, 'rr_nivometeo':rr_nivometeo, f'rr_{args.product}':rr_antilope, 'ndays':nb_values, 'massif_number':massif_number,
            'lats':lats, 'lons':lons, 'num_poste':num_poste}
    #workdict = {'elevation':elevations, 'rr_nivometeo':rr_nivometeo, f'rr_{args.product}':rr_antilope, 'ndays':nb_values, 'massif_number':massif_number, 'lats':lats, 'lons':lons}
    df_stat   = pd.DataFrame(workdict)
    #workdf = workdf.loc[workdf['ndays']>nb_obs_min] # Consider only points with at least 100 observations

    # III- Visualisations des données
    #################################
    if args.massif is not None:
        # Focus sur un unique massif (carte)
        workdf = df.loc[df['massif_number'] == args.massif]
        plot_massif(workdf, args.massif, suffix=suffix, product=args.product)
        daily_scatterplot(workdf, args.datebegin, args.dateend, massif=args.massif, suffix=suffix, product=args.product)
    elif args.subdomain is not None:
        massifs   = subdomain_map[args.subdomain]
        workdf = df.loc[df['massif_number'].isin(massifs)]
        plot_massif(workdf, massifs, subdomain=args.subdomain, suffix=suffix, product=args.product)
        daily_scatterplot(workdf, args.datebegin, args.dateend, subdomain=args.subdomain, suffix=suffix, product=args.product)
    else:
        # 1. Plot rain-gauges informations
        plot_obs(lats.to_numpy(), lons.to_numpy(), elevations.to_numpy(), args.datebegin.strftime('%Y%m%d'), args.dateend.strftime('%Y%m%d'))
        # 2. Raw sactter plot of all availbale stations
        raw_scatterplot(df_stat['rr_nivometeo'].to_numpy(), df_stat[f'rr_{args.product}'].to_numpy(), df_stat['elevation'].to_numpy(), args.datebegin, args.dateend, suffix=suffix, product=args.product)
        # 3. Scatter plot with stations sorted by elevation range
        elevation_scatterplot(df_stat, args.datebegin, args.dateend, suffix=suffix, product=args.product)
        # 4. Daily scatter plot
        daily_scatterplot(df, args.datebegin, args.dateend, suffix=suffix, product=args.product)

        # 5. Maps
        #for domain in ['alpes', 'pyrenees', 'corse']:
        #for domain in ['alpes', 'pyrenees']:
        for domain in ['alpes']:
            plot_full_domain(domain, lats.to_numpy(), lons.to_numpy(), df_stat, suffix=suffix, product=args.product)
            plot_massif(df.loc[df['massif_number'].isin(map_massifs[domain])], suffix=suffix, product=args.product, domain=domain)



