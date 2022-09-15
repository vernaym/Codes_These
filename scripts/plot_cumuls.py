#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os, sys
from datetime import datetime,timedelta
import pandas as pd  # Version 0.25.3
import numpy as np
import xarray as xr
#import copy

import matplotlib.pyplot as plt

import argparse

from pyproj import Proj, transform

#from sklearn.linear_model import LinearRegression, RANSACRegressor
#from sklearn.datasets import make_regression
#from sklearn.metrics import mean_squared_error, r2_score

#from scipy.stats import gaussian_kde

import statistics

##############################################################################################
##############################################################################################

datadir = '/home/vernaym/These/DATA'
savedir = '/home/vernaym/These/figures'

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

blacklist = [5063407, 5063410, 38191408]  # La Meije, LA GRAVE, Huez 2350

landmarks = {
        "Alpe d'Huez" : dict(lon=6.070, lat=45.092, alt=1800, marker='o'),
        "Les 2 Alpes" : dict(lon=6.127, lat=45.013, alt=1800, marker='o'),
        "Lautaret"    : dict(lon=6.408, lat=45.038, alt=2058, marker='X'),
        "La Meije"    : dict(lon=6.311, lat=45.008, alt=3500, marker='^'),  # real alt = 3984
        "Pic Blanc"   : dict(lon=6.131, lat=45.128, alt=3000, marker='^'),  # real alt = 3333
    }

def parse_command_line():
    description = "Evaluation of RADAR products (ANTILOPE or PANTHERE) using nivo-météo network observations"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-b', '--datebegin', help='Begining date of extraction, format YYYYMMDDHH or YYMMDDHH', required=True)
    parser.add_argument('-e', '--dateend', help = 'Final date of extraction (default=datebegin)')
    parser.add_argument('-d', '--domain', nargs='+', help='Domain of the file', choices=['alp', 'pyr', 'cor', 'GrandesRousses'], default='GrandesRousses')
    parser.add_argument('-w', '--workdir', help='Runing directory (default for guppy)', default='/home/mrns/vernaym/workdir/extraction_antilope')
    parser.add_argument('-m', '--massif', help='PLot for a specific massif', default=None, type=int)
    parser.add_argument('-t', '--threshold', default=None, help='Threshold of precipitation (mm) to apply in the data to consider', type=int)
    parser.add_argument('-p', '--product', default='antilopejp1', help='Product to deal with', choices=['antilope', 'antilopejp1', 'panthere', 'kriging'])
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

# Definition of gamma distribution
def gamma_shape_PDF(x, k=3, theta=1):
    import math
    if isinstance(x, np.ndarray):
        return list(map(lambda t: t**(k-1)*np.exp(-t/theta)/(theta**k*math.gamma(k)) if t > 0 else None, x))
    else:
        return x**(k-1)*np.exp(-x/theta)/(theta**k*math.gamma(k)) if x > 0 else None

# Definition of SCGD
def SCGD_shape_PDF(x, k=1., theta=1., delta=0., plot_distribution=False, plot_parameters=False):
    if not (isinstance(x, (list, np.ndarray))):
        x = np.array([x])
    # On n'impose pas de condition sur delta : on peut vouloir translater la distribution vers la droite ou vers la gauche
#    if delta > 0:
#        # WARNING : dans Schuerer and Hammil 2015 and Nusu 2019 delat est définit comme > 0, ce qui entraine un
#        # décallage vers la droite plutot que vers la gauche...
#        raise ValueError('Error : delta parameter must be <= 0')
    dy = 0.01

    # WARNING : la distribution obtenue sera discontinue en 0 et potentiellement accordera trop / trop peu de
    # poids aux precipitations nulles
    y = np.arange(delta, theta*(k-1)+delta+3*k*theta**2, dy)  # WARNING : 'y' doit ABSOLUMENT couvrir toute la distribution
                                    # de gamma sinon la probabilité en 0 est artificiellement surestimée !
    gamma = np.array(list(map(
        lambda t: gamma_shape_PDF(t-delta, k=k, theta=theta) if t > delta else 0, y)))
    scgd0 = 1 - sum(gamma[y>0]*dy)  # Calcul de la probabilité résiduelle en 0

    if plot_distribution:
        fig,ax = plt.subplots()
        color = next(ax._get_lines.prop_cycler)['color']
        # Plot over a smaller range for better lisibility
        ymin = theta*(k-1)+delta-k*theta**2
        ymax = theta*(k-1)+delta+1.5*k*theta**2
        ax.plot(y[y<ymax], gamma[y<ymax], linestyle=':', color=color)
        ax.plot(y[(y>0) & (y<ymax)], gamma[(y>0) & (y<ymax)],
                label=f'SCGD (k={k:0.2}, theta={theta:0.2}, delta={delta:0.2}, mu={mu1:0.2}, bias={mu0:0.2}, sigma={sigma:0.2})', color=color)
        plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        plt.axvline(x=Y, color='k', linestyle='--', label=f'Observation : {Y:0.2}')
        plt.axvline(x=mu1, color='red', linestyle='--', label=f'Observation + mean bias : {mu1:0.2}')
        plt.axvline(x=mu, color='blue', linestyle='--', label=f'Artificial max : {mu:0.2}')
        if scgd0 > 0.0001:
            ax.plot(0, scgd0, marker='.', markersize=20, markeredgecolor=color, markeredgewidth=2,
                     color='none', markerfacecolor='lightgrey', label=f'P(0)={scgd0:0.3}')
            ax.fill_between(x=y, y1=gamma, where=(delta < y) & (y < 0), color='lightgrey')
        #ax.bar(0, scgd0, width=0.2, align='edge', color=color)
        #ax.bar(0, scgd0, width=0.2, color=color)
        plt.legend(loc ="upper right")
        fig.savefig(f"PDF/station_{station}_{Y}mm.svg", format='svg')

    if plot_parameters:
        #plt.plot(Y, k, linestyle='', marker='.', color='blue', label='k')
        #plt.plot(Y, theta, linestyle='', marker='+', color='blue', label='theta')
        plt.plot(Y, delta, linestyle='', marker='+', color='k', label='delta')
        plt.plot(Y, scgd0, linestyle='', marker='+', color='red', label='P(0)')
        plt.plot(Y,gamma_shape_PDF(0.1-delta, k=k, theta=theta), marker='+', color='blue', label='P0+')

        if Y == 0:
            plt.legend(loc ="upper right")

    return np.array(list(map(lambda t: gamma_shape_PDF(t-delta, k=k, theta=theta) if t > 0
                    else scgd0 if t == 0 else 0, x)))

def read_ensemble(datebegin, dateend, domain='GrandesRousses'):
    pearome = dict()
    for member in range(1,17):
        filename = f'pearome_{member:03d}_{datebegin}_{dateend}_{domain}.nc'
        filename = os.path.join(datadir, filename)
        pearome[member] = xr.open_dataset(filename)

    return pearome

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

def add_obs_nivometeo():
    nivometeo = pd.read_csv(os.path.join(datadir, 'obs_nivometeo_daily_RR_20211201_20220430.csv'), sep=';', parse_dates=['Q.dat'],
        dtype={'Q.num_poste':int, 'poste_nivo.nom_usuel':str, 'poste_nivo.alti':int, 'poste_nivo.lat_dg':float, 'poste_nivo.lon_dg':float, 'poste_nivo.massif_nivo':int, 'Q.rr':float})

    dist = lambda dx,dy: np.sqrt(dx**2+dy**2)

    nivometeo = nivometeo.loc[(nivometeo['poste_nivo.lat_dg']>=latmin) & (nivometeo['poste_nivo.lat_dg']<=latmax) & (nivometeo['poste_nivo.lon_dg']>=lonmin) & (nivometeo['poste_nivo.lon_dg']<=lonmax)]
    keep = list()
    biais_antilope = list()
    biais_krigeage = list()
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
                biais_krigeage.append(krigeage.rr_cumul[idx].data[0][0]-cumul)
            else:
                print(f'Station {num_poste} is droped (too many missing obs : {missing_days:d})')

    #print(first_day, last_day)
    #print(keep)
    return biais_antilope, biais_krigeage, lon, lat

def nextax(i,j):
    j = j + 1
    if j==2:
        j = 0
        i = i + 1
    return i, j

def plot(data, vmin, vmax):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,7))
    fig.suptitle('Total precipitation between {0:s} and {1:s}'.format(args.datebegin.strftime('%Y%m%d'), args.dateend.strftime('%Y%m%d')), fontsize=16)

    i = 0
    j = 0
    for product, field in data.items():
        if product == 'ANTILOPEVSAROME':
            field.rr_cumul.plot(ax=ax[i,j], cmap=plt.cm.gist_ncar)
        else:
            field.rr_cumul.plot(ax=ax[i,j], vmin=vmin, vmax=vmax, cmap=plt.cm.gist_ncar)
        ax[i,j].set_title(product)
        ax[i,j].axis('off')
#        if args.datebegin == datetime(2021, 12, 1) and args.dateend == datetime(2022, 4, 30):
#            for (c,x,y) in zip(biais_antilope, lon, lat):
#                ax[0,0].annotate('{0:.2f}'.format(c), (x, y))
        plt.sca(ax[i,j])
        # Add landmarks
        for landmark, infos in landmarks.items():
            plt.plot(infos['lon'], infos['lat'], marker=infos['marker'], color='red', markersize=5)
            plt.annotate(landmark, (infos['lon']+0.003, infos['lat']+0.003), color='red', fontsize=12)
        i,j = nextax(i,j)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    fig.savefig(os.path.join(savedir, 'CUMULS_{0:s}_{1:s}.pdf'.format(args.datebegin.strftime('%Y%m%d'), args.dateend.strftime('%Y%m%d'))))

if __name__ == "__main__":
    args = parse_command_line()
    extract_period = date_range(args.datebegin, args.dateend)
    mnt = xr.open_dataset(os.path.join(datadir, "MNT_GrandesRousses.nc"))
    mnt_proj = proj_mnt(mnt)
    panthere = xr.open_dataset(os.path.join(datadir, 'PANTHERE_CUMUL_{0:s}_{1:s}.nc'.format(args.datebegin.strftime('%Y%m%d%H%M'), args.dateend.strftime('%Y%m%d%H%M'))))
    antilope = xr.open_dataset(os.path.join(datadir, 'CUMUL_ANTILOPEQ_GrandesRousses_{0:s}_{1:s}.nc'.format(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'))))
    arome    = xr.open_dataset(os.path.join(datadir, 'arome_{0:s}_{1:s}_GrandesRousses.nc'.format(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'))))
    arome = arome.sum('time').rename({'rr':'rr_cumul'})
    pearome  = xr.open_dataset(os.path.join(datadir, 'pearome_001_{0:s}_{1:s}_GrandesRousses.nc'.format(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'))))
    pearome = pearome.sum('time').rename({'rr':'rr_cumul'})
#    krigeage = xr.open_dataset(os.path.join(datadir, 'CUMUL_krigeage_{0:s}_{1:s}.nc'.format(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'))))
#    safran   = xr.open_dataset(os.path.join(datadir, 'CUMUL_SAFRAN_GrandesRousses_{0:s}_{1:s}.nc'.format(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'))))
#    safran = safran.rr.sum(axis=0).interp(elevation=mnt_proj, method='nearest')

    if args.datebegin == datetime(2021, 12, 1) and args.dateend == datetime(2022, 4, 30):
        biais_antilope, biais_krigeage, lon, lat = add_obs_nivometeo()

    # Interp all data on the ANTILOPE grid over the GrandesRousses domain
    antilope = antilope.where((antilope.lon>=lonmin) & (antilope.lon<=lonmax) & (antilope.lat<=latmax) & (antilope.lat>=latmin), drop=True)
    panthere = panthere.where((panthere.lon>=lonmin) & (panthere.lon<=lonmax) & (panthere.lat<=latmax) & (panthere.lat>=latmin), drop=True)
    arome = arome.where((arome.lon>=lonmin) & (arome.lon<=lonmax) & (arome.lat<=latmax) & (arome.lat>=latmin), drop=True)
    pearome = pearome.where((pearome.lon>=lonmin) & (pearome.lon<=lonmax) & (pearome.lat<=latmax) & (pearome.lat>=latmin), drop=True)

    #krigeage = krigeage.interp(lon=antilope.lon, lat=antilope.lat, method='nearest')
    #safran   = safran.where((safran.lon>=lonmin) & (safran.lon<=lonmax) & (safran.lat<=latmax) & (safran.lat>=latmin), drop=True)
    #panthere = panthere.interp(lon=antilope.lon, lat=antilope.lat, method='nearest')

#    vmax = max([np.max(antilope.rr_cumul), np.max(panthere.rr_cumul), np.max(krigeage.rr_cumul), np.min(safran)])
#    vmin = min([np.min(antilope.rr_cumul), np.min(panthere.rr_cumul), np.min(krigeage.rr_cumul), np.min(safran)])

    antilopevsarome = (antilope / np.max(antilope.rr_cumul)) / (arome / np.max(arome.rr_cumul))

    #vmax = max([np.max(antilope.rr_cumul), np.max(panthere.rr_cumul), np.max(arome.rr_cumul), np.max(pearome.rr_cumul)])
    #vmin = min([np.min(antilope.rr_cumul), np.min(panthere.rr_cumul), np.min(arome.rr_cumul), np.min(pearome.rr_cumul)])
    vmax = max([np.max(antilope.rr_cumul), np.max(panthere.rr_cumul), np.max(arome.rr_cumul)])
    vmin = min([np.min(antilope.rr_cumul), np.min(panthere.rr_cumul), np.min(arome.rr_cumul)])
    data = dict(
        ANTILOPE = antilope,
        PANTHERE = panthere,
        AROME    = arome,
        #PEAROME  = pearome,
        ANTILOPEVSAROME  = antilopevsarome,
    )

    plot(data, vmin, vmax)




