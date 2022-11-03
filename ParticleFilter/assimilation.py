#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os, sys
from datetime import datetime,timedelta
import pandas as pd  # Version 0.25.3
import numpy as np
import xarray as xr
import glob
#import copy

import argparse

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # F401 unused import --> to ignore !
from matplotlib.text import Annotation
from matplotlib import offsetbox
import seaborn as sns

from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.proj3d import proj_transform

import time

##############################################################################################
# TODO : Save number of selected members for each pixel
# TODO : localisation : build a bigger ensemble by taking into account values of
# neighboring pixels in space and time
# TODO : plot one specific point
# TODO : Add option to switch on/off the observation error mask
##############################################################################################

datadir = '/home/vernaym/These/DATA'

# Domaine des Grandes Rousses
domain_coords = dict(
        GrandesRousses = dict(latmax=45.240, latmin=44.990, lonmin=6.010, lonmax = 6.490),
        alp            = dict(latmax=46.450, latmin=44.100, lonmin=5.400, lonmax=7.200),
)

suffix = dict(hourly='H', daily='Q')
timestep = dict(hourly=1, daily=24)

norm = plt.Normalize()

zoom = dict(num_poste=73306403, lat=45.160833, lon=6.463500)
zoom = dict(num_poste=73173400, lat=45.227667, lon=6.404000)
zoom = dict(num_poste=38020400, lat=45.057167, lon=6.076333)

landmarks = {
        "Alpe d'Huez" : dict(lon=6.070, lat=45.092, alt=1800, marker='o'),
        "Les 2 Alpes" : dict(lon=6.127, lat=45.013, alt=1800, marker='o'),
        "Lautaret"    : dict(lon=6.408, lat=45.038, alt=2058, marker='X'),
        "La Meije"    : dict(lon=6.311, lat=45.008, alt=3500, marker='^'),  # real alt = 3984
        "Pic Blanc"   : dict(lon=6.131, lat=45.128, alt=3000, marker='^'),  # real alt = 3333
        "Valloire"    : dict(lon=6.463500, lat=45.160833, alt=3000, marker='p'),
    }

def parse_command_line():
    description = "Evaluation of RADAR products (ANTILOPE or PANTHERE) using nivo-météo network observations"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-b', '--datebegin', help='Begining date of extraction, format YYYYMMDDHH or YYMMDDHH', required=True)
    parser.add_argument('-e', '--dateend', help = 'Final date of extraction (default=datebegin)')
    parser.add_argument('-d', '--domain', help='Domain of the file', choices=['alp', 'pyr', 'cor', 'GrandesRousses'], default='GrandesRousses')
    parser.add_argument('-w', '--workdir', help='Runing directory', default='/home/vernaym/workdir/ASSIMILATION')
    parser.add_argument('-m', '--massif', help='PLot for a specific massif', default=None, type=int)
    parser.add_argument('-t', '--threshold', default=None, help='Threshold of precipitation (mm) to apply in the data to consider', type=int)
    parser.add_argument('-p', '--plot', action='store_true', default=False, help='Plot assimilated fields')
    parser.add_argument('-f', '--frequency', choices=['hourly', 'daily'], help='Assimilation frequency')
    parser.add_argument('-l', '--lpn', action='store_true', help='Take into account the rain-snow limit')
    parser.add_argument('-g', '--gridded', default=False, action='store_true', help='Gridded assimilation if True, or assimilation only at nivometeo locations if False')

    args = parser.parse_args()

    args.datebegin = get_date(args.datebegin)
    if args.dateend:
        args.dateend = get_date(args.dateend)
    else:
        args.dateend = args.datebegin + timedelta(hours=24)

    return args

def speedtest(function):
    def wrapper(*args, **kw):
        t1 = time.time()
        result = function(*args, **kw)
        t2 = time.time()
        print('The method {0:s} took {1:f}ms'.format(function.__name__, (t2-t1)*1000.0))
        return result
    return wrapper

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
    start = start.replace(hour=6) - timedelta(hours=24) + timedelta(hours=dt)
    end = end.replace(hour=6)
    dates = list()
    while start <= end:
        dates.append(start)
        start += timedelta(hours=dt)

    return dates

def goto(path):
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)

@speedtest
def read_ensemble(datebegin, dateend, frequency, domain):

    #filenames = [os.path.join(datadir, f"aspearome_{mb:03d}_{datebegin}_{dateend}_{domain}_{frequency}.nc") for mb in range(1,17)]
    #filenames = [os.path.join(datadir, f"aspearome_{mb:03d}_2021073106_2022070106_{domain}_{frequency}.nc") for mb in range(1,17)]
    # TODO : problème de taille des données pour une assimilation horaire sur le domaine alp complet
    # - réduire le domaine lors de la lecture (Extraction_PEAROME_bdap.py) ?
    # - réduire la durée au maximum ?
    # - Réduire la periode des fichiers (mensuel/journalier) ?  ==> plusieur lectures
    filenames = [os.path.join(datadir, f"aspearome_{mb:03d}_2021102806_2022060206_{domain}_hourly.nc") for mb in range(1,17)]

    # open_mfdataset returns a dask.array<chunksize=(...), meta=np.ndarray> object that divides arrays into many small pieces, called chunks, 
    # each of which is presumed to be small enough to fit into memory in order to avoid a memory overload. 
    # The compute() method effectively load data into memory so it must be called as late as possible to reduce memory use as well as
    # running time.
    # See : https://docs.xarray.dev/en/stable/user-guide/dask.html
    #pearome = xr.open_mfdataset(filenames, combine='nested', concat_dim='member').compute()  # Impossible avec des fichiers trop volumineux
    pearome = xr.open_mfdataset(filenames, combine='nested', concat_dim='member')
    pearome['member'] = np.arange(1,17)
    if frequency == 'daily':
        # Convert hourly precipitation into 24h precipitation between 6h J-1 and 6h J
        # Problem : the xarray tools to do that allows only accumulations between
        # 0h and 23h.
        # solution : shift time serie by 7h, compute 24h accumulations and
        # shift back !
        pearome['time'] = pearome.time-np.timedelta64(7, 'h')
        pearome = pearome.resample(time='D').sum(dim='time')  # !!! VERY SLOW !!!
        pearome['time'] = pearome.time+np.timedelta64(30, 'h')

    return pearome

@speedtest
def read_nivometeo_obs(domain='alp'):

    nivometeo = pd.read_csv(os.path.join(datadir, 'obs_nivometeo_daily_RR_20211201_20220430.csv'), sep=';', parse_dates=['date'], header=0,
            names=['date', 'num_poste', 'nom', 'alti', 'lat', 'lon', 'massif', 'obs', 'unused'],
            usecols=['date', 'num_poste', 'nom', 'alti', 'lat', 'lon', 'obs'],
            dtype={'num_poste':int, 'nom':str, 'alti':int, 'lat':float, 'lon':float, 'obs':float},
        )

    latmax = domain_coords[domain]['latmax']
    latmin = domain_coords[domain]['latmin']
    lonmin = domain_coords[domain]['lonmin']
    lonmax = domain_coords[domain]['lonmax']
    nivometeo = nivometeo.loc[(nivometeo['lat']>=latmin) & (nivometeo['lat']<=latmax) & (nivometeo['lon']>=lonmin) & (nivometeo['lon']<=lonmax)]  # Select area
    nivometeo.date = nivometeo.date + pd.Timedelta("1d6h")   #BDClim extraction for date ymd is the observation from ymd6h to ym(d+1)6h
    #nivometeo.groupby('num_poste')['nom', 'lat', 'lon', 'alti'].agg(set)
    #nivometeo = nivometeo.set_index(['num_poste', 'lat', 'lon', 'nom', 'alti', 'date'])  # Utilité de passer en index ?
    nivometeo.set_index(['num_poste','date'], inplace=True)

    return nivometeo.to_xarray()

def read_ensemble_old(datebegin, dateend, frequency, domain='GrandesRousses'):

    pearome = dict()
    for member in range(1,17):
        filename = f'aspearome_{member:03d}_{datebegin}_{dateend}_{domain}_{frequency}.nc'
        filename = os.path.join(datadir, filename)
        if not os.path.exists(filename):
            print(f'WARNING : file {filename} does not exist, using default file')
            filename = os.path.join(datadir, f'aspearome_{member:03d}_2021080106_2022070106_GrandesRousses.nc')
        if not os.path.exists(filename):
            print(f'WARNING : no file named {filename} under {datadir}, using default file aspearome_{member:03d}_2021073106_2022070106_GrandesRousses_{frequency}.nc')
            filename = os.path.join(datadir, f'aspearome_{member:03d}_2021073106_2022070106_GrandesRousses_{frequency}.nc')
        model = xr.open_dataset(filename).clip(0)  # Avoid <0 values
        # Extract domain of interest :
        model = model.where((model.lon>=lonmin-0.25) & (model.lon<=lonmax+0.25) & (model.lat>=latmin-0.25) & (model.lat<=latmax+0.25), drop=True)
        pearome[member] = model

    return pearome

class Annotation3D(Annotation):
    """ From : https://datascience.stackexchange.com/questions/11430/how-to-annotate-labels-in-a-3d-matplotlib-scatter-plot"""

    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)

class ImageAnnotations3D():
    """ From : https://discuss.dizzycoding.com/matplotlib-3d-scatter-plot-with-images-as-annotations/ """

    #def __init__(self, xyz, imgs, ax3d,ax2d):
    def __init__(self, xyz, text, ax3d, ax2d):
        self.xyz = xyz
        #self.imgs = imgs
        self.text = text
        self.ax3d = ax3d
        self.ax2d = ax2d
        self.annot = []
        #for s,im in zip(self.xyz, self.imgs):
        for s,txt in zip(self.xyz, self.text):
            x,y = self.proj(s)
            self.annot.append(self.annotation(txt,[x,y]))
            #self.annot.append(self.image(im,[x,y]))
        self.lim = self.ax3d.get_w_lims()
        self.rot = self.ax3d.get_proj()
        self.cid = self.ax3d.figure.canvas.mpl_connect("draw_event",self.update)

        self.funcmap = {"button_press_event" : self.ax3d._button_press,
                        "motion_notify_event" : self.ax3d._on_move,
                        "button_release_event" : self.ax3d._button_release}

        self.cfs = [self.ax3d.figure.canvas.mpl_connect(kind, self.cb) 
                        for kind in self.funcmap.keys()]

    def cb(self, event):
        event.inaxes = self.ax3d
        self.funcmap[event.name](event)

    def proj(self, X):
        """ From a 3D point in axes ax1, 
            calculate position in 2D in ax2 """
        x,y,z = X
        x2, y2, _ = proj3d.proj_transform(x,y,z, self.ax3d.get_proj())
        tr = self.ax3d.transData.transform((x2, y2))
        return self.ax2d.transData.inverted().transform(tr)

    def image(self, arr, xy):
        """ Place an image (arr) as annotation at position xy """
        im = offsetbox.OffsetImage(arr, zoom=2)
        im.image.axes = ax
        ab = offsetbox.AnnotationBbox(im, xy, xybox=(-30., 30.),
                            xycoords='data', boxcoords="offset points",
                            pad=0.3, arrowprops=dict(arrowstyle="->"))
        self.ax2d.add_artist(ab)
        return ab

    def annotation(self, ann, xy):
        """ Place an annotation (ann) at position xy """
        #an = Annotation3D(ann, xy, arrowprops=dict(arrowstyle="-|>", ec='black'))
        an = Annotation(ann, xy, arrowprops=dict(arrowstyle="-|>", ec='black'),)
            #                textcoords="offset points")
        self.ax2d.add_artist(an)
        return an

    def update(self, event):
        if np.any(self.ax3d.get_w_lims() != self.lim) or np.any(self.ax3d.get_proj() != self.rot):
            self.lim = self.ax3d.get_w_lims()
            self.rot = self.ax3d.get_proj()
            for s,ab in zip(self.xyz, self.annot):
                ab.xy = self.proj(s)

def plot3D(X, Y, Z, colors, date):
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

    # Create a dummy axes to place annotations to
    ax2 = fig.add_subplot(111,frame_on=False) 
    ax2.axis("off")
    ax2.axis([0,1,0,1])

    #ia = ImageAnnotations3D(np.array([[6.073, 45.095, 1800],]), "Alpe d'huez", ax, ax2 )
    ia = ImageAnnotations3D(
            np.array([[d['lon'], d['lat'], d['alt']] for d in landmarks.values()]),
            list(landmarks.keys()), ax, ax2)

    #ax.set_zlim(0., np.max(Z))
    ax.set_zlim(0., 3500.)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.cm.coolwarm), ax=ax, shrink=0.75, aspect=8, label=f'ANTILOPE precipitation (mm)')
    plt.savefig(f'{date}/OBS_3D_{date}.pdf', format='pdf')


#def plot_field(field, ax, vmin, vmax, title, cmap=plt.cm.YlGnBu):
def plot_field(field, ax, vmin, vmax, title, cmap='viridis'):
    im = field.plot(ax=ax, add_colorbar=False, vmin=vmin, vmax=vmax, cmap=cmap)
    for landmark, infos in landmarks.items():
        ax.plot(infos['lon'], infos['lat'], marker=infos['marker'], color='red', markersize=4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)

    return im

@speedtest
def finalize_fig(figure, imm, label, outname):
    # Saving figures is by far the slowest part (~1s per figure)

    #t1 = time.time()
    figure.tight_layout()
    figure.subplots_adjust(right=0.85)
    cbar_ax = figure.add_axes([0.90, 0.15, 0.05, 0.7])
    figure.colorbar(imm, cax=cbar_ax, label=label)
    #t2 = time.time()
    #print(f'Finalising figures took {(t2-t1)*1000.}ms')
    figure.savefig(outname, format='pdf')
    #t3 = time.time()
    #print(f'Saving figures took {(t3-t2)*1000.}ms')

@speedtest
def read_obs(args):
    filename = f'ANTILOPE{suffix[args.frequency]}_{args.datebegin.strftime("%Y%m%d%H")}_{args.dateend.strftime("%Y%m%d%H")}_{args.domain}.nc'
    if not os.path.exists(filename):
        print(f'WARNING : file {filename} does not exist, looking for it under {datadir}')
        filename = os.path.join(datadir, filename)
    if not os.path.exists(filename):
        #print(f'WARNING : no file named {filename} under {datadir}, using default file ANTILOPEQ_2021080106_2022070106_GrandesRousses.nc')
        #filename = os.path.join(datadir, f'ANTILOPE{suffix[args.frequency]}_2021073106_2022070106_GrandesRousses.nc')
        print(f'WARNING : no file named {filename} under {datadir}, using default file ANTILOPEH_2021103000_2022060200_alp.nc')
        filename = os.path.join(datadir, f'ANTILOPEH_2021103000_2022060200_alp.nc')
    if os.path.exists(filename):
        antilope = xr.open_dataset(filename)
        antilope = antilope.transpose('lat', 'lon', 'time')  # TODO : Fix the dataset in the generation script
        # WARNING :  la commande suivante réduit sensibement le domaine, attention aux comparaisons entre figures (en particulier avec les CUMULS)
        latmax = domain_coords[args.domain]['latmax']
        latmin = domain_coords[args.domain]['latmin']
        lonmin = domain_coords[args.domain]['lonmin']
        lonmax = domain_coords[args.domain]['lonmax']
        antilope = antilope.where((antilope.lon>=lonmin) & (antilope.lon<=lonmax) & (antilope.lat>=latmin) & (antilope.lat<=latmax), drop=True)
        # Pour une assimilation quotidienne, sommer les cumuls horaires
        if args.frequency == 'daily' and  'ANTILOPEH' in filename:
            # Convert hourly precipitation into 24h precipitation between 6h J-1 and 6h J
            # Problem : the xarray tools to do that allows only accumulations between
            # 0h and 23h.
            # solution : shift time serie by 7h, compute 24h accumulations and
            # shift back !
            antilope['time'] = antilope.time-np.timedelta64(7, 'h')
            antilope = antilope.resample(time='D').sum(dim='time')  # !!! VERY SLOW !!!
            antilope['time'] = antilope.time+np.timedelta64(30, 'h')
    else:
        print(f'ERROR : file {filename} does not exist')
        sys.exit(1)
    return antilope


class ParticleFilter(object):

    def __init__(self, period, obs, ensemble, nivometeo, plot, frequency, gridded):

        #goto(self.date_str)
        #self.nline, self.ncol = np.shape(obs.rr.data)
        self.period = period
        self.frequency = frequency

        # To look at one specific point
        self.zoom_x = np.argwhere(obs.lat.data == nearest(obs.lat, zoom['lat']))[0][0]
        self.zoom_y = np.argwhere(obs.lon.data == nearest(obs.lon, zoom['lon']))[0][0]
        self.zoom_lon = nearest(obs.lon, zoom['lon'])
        self.zoom_lat = nearest(obs.lat, zoom['lat'])

        self.radar = obs.transpose('lon','lat','time')
        self.ensemble = ensemble
        self.members = self.ensemble.member
        self.Ne = len(self.members)
        self.plot = plot

        # Plot observation field
        if self.plot:
            self.plot_obs()

        self.gridded = gridded
        self.nivometeo = nivometeo

    # Definition of gamma distribution
    def gamma_shape_PDF(self, x, k=3., theta=1.):
        #import math  # math.gamma does not work with arrays
        from scipy.special import gamma

        return x**(k-1)*np.exp(-x/theta)/(theta**k*gamma(k))

#    if isinstance(x, np.ndarray):
#        return list(map(lambda t: t**(k-1)*np.exp(-t/theta)/(theta**k*gamma(k)) if t > 0 else None, x))
#    else:
#        return x**(k-1)*np.exp(-x/theta)/(theta**k*gamma(k)) if x > 0 else None

    # Definition of SCGD
    def SCGD_shape_PDF(self, x, mu, k=1., theta=1., delta=0., plot_distribution=False, plot_parameters=False):
#    if not (isinstance(x, (list, np.ndarray))):
#        x = np.array([x])
        # On n'impose pas de condition sur delta : on peut vouloir translater la distribution vers la droite ou vers la gauche
#    if delta > 0:
#        # WARNING : dans Schuerer and Hammil 2015 and Nusu 2019 delat est définit comme > 0, ce qui entraine un
#        # décallage vers la droite plutot que vers la gauche...
#        raise ValueError('Error : delta parameter must be <= 0')

        # On veut des poids relatifs, pas besoin de modifier artificiellement la probabilité de précipitations nulles :
        # la valeur fournie par la SCGD convient parfaitement et assure la continuité des poids.
        # Le fait de ne pas avoir la somme des poids à priori égale à 1 (la "masses" en desous de 0 est perdue) n'est 
        # pas important puisque les poids sont normés à posteriori.
        # WARNING : la distribution obtenue sera discontinue en 0 et potentiellement accordera trop / trop peu de
        # poids aux precipitations nulles
#    dy = 0.01
#    num = np.max(((4*k + delta) / dy).astype(int))
#    y = np.linspace(-delta, 4*k, num=num)  # WARNING : 'y' doit ABSOLUMENT couvrir toute la distribution
#                                    # de gamma sinon la probabilité en 0 est artificiellement surestimée !
#    gamma = np.array(list(map(
#        lambda t: gamma_shape_PDF(t+delta, k=k, theta=theta) if t > delta else 0, y)))
#    scgd0 = 1 - sum(gamma[y>0]*dy)  # Calcul de la probabilité résiduelle en 0

        if plot_distribution:
            dy = 0.01
            num = np.max(((4*k + delta) / dy).astype(int))
            y = np.linspace(-delta, 20*k, num=num)  # WARNING : 'y' doit ABSOLUMENT couvrir toute la distribution
                                            # de gamma sinon la probabilité en 0 est artificiellement surestimée !
            gamma = np.array(list(map(
                #lambda t: gamma_shape_PDF(t+delta, k=k, theta=theta) if t > delta else 0, y)))
                lambda t: self.gamma_shape_PDF(t+delta, k=k, theta=theta) if t > 0 else 0, y)))
            fig,ax = plt.subplots()
            color = next(ax._get_lines.prop_cycler)['color']
            # Plot over a smaller range for better lisibility
            ymin = theta*(k-1)+delta-k*theta**2
            ymax = theta*(k-1)+delta+10*k*theta**2
            ax.plot(x, self.gamma_shape_PDF(x+delta, k=k, theta=theta), linestyle='', marker='+', markersize=10.)
            ax.plot(y[y<ymax], gamma[y<ymax], linestyle=':', color=color)
            ax.plot(y[(y>0) & (y<ymax)], gamma[(y>0) & (y<ymax)],
                    #label=f'SCGD (k={k:0.2}, theta={theta:0.2}, delta={delta:0.2}, mu={mu1:0.2}, bias={mu0:0.2}, sigma={sigma:0.2})', color=color)
                    label=f'SCGD (k={k:0.2}, theta={theta:0.2}, delta={delta:0.2})', color=color)
            plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            #mu = k*theta 
            plt.axvline(x=mu, color='k', linestyle='--', label=f'Observation : {mu:0.2}')
            #plt.axvline(x=mu1, color='red', linestyle='--', label=f'Observation + mean bias : {mu1:0.2}')
            #plt.axvline(x=mu, color='blue', linestyle='--', label=f'Artificial max : {mu:0.2}')
#        if scgd0 > 0.0001:
#            ax.plot(0, scgd0, marker='.', markersize=20, markeredgecolor=color, markeredgewidth=2,
#                     color='none', markerfacecolor='lightgrey', label=f'P(0)={scgd0:0.3}')
#            ax.fill_between(x=y, y1=gamma, where=(delta < y) & (y < 0), color='lightgrey')
            #ax.bar(0, scgd0, width=0.2, align='edge', color=color)
            #ax.bar(0, scgd0, width=0.2, color=color)
            plt.xlabel('24-hour precipitation (mm)')
            plt.ylabel('Weight')
            plt.legend(loc ="upper right")
            fig.savefig(f"PDF00.pdf", format='pdf')

        if plot_parameters:
            #plt.plot(Y, k, linestyle='', marker='.', color='blue', label='k')
            #plt.plot(Y, theta, linestyle='', marker='+', color='blue', label='theta')
            plt.plot(Y, delta, linestyle='', marker='+', color='k', label='delta')
            #plt.plot(Y, scgd0, linestyle='', marker='+', color='red', label='P(0)')
            plt.plot(Y,self.gamma_shape_PDF(0.1+delta, k=k, theta=theta), marker='+', color='blue', label='P0+')

            if Y == 0:
                plt.legend(loc ="upper right")

#    return np.array(list(map(lambda t: gamma_shape_PDF(t-delta, k=k, theta=theta) if t > 0
#                    else scgd0 if t == 0 else 0, x)))
        draw = self.gamma_shape_PDF(x+delta, k=k, theta=theta)
        #draw[np.where(x<0)] = 0  # x is already a precipitation field with >0 values
        return draw

    def plot_obs(self):
        #mnt = xr.open_dataset(os.path.join(datadir, "MNT_GrandesRousses.nc"))
        mnt = xr.open_dataset('/home/vernaym/QGIS/MNT/DEM_ALPES_WGS84_250m_bilinear.nc')  # Pour tracer sur toutes les Alpes
        # Plot ANTILOPE precipitation field
        fig = plt.figure(figsize=(18,8))
        #radar.transpose('lat', 'lon').rr.plot(vmin=rrmin, vmax=rrmax, cbar_kwargs={'label': "24 hour precipitation (mm)"})  # quadmesh object
        radar.rr.plot(vmin=self.rrmin, vmax=self.rrmax, cbar_kwargs={'label': "24 hour precipitation (mm)"})  # quadmesh object
        # Add landmarks
        for landmark, infos in landmarks.items():
            plt.plot(infos['lon'], infos['lat'], marker=infos['marker'], color='red', markersize=10)
            plt.annotate(landmark, (infos['lon']+0.003, infos['lat']+0.003), color='red', fontsize=20)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_title(f'Date {self.date_str}', fontsize=20)
        ax.axes.get_xaxis().get_label().set_visible(False)
        ax.axes.get_yaxis().get_label().set_visible(False)
        fig.tight_layout()
        fig.savefig(f'{self.date_str}/OBS_{self.date_str}.pdf', format='pdf')

        # Plot 3D ANTILOPE precipitation field
        tmp = mnt.interp(lon=self.radar.lon, lat=self.radar.lat, method='nearest')  # Pour interpoller le MNT sur la grille ANTILOPE
        # WARNING :  la commande suivante réduit sensibement le domaine, attention aux comparaisons entre figures (en particulier avec les CUMULS)
        #tmp = tmp.where((tmp.lon>=lonmin) & (tmp.lon<=lonmax) & (tmp.lat>=latmin) & (tmp.lat<=latmax), drop=True)
        X, Y = np.meshgrid(tmp['lon'].values, tmp['lat'].values)
        Z = np.nan_to_num(tmp['Band1'].values)
        # define pixel colors
        #colors = plt.cm.coolwarm(norm(np.nan_to_num(radar.transpose('lat', 'lon').rr.data)))
        colors = plt.cm.coolwarm(norm(np.nan_to_num(self.radar.rr.data)))
        plot3D(X, Y, Z, colors, self.date_str)

    @speedtest
    def pdf_parameters(self):
        # I. Define PDF parameters
        #-------------
        parameters = self.radar.copy()  # TODO : vérifier si ce n'est pas trop couteux (de toutes façon l'bs n'a pas de raison d'être modifiée)
        parameters = parameters.rename({'rr':'mu'})  #  Mode=observation (WARNING : mu is NOT the mean) TODO : check if the ensemble after assimilation is not biased

        # Observation error
        # Import multiplicative mask to increase observation error where necessary

        # MASK NOT YET AVAILABLE FOR THE WHOLE ALP DOMAIN ==> TODO
        #mask = xr.open_dataset(os.path.join(datadir, "mask_error_antilope_GrandesRousses.nc"))
        #parameters['sigma'] = (0.261 + 0.263 * parameters['mu'])*mask.mask  # According to the linear regression of ANTILOPE RMSE vs ANTILOPE RR
        parameters['sigma'] = (0.261 + 0.263 * parameters['mu'])  # According to the linear regression of ANTILOPE RMSE vs ANTILOPE RR

        ####################################################################################################################################################################
        # USELESS with gaussian distribution
        # shift of the gamma PDF ==> this defines the weight given to 0mm forecats (=0 if delta=0) !!
        #parameters = parameters.assign(delta=lambda x: 1/(1+x.mu))  # TODO : find a better shift than 10/mu
        #parameters.delta.data[np.isinf(parameters.delta.data)] = 0  # Si l'obs est nulle on veut imposer une PDF exponentielle décroissante (k=1) sans translation
        #print(parameters['delta'].data[self.zoom_x, self.zoom_y])
        #delta = np.zeros(np.shape(mu))
        #delta[np.where(mu>0)] = 1 / mu[np.where(mu>0)]
        #
        # TODO : Intégrer delta au calcul de k ?
        # ======================================
        #parameters['k'] = (2*parameters.sigma**2+parameters.mu**2+np.sqrt((parameters.mu**2*(4*parameters.sigma**2+parameters.mu**2))))/(2*parameters.sigma**2)  # shape parameter of the Gamma PDF
        #k = (2*sigma**2+mu**2+np.sqrt((mu**2*(4*sigma**2+mu**2))))/(2*sigma**2)  # shape parameter of the Gamma PDF
        #parameters.k.where(parameters.mu==0).data = parameters.mu.where(parameters.mu==0).data / parameters.mu.where(parameters.mu==0).data   # k>1 if Y>0 else k==1
        #parameters["k"] = xr.where(parameters.mu==0, 1, parameters.k)   # k>1 if Y>0 else k==1
        #k[np.where(mu==0)] = 1
        #parameters['theta'] = parameters.mu / (parameters.k-1)  # Scale parameter of the Gamma PDF
        #parameters['theta'] = xr.where(parameters.k==1, 3*parameters.k, parameters.theta)  # Set theta=3 by default if k=1. TODO : In this case the scale parameter could depend on neighboring observations
        #theta = mu / (k-1)
        ####################################################################################################################################################################

        self.parameters = parameters

        # TODO : plot PDF for some pixels
        # gamma_shape_PDF(Y, k=k, theta=theta)
        if self.plot:
            self.plot_sigma()
            #self.plot_parameters()

    def plot_sigma(self):
        fig, ax = plt.subplots(figsize=(14,6))
        cmap = 'nipy_spectral'
        im = self.parameters['sigma'].plot(ax=ax[i,j], cmap=cmap, cbar_kwargs=dict(label='Standard deviation (mm)'))
        for landmark, infos in landmarks.items():
                ax[i,j].plot(infos['lon'], infos['lat'], marker=infos['marker'], color='red', markersize=4)
        fig.tight_layout()
        fig.savefig(f"{self.date_str}/PDF_std_{self.date_str}.pdf", format='pdf')

    def plot_parameters(self):
        label_map = dict(sigma='Standard deviation sigma (mm)', k='Shape parameter (k)', theta='Scale parameter (theta)', delta='Shift parameter (delta)')
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,7))
        i = 0
        j = 0
        for param in ['sigma', 'k', 'theta', 'delta']:
            if param == 'sigma':
                cmap = 'nipy_spectral'
            else:
                cmap = 'viridis'
            im = self.parameters[param].plot(ax=ax[i,j], cmap=cmap, cbar_kwargs=dict(label=label_map[param]))
            for landmark, infos in landmarks.items():
                ax[i,j].plot(infos['lon'], infos['lat'], marker=infos['marker'], color='red', markersize=4)
            ax[i,j].set_aspect('equal')
            ax[i,j].axis('off')
            ax[i,j].set_title(param)
            j = j + 1
            if j == 2:
                j = 0
                i = i +1
        fig.tight_layout()
        fig.savefig(f"{self.date_str}/PDF_parameters_{self.date_str}.pdf", format='pdf')

#        im1 = ax[0,0].imshow(sigma, cmap='nipy_spectral')
#        plt.colorbar(im1, ax=ax[0,0], label='sigma (mm)')
#        im2 = ax[0,1].imshow(k, cmap='viridis')
#        plt.colorbar(im2, ax=ax[0,1], label='k')
#        im3 = ax[1,0].imshow(theta, cmap='viridis')
#        plt.colorbar(im3, ax=ax[1,0], label='Theta (mm)')
#        im4 = ax[1,1].imshow(delta, cmap='viridis')
#        plt.colorbar(im4, ax=ax[1,1], label='delta (mm)')
#        plt.tight_layout()
#        fig.savefig(f"PDF_parameters_{date_str}.pdf", format='pdf')

        #SCGD_shape_PDF(Y, k=k, theta=theta, delta=delta, plot_parameters=False)
        #-----------------------------------------------------------------------

#    @speedtest
    def resample(self, weights, Ne):
        """ Ne is the number of members to draw for the new enesmble """
        import random
        #Ne = len(self.members)
        step = 1/Ne
        # Sort particules on [0,1[ according to their weight
        cumulated_weights = np.cumsum(weights, axis=0)
        #for i in range(1, Ne+1):
        selected_particles = list()
        # Random draw between [0, 1/Ne[
        rdm = random.uniform(0, step)
        #rdm = np.random.random_sample(np.shape(cumulated_weights[0]))*step
        while rdm <= 1:
            # Select particle indicies in wich rdm falls
            #selected_particles.append(np.searchsorted(cumulated_weights, rdm)+1)
            #selected_particles.append(np.apply_along_axis(lambda a: a.searchsorted(rdm), axis=0, arr=cumulated_weights)+1)
            selected_particles.append(int(np.apply_along_axis(lambda a: a.searchsorted(rdm), axis=0, arr=cumulated_weights)))  # add index value (int)
            # Go 1 step forward and start again
            rdm += step
        return selected_particles

    def normal_dist(self, x , mean , sd):
        # TODO : check why sum(norm) >> 1
        prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
        return prob_density

#    @speedtest
    def weighting(self, x, mu, sigma, plot_distribution=False, plot_parameters=False, **kw):

        draw = self.normal_dist(x, mu, sigma)

        if plot_distribution:
            dy = 0.01
            num = np.max((mu+3*sigma)/dy).astype(int)
            y = np.linspace(0, mu+3*sigma, num=num)
            norm = self.normal_dist(y, mu, sigma)
            fig,ax = plt.subplots()
            color = next(ax._get_lines.prop_cycler)['color']
            # Plot over a smaller range for better lisibility
            ymin = mu-3*sigma
            ymax = mu+3*sigma
            ax.plot(x, draw, linestyle='', marker='+', markersize=10.)
            ax.plot(y[(y>0) & (y<ymax)], norm[(y>0) & (y<ymax)],
                    label=f'Norm(mu={mu:0.2},sigma={sigma:0.2})', color=color)
            plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            plt.axvline(x=mu, color='k', linestyle='--', label=f'Observation : {mu:0.2}')
            plt.xlabel('Precipitation (mm)')
            plt.ylabel('Weight')
            plt.legend(loc ="upper right")
            filename = "distribution"
            if "num_poste" in kw.keys():
                filename = '_'.join([filename, str(kw['num_poste'])])
            if 'date' in kw.keys():
                filename = '_'.join([filename, kw['date'].strftime('%Y%m%d%H')])

            fig.savefig(f"{filename}.pdf", format='pdf')
            plt.close('all')

        return draw

    def selection_unique(self, assimilation_period, localisation_lat, localisation_lon):
        t1 = time.time()
        #self.ensemble.sel({'time':assimilation_period, 'lat':localisation_lat, 'lon':localisation_lon})
        self.ensemble.sel({'time':assimilation_period, 'lat':localisation_lat, 'lon':localisation_lon}).rr.data.flatten()
        t2 = time.time()

        return t2-t1

    @speedtest
    def selection_sequentielle(self, assimilation_period, localisation_lat, localisation_lon):
        t1 = time.time()
        sel1 = self.ensemble.sel({'time':assimilation_period})
        sel2 = sel1.sel({'lat':localisation_lat})
        sel3 = sel2.sel({'lon':localisation_lon})
        sel4 = sel3.rr.data.flatten()
        t2= time.time()

        return t2-t1

#    @speedtest
    def gridded_assimilation(self, date, idd, localized_period, obs_date, raw_date, parameters_date):
        """ Loop over all the domain's pixels."""
        # TODO : avec la localisation on ne peut pas traiter les pixels sur les bords du domaine,
        # il faut donc une verrue pour tronquer le domaine
        # WARNING : virer les pixels du bord des visualisation/evaluations ensuite
        # TODO : test de rapidité de la selection par étapes ou en une fois...
        # TODO : gérer une assimilation pour une liste de points donnée

        xloc = 3  # TODO : à paramétriser
        yloc = 3  # TODO : à paramétriser

        self.nb_out_raw = np.zeros((self.nlon, self.nlat))  # Count number of obs outside raw ensemble
        self.nb_out_loc = np.zeros((self.nlon, self.nlat))  # Count number of obs outside localized ensemble
        self.inflation = np.zeros((self.nlon, self.nlat))  # Count number of time inflation was used

        for idx,lon in enumerate(self.radar.lon.data[xloc:-xloc]):  # TODO : rajouter les bords du domaines
            idx = idx + xloc
            localisation_lon = [self.radar.lon.data[idx+dlon] for dlon in range(-xloc,xloc+1)]  # TODO : modifier la zone de localisation (--> cerlce)
            localized_lon = localized_period.sel({'lon':localisation_lon})
            obs_lon = obs_date.sel(lon=lon)
            raw_lon = raw_date.sel(lon=lon)
            parameters_lon = parameters_date.sel(lon=lon)

            for idy,lat in enumerate(self.radar.lat.data[yloc:-yloc]):  # TODO : rajouter les bords du domaines
                idy = idy + yloc
                localisation_lat = [self.radar.lat.data[idy+dlat] for dlat in range(-yloc,yloc+1)]  # TODO : modifier la zone de localisation (--> cerlce)
                localized_lat = localized_lon.sel({'lat':localisation_lat})
                raw_localized = localized_lat.rr.data.flatten()  # "Super ensemble"
                obs = obs_lon.sel({'lat':lat})
                raw = raw_lon.sel({'lat':lat})
                parameters = parameters_lon.sel({'lat':lat})

                ##############################################################################################################################
                # Speed test
                #time_selection_unique += self.selection_unique(assimilation_period, localisation_lat, localisation_lon)
                #time_selection_sequentielle += self.selection_sequentielle(assimilation_period, localisation_lat, localisation_lon)
                # For 330 dates over the Grandes Rousses domaine the result of the speed test is :
                # Total time for unique selection :  178115.16904830933
                # Total time for sequential selection :  217705.75380325317
                # BUT the sequential selectionis far more efficient since there are fewer calls
                ##############################################################################################################################

                # Data selection already done before to optimse running time
                #obs, raw, raw_localized, parameters = self.select_data(date, lon, lat, assimilation_period, localisation_lat, localisation_lon)
                #obs, raw, parameters = self.select_data(date, lon, lat)

                # Is the observation outside the ensemble ?
                if (np.min(raw) > obs) or (np.max(raw) < obs):
                    self.nb_out_raw[idx, idy] += 1
                if (np.min(raw_localized) > obs) or (np.max(raw_localized) < obs):
                    self.nb_out_loc[idx, idy] += 1

                new, inflation = self.assimilation(date, obs, raw, raw_localized, parameters, lat, lon, idx, idy)

                self.inflation[idx,idy] = inflation
                # TODO : remplir une liste plutot que boucler
                for member, field in self.newlocalfield.items():
                    field[idx,idy,idd]  = new[member-1]  # fill new member

        if self.plot:  # Not with localisation
            plot_weights(date, weights)

#    @speedtest
    def ponctual_assimilation(self, date, idd, localized_period, obs_date, raw_date, parameters_date):
        stop = False
        xloc = 3  # TODO : à paramétriser
        yloc = 3  # TODO : à paramétriser
        assimilation_points = zip(self.nivometeo.num_poste.data, np.max(self.nivometeo.lat, axis=1).data, np.max(self.nivometeo.lon, axis=1).data)
        self.nb_out_raw = np.zeros(self.nposte)  # Count number of obs outside raw ensemble
        self.nb_out_loc = np.zeros(self.nposte)  # Count number of obs outside localized ensemble
        self.inflation = np.zeros(self.nposte)  # Count number of time inflation was used
        for idp, (num_poste, lat, lon) in enumerate(assimilation_points):
            lat = nearest(self.radar.lat, lat)  # Latitude of the corresponding antilope pixel
            lon = nearest(self.radar.lon, lon)  # Longitude of the corresponding antilope pixel
            idy = np.where(self.radar.lat.data==lat)[0][0]  # index of the corresponding antilope pixel latitude
            idx = np.where(self.radar.lon.data==lon)[0][0]  # index of the corresponding antilope pixel longitude
            localisation_lat = [self.radar.lat.data[idy+dlat] for dlat in range(-yloc,yloc+1)]  # Latitudes to consider for localisation. TODO : use a circle
            localisation_lon = [self.radar.lon.data[idx+dlon] for dlon in range(-xloc,xloc+1)]  #Longitudes to consider for localisation. TODO : use a circle
            raw_localized = localized_period.sel({'lat':localisation_lat, 'lon':localisation_lon})
            if self.frequency == 'hourly':
                import pdb
                pdb.set_trace()
                t1 = time.time()
                raw_localized.compute()  # Load data now
                t2 = time.time()
                print(f'reading "raw_localized" took {(t2-t1)*1000.}ms')
            obs = obs_date.sel({'lat':lat, 'lon':lon})
            raw = raw_localized.sel({'time':date, 'lat':lat, 'lon':lon})
            raw_localized = raw_localized.rr.data.flatten()  # "Super ensemble"
            parameters = parameters_date.sel({'lat':lat, 'lon':lon})

            # Is the observation outside the ensemble ?
            if (np.min(raw) > obs) or (np.max(raw) < obs):
                self.nb_out_raw[idp] += 1
            if (np.min(raw_localized) > obs) or (np.max(raw_localized) < obs):
                self.nb_out_loc[idp] += 1

            new, inflation = self.assimilation(date, obs, raw, raw_localized, parameters, lat, lon, idx, idy, num_poste=num_poste)

            self.inflation[idp] = inflation
            for member, field in self.newlocalfield.items():
                field[idp,idd]  = new[member-1]  # fill new member

#    @speedtest
    def assimilation(self, date, obs, raw, raw_localized, parameters, lat, lon, idx, idy, num_poste=None):

        nb_new_member = 0
        sigma = float(parameters.sigma.data)/2.
        inflation = 0
        while ((nb_new_member < 4) and (inflation <= 8)):  # Security to avoid infinite loops

            # 2.a Inflation
            #---------------
            # BUT : lorsque l'obs est en dehors de l'ensemble, on augmente l'erreur d'obs pour assurer qu'un
            # nombre suffisant de membres est selectionné.
            # TODO : s'assurer que c'est bien compatible avec la technique de localisation
            inflation += 1
            sigma = 2*sigma  # TODO : facteur d'inflation à redéfinir

            # 2.b Weighting
            #-------------
            #weights = self.weighting(raw_localized, parameters.mu.data, sigma, plot_distribution=True)
            weights = self.weighting(raw_localized, parameters.mu.data, sigma)
            weights = weights / np.sum(weights)

            # 3. Resampling
            #--------------
            # ECC is the order of members indicies (from 0 to 15 !) sorted by increasing precipitation
            selection_locale = self.resample(weights, self.Ne)  # Idicies of selected particles from the "super-ensemble"
            nb_new_member = len(np.unique(selection_locale))

            if inflation >= 2:
                print(f'Iteration {inflation} for pixel ({idx}, {idy})')

        tmp = np.sort(raw_localized[selection_locale])

        # 4. ECC for consistency with raw members
        #----------------------------------------
        ecc = np.argsort(raw.rr.data, axis=0)
        new = tmp[ecc]
        # To plot the posterior distribution:
        # self.weighting(new, parameters.mu.data, parameters.sigma.data, plot_distribution=True)

        # To plot data for one specific point / date
        #if num_poste == 74134400:
            #self.plot_assimilation(raw.rr.data, new, obs.rr, num_poste, date)
            #self.weighting(raw_localized, parameters.mu.data, sigma, plot_distribution=True, num_poste=num_poste, date=date)

        return new, inflation

    def plot_assimilation(self, raw, assim, obs, num_poste, date):
        """ References :
        https://stackoverflow.com/questions/64646449/how-to-create-asymmetric-violin-plot-in-python-using-matplotlib
        https://seaborn.pydata.org/generated/seaborn.violinplot.html
        """
        fig, ax = plt.subplots()
        data = pd.DataFrame({'raw':raw, 'assim':assim})
        data = data.melt()
        data['dummy'] = 0
        sns.violinplot(data=data, split=True, y='value', hue='variable', x='dummy', inner="stick", palette=['sandybrown', 'skyblue'])
        plt.plot(obs, marker='_', markersize=30, markeredgewidth=3, color='red')
        # TODO : ploter l'obs nivometeo correspondante...
        fig.savefig(f'assim_{num_poste}_{date}.pdf', formatout='pdf',  bbox_inches='tight')


    @speedtest
    def run(self):
        """ 
        Main Particle Filter method that loop over the assimilation dates and grid points.
        TODO : compléter la doc sur la méthode
        """

        if self.plot:
            # On profite de la loop sur les membres pour tracer les ensembles bruts avant et après interpolation
            fig1, axes1 = plt.subplots(nrows=4, ncols=4, figsize=(16, 8))
            fig2, axes2 = plt.subplots(nrows=4, ncols=4, figsize=(16, 8))
        i = 0
        j = 0

        # Initialisation of output fields
        if self.gridded:
            self.nlon, self.nlat = len(self.radar.lon), len(self.radar.lat)
            null  = np.empty((self.nlon, self.nlat, len(self.period)))  # 2D (lat/lon) field
        else:
            self.nposte = len(self.nivometeo.num_poste)
            null = np.empty((self.nposte, len(self.period)))
        self.newlocalfield = {m:null.copy() for m in range(1, self.Ne+1)}
        time_selection_unique = 0
        time_selection_sequentielle = 0
        for idd,date in enumerate(self.period):
            print(date)
            # On réduit le dataset maintenant pour gagner du temps ensuite
            if self.frequency == 'hourly':
                assimilation_period = [date + timedelta(hours=dt) for dt in range(-2,3)]
                localized_period = self.ensemble.sel({'time':assimilation_period})  # Do not load data now (crash) !
            else:
                assimilation_period = [date]
                t1 = time.time()
                localized_period = self.ensemble.sel({'time':assimilation_period}).compute()  # Load data into memory now ==> very slow
                t2 = time.time()
                print(f'reading "raw_localized" took {(t2-t1)*1000.}ms')
            obs_date = self.radar.sel(time=date)
            raw_date = localized_period.sel(time=date)
            parameters_date = self.parameters.sel({'time':date})

            self.date_str = date.strftime('%Y%m%d%H')
            if not os.path.exists(self.date_str):
                os.makedirs(self.date_str)

            if self.gridded:
                self.gridded_assimilation(date, idd, localized_period, obs_date, raw_date, parameters_date)
            else:
                self.ponctual_assimilation(date, idd, localized_period, obs_date, raw_date, parameters_date)

    @speedtest
    def plot_weights(self, date, weights):
        # TODO : find a way to plot weights...
        pass

#    @speedtest
    def select_data(self, date, lon, lat, assimilation_period, localisation_lat, localisation_lon):
        """ Unsed """

        # TODO : améliorer la performance de cette méthode (~2ms * nombre de pixel * nombre de dates...)
        # TODO : Extraire un domaine légèrement plus grand que le domaine couvert par l'assimilation pour
        # appliquer le localisation aux points en hors du domaine
        # TODO : untiliser une distance (cerle) plutot qu'on rectangle pour la localisation
        # TODO : passer les paramètres de localisation (rayon, fenetre temporelle) en argument pour plus
        # de flexibilité (etude de sensibilité, dépendance à la situation,...)
        #if (lon == self.zoom_lon) and (lat == self.zoom_lat):

        # Localisation
        #----------------
        t1 = time.time()
        # WARNING : la lecture de l'ensemble localisé représente >50% du temps pour une itération de date
        #raw_localized = self.ensemble.sel({'time':assimilation_period, 'lon':localisation_lon, 'lat':localisation_lat}).rr.data.flatten()  # "Super ensemble"
        t2 = time.time()
        #print((t2-t1)*1000.0)
        t3 = time.time()
        #print((t3-t2)*1000.0)
        raw = self.ensemble.sel({'time':date, 'lon':lon, 'lat':lat}).rr.data
        t4 = time.time()
        #print((t4-t3)*1000.0)
        obs = self.radar.sel({'time':date, 'lon':lon, 'lat':lat})
        t5 = time.time()
        #print((t5-t4)*1000.0)
        parameters = self.parameters.sel({'time':date, 'lon':lon, 'lat':lat})
        t6 = time.time()
        #print((t6-t5)*1000.0)

        #return obs, raw, raw_localized, parameters
        return obs, raw, parameters

    def selection(self):
        self.global_selection()
        self.local_selection()

    def global_selection(self):
        # 1 Weighting
        total_global = sum(self.weight.values())
        global_weights =  {k: v/total_global for k, v in self.weight.items()}
        # 2 Resampling
        self.selection_globale = self.resample(list(global_weights.values()))

    def plot_one_pixel(self, x, y):
        model = [values.rr.data[x][y] for values in self.ensemble.values()]
        #self.SCGD_shape_PDF(model, self.parameters.mu.data[x][y], k=self.parameters.k.data[x][y], theta=self.parameters.theta.data[x][y],delta=self.parameters.delta.data[x][y], plot_distribution=True)
        self.weighting(model, self.parameters.mu.data[x][y], self.parameters.sigma.data[x][y], plot_distribution=True)

    @speedtest
    def output(self, localfields, globalfields):
        # This method takes ~3.5 s but most of the time (~3.3s) is spent
        # while saving figures.
        # TODO : trouver un moyen de plotter les champs de poids

        if self.gridded:
            out_raw = xr.DataArray(
                    name   = 'out_raw',
                    data   = self.nb_out_raw,
                    dims   = ["lon", "lat"],
                    coords = dict(lon=self.radar.lon, lat=self.radar.lat),
                    attrs  = dict(description="Number of assimilation step when the observation was outside the raw ensemble"),
                )
            out_loc = xr.DataArray(
                    name   = 'out_loc',
                    data   = self.nb_out_loc,
                    dims   = ["lon", "lat"],
                    coords = dict(lon=self.radar.lon, lat=self.radar.lat),
                    attrs  = dict(description="Number of assimilation step when the observation was outside the localized ensemble"),
                )
            inflation = xr.DataArray(
                    name   = 'inflation',
                    data   = self.inflation,
                    dims   = ["lon", "lat"],
                    coords = dict(lon=self.radar.lon, lat=self.radar.lat),
                    attrs  = dict(description="Number of assimilation step when inlfation was used"),
                )
        else:
            out_raw = xr.DataArray(
                    name   = 'out_raw',
                    data   = self.nb_out_raw,
                    dims   = ["num_poste"],
                    coords = dict(num_poste=self.nivometeo.num_poste.data),
                    attrs  = dict(description="Number of assimilation step when the observation was outside the raw ensemble"),
                )
            out_loc = xr.DataArray(
                    name   = 'out_loc',
                    data   = self.nb_out_loc,
                    dims   = ["num_poste"],
                    coords = dict(num_poste=self.nivometeo.num_poste.data),
                    attrs  = dict(description="Number of assimilation step when the observation was outside the localized ensemble"),
                )
            inflation = xr.DataArray(
                    name   = 'inflation',
                    data   = self.inflation,
                    dims   = ["num_poste"],
                    coords = dict(num_poste=self.nivometeo.num_poste.data),
                    attrs  = dict(description="Number of assimilation step when inlfation was used"),
                )

        out_raw.to_netcdf(f"nb_obs_outside_raw_ensemble_{self.period[0].strftime('%Y%m%d%H')}_{self.period[-1].strftime('%Y%m%d%H')}_{self.frequency}.nc")
        out_loc.to_netcdf(f"nb_obs_outside_localized_ensemble_{self.period[0].strftime('%Y%m%d%H')}_{self.period[-1].strftime('%Y%m%d%H')}_{self.frequency}.nc")
        inflation.to_netcdf(f"inflation_{self.period[0].strftime('%Y%m%d%H')}_{self.period[-1].strftime('%Y%m%d%H')}_{self.frequency}.nc")


        if self.plot:
            #fig1, ax1 = plt.subplots(nrows=4, ncols=4, figsize=(16, 8))
            fig2, ax2 = plt.subplots(nrows=4, ncols=4, figsize=(16, 8))
            #fig3, axes3 = plt.subplots(nrows=4, ncols=4, figsize=(16, 8))
        i = 0
        j = 0
        for m in range(1, self.Ne+1):
            #t1 = time.time()
            #localfields.loc[{'time':self.date, 'member':m}] = self.newlocalfield[m]  # self.newlocalfield is a numpy array
            localfields.loc[{'member':m}] = self.newlocalfield[m]  # self.newlocalfield is a numpy array
            #t2 = time.time()
            #print(f'Wrinting localfields took {(t2-t1)*1000.}ms')
            #globalfields.loc[{'time':self.date, 'member':m}] = self.ensemble[self.selection_globale[m-1]].rr.data  # self.ensemble is a DataArray
            #t3 = time.time()
            #print(f'Wrinting globalfields took {(t3-t2)*1000.}ms')

            if self.plot:
                self.rrmin = 0.
                self.rrmax = max(
                        np.nanmax(self.ensemble.sel(time=date).rr.data),
                        np.nanmax(self.radar.sel(time=date).rr.data)
                        )
                # Plot local weight fields
                vmin = 0
                #vmax = 0.25  # TODO : vmax=f(sigma) [peut être renvoyé par la fonction de la PDF comme le max de probabilité]
                #vmax = np.max(gamma_shape_PDF(parameters.mu.data, k=parameters.k.data, theta=parameters.theta.data))
                #vmax = max([np.max(ww.data) for ww in self.local_weights.values()])
                #im3 = plot_field(self.local_weights[m], axes3[i,j], vmin, vmax, title=f'member {m:03d}, total weight={self.weight[m]:.3f}', cmap=plt.cm.Greys)

                #im1 = plot_field(globalfields.loc[{'time':self.date, 'member':m}], ax1[i,j], self.rrmin, self.rrmax, title=f'Member {self.selection_globale[m-1]:03d}')
                im2 = plot_field(localfields.loc[{'time':self.date, 'member':m}], ax2[i,j], self.rrmin, self.rrmax, title=f'New member {m:03d}')
                #t4 = time.time()
                #print(f'Plotting took {(t4-t3)*1000.}ms')
            j = j + 1
            if j==4:
                j = 0
                i = i + 1

        if self.plot:
            #finalize_fig(fig1, im1, label='24-hour precipitation (mm)', outname=f'{self.date_str}/ASSIM_globale_{self.date_str}.pdf')
            finalize_fig(fig2, im2, label='24-hour precipitation (mm)', outname=f'{self.date_str}/ASSIM_locale_{self.date_str}.pdf')
            #finalize_fig(fig3, im3, label='Weight', outname=f'{self.date_str}/WEIGHTS_{self.date_str}.pdf')
            #t5 = time.time()
            #print(f'Finalisation of figures took {(t5-t4)*1000.}ms')

        plt.close('all')

        return localfields, globalfields

def nearest(array, value):
            # Find element of "array" the closer to "value"
            return float(array[np.abs(array - value).argmin()].data)

def plot_chrono(time, obs, raw, assim):
    fig = plt.figure()
    obs = obs.sel({'lat':nearest(obs.lat, zoom['lat']), 'lon':nearest(obs.lon, zoom['lon'])}).sel(time=time)
    #plt.plot(time, np.cumsum(obs.rr.data), label='Antilope', color='k')
    plt.plot(time, obs.rr.data, label='Antilope', color='k')
#    fig1,ax1 = plt.subplots((4,4))nrows=2, ncols=2, figsize=(16,7)
#    fig2,ax2 = plt.subplots((4,4))
#    i = 0
#    j = 0
    for member in assim.member.data:
        chrono_raw = raw[member].sel({'lat':nearest(raw[member].lat, zoom['lat']), 'lon':nearest(raw[member].lon, zoom['lon'])}).sel(time=time)
        if member == 1:
            rawlabel = 'Raw ensemble'
            assimlabel = 'After assimilation'
        else:
            rawlabel = None
            assimlabel = None
        #plt.plot(time, np.cumsum(chrono_raw.rr.data), color='sandybrown', label=rawlabel)
        plt.plot(time, chrono_raw.rr.data, color='sandybrown', label=rawlabel)
        chrono_assim = assim.sel({'lat':nearest(assim.lat, zoom['lat']), 'lon':nearest(assim.lon, zoom['lon']), 'member':member}).sel(time=time)
        #plt.plot(time, np.cumsum(chrono_assim.data), color='skyblue', label=assimlabel)
        plt.plot(time, chrono_assim.data, color='skyblue', label=assimlabel)
#        j = j + 1
#        if j==4:
#            j = 0
#            i = i + 1
#    fig1.savefig("Chronologie_raw.pdf", format='pdf')
#    fig2.savefig("Chronologie_assim.pdf", format='pdf')
    plt.legend()
    plt.savefig(f"Chronologie_{time[0].strftime('%Y%m%d%H')}_{time[-1].strftime('%Y%m%d%H')}.pdf", format='pdf')

if __name__ == "__main__":
    args = parse_command_line()

    #goto(args.workdir)
    extract_period = date_range(args.datebegin, args.dateend, dt=timestep[args.frequency])

    antilope = read_obs(args)
    pearome = read_ensemble(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'), args.frequency, args.domain)
    if args.gridded:
        localfields = xr.DataArray(
                name   = 'rr',
                dims   = ["lon", "lat", "time", "member"],  # TODO homogénéiser l'ordre des coordonnées
                coords = dict(lon=antilope.lon, lat=antilope.lat, time=extract_period, member=range(1,17)),
                attrs  = dict(description="24 hour precipitation",units="mm"),
            )
        globalfields = xr.DataArray(
                name   = 'rr',
                dims   = ["lat", "lon", "time", "member"],  # TODO homogénéiser l'ordre des coordonnées
                coords = dict(lon=antilope.lon, lat=antilope.lat, time=extract_period, member=range(1,17)),
                attrs  = dict(description="24 hour precipitation",units="mm"),
            )
        nivometeo = None
    else:
        nivometeo = read_nivometeo_obs()
        localfields = xr.DataArray(
                name   = 'rr',
                dims   = ["num_poste", "time", "member"],  # TODO homogénéiser l'ordre des coordonnées
                coords = dict(num_poste=nivometeo.num_poste.data, time=extract_period, member=range(1,17)),
                attrs  = dict(description="24 hour precipitation",units="mm"),
            )
        globalfields = None

    # TODO : on doit peut être pouvoir se passer de la boucle temporelle en utilisant les fonctionalités de xarray
    #date = args.datebegin
    #while date <= args.dateend:
    #r date in extract_period:
    interp_ensemble = pearome.interp(lon=antilope.lon, lat=antilope.lat).clip(0)  # Avoid <0 precipitation values
    pf = ParticleFilter(extract_period, antilope, interp_ensemble, nivometeo, args.plot, args.frequency, args.gridded)
    pf.pdf_parameters()
    pf.run()  # Compute weight fields before normalisation + plot raw/interp fields
    #pf.selection()  # Global and local selections

    localfields, globalfields = pf.output(localfields, globalfields)

#    plot_chrono(extract_period, antilope, pearome, localfields)

    localfields.to_netcdf(f"Assimilation_locale_{args.datebegin.strftime('%Y%m%d%H')}_{args.dateend.strftime('%Y%m%d%H')}_{args.frequency}_{args.domain}.nc")
    #globalfields.to_netcdf(f"Assimilation_globale_{args.datebegin.strftime('%Y%m%d%H')}_{args.dateend.strftime('%Y%m%d%H')}_{args.frequency}.nc")

#        date = date + timedelta(days=1)

#        import pdb
#        pdb.set_trace()
