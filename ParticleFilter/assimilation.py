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
##############################################################################################

datadir = '/home/vernaym/These/DATA'

# Domaine des Grandes Rousses
domain_coords = dict(
        GrandesRousses = dict(latmax=45.240, latmin=44.990, lonmin=6.010, lonmax = 6.490),
        alp            = dict(latmax=46.450, latmin=44.100, lonmin=5.400, lonmax=7.200),
)

landmarks = {
        "Alpe d'Huez" : dict(lon=6.070, lat=45.092, alt=1800, marker='o'),
        "Les 2 Alpes" : dict(lon=6.127, lat=45.013, alt=1800, marker='o'),
        "Lautaret"    : dict(lon=6.408, lat=45.038, alt=2058, marker='X'),
        "La Meije"    : dict(lon=6.311, lat=45.008, alt=3500, marker='^'),  # real alt = 3984
        "Pic Blanc"   : dict(lon=6.131, lat=45.128, alt=3000, marker='^'),  # real alt = 3333
    }

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
    parser.add_argument('-m', '--mask', help='Switch observation error mask on/off', choices=[1,2,3,4,5], default=None, type=int)
    parser.add_argument('-c', '--debiasing', help='Apply bias correction to observation (0=constant bias, 1=pseudo-kriging, 2=estimation based on homogeneity)', default=None, type=int, choices=[0,1,2])
    parser.add_argument('-t', '--threshold', default=None, help='Threshold of precipitation (mm) to apply in the data to consider', type=int)
    parser.add_argument('-p', '--plot', action='store_true', default=False, help='Plot assimilated fields')
    parser.add_argument('-f', '--frequency', choices=['hourly', 'daily'], help='Assimilation frequency')
    parser.add_argument('-l', '--localisation', default=None, help='Spatial localisation distance (switch localisation on).', type=int)
    parser.add_argument('-g', '--gridded', default=False, action='store_true', help='Gridded assimilation if True, or assimilation only at nivometeo locations if False')
    parser.add_argument('-y', '--likelyhood', default='normal', choices=['normal', 'gamma'], help='Likelihood distribution')

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
                    print(f'The date {a_string} provided is not in a good format (YYYYMMDDHH or YYMMDDHH or YYYYMMDDHHMM or YYMMDD or YYYYMMDD)')
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
    """
    Read ensemble data from multiple netcdf files.
    When the domain is large (for example the french Alps : ~120*150 grid cells) the computation is complex to avoid memory crashes.

    The "open_mfdataset" method uses dask parallelisation to optimise computation time. It breaks up the data into "chunks" so that
    operations can be parallelized more efficiently. By default, chunks will be chosen to load entire input files into memory at once,
    which results in memory crashes.
    It is advised to specify small chunks (here a multiple of 24 time steps seem optimal : ~0.6s by date iteration vs >1.2 for other
    chunk sizes.
    It is not possible to chunk on lat/lon coordinates since there is an interpolation step early in the algorithm.

    Documentation :
    - open_mfdataset : https://docs.xarray.dev/en/stable/generated/xarray.open_mfdataset.html#id4
    - Dask parallelization : https://docs.xarray.dev/en/stable/user-guide/dask.html#chunking-and-performance
    - How to set chunks : https://docs.dask.org/en/latest/array-chunks.html

    """

    #filenames = [os.path.join(datadir, f"aspearome_{mb:03d}_{datebegin}_{dateend}_{domain}_{frequency}.nc") for mb in range(1,17)]
    #filenames = [os.path.join(datadir, f"aspearome_{mb:03d}_2021073106_2022070106_{domain}_{frequency}.nc") for mb in range(1,17)]
    if domain == 'alp':
        filenames = [os.path.join(datadir, f"aspearome_{mb:03d}_2021102806_2022060206_{domain}_hourly.nc") for mb in range(1,17)]
    elif domain == 'GrandesRousses':
        filenames = [os.path.join(datadir, f"aspearome_{mb:03d}_2021073106_2022070106_{domain}_hourly.nc") for mb in range(1,17)]

    # open_mfdataset returns a dask.array<chunksize=(...), meta=np.ndarray> object that divides arrays into many small pieces, called chunks, 
    # each of which is presumed to be small enough to fit into memory in order to avoid a memory overload. 
    # The compute() method effectively load data into memory so it must be called as late as possible to reduce memory use as well as
    # running time.
    # See : https://docs.xarray.dev/en/stable/user-guide/dask.html
    #pearome = xr.open_mfdataset(filenames, combine='nested', concat_dim='member').compute()  # Impossible avec des fichiers trop volumineux

    # Chunks of a multiple of 24 time steps seem optimal (~0.6s by iteration vs >1.2 for other chunk sizes)
    pearome = xr.open_mfdataset(filenames, combine='nested', concat_dim='member', chunks={'time': 24})  # Setting chunks is critical (read the doc !)
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


def plot_field(field, ax, vmin, vmax, title=None, cmap=plt.cm.YlGnBu):
#def plot_field(field, ax, vmin, vmax, title, cmap='viridis'):
#    import pdb
#    pdb.set_trace()
    im = field.plot(ax=ax, add_colorbar=False, vmin=vmin, vmax=vmax, cmap=cmap)
    for landmark, infos in landmarks.items():
        ax.plot(infos['lon'], infos['lat'], marker=infos['marker'], color='red', markersize=4)
    ax.set_aspect('equal')
    ax.axis('off')
    if title is not None:
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
        if args.domain == 'GrandesRousses':
            print(f'WARNING : no file named {filename} under {datadir}, using default file ANTILOPEQ_2021080106_2022070106_GrandesRousses.nc')
            filename = os.path.join(datadir, f'ANTILOPE{suffix[args.frequency]}_2021073106_2022070106_GrandesRousses.nc')
        elif args.domain == 'alp':
            print(f'WARNING : no file named {filename} under {datadir}, using default file ANTILOPEH_2021103000_2022060200_alp.nc')
            filename = os.path.join(datadir, f'ANTILOPEH_2021103000_2022060200_alp.nc')
    if os.path.exists(filename):
        antilope = xr.open_dataset(filename)
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

    def __init__(self, period, obs, ensemble, nivometeo, plot, frequency, gridded, localisation, mask, debiasing, domain, likelyhood):

        #goto(self.date_str)
        #self.nline, self.ncol = np.shape(obs.rr.data)
        self.period = period
        self.frequency = frequency

        # To look at one specific point
        self.zoom_x = np.argwhere(obs.lat.data == nearest(obs.lat, zoom['lat']))[0][0]
        self.zoom_y = np.argwhere(obs.lon.data == nearest(obs.lon, zoom['lon']))[0][0]
        self.zoom_lon = nearest(obs.lon, zoom['lon'])
        self.zoom_lat = nearest(obs.lat, zoom['lat'])

        #self.radar = obs.transpose('lon','lat','time')
        self.radar = obs
        self.ensemble = ensemble
        self.members = self.ensemble.member
        self.Ne = len(self.members)
        self.plot = plot

        self.gridded = gridded
        self.nivometeo = nivometeo
        self.localisation = localisation
        self.mask = mask
        self.debiasing = debiasing
        self.domain = domain
        self.likelyhood = likelyhood

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

    def plot_obs(self, field):
        mnt = xr.open_dataset('/home/vernaym/QGIS/MNT/DEM_ALPES_WGS84_250m_bilinear.nc')  # Pour tracer sur toutes les Alpes
        # Plot ANTILOPE precipitation field
        fig = plt.figure(figsize=(18,8))
        field.rr.plot(vmin=0, vmax=self.rrmax, cmap=plt.cm.YlGnBu, cbar_kwargs={'label': "24 hour precipitation (mm)"})  # quadmesh object
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
#        tmp = mnt.interp(lon=self.radar.lon, lat=self.radar.lat, method='nearest')  # Pour interpoller le MNT sur la grille ANTILOPE
#        # WARNING :  la commande suivante réduit sensibement le domaine, attention aux comparaisons entre figures (en particulier avec les CUMULS)
#        #tmp = tmp.where((tmp.lon>=lonmin) & (tmp.lon<=lonmax) & (tmp.lat>=latmin) & (tmp.lat<=latmax), drop=True)
#        X, Y = np.meshgrid(tmp['lon'].values, tmp['lat'].values)
#        Z = np.nan_to_num(tmp['Band1'].values)
#        # define pixel colors
#        #colors = plt.cm.coolwarm(norm(np.nan_to_num(radar.transpose('lat', 'lon').rr.data)))
#        colors = plt.cm.coolwarm(norm(np.nan_to_num(self.radar.rr.data)))
#        plot3D(X, Y, Z, colors, self.date_str)

    @speedtest
    def pdf_parameters(self):
        # I. Define PDF parameters
        #-------------------------
        parameters = self.radar.copy()  # TODO : vérifier si ce n'est pas trop couteux (de toutes façon l'obs n'a pas de raison d'être modifiée)
        if self.debiasing == 0:
            ratio = 0.825  # Homogeneous debiasing based on the mean antilope/ref ratio
        elif self.debiasing == 1:
            mask = xr.open_dataset(os.path.join("/home/vernaym/workdir/ASSIMILATION/mask", f"estimated_ratio_{self.domain}.nc"))
            ratio = mask.ratio
        elif self.debiasing == 2:
            mask = xr.open_dataset(os.path.join("/home/vernaym/workdir/ASSIMILATION/mask", f"estimated_ratio2_loc25_seuil_0.1_{self.domain}.nc"))
            ratio = mask.ratio
        else:
            ratio = 1  # No debiasing
        parameters['mu'] = parameters['rr'] / ratio  # TODO : check if the ensemble after assimilation is not biased

        # Observation error
        # Import multiplicative mask to increase observation error where necessary
        if self.mask is not None:
            #try:
            #mask = xr.open_dataset(os.path.join(datadir, f"mask_error_antilope_{self.domain}.nc"))
            if self.mask in [1,2]:
                mask = xr.open_dataset(os.path.join("/home/vernaym/workdir/ASSIMILATION/mask", f"mask{self.mask}_loc10_seuil0.6_{self.domain}.nc"))
                parameters['sigma'] = (0.261 + 0.263 * parameters['rr'])*mask.mask  # According to the linear regression of ANTILOPE RMSE vs ANTILOPE RR
            if self.mask in [3,4]:
                mask = xr.open_dataset(os.path.join("/home/vernaym/workdir/ASSIMILATION/mask", f"mask{self.mask}_loc10_{self.domain}.nc"))
                parameters['sigma'] = (0.261 + 0.263 * parameters['rr'])*mask.mask  # According to the linear regression of ANTILOPE RMSE vs ANTILOPE RR
            elif self.mask in [5]:
                if self.debiasing is None:
                    mask = xr.open_dataset(os.path.join("/home/vernaym/workdir/ASSIMILATION/mask", f"estimated_ratio2_loc25_seuil_0.1_{self.domain}.nc"))
                    ratio = mask.ratio
                # TODO : augmenter l'erreur d'observation
                mask = xr.open_dataset(os.path.join("/home/vernaym/workdir/ASSIMILATION/mask", f"mask{self.mask}_loc25_seuil0.1_{self.domain}.nc"))
                parameters['sigma'] = 0.261 * mask.mask + 0.263 * parameters['rr'] * (1+np.abs(1 - ratio))
                #parameters['sigma'] = (0.261 + 0.263 * parameters['rr'] * (1+np.abs(1 - ratio))) * mask.mask

#            except FileNotFoundError as e:
#                print(e)
#                print('WARNING : no observation error mask')
#                self.mask = None
#                parameters['sigma'] = (0.261 + 0.263 * parameters['mu'])  # No mask
        else:
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

        return self.mask

    def plot_sigma(self, field):
        fig, ax = plt.subplots(figsize=(14,6))
        #cmap = plt.cm.YlGnBu
        cmap = plt.cm.Greys
        im = field['sigma'].plot(ax=ax, cmap=cmap, cbar_kwargs=dict(label='Standard deviation (mm)'))
        for landmark, infos in landmarks.items():
                ax.plot(infos['lon'], infos['lat'], marker=infos['marker'], color='red', markersize=4)
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

    def plot_ensemble(ensemble, label):
        fig,ax = plt.subplots(nrows=4, ncols=4, figsize=(16,7))
        i = 0
        j = 0
        for member in ensemble.member.data:
            member.plot(ax=ax[i,j], cmap=plt.cm.YlGnBu)
            j = j + 1
            if j==4:
                j = 0
                i = i + 1
        fig.savefig(f"{self.date_str}/{label}_{self.date_str}.pdf", format='pdf')
#    fig, ax = plt.subplots(figsize=(4,4))
#    i = 0
#    j = 0
#    for member in ensemble:
#        if member == 1 and label is not None:
#            member.plot(ax[i,j], cmap=plt.cm.YlGnBu, label=rawlabel)
#        else:
#            member.plot(ax[i,j], cmap=plt.cm.YlGnBu)
#        j = j + 1
#        if j==4:
#            j = 0
#            i = i + 1
#    plt.legend()

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

    def normal_dist(self, sample, mean, sd):
        # TODO : check why sum(norm) >> 1
        #prob_density = (np.pi*sd) * np.exp(-0.5*((sample-mean)/sd)**2)
        prob_density =  np.exp(-0.5*((sample-mean)/sd)**2) / (sd*np.sqrt(2*np.pi))

        return prob_density

#    @speedtest
    def gamma_dist(self, sample, mu, sd):
        """ Definition of gamma distribution """
        #import math  # math.gamma does not work with arrays
        from scipy.special import gamma

        if mu > 0:
            k     = mu**2 / sd + 1  # A prouver
            theta = sd / mu  # A prouver
            prob_density = sample**(k-1)*np.exp(-sample/theta)/(theta**k*gamma(k))
        else:
            prob_density = np.exp(-sample/sd)  # décroissance exponentielle

        return prob_density

#    @speedtest
    def weighting(self, x, mu, sigma, obs, plot_distribution=False, **kw):

        if self.likelyhood == 'normal':
            draw = self.normal_dist(x, mu, sigma)
        elif self.likelyhood == 'gamma':
            draw = self.gamma_dist(x, mu, sigma)

        if plot_distribution:
            dy = 0.01
            num = np.max((mu+15*sigma)/dy).astype(int)
            y = np.linspace(0, mu+15*sigma, num=num)
            if self.likelyhood == 'normal':
                norm = self.normal_dist(y, mu, sigma)
            elif self.likelyhood == 'gamma':
                gamma = self.gamma_dist(y, mu, sigma)

            fig,ax = plt.subplots()
            color = next(ax._get_lines.prop_cycler)['color']
            # Plot over a smaller range for better lisibility
            ymin = mu-3*sigma
            ymax = mu+3*sigma
            ax.plot(x, draw, linestyle='', marker='+', markersize=10.)
            if self.likelyhood == 'normal':
                ax.plot(y[(y>=0) & (y<ymax)], norm[(y>=0) & (y<ymax)],
                        label=f'Norm(mu={mu:0.2},sigma={sigma:0.2})', color=color)
            elif self.likelyhood == 'gamma':
                ax.plot(y[(y>=0) & (y<ymax)], gamma[(y>=0) & (y<ymax)],
                        label=f'gamma(mu={mu:0.2},sigma={sigma:0.2})', color=color)
            plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            plt.axvline(x=obs, color='k', linestyle='--', label=f'Observation : {obs:0.2}')
            plt.xlabel('Precipitation (mm)')
            plt.ylabel('Weight')
            plt.legend(loc ="upper right")
            filename = "distribution"
            if "num_poste" in kw.keys():
                filename = '_'.join([filename, str(kw['num_poste'])])
            if 'date' in kw.keys():
                filename = '_'.join([filename, kw['date'].strftime('%Y%m%d%H')])

            if not os.path.exists(self.date_str):
                os.makedirs(self.date_str)
            fig.savefig(f"{self.date_str}/{filename}.pdf", format='pdf')
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
    def gridded_assimilation(self, date, idd, localized_period, parameters_date):
        """ Loop over all the domain's pixels."""

        assim_lat = self.radar.lat.data
        assim_lon = self.radar.lon.data
        for idx,lon in enumerate(assim_lon):
            parameters_lon = parameters_date.sel(lon=lon)
            if self.localisation is not None:
                localisation_lon = np.array(
                        [self.radar.lon.data[idx+dlon] if idx+dlon>=0 and idx+dlon<len(assim_lon) else np.nan
                            for dlon in range(-self.localisation, self.localisation+1)]
                    )
                # On enlève les valeurs manquantes (pour les points en bord de domaine)
                # ==> la localisation est réduite en bordure de domaine !
                localisation_lon = localisation_lon[~np.isnan(localisation_lon)]
                localized_lon = localized_period.sel({'lon':localisation_lon})
            else:
                localized_lon = localized_period.sel({'lon':lon})

            for idy,lat in enumerate(assim_lat):
                parameters = parameters_lon.sel(lat=lat)
                obs = parameters.rr.data
                if self.localisation:
                    localisation_lat = np.array(
                            [self.radar.lat.data[idy+dlat] if idy+dlat>=0 and idy+dlat<len(assim_lat) else np.nan
                                for dlat in range(-self.localisation,self.localisation+1)]
                        )
                    # On enlève les valeurs manquantes (pour les points en bord de domaine)
                    # ==> la localisation est réduite en bordure de domaine !
                    localisation_lat = localisation_lat[~np.isnan(localisation_lat)]
                    raw_localized = localized_lon.sel({'lat':localisation_lat})
                    raw = raw_localized.sel({'time':date, 'lat':lat, 'lon':lon}).rr.data
                    raw_localized = raw_localized.rr.data.flatten()  # "Super ensemble"
                else:
                    raw_localized = localized_lon.sel({'lat':lat}).rr.data.flatten()
                    raw = raw_localized

                ##############################################################################################################################
                # Speed test
                #time_selection_unique += self.selection_unique(assimilation_period, localisation_lat, localisation_lon)
                #time_selection_sequentielle += self.selection_sequentielle(assimilation_period, localisation_lat, localisation_lon)
                # For 330 dates over the Grandes Rousses domaine the result of the speed test is :
                # Total time for unique selection :  178115.16904830933
                # Total time for sequential selection :  217705.75380325317
                # BUT the sequential selectionis far more efficient since there are fewer calls
                ##############################################################################################################################

                # Is the observation outside the ensemble ?
                if (np.min(raw) > obs) or (np.max(raw) < obs):
                    self.nb_out_raw[idy, idx] += 1
                if (np.min(raw_localized) > obs) or (np.max(raw_localized) < obs):
                    self.nb_out_loc[idy, idx] += 1

                new, inflation, sigma = self.assimilation(date, raw, raw_localized, parameters, lat, lon, idx, idy)

                self.erreur_obs[idy,idx,idd] = sigma

                if inflation >= 2:
                    print(f'Iteration {inflation} for pixel ({idx}, {idy})')
                self.inflation[idy,idx] += int(inflation)

                # TODO : remplir une liste plutot que boucler
                for member, field in self.newlocalfield.items():
                    field[idy,idx,idd]  = new[member-1]  # fill new member

        if self.plot:
            fig1,ax1 = plt.subplots(nrows=4, ncols=4, figsize=(16,7))
            fig2,ax2 = plt.subplots(nrows=4, ncols=4, figsize=(16,7))
            i = 0
            j = 0
            raw = localized_period.sel({'time':date})
            self.rrmin = 0.
            self.rrmax = max(
                    np.nanmax(raw.rr.data),
                    np.nanmax(parameters_date.rr.data)
                    )
            #self.rrmax = np.nanmax(parameters_date.rr.data)
            for member in range(1,17):
                assim = xr.DataArray(
                    name   = 'rr',
                    data   = self.newlocalfield[member][:,:,0],
                    dims   = ["lat", "lon"],
                    coords = dict(lon=raw.lon, lat=raw.lat),
                )
                #raw.sel({'member':member}).rr.plot(ax=ax1[i,j], cmap=plt.cm.YlGnBu)
                #im1 = plot_field(raw, ax1[i,j], self.rrmin, self.rrmax, title=f'Member {self.selection_globale[m-1]:03d}')
                im1 = plot_field(raw.sel({'member':member}).rr, ax1[i,j], self.rrmin, self.rrmax)
                #assim.plot(ax=ax2[i,j], cmap=plt.cm.YlGnBu)
                im2 = plot_field(assim, ax2[i,j], self.rrmin, self.rrmax)
                ax1[i,j].set_title(None)
                ax2[i,j].set_title(None)
                j = j + 1
                if j==4:
                    j = 0
                    i = i + 1
            if not os.path.exists(f'{self.date_str}'):
                os.makedirs(f'{self.date_str}')
            finalize_fig(fig1, im1, label='24-hour precipitation (mm)', outname=f'{self.date_str}/RAW_{self.date_str}.pdf')
            finalize_fig(fig2, im2, label='24-hour precipitation (mm)', outname=f'{self.date_str}/ASSIM_{self.date_str}.pdf')
#            fig1.savefig(f"{self.date_str}/RAW_{self.date_str}.pdf", format='pdf', layout='tight')
#            fig2.savefig(f"{self.date_str}/ASSIM_{self.date_str}.pdf", format='pdf', layout='tight')
            #plot_weights(date, weights)

    @speedtest
    def ponctual_assimilation(self, date, idd, localized_period, parameters_date):
        stop = False

        assimilation_points = zip(self.nivometeo.num_poste.data, np.max(self.nivometeo.lat, axis=1).data, np.max(self.nivometeo.lon, axis=1).data)
        for idp, (num_poste, lat, lon) in enumerate(assimilation_points):
            #print(num_poste)
            lat = nearest(self.radar.lat, lat)  # Latitude of the corresponding antilope pixel
            lon = nearest(self.radar.lon, lon)  # Longitude of the corresponding antilope pixel
            idy = np.where(self.radar.lat.data==lat)[0][0]  # index of the corresponding antilope pixel latitude
            idx = np.where(self.radar.lon.data==lon)[0][0]  # index of the corresponding antilope pixel longitude
            if self.localisation is not None:
                localisation_lat = np.array([self.radar.lat.data[idy+dlat] if idy+dlat>=0 and idy+dlat<len(self.radar.lat) else np.nan for dlat in range(-self.localisation, self.localisation+1)])
                localisation_lon = np.array([self.radar.lon.data[idx+dlon] if idx+dlon>=0 and idx+dlon<len(self.radar.lon) else np.nan for dlon in range(-self.localisation, self.localisation+1)])
                localisation_lat = localisation_lat[~np.isnan(localisation_lat)]
                localisation_lon = localisation_lon[~np.isnan(localisation_lon)]
            else:
                localisation_lat = lat
                localisation_lon = lon
            raw_localized = localized_period.sel({'lat':localisation_lat, 'lon':localisation_lon})
            if self.frequency == 'hourly':
                t1 = time.time()
                raw_localized.compute()  # Load data now
                t2 = time.time()
                #print(f'reading "raw_localized" took {(t2-t1)*1000.}ms')
#            obs = obs_date.sel({'lat':lat, 'lon':lon})
            if self.localisation is not None:
                raw = raw_localized.sel({'time':date, 'lat':lat, 'lon':lon}).rr.data
                raw_localized = raw_localized.rr.data.flatten()  # "Super ensemble"
            else:
                raw_localized = raw_localized.rr.data.flatten()
                raw = raw_localized
            parameters = parameters_date.sel({'lat':lat, 'lon':lon})
            obs = parameters.rr.data

            # Is the observation outside the ensemble ?
            if (np.min(raw) > obs) or (np.max(raw) < obs):
                self.nb_out_raw[idp] += 1
            if (np.min(raw_localized) > obs) or (np.max(raw_localized) < obs):
                self.nb_out_loc[idp] += 1

            new, inflation, sigma = self.assimilation(date, raw, raw_localized, parameters, lat, lon, idx, idy, num_poste=num_poste)

            self.erreur_obs[idp,idd] = sigma

            if inflation >= 2:
                print(f'{inflation} iterations for station {num_poste}')
            self.inflation[idp] += int(inflation)

            for member, field in self.newlocalfield.items():
                field[idp,idd]  = new[member-1]  # fill new member

#    @speedtest
    def assimilation(self, date, raw, raw_localized, parameters, lat, lon, idx, idy, num_poste=None):

        nb_new_member = 0
        sigma = float(parameters.sigma.data)/2.
        obs = parameters.rr.data
        mu = parameters.mu.data

        inflation = 0
        # TODO : avec la loi gamma, si mu>0 et max(raw/raw_localized)=0 ==> ne pas entrer dans la boucle (inutile, ce cas spécifique est traité à part)
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
            #weights = self.weighting(raw_localized, mu, sigma, obs, plot_distribution=True)
            weights = self.weighting(raw_localized, mu, sigma, obs)
            weights = weights / np.sum(weights)

            # 3. Resampling
            #--------------
            # ECC is the order of members indicies (from 0 to 15 !) sorted by increasing precipitation
            selection_locale = self.resample(weights, self.Ne)  # Idicies of selected particles from the "super-ensemble"
            nb_new_member = len(np.unique(selection_locale))

        if inflation == 9:
            # Generally occurs with gamma likelyhood when obs > 0 and all members == 0
            new = raw
        else:
            tmp = np.sort(raw_localized[selection_locale])

            # 4. ECC for consistency with raw members
            #----------------------------------------
            ecc = np.argsort(raw, axis=0)
            new = np.empty(len(tmp))
            new[:] = np.nan
            for idx, value in enumerate(tmp):
                new[ecc[idx]] = value

        # TODO : virer la ligne suivante qui court-circuite l'ECC
        #new = raw_localized[selection_locale]

        # To plot the posterior distribution:
        # self.weighting(new, parameters.mu.data, parameters.sigma.data, plot_distribution=True)

        # To plot data for one specific point / date
#        if num_poste == 73150400:
#            self.weighting(raw_localized, parameters.mu.data, sigma, obs, plot_distribution=True, num_poste=num_poste, date=date)
        #if self.plot:
        #    self.plot_assimilation(raw, new, obs, num_poste, date)
        #    self.weighting(raw_localized, parameters.mu.data, sigma, obs, plot_distribution=True, num_poste=num_poste, date=date)

        return new, inflation, sigma

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
        if num_poste == None:
            fig.savefig(f'assim_{date}.pdf', formatout='pdf',  bbox_inches='tight')
        else:
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
            null  = np.empty((self.nlat, self.nlon, len(self.period)))  # 2D (lat/lon) field
            self.nb_out_raw = np.zeros((self.nlat, self.nlon))  # Count number of obs outside raw ensemble
            self.nb_out_loc = np.zeros((self.nlat, self.nlon))  # Count number of obs outside localized ensemble
            self.inflation = np.zeros((self.nlat, self.nlon))  # Count number of time inflation was used
        else:
            self.nposte = len(self.nivometeo.num_poste)
            null = np.empty((self.nposte, len(self.period)))
            self.nb_out_raw = np.zeros(self.nposte)  # Count number of obs outside raw ensemble
            self.nb_out_loc = np.zeros(self.nposte)  # Count number of obs outside localized ensemble
            self.inflation = np.zeros(self.nposte)  # Count number of time inflation was used
        self.newlocalfield = {m:null.copy() for m in range(1, self.Ne+1)}
        self.erreur_obs = null.copy()
        time_selection_unique = 0
        time_selection_sequentielle = 0
        time_localisation = 3  # TODO : à passer en paramètre
        for idd,date in enumerate(self.period):
            print(date)
            t1 = time.time()
            # On réduit le dataset maintenant pour gagner du temps ensuite
            if self.frequency == 'hourly' and self.localisation is not None:
                assimilation_period = [date + timedelta(hours=dt) for dt in range(-time_localisation,time_localisation+1)]
            else:
                assimilation_period = [date]
            localized_period = self.ensemble.sel({'time':assimilation_period}).compute()  # Load data into memory now
            #print(f'reading "raw_localized" took {(t2-t1)*1000.}ms')
#            obs_date = self.radar.sel(time=date)
#            if self.frequency == 'hourly' and self.localisation:
#                raw_date = localized_period.sel(time=date)
#            else:
#                raw_date = localized_period
            parameters_date = self.parameters.sel({'time':date})

            self.date_str = date.strftime('%Y%m%d%H')
            if self.plot:
                if not os.path.exists(self.date_str):
                    os.makedirs(self.date_str)

            if self.gridded:
                self.gridded_assimilation(date, idd, localized_period, parameters_date)
            else:
                self.ponctual_assimilation(date, idd, localized_period, parameters_date)

            if self.plot:
                self.plot_sigma(parameters_date)
                self.rrmin = 0.
                self.rrmax = max(
                        np.nanmax(localized_period.rr.data),
                        np.nanmax(parameters_date.rr.data)
                        )
                #self.rrmax = np.nanmax(parameters_date.rr.data)
                self.plot_obs(parameters_date)
            t2 = time.time()
            print(f'Assimilation for date {self.date_str} took {(t2-t1)*1000.}ms')


    @speedtest
    def plot_weights(self, date, weights):
        # TODO : find a way to plot weights...
        pass

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
        self.weighting(model, self.parameters.mu.data[x][y], self.parameters.sigma.data[x][y], self.parameters.rr.data[x][y], plot_distribution=True)

    def add_landmarks(self, ax):
        # Add landmarks
        for landmark, infos in landmarks.items():
            ax.plot(infos['lon'], infos['lat'], marker=infos['marker'], color='red', markersize=5)
            ax.annotate(landmark, (infos['lon']+0.003, infos['lat']+0.003), color='red', fontsize=12)

    @speedtest
    def output(self, localfields, globalfields):
        # This method takes ~3.5 s but most of the time (~3.3s) is spent
        # while saving figures.
        # TODO : trouver un moyen de plotter les champs de poids

        if self.gridded:
            out_raw = xr.DataArray(
                    name   = 'out_raw',
                    data   = self.nb_out_raw,
                    dims   = ["lat", "lon"],
                    coords = dict(lon=self.radar.lon, lat=self.radar.lat),
                    attrs  = dict(description="Number of assimilation step when the observation was outside the raw ensemble"),
                )
            out_loc = xr.DataArray(
                    name   = 'out_loc',
                    data   = self.nb_out_loc,
                    dims   = ["lat", "lon"],
                    coords = dict(lon=self.radar.lon, lat=self.radar.lat),
                    attrs  = dict(description="Number of assimilation step when the observation was outside the localized ensemble"),
                )
            inflation = xr.DataArray(
                    name   = 'inflation',
                    data   = self.inflation,
                    dims   = ["lat", "lon"],
                    coords = dict(lon=self.radar.lon, lat=self.radar.lat),
                    attrs  = dict(description="Number of assimilation step when inlfation was used"),
                )
            erreur_obs = xr.DataArray(
                    name   = 'erreur_obs',
                    data   = self.erreur_obs,
                    dims   = ["lat", "lon", "time"],
                    coords = dict(lon=self.radar.lon, lat=self.radar.lat, time=self.period),
                    attrs  = dict(description="Observation error"),
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
            erreur_obs = xr.DataArray(
                    name   = 'erreur_obs',
                    data   = self.erreur_obs,
                    dims   = ["num_poste", "time"],
                    coords = dict(num_poste=self.nivometeo.num_poste.data, time=self.period),
                    attrs  = dict(description="Observation error"),
                )
        outname1 = f"nb_obs_outside_raw_ensemble_{self.period[0].strftime('%Y%m%d%H')}_{self.period[-1].strftime('%Y%m%d%H')}_{self.frequency}_{self.domain}"
        outname2 = f"nb_obs_outside_localized_ensemble_{self.period[0].strftime('%Y%m%d%H')}_{self.period[-1].strftime('%Y%m%d%H')}_{self.frequency}_{self.domain}"
        outname3 = f"inflation_{self.period[0].strftime('%Y%m%d%H')}_{self.period[-1].strftime('%Y%m%d%H')}_{self.frequency}_{self.domain}"
        outname4 = f"observation_error_{self.period[0].strftime('%Y%m%d%H')}_{self.period[-1].strftime('%Y%m%d%H')}_{self.frequency}_{self.domain}"
        if self.localisation is not None:
            outname1 = '_'.join([outname1, f'localisation{self.localisation}'])
            outname2 = '_'.join([outname2, f'localisation{self.localisation}'])
            outname3 = '_'.join([outname3, f'localisation{self.localisation}'])
        if self.mask is not None:
            outname1 = '_'.join([outname1, f'mask{self.mask}'])
            outname2 = '_'.join([outname2, f'mask{self.mask}'])
            outname3 = '_'.join([outname3, f'mask{self.mask}'])
        if self.debiasing is not None:
            outname1 = '_'.join([outname1, f'debiasing{self.debiasing}'])
            outname2 = '_'.join([outname2, f'debiasing{self.debiasing}'])
            outname3 = '_'.join([outname3, f'debiasing{self.debiasing}'])
        if self.gridded:
            #for (outname, field) in zip([outname1, outname2, outname3, outname4], [out_raw, out_loc, inflation, erreur_obs]):
            for (outname, field) in zip([outname1, outname2, outname3], [out_raw, out_loc, inflation]):
                fig, ax = plt.subplots(figsize=(12,6))
                field.plot(ax=ax, cmap=plt.cm.Greys)
                self.add_landmarks(ax)
                fig.savefig(f'{outname}.pdf', format='pdf', bbox_inches='tight')
        out_raw.to_netcdf(f"{outname1}.nc")
        out_loc.to_netcdf(f"{outname2}.nc")
        inflation.to_netcdf(f"{outname3}.nc")
        erreur_obs.to_netcdf(f"{outname4}.nc")

#        if self.plot:
#            #fig1, ax1 = plt.subplots(nrows=4, ncols=4, figsize=(16, 8))
#            fig2, ax2 = plt.subplots(nrows=4, ncols=4, figsize=(16, 8))
#            #fig3, axes3 = plt.subplots(nrows=4, ncols=4, figsize=(16, 8))
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

#        if self.plot:
#            #finalize_fig(fig1, im1, label='24-hour precipitation (mm)', outname=f'{self.date_str}/ASSIM_globale_{self.date_str}.pdf')
#            figname = f'{self.date_str}/ASSIM_locale_{self.date_str}'
#            if self.localisation is not None:
#                figname = '_'.join([figname, f'localisation{self.localisation}'])
#            if self.mask is not None:
#                figname = '_'.join([figname, f'mask{self.mask}'])
#            if self.debiasing is not None:
#                figname = '_'.join([figname, f'debiasing{self.debiasing}'])
#            finalize_fig(fig2, im2, label='24-hour precipitation (mm)', outname=f'{figname}.pdf')
#            #finalize_fig(fig3, im3, label='Weight', outname=f'{self.date_str}/WEIGHTS_{self.date_str}.pdf')
#            #t5 = time.time()
#            #print(f'Finalisation of figures took {(t5-t4)*1000.}ms')

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
    t0 = time.time()
    args = parse_command_line()
    #goto(args.workdir)
    extract_period = date_range(args.datebegin, args.dateend, dt=timestep[args.frequency])

    antilope = read_obs(args)
    pearome = read_ensemble(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'), args.frequency, args.domain)
    if args.gridded:
        localfields = xr.DataArray(
                name   = 'rr',
                dims   = ["lat", "lon", "time", "member"],
                coords = dict(lon=antilope.lon, lat=antilope.lat, time=extract_period, member=range(1,17)),
                attrs  = dict(description="24 hour precipitation",units="mm"),
            )
        globalfields = xr.DataArray(
                name   = 'rr',
                dims   = ["lat", "lon", "time", "member"],
                coords = dict(lon=antilope.lon, lat=antilope.lat, time=extract_period, member=range(1,17)),
                attrs  = dict(description="24 hour precipitation",units="mm"),
            )
        nivometeo = None
    else:
        nivometeo = read_nivometeo_obs()
        localfields = xr.DataArray(
                name   = 'rr',
                dims   = ["num_poste", "time", "member"],
                coords = dict(num_poste=nivometeo.num_poste.data, time=extract_period, member=range(1,17)),
                attrs  = dict(description="24 hour precipitation",units="mm"),
            )
        globalfields = None

    interp_ensemble = pearome.interp(lon=antilope.lon, lat=antilope.lat).clip(0)  # Avoid <0 precipitation values
    pf = ParticleFilter(extract_period, antilope, interp_ensemble, nivometeo, args.plot, args.frequency, args.gridded, args.localisation, args.mask, args.debiasing, args.domain, args.likelyhood)
    mask = pf.pdf_parameters()
    pf.run()  # Compute weight fields before normalisation + plot raw/interp fields
    #pf.selection()  # Global and local selections

    localfields, globalfields = pf.output(localfields, globalfields)

#    plot_chrono(extract_period, antilope, pearome, localfields)
    outname = f"Assimilation_locale_{args.datebegin.strftime('%Y%m%d%H')}_{args.dateend.strftime('%Y%m%d%H')}_{args.frequency}_{args.domain}"
    if args.localisation is not None:
        outname = '_'.join([outname, f'localisation{args.localisation}'])
    if mask is not None:
        outname = '_'.join([outname, f'mask{mask}'])
    if args.debiasing is not None:
        outname = '_'.join([outname, f'debiasing{args.debiasing}'])
    localfields.to_netcdf(f"{outname}.nc")
    #globalfields.to_netcdf(f"Assimilation_globale_{args.datebegin.strftime('%Y%m%d%H')}_{args.dateend.strftime('%Y%m%d%H')}_{args.frequency}.nc")

    tfin = time.time()
    print(f'Total execution time : {(tfin-t0)/60.} minutes')
