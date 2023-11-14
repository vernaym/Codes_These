#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os, sys
from datetime import datetime,timedelta
import pandas as pd  # Version 0.25.3
import numpy as np
import scipy
from scipy.ndimage import uniform_filter
from scipy.spatial.distance import cdist
from scipy import sparse
from scipy.sparse import csc_matrix, csr_matrix, dia_matrix
from scipy.sparse import linalg as splinalg
from scipy.sparse.linalg import inv, spsolve
from scipy.spatial import cKDTree
from scipy.sparse import diags
from scipy.stats import norm, gamma
import math
import xarray as xr
from pykrige.uk import UniversalKriging
import glob
import shapefile
from shapely.geometry import Point, Polygon

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
import palettable

import plotly.express as px

#plt.rcParams["figure.autolayout"] = True

from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.proj3d import proj_transform

import time

from These.radar import Preprocessing_ANTILOPE
from These.scripts import make_mask

##############################################################################################
# TODO : Save number of selected members for each pixel
##############################################################################################

datadir = '/home/vernaym/These/DATA'

domain_coords = dict(
        GrandesRousses = dict(latmax=45.240, latmin=44.990, lonmin=6.010, lonmax = 6.490),
        NorthernAlps   = dict(lonmin=6.0, lonmax=6.9, latmin=45.6, latmax=46.35),
        CentralAlps    = dict(lonmin=5.6, lonmax=7.0, latmin=45.0, latmax=45.6),
        SouthernAlps   = dict(lonmin=5.7, lonmax=7.0, latmin=44.2, latmax=45.0),
        HauteSavoie    = dict(lonmin=5.82, lonmax=7.05, latmin=45.5, latmax=46.29),
        MontBlanc      = dict(lonmin=6.45, lonmax=7.1, latmin=45.65, latmax=46.1),
        Savoie         = dict(lonmin=6.06, lonmax=7.1, latmin=45.15, latmax=45.7),
        Isere          = dict(lonmin=5.54, lonmax=6.19, latmin=44.89, latmax=45.16),
        Brianconnais   = dict(lonmin=6.48, lonmax=6.95, latmin=44.67, latmax=44.95),
        HautesAlpes    = dict(lonmin=5.9, lonmax=7, latmin=44, latmax=45.1),
        Vercors        = dict(lonmin=5.4, lonmax=6, latmin=44.75, latmax=45.4),
        AlpesSud       = dict(lonmin=6.56, lonmax=6.92, latmin=44.18, latmax=44.49),
        alp            = dict(latmax=46.450, latmin=44.100, lonmin=5.400, lonmax=7.200),
)

figsize = dict(
        alp            = dict(singleplot=(14,16), ensembleplot=(32,20)),
        GrandesRousses = dict(singleplot=(15,7), ensembleplot=(16,7)),
        HauteSavoie    = dict(singleplot=(12,12), ensembleplot=(12,12)),
        MontBlanc      = dict(singleplot=(15,10), ensembleplot=(15,10)),
        Savoie         = dict(singleplot=(16,8), ensembleplot=(16,7)),
        HautesAlpes    = dict(singleplot=(15,10), ensembleplot=(15,10)),
        Vercors        = dict(singleplot=(15,12), ensembleplot=(15,12)),
        Isere          = dict(singleplot=(16,8), ensembleplot=(16,7)),
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

pltnorm = plt.Normalize()

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

# Parameters to compute Euclidian distance between all points in the domain
ld = 0.1 # correlation lenght. WARNING : ne pas trop augmenter la distance de correlation (analyse trop proche de l'obs ==> perte de dispersion)
# ld = 0.02 marche plutot bien (sous dispersion), ld=0.03 pas du tout !!!
max_dist = ld*2
#max_dist = 0.5  # Memory limit reached at 0.2 for domain Alp. WARNING : very high analysis sensibility to this parameter !!

def parse_command_line():
    description = "Evaluation of RADAR products (ANTILOPE or PANTHERE) using nivo-météo network observations"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-b', '--datebegin', help='Begining date of extraction, format YYYYMMDDHH or YYMMDDHH', required=True)
    parser.add_argument('-e', '--dateend', help = 'Final date of extraction (default=datebegin)')
    parser.add_argument('-d', '--domain', help='Domain of the file', choices=domain_coords.keys(), default='GrandesRousses')
    parser.add_argument('-w', '--workdir', help='Runing directory', default='/home/vernaym/workdir/ASSIMILATION')
    parser.add_argument('-a', '--assimilation', help='Assimilation method (Particle Filter, Ensemble Kalman Filter or Random Sampling)', choices=['pf', 'enkf', 'rs'], default='enkf')
    parser.add_argument('-m', '--mask', help='Switch observation error mask on/off', choices=[1,2,3,4,5,6,7,8,9], default=None, type=int)
    parser.add_argument('-c', '--debiasing', help='Apply bias correction to observation (0=constant bias, 1=pseudo-kriging, 2=estimation based on homogeneity,3=smoothing+scores)', default=None, type=int, choices=[0,1,2,3])
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

def sparse_cholesky(A): # The input matrix A must be a sparse symmetric positive-definite.
    """ 
    From https://gist.github.com/omitakahiro/c49e5168d04438c5b20c921b928f1f5d
    """
    n = A.shape[0]
    LU = splinalg.splu(A,diag_pivot_thresh=0) # sparse LU decomposition

    if ( LU.perm_r == np.arange(n) ).all() and ( LU.U.diagonal() > 0 ).all(): # check the matrix A is positive definite.
        return LU.L.dot( sparse.diags(LU.U.diagonal()**0.5) )
    else:
        sys.exit('The matrix is not positive definite')

@speedtest
def read_ensemble(datebegin, dateend, frequency, domain, antilope):
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
    if domain == 'GrandesRousses':
        filenames = [os.path.join(datadir, f"aspearome_{mb:03d}_2021073106_2022070106_{domain}_hourly.nc") for mb in range(1,17)]
    else:
        filenames = [os.path.join(datadir, f"aspearome_{mb:03d}_2021102806_2022060206_alp_hourly.nc") for mb in range(1,17)]
    filenames = [os.path.join(datadir, f"aspearome_{mb:03d}_2021102806_2022060206_alp_hourly.nc") for mb in range(1,17)]


    # open_mfdataset returns a dask.array<chunksize=(...), meta=np.ndarray> object that divides arrays into many small pieces, called chunks, 
    # each of which is presumed to be small enough to fit into memory in order to avoid a memory overload. 
    # The compute() method effectively load data into memory so it must be called as late as possible to reduce memory use as well as
    # running time.
    # See : https://docs.xarray.dev/en/stable/user-guide/dask.html
    #pearome = xr.open_mfdataset(filenames, combine='nested', concat_dim='member').compute()  # Impossible avec des fichiers trop volumineux

    # Chunks of a multiple of 24 time steps seem optimal (~0.6s by iteration vs >1.2 for other chunk sizes)
    pearome = xr.open_mfdataset(filenames, combine='nested', concat_dim='member', chunks={'time': 24})  # Setting chunks is critical (read the doc !)
    pearome = pearome.interp(lon=antilope.lon, lat=antilope.lat).clip(0)  # Avoid <0 precipitation values
    sel_lat = np.round(np.arange(domain_coords[domain]['latmin']-max_dist, domain_coords[domain]['latmax']+max_dist, 0.01), 2)
    sel_lon = np.round(np.arange(domain_coords[domain]['lonmin']-max_dist, domain_coords[domain]['lonmax']+max_dist, 0.01), 2)
    pearome = pearome.sel({'lat':np.intersect1d(sel_lat, pearome.lat.data), 'lon':np.intersect1d(sel_lon, pearome.lon.data)})

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

def add_boundaries(ax):
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
        ax.plot(x,y,color='k')

def add_cities(latmin, latmax, lonmin, lonmax):
    cities = pd.read_csv(os.path.join('/home/vernaym/safran/monitoring/', 'cities.csv'), sep=',')
    tmp = cities[(cities.population>10000) & (cities.lat>=latmin) & (cities.lat<=latmax) & (cities.lng>=lonmin) & (cities.lng<=lonmax)]
    plt.plot(tmp.lng, tmp.lat, marker='.', linestyle='')
    for idx in tmp.index:
        plt.text(tmp.lng[idx], tmp.lat[idx], tmp.city[idx], alpha=0.5)

def plot_correlation(ax, field, corr, point=1000):
    # TODO : give corr matrix and extract point / cooordinates here

    # Plot correlation matrix
    if not os.path.exists(f'codistance_matrix_correlation{max_dist}.png'):
        fig2,ax2 = plt.subplots(figsize=(20,20))
        ax2.spy(codist, precision=0.5)
        fig2.savefig(f'codistance_matrix_correlation{max_dist}.png', format='png')
        plt.close(fig2)

    corr = corr.where(corr>0)
    cml = corr.plot(ax=ax, cmap=plt.cm.Greys, add_colorbar=False, alpha=0.3)
    circle = plt.Circle((6.27, 45.07), max_dist, color='red', fill=False, linewidth=4)
    ax.add_artist(circle)

    return ax

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

    #ax.set_zlim(0., np.nanmax(Z))
    ax.set_zlim(0., 3500.)
    fig.colorbar(cm.ScalarMappable(norm=pltnorm, cmap=plt.cm.coolwarm), ax=ax, shrink=0.75, aspect=8, label=f'ANTILOPE precipitation (mm)')
    plt.savefig(f'{date}/OBS_3D_{date}.pdf', format='pdf')


def plot_field(field, ax, vmin, vmax, domain, title=None, cmap=plt.cm.YlGnBu):
    latmax = domain_coords[domain]['latmax']
    latmin = domain_coords[domain]['latmin']
    lonmin = domain_coords[domain]['lonmin']
    lonmax = domain_coords[domain]['lonmax']
    sel_lat = np.round(np.arange(latmin, latmax, 0.01), 2)
    sel_lon = np.round(np.arange(lonmin, lonmax, 0.01), 2)
    field = field.sel({'lat':np.intersect1d(sel_lat, field.lat.data), 'lon':np.intersect1d(sel_lon, field.lon.data)})

    im = field.plot(ax=ax, add_colorbar=False, vmin=vmin, vmax=vmax, cmap=cmap)
#    for landmark, infos in landmarks.items():
#        ax.plot(infos['lon'], infos['lat'], marker=infos['marker'], color='red', markersize=4)
#    add_boundaries(ax)
    ax.set_aspect('equal')
    ax.axis('off')
    if title is not None:
        ax.set_title(title)

    return im

def plot_distribution(ax, mean, sd, ensemble=None, label=None, color=None, distribution='norm', linewidth=1):
    x = np.linspace(0, 80, 10000)
    if color == None:
        color = next(ax._get_lines.prop_cycler)['color']

    if distribution == 'norm':
        ax.plot(x, norm.pdf(x, loc=mean, scale=sd), color=color, label=label, linestyle='--', linewidth=linewidth)
        ax.bar(mean, 1, width=0.3, color=color, label="Ensemble mean")
        if ensemble is not None:
            #ax.bar(ensemble, norm.pdf(ensemble, loc=mean, scale=sd), 'r-', color=color, label='members', linewidth=linewidth)
            ax.bar(ensemble, norm.pdf(ensemble, loc=mean, scale=sd), width=0.1, color=color)
    elif distribution == 'gamma':
        #k = mean**2/sd
        #theta = sd/mean
        #on veut que mu soit le mode de la distribution gamma (< à la moyenne)
        theta = (np.sqrt(mean**2+4*sd)-mean)/2
        k     = 4*sd/(np.sqrt(mean**2+4*sd)-mean)**2
        ax.plot(x, gamma.pdf(x, k, scale=theta), color=color, label=label, linestyle='--', linewidth=0.5)
    elif distribution == 'EGP':
        pass

    return ax

@speedtest
def finalize_fig(figure, imm, label, outname):
    # Saving figures is by far the slowest part (~1s per figure)

    #t1 = time.time()
    figure.tight_layout()
    figure.subplots_adjust(right=0.85)
    cbar_ax = figure.add_axes([0.87, 0.05, 0.03, 0.9])
    cb = figure.colorbar(imm, cax=cbar_ax)
    cb.ax.tick_params(labelsize=20)
    cb.set_label(label, size=24)
    #t2 = time.time()
    #print(f'Finalising figures took {(t2-t1)*1000.}ms')
    figure.savefig(outname, format='pdf')
    plt.close(figure)
    #t3 = time.time()
    #print(f'Saving figures took {(t3-t2)*1000.}ms')

@speedtest
def read_obs(args):
    #filename = f'ANTILOPE{suffix[args.frequency]}_{args.datebegin.strftime("%Y%m%d%H")}_{args.dateend.strftime("%Y%m%d%H")}_alp.nc'
    filename = f'ANTILOPEQ_2021102900_2022060200_alp.nc'
    if not os.path.exists(filename):
        print(f'WARNING : file {filename} does not exist, looking for it under {datadir}')
        filename = os.path.join(datadir, filename)
    if not os.path.exists(filename):
        print(f'WARNING : no file named {filename} under {datadir}, using default file ANTILOPEH_2021103000_2022060200_alp.nc')
        filename = os.path.join(datadir, f'ANTILOPEH_2021103000_2022060200_alp.nc')

        antilope = xr.open_dataset(filename, chunks={'time': 24})  # WARNING : works with xarray-2022.3.0 but not xarray-2023.1.0
        #antilope = antilope.where((antilope.lon>=lonmin) & (antilope.lon<=lonmax) & (antilope.lat>=latmin) & (antilope.lat<=latmax), drop=True)
        # Pour une assimilation quotidienne, sommer les cumuls horaires
        if args.frequency == 'daily' and  'ANTILOPEH' in filename:
            # Convert hourly precipitation into 24h precipitation between 6h UTC J-1 and 6h UTC J
            # Problem : the xarray tools to do that allows only accumulations between
            # 0h and 24h.
            # solution : shift time serie by 6h, compute 24h accumulations and
            # shift back !
            antilope['time'] = antilope.time-np.timedelta64(7, 'h')
            #antilope['time'] = antilope.time-np.timedelta64(6, 'h')
            antilope = antilope.resample(time='D').sum(dim='time')  # !!! VERY SLOW !!!
            antilope['time'] = antilope.time+np.timedelta64(30, 'h')
    else:
        antilope = xr.open_dataset(filename)

    # Extract sub-domain
    latmax = domain_coords[args.domain]['latmax']
    latmin = domain_coords[args.domain]['latmin']
    lonmin = domain_coords[args.domain]['lonmin']
    lonmax = domain_coords[args.domain]['lonmax']
    sel_lat = np.round(np.arange(latmin-max_dist, latmax+max_dist, 0.01), 2)
    sel_lon = np.round(np.arange(lonmin-max_dist, lonmax+max_dist, 0.01), 2)
    antilope = antilope.sel({'lat':np.intersect1d(sel_lat, antilope.lat.data), 'lon':np.intersect1d(sel_lon, antilope.lon.data)})

    return antilope


class Assimilation(object):

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
            num = np.nanmax(((4*k + delta) / dy).astype(int))
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

    def plot_obs(self, field, var='rr', domain=None, correlation_area=False, plot3D=False, text1=None):

        if var == 'rr':
            savename = f'{self.date_str}/OBS_{self.date_str}_{domain}.pdf'
        elif var == 'mu':
            savename = f'{self.date_str}/CLIM_DEBIASED_OBS_{self.date_str}_{domain}.pdf'
        elif var == 'db':
            savename = f'{self.date_str}/DYN_DEBIASED_OBS_{self.date_str}_{domain}.pdf'
        elif var == 'obs':
            savename = f'{self.date_str}/ASSIMILATED_OBS_{self.date_str}_{domain}.pdf'
        elif var == 'diff':
            savename = f'{self.date_str}/DIFF_OBS_{self.date_str}_{domain}.pdf'


        if domain is None:
            domain = self.domain

        if not os.path.exists(savename) or var in ['obs', 'diff']:

            latmin = domain_coords[domain]['latmin']
            latmax = domain_coords[domain]['latmax']
            lonmin = domain_coords[domain]['lonmin']
            lonmax = domain_coords[domain]['lonmax']
            sel_lat = np.round(np.arange(latmin, latmax, 0.01), 2)
            sel_lon = np.round(np.arange(lonmin, lonmax, 0.01), 2)

            # Add correlation area
            if correlation_area:
                point = 1200
                corr = xr.DataArray(
                        name   = 'correlation',
                        data   = self.pond.getrow(point).toarray()[0].reshape((len(field.lat), len(field.lon))),
                        dims   = ["lat", "lon"],
                        coords = dict(lon=field.lon, lat=field.lat),
                    )

                corr = corr.sel({'lat':np.intersect1d(sel_lat, corr.lat.data), 'lon':np.intersect1d(sel_lon, corr.lon.data)})

            field = field.sel({'lat':np.intersect1d(sel_lat, field.lat.data), 'lon':np.intersect1d(sel_lon, field.lon.data)})

            # Plot ANTILOPE precipitation field
#            fig = plt.figure(figsize=figsize[domain]['singleplot'])
            fig,ax = plt.subplots(figsize=figsize[domain]['singleplot'])
            #fig = plt.figure(figsize=(14,16))  Alps

            #field[var].plot(vmin=0, vmax=self.rrmax, cmap=plt.cm.YlGnBu, cbar_kwargs={'label': "24 hour precipitation (mm)", 'labelsize':18})  # quadmesh object
            if var == 'diff':
                im = field[var].plot(ax=ax, add_colorbar=False, cmap=plt.cm.coolwarm)  # quadmesh object
            else:
                im = field[var].plot(ax=ax, vmin=0, vmax=self.rrmax, add_colorbar=False, cmap=plt.cm.YlGnBu)  # quadmesh object

            figure = px.imshow(field[var].data, color_continuous_scale='YlGnBu', origin='lower')  # https://plotly.com/python/2D-Histogram/
#            fig.add_scattermapbox(lat=field.lat, lon=field.lon,marker_size=field['sigma'],marker_symbol='x',showlegend = False)  # https://stackoverflow.com/questions/68762104/plotly-adding-scatter-geo-points-and-traces-on-top-of-density-mapbox
            lat, lon = np.meshgrid(range(len(field.lat.data)), range(len(field.lon.data)))
            figure.add_scatter(
                    x=lon.flatten(),
                    y=lat.flatten(),
                    mode = 'markers',
                    marker = dict(
                        symbol='x-thin',
                        size=np.abs(np.transpose(field['sigma'].data).flatten()),
                        color='grey'
                    ),
#                    color_discrete_sequence=['grey']
                    )
#                    ).update_traces(marker=dict(color='grey'))
#            figure.show()

            if correlation_area:
                # Add correlation area
                ax = plot_correlation(ax, field, corr, point=1200)

            # Add landmarks
            if domain == 'GrandesRousses':
                for landmark, infos in landmarks.items():
                    plt.plot(infos['lon'], infos['lat'], marker=infos['marker'], color='red', markersize=10)
                    plt.annotate(landmark, (infos['lon']+0.003, infos['lat']+0.003), color='red', fontsize=20)
            add_boundaries(ax)
            add_cities(latmin, latmax, lonmin, lonmax)

            if text1 is not None:
                for lon, lat, text in text1:
                    if lon>=lonmin and lon<=lonmax and lat>=latmin and lat<=latmax:
                        text = f'{text:.2f}'
                        ax.text(lon, lat, text, fontsize=14)

            # Add colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cb = fig.colorbar(im, cax=cbar_ax)
            cb.ax.tick_params(labelsize=18)
            cb.set_label("24h precipitation(mm)", size=18)

            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            ax.set_aspect('equal')
            #ax.axis('off')  # To add lat/lon
            ax.set_title(f'Date {self.date_str}', fontsize=20)
            ax.axes.get_xaxis().get_label().set_visible(False)
            ax.axes.get_yaxis().get_label().set_visible(False)
            fig.tight_layout()

            fig.savefig(savename, format='pdf')

        # Plot 3D ANTILOPE precipitation field
        if plot3D:
            mnt = xr.open_dataset('/home/vernaym/QGIS/MNT/DEM_ALPES_WGS84_250m_bilinear.nc')  # Pour tracer sur toutes les Alpes
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
        elif self.debiasing == 3:
            #mask = xr.open_dataset(os.path.join("/home/vernaym/These/DATA/mask", f"Estimated_ratio_{self.domain}_0.15_15.nc"))
            mask = xr.open_dataset(os.path.join(f"Estimated_ratio.nc"))
            #ratio = mask.rr
            try:
                ratio = mask.Ratio
            except AttributeError:
                ratio = mask.ratio
        else:
            ratio = 1  # No debiasing
        parameters['mu'] = parameters['rr'] / ratio  # TODO : check if the ensemble after assimilation is not biased
        self.ratio = ratio

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
            elif self.mask in [6]:
                #mask = xr.open_dataset(os.path.join("/home/vernaym/workdir/ASSIMILATION/mask", f"Observation_error_smoothingsize20_{self.domain}.nc"))
                mask = xr.open_dataset(os.path.join("/home/vernaym/workdir/ASSIMILATION/mask", f"mask5_loc10_seuil0.1_{self.domain}.nc"))
                #mask = xr.open_dataset(os.path.join("/home/vernaym/workdir/ASSIMILATION/mask", f"Observation_error_smoothingsize30.nc"))
                #parameters['sigma'] = (0.261 + 0.263 * parameters['rr']) * np.abs(mask.rr)
                #parameters['sigma'] =  np.abs(mask.rr)
                parameters['sigma'] =  np.abs(mask.mask)
                #parameters['sigma'] = np.abs(mask.rr) * 0.261 + 0.263 * parameters['rr']
                #parameters['sigma'] = np.abs(mask.rr)
            elif self.mask in [7]:
                mask = xr.open_dataset(os.path.join("/home/vernaym/These/DATA/mask", f"Observation_error_absolute_value_smoothingsize15_{self.domain}.nc"))
                parameters['sigma'] =  np.abs(mask.rr)
            elif self.mask in [8]:
                mask = xr.open_dataset(os.path.join("/home/vernaym/These/DATA/mask", f"Observation_error_15_0.15_alp.nc"))
                parameters['sigma'] =  np.abs(mask.rr)
            elif self.mask in [9]:
                mask = xr.open_dataset(os.path.join(f"Observation_error.nc"))
                #parameters['sigma'] =  mask.rr  # Pas de valeur absolue pour le calcul des covariances !
                #parameters['sigma'] =  mask.ratio  # Pas de valeur absolue pour le calcul des covariances !
                try:
                    parameters['sigma'] =  mask.Uncertainty
                except AttributeError:
                    parameters['sigma'] =  mask.error
                #parameters['sigma'] =  (0.261 + 0.263 * parameters['rr'])*np.abs(mask.rr)  # PF

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
        fig, ax = plt.subplots(figsize=figsize[self.domain]['singleplot'])
        #cmap = plt.cm.YlGnBu
        cmap = plt.cm.Greys
        im = field['sigma'].plot(ax=ax, cmap=cmap, cbar_kwargs=dict(label='Standard deviation (mm)'))
        for landmark, infos in landmarks.items():
                ax.plot(infos['lon'], infos['lat'], marker=infos['marker'], color='red', markersize=4)
        fig.tight_layout()
        fig.savefig(f"{self.date_str}/PDF_std_{self.date_str}.pdf", format='pdf')

    def plot_parameters(self):
        label_map = dict(sigma='Standard deviation sigma (mm)', k='Shape parameter (k)', theta='Scale parameter (theta)', delta='Shift parameter (delta)')
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize[self.domain]['singleplot'])
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

    def plot_ensemble(ensemble, label, domain=None):
        if domain is None:
            domain = self.domain
        fig,ax = plt.subplots(nrows=4, ncols=4, figsize=figsize[domain]['ensembleplot'])
        #fig,ax = plt.subplots(nrows=2, ncols=8, figsize=(16,16))  # Alps
        i = 0
        j = 0
        for member in ensemble.member.data:
            member.plot(ax=ax[i,j], cmap=plt.cm.YlGnBu)
            j = j + 1
            if j==4:
                j = 0
                i = i + 1
        fig.savefig(f"{self.date_str}/{label}_{self.date_str}_{domain}.pdf", format='pdf')
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

    def plot_super_ensemble(self, point, oldfield, newfield, mu, std, pond, product, label=None, ax=None, reference=None, initial_obs=None):

        if ax is None:
            now = True
            fig, ax = plt.subplots()
        else:
            now = False

        weights = pond.getrow(point).toarray()[0]
        oldobs = oldfield[point]
        newobs = newfield[point]
        #obsweight = weights[point]/np.sum(weights)
        obsweight = weights[point]

        ax.hist(oldfield, density=True, bins=np.arange(np.floor(np.nanmin(oldfield))-0.1, np.ceil(np.nanmax(oldfield)) + 0.1, 0.1), weights=weights/np.sum(weights), label='Neighborhood distribution', alpha=0.5)
        #ax.bar(oldfield, weights/np.sum(weights))
        #ax.plot(obs, obsweight, marker='+', color='orange', label='Initial Observation')
        ax.bar(mu, 2, width=0.005, color='k')

        # Plot the actual distribution used for the assimilation :
        plot_distribution(ax, mu, std, distribution='norm', linewidth=1, label=f'{product} distribution', color='k')  # mu est la valeur du pixel
#        plot_distribution(ax, obs, std, distribution='norm', linewidth=1, label='Observation distribution')  # mu est la valeur du pixel

        # To test a new method (the goal is that it gives the same distribution as the red one in the final version) :
#        mean = np.sum(weights*oldfield)/np.sum(weights)
#        ax.bar(mean, 2.54, width=0.005, color='k')
#        plot_distribution(ax, mean, std, distribution='norm', linewidth=1, color='k')  # mu est la valeur du pixel

        ax.plot(oldobs, obsweight, marker='.', color='k', linestyle='', markersize=10)
        ax.bar(oldobs, 2, width=0.1, color='k', label='Before correction')
        ax.bar(newobs, 2, width=0.1, color='green', label='After correction')

        if reference is not None:
            ax.bar(reference, 2, width=0.1, color='red', label='Reference Observation')

        if initial_obs is not None:
            ax.bar(initial_obs[point], 2, width=0.1, color='blue', label='Initial Observation')

#        newobs = (obs*obsweight + mean * np.nanmean(weights[weights>0])) / (obsweight+np.nanmean(weights[weights>0]))
#        #sd   = np.sum(weights*(oldfield-obs)**2)/np.sum(weights)
#        sd   = np.sum(weights*(oldfield-mean)**2)/np.sum(weights)
#        plot_distribution(ax, newobs, sd, distribution='norm', linewidth=1, color='blue', label='Observation distribution')

        if now:
            # Set figure boundaries
            ax.set_ylim(bottom=0, top=0.5)
            ax.set_xlim(left=np.nanmin(oldfield)-std, right=np.nanmax(oldfield)+std)
            #ax.set_xlim(left=0, right=3)
            ax.set_xlabel('Precipitation (mm)')
            ax.set_ylabel('Probability')
            ax.legend()
#            if not os.path.exists(f'{self.date_str}/distributions'):
#                os.makedirs(f'{self.date_str}/distributions')
#            fig.savefig(f'{self.date_str}/distributions/DISTRIBUTION_{product}_{point}.pdf')
            if not os.path.exists(f'distributions'):
                os.makedirs(f'distributions')
            fig.savefig(f'distributions/DISTRIBUTION_{product}.pdf')
            plt.close(fig)

    @speedtest
    def background_error_covariance(self, ensemble):
        """
        B = 1/N * sum((Xi-Xmean)(Xi-Xmean)')
        """

        # TODO : assurer une erreur d'ébauche minimale si la dispersion est nulle (par ex tous les membres ratent une précipitation observée)
        # cf Lussana et al., 2021 : https://npg.copernicus.org/articles/28/61/2021/

        # TODO : utiliser une moyenne pondérée par le likelyhood des membres comment dans Atencia 2020 ?

        ensemble_mean = ensemble.mean('member').rr.data  # flatten is optionnal since 'outer' method already flattens a 2D array
#        M = np.nanmean(ensemble_mean)
#        P = np.outer(ensemble_mean-M, ensemble_mean-M)/len(ensemble_mean)
        #P = np.empty((len(ensemble_mean), len(ensemble_mean)))

        #B = csc_matrix(np.shape(self.pond))
        B = dia_matrix(np.shape(self.pond))
# To add a static background error :
        #X = 2*scipy.sparse.eye(self.pond.shape[0])
        #B = B + X.dot(self.pond.dot(X))
        X = diags([5]*self.pond.shape[0], 0)
        B = B + X.dot(self.pond.dot(X))

        for mb in ensemble.member.data:
            member = ensemble.sel(member=mb).rr.data
            #B = B + (member-ensemble_mean)**2
            #B = B + np.diag((member-ensemble_mean)**2)
            #scipy.linalg.cholesky(self.pond.toarray())
            X = diags((member-ensemble_mean).flatten(), 0)
            B = B + X.dot(self.pond.dot(X))
            #B = B + self.pond.multiply(np.outer(member-ensemble_mean, member-ensemble_mean))  # elementwive multiplication
#        B = B/len(ensemble.member)**2  # TODO check denominator
        B = B/(len(ensemble.member)-1)

        B.data = np.nan_to_num(B.data, copy=False)
        #B = np.diag(B)
#        B = np.sqrt(B)
        #B.reshape(len(ensemble.lat), len(ensemble.lon))  # To reshape back as 2D field

#        M = ensemble.mean('member').rr.data
#        B = np.empty(np.shape(M))
#        for mb in ensemble.member.data:
#            member = ensemble.sel(member=mb).rr.data
#            B = B + (member-M)*(member-M)
#        B = np.sqrt(B)

        return B

    @speedtest
    def background_error_covariance_new(self, ensemble, obs, stdobs, replacement_strategy='max_weight'):
        """
        But : calculer les statistiques de la distribution d'ébauche.


        dispersion d'un "super ensemble" dont le poid de chaque memebre
        est mondéré par la matrice "self.pond".

        Méthode :
        ---------

        1. Calcul d'une distribution de probabilité pour chaque membre avec les valeurs
        du voisinage (pondération avec 'self.pond')
            --> mu(i), sd(i)
        Remplacement de la valeur d'ébauche par mu(i) (ce qui permet de mettre des précipitations
        sur un membre sans précipitations)

        2. Calcul de la dispersion finale du nouvel ensemble de 16 valeurs mu(i)
            --> B= 1/N * sum((mu(i)-mu*)(mu(i)-mu*)')
        TODO : vérifier si le calcul est bien équivalent au calcul de la dispersion avec
        le super-super-ensemble mélangeant tous les membres et tous les pixels !

        Identification des pixels à prendre en compte par la matrice 'super_ensemble'
        """

        super_ensemble = self.pond.copy()
        super_ensemble[super_ensemble.nonzero()] = 1  # Position of pixels to inclue in the spread computation = L

        new_ensemble = xr.DataArray(
            name   = 'rr',
            dims   = ["member", "lat", "lon"],
            coords = dict(lon=ensemble.lon, lat=ensemble.lat, member=ensemble.member),
        )

        # 1. calcul de la distribution locale de chaque membre
        std = dict()
        for mb in ensemble.member.data:
            member = ensemble.sel(member=mb).rr.data
            X = member.flatten()

            # Goal : compute the likelyhood of each pixel of the super-ensemble
            # The final weight of a pixel is the product of distance weight and the likelyhood

            # Computation of the likelyhood of each pixel of the super-ensemble
            O = super_ensemble.dot(diags(obs.data.flatten()))  # Observation
            M = super_ensemble.dot(diags(X))  # Backgound member
            S = super_ensemble.dot(diags(1/stdobs.diagonal()))
            A = (M-O).multiply(S)
            likelyhood = -A.multiply(A)
            np.exp(likelyhood.data, out=likelyhood.data)  # likelyhood = exp(-((O-M).S)**2)

#            X = super_ensemble.dot(diags(member.flatten(), 0))-se_mean   # M.diag(x)-diag(e).M
            # TODO : revoir la pondération pour assurer que une erreur statique importante a un poids moins élevé qu'un point très loin avec une faible erreur statique
            # TODO : pondérer les poids avecle likelyhood de l'obs
            pond = self.pond.multiply(likelyhood)  # likelyhood ponderation

            weight = pond.sum(axis=1).getA1()  # The sum of the weights (axis=1 <==> sum over rows)

            # TODO : remplacer la valeur initial de chaque pixel par la valeur du super-ensemble avec le poids (~likelyhood) le plus élevé
            #replacement_strategy = 'mean'  # !! TODO : TMP !!
            #replacement_strategy = 'toward_mean'!! TMP !!
            #mean, std[mb] = self.get_parameters(member, pond, weight=weight, super_ensemble=super_ensemble, replacement_strategy=replacement_strategy)
            #mean, std[mb] = self.get_parameters(member, pond, weight=weight, super_ensemble=super_ensemble, replacement_strategy='mean')  # Standard WMA --> modify backgound
            mean, std[mb] = self.get_parameters(member, pond, weight=weight, super_ensemble=super_ensemble, replacement_strategy='keep')  # Do not modify Background !

            # To avoid to smooth member but allow precipitation on pixels originaly without precipitation
            # --> Now useless since it is considered in the mean computation in the 'get_parameters' method
            # The pixel value can be slightly modified but since it has the highest weight it remains close from
            # the original value but can become >0 if the original value is 0 (exactly what we want !)

            # Plot data
            if self.plot:
                #point = 2059  #max obs 20220110
                #point = 1988  #max std 20220110
                #point = 887 #max obs 20210825
                #point = 78 # min obs 20211230
                #point = 2065 # max obs 20211230
                point = 1200 # To match the illustrastion of the localization area
                self.plot_super_ensemble(point, X, mean[point], std[mb][point], pond, f'Background ({mb})')

            # !! WARNING : modification des champs !!
            # Choisir entre les 3 solutions suivantes :

            if replacement_strategy == 'max_weight':
                # 1. take all new values (weighted average between the original value and the average of
                # the local super-ensemble weighted by the average weight
                # --> cela a tendance à lisser le champs en diminuant/augmenatant les valeurs extremes !!
                new_ensemble.loc[{'member':mb}] = mean.reshape(len(ensemble.lat), len(ensemble.lon))
            elif replacement_strategy == 'keep':
                # 2. Keep the original (debiased) field
                new_ensemble.loc[{'member':mb}] = member
            elif replacement_strategy == 'only_zeros':
                # 3. Change only values when the original value is 0 and the local weighted average is >0
                mean[np.where(X > 0)] = X[np.where(X > 0)]
                new_ensemble.loc[{'member':mb}] = mean.reshape(len(ensemble.lat), len(ensemble.lon))
            elif replacement_strategy == 'mean':
                # 4. Change values by the mean local average weighted by the observation likelyhood
                new_ensemble.loc[{'member':mb}] = mean.reshape(len(ensemble.lat), len(ensemble.lon))

        # 2. Calcul de la dispersion
        # TODO : Choisir la méthode de calcul de la dispersion
        # a) prise en compte de tous les pixels de tous les membres (super-super-ensemble) :
        # a.1 autour de la moyenne de l'ensemble initial ?
        # a.2 autour de la moyenne de l'ensemble modifié ?
        # a.3 autour de la moyenne pondérée ?
        #
        # b) calcul de la disperion des nouveaux membres 
        # 1. sans pondération par la dispersion locale
        # 2. avec pondération par la dispersion locale de chaque membre ?
        #
        # c) Somme des dispersions individuelles (moyenne des dispersions)

        ensemble_mean = new_ensemble.mean('member').data.flatten()  # mean of the modified ensemble
        initial_mean = ensemble.mean('member').rr.data.flatten()  # Mean of the original ensemble

        B = np.zeros(len(ensemble.lat)*len(ensemble.lon))
        B = diags(B, 0)
        for mb in ensemble.member.data:
            initial_member = ensemble.sel(member=mb).rr.data
            initial_data = diags(initial_member.flatten())
            member = new_ensemble.sel(member=mb).data
            data = diags(member.flatten())

            # TODO : il faut quand même diviser par la somme des poids à la fin avec les méthodes a1 et a2 !!
            # Methode a.1
            #sd = self.get_std(initial_data, ensemble.mean('member').rr.data.flatten(), pond, weight=weight, super_ensemble=super_ensemble)
            # Methode a.2
            #sd = self.get_std(initial_data, ensemble_mean, pond, weight=weight, super_ensemble=super_ensemble)

            # Methode b.1
            #sd = (member.flatten() - ensemble_mean)**2
            sd = (initial_member.flatten() - initial_mean)**2
            # Methode b.2  TODO : implémenter la pondération par la dispersion locale de chaque memebre
            #sd = (data - ensemble_mean)**2

            # Methode c
            # Si cette méthode est retenue, cette deuxième boucle sur les membres est inutile !
            # ==> sous dispersif
            #sd = std[mb]

            B = B + diags(sd, 0)

#            X = super_ensemble.dot(diags(member.flatten(), 0))-se_mean   # M.diag(x)-diag(e).M
#            B = B + X.multiply(X).multiply(pond).sum(axis=1).getA1()  # X*XT.Pond  (Attention à l'ordre des opérations !)

        B = np.sqrt(B/16)
        #B.data = np.nan_to_num(B.data, copy=False)
#        B = np.nan_to_num(B)
#        B = diags(B, 0)

        # TODO : plot the distribution of the super-super-ensemble and the associated Gaussian
        # TODO : construire explicitement le super ensemble pour le plotter
        # TODO : plotter le super-ensemble de chaque membre lors des itérations
        #self.plot_super_ensemble(point, np.zeros(np.shape(ensemble_mean)), ensemble_mean[point], B.diagonal()[point], pond, f'super_ensemble')

        ensemble = ensemble.rename({'rr':'raw'})
        ensemble = ensemble.update({'rr':new_ensemble})

        return B, ensemble

    @speedtest
    def ensemble_dispersion(self, ensemble):
        """
        D = sqrt(sum((Xi-Xmean)(Xi-Xmean)')/(N-1))
        """
        members = ensemble.member.data
        N = len(ensemble.member)
        M = self.ensemble_mean(ensemble)
        D = np.zeros(np.shape(M))
        for mb in members:
            member = ensemble.sel(member=mb).data
            D = D + (member-M)**2
        D = np.sqrt(D / (N-1))

        return D

    @speedtest
    def ensemble_mean(self, ensemble):
        """
        M = sum(Xi)/N
        """
        M = ensemble.mean('member').data

        return M

#    @speedtest
    def observation_ECM_new(self, parameters, date, plot=None):

        initial_obs = parameters.rr.data.flatten()
        std = np.abs(parameters.sigma.data)
        Rstat = diags(std.flatten())
        #pond = self.pond.dot(diags(np.exp(-std).flatten(), 0))  # Pondération par la distance et l'erreur statique !! ATTENTION A L'ORDRE !!
        #pond = self.pond.dot(diags(1/std.flatten(), 0))  # std>1 par construction
        uncertainty = std + parameters.error.data
        pond = self.pond.dot(diags(1/uncertainty.flatten(), 0))  # WARNING : error NOT >1 par construction
        #pond = self.pond.dot(diags(1/(std.flatten()*(1+parameters.error.data.flatten())), 0))  # WARNING : error NOT >1 par construction
        #pond = self.pond.dot(diags(np.exp(-parameters.error.data).flatten(), 0))  # WARNING : error NOT >1 par construction
        #pond = self.pond.dot(diags(1/parameters.error.data.flatten(), 0))

        #obs = parameters.mu.data.flatten()  # Climatological de-biasing
        obs = parameters.db.data.flatten()  # Dynamic de-biasing
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO : TMP (to see the perf of WMA correction only)
        #obs = parameters.rr.data.flatten()
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # read AROME mean vertical gradient (TEST !)
        #fic_gradient = os.path.join('/home/vernaym/workdir/ASSIMILATION/mask/alp', f'arome_gradient.nc')
        fic_gradient = os.path.join('/home/vernaym/These/DATA', f'CUMUL_AROME.nc')
        gradient = xr.open_dataarray(fic_gradient)
        gradient.data = uniform_filter(gradient.data, 10)
        gradient = gradient.sel({'lat':np.intersect1d(parameters.lat, gradient.lat), 'lon':np.intersect1d(parameters.lon, gradient.lon)})

        newfield, mean, sd = Preprocessing_ANTILOPE.dynamic_correction(obs, pond)
        #newfield, mean, sd = Preprocessing_ANTILOPE.dynamic_correction(obs, pond, gradient=gradient.data.flatten(), uncertainty=uncertainty)  # Use AROME mean vertical gradient
        #newfield, mean, sd = Preprocessing_ANTILOPE.dynamic_correction(obs, pond, qq_adjustment=True)  # qq adjustment add >0 bias !

        #Rdyn = diags(sd, 0)
        new_obs = xr.DataArray(
            data   = newfield.reshape((len(parameters.lat), len(parameters.lon))),
            name   = 'obs',
            dims   = ["lat", "lon"],
            coords = dict(lon=parameters.lon, lat=parameters.lat)
        )

        Rstat = diags(sd, 0)  # WARNING : variable name not adapted anymore
        #Rdyn = diags(np.sqrt(sd*np.abs(new_obs.data - parameters.db.data).flatten()), 0)
        #error = parameters.error.data
        #error = uniform_filter(error, 10)
        #Rdyn = diags(error.flatten(), 0)
        error = np.abs(new_obs.data - parameters.mu.data)
        #error = np.abs(new_obs.data - parameters.rr.data)  # TODO : try this error formulation
        error = uniform_filter(error, 5)  # TODO : try without error smoothing
        Rdyn = diags(error.flatten(), 0)
        R = dia_matrix(Rdyn+Rstat)

        # Plot data
        if plot is not None:
            #point = 2059  #max obs 20220110
            #point = 1988  #max std 20220110
            #point = 887 #max obs 20210825
            point = 2065
            point = 1200 # To match the illustrastion of the localization area
            #point = 78 # min obs 20211230
            #point = np.where(obs==np.nanmin(obs))[0][0]
            #point = np.where(obs==np.nanmax(obs))[0][0]
            num_poste = plot['num_poste']
            date = plot['date']
            nivometeo = read_nivometeo_obs()
            if np.datetime64(date) in nivometeo.date:
                ref = nivometeo.loc[{'num_poste':num_poste, 'date':np.datetime64(date)}].obs.data
                if not np.isnan(ref):
                    lat,lon = np.meshgrid(parameters.lat, parameters.lon)
                    point = np.where((lat.flatten()==plot['lat']) & (lon.flatten()==plot['lon']))[0][0]
                    #self.plot_super_ensemble(point, obs, mean[point], sd[point], pond, f'Observation_{num_poste}_{date}', reference=ref, initial_obs=initial_obs)
                    self.plot_super_ensemble(point, obs, newfield, mean[point], R.diagonal()[point], pond, f'Observation_{num_poste}_{date}', reference=ref, initial_obs=initial_obs)

        return R, Rstat, Rdyn, new_obs

    def get_parameters(self, field, pond, weight=None, super_ensemble=None, replacement_strategy='keep'):

        field[np.isnan(field)] = 0.0
        initial_field = field.flatten()
        X = diags(field.flatten(), 0)

        # 1. Calcul de la moyenne pondérée par la distance ET l'erreur statique
        if weight is None:
            pond.data[np.isnan(pond.data)] = 0.0
            weight = pond.sum(axis=1).getA1()  # The sum of the weights (axis=1 <==> sum over rows)

        mean = pond.dot(X).sum(axis=1).getA1()  # getA1 transforms the 1*N matrix object into a 1D np.array
        mean = mean / weight

        # Transformation of the observation field :
        # The goal is to update a value toward the weighted average of the neighborhood in proportion of its weight
        # and the average weight in the neighborhood.
        # The goal is to ensure that :
        # - only pixels with a significantly lower weight than the pixels around are updated (since the pixel weight is
        # not penalised by the distance weighting
        # - the standard deviation around this value is higher than the one around the super-ensemble weighted mean,
        # especially if the original pixel value stands far from the mass accumulation of the super ensemble.
        # Background fields are (almost) not impacted since the weights only take into account the distance, the target pixel
        # always has the highest weight ==> not true : small precipitation cores and precpitation extrems are smmothed
        #
        # TODO : trouver une solution pour changer la valeur de l'obs quand la moyenne de la distribution est plus
        # pertinente, sans trop affecter les valeurs extrêmes (seuil sur l'erreur statique ?)

        # 2. Calcul de la dispersion et mise à jour des champs
        # X is the original field as diagonal matrix)
        if replacement_strategy == 'keep':  # Default behavior (for observation)
            newfield = initial_field
            # 2.1 Computation of the dispersion around the original value (--> increase Rdyn !)
            sd = self.get_std(X, initial_field, pond, weight=weight, super_ensemble=super_ensemble)
        elif replacement_strategy == 'toward_mean':
            pixel_weight = pond.diagonal()  # = exp(-erreur_statique) pour l'obs et =likelyhood du pixel pour les membres de l'ensemble
            sums = pond.sum(axis=1).A1
            # To weight against the average weight in the neighborhood
            nb_nonzero = (pond != 0).sum(0).getA1()  # Count non zero elements of each row
            meanweight = sums / nb_nonzero
            newfield = (initial_field * pixel_weight + mean * meanweight) / (pixel_weight + meanweight)
            #newfield = (initial_field * pixel_weight + mean * (1-pixel_weight))  # Plus impactant à priori !
            # 2.2 Computation of the dispersion around the local ensemble mean value
            ######## TODO : TMP (to increase dynamic error)  ########
            #nopond = pond.copy()
            #nopond[nopond.nonzero()] = 1  # Compute dispersion without ponderartion to increase the error
            #weight = nopond.sum(axis=1).getA1()  # The sum of the weights (axis=1 <==> sum over rows)
            ##########################################################
            # TODO : calculer l'erreur sur l'observation itnitiale (avant même débiaisage) pour éviter une sous-estimation de
            # l'erreur due au lissage du champ par les différentes méthodes de correction
            #sd = self.get_std(X, mean, nopond, weight=weight, super_ensemble=super_ensemble)
            sd = self.get_std(X, newfield, pond, weight=weight, super_ensemble=super_ensemble)
        elif replacement_strategy == 'max_weight':  # To pull background members toward the observation
            # Get maximum weight of each line of the pond matrix
            idx = pond.argmax(axis=1).A1  # get the index of the maximum value of each line
            newfield = initial_field[idx]
            # 2.3 Computation of the dispersion around the new mean mean value
            sd = self.get_std(X, newfield, pond, weight=weight, super_ensemble=super_ensemble)
        elif replacement_strategy == 'mean':  # To replace background value by the average weighted by observation likelyhood
            newfield = mean
            sd = self.get_std(X, mean, pond, weight=weight, super_ensemble=super_ensemble)

        sd = sd + np.abs(initial_field-newfield)  # Add displacement to error (--> increase error)  !! WARNING : check for double penalty !!
        # TODO : Static error must be reduced in cases where all precipitation are 0mm
#        sd = sd + np.sqrt(0.263 * np.square(newfield))  # Add error proportionnal to precipitation intensity

        return newfield, sd

    def get_std(self, data, mean, pond, weight=None, super_ensemble=None):

        if weight is None:
            weight = pond.sum(axis=1).getA1()  # The sum of the weights (axis=1 <==> sum over rows)
        if super_ensemble is None:
            super_ensemble = pond.copy()  # WARNING : make a copy or pond will change when super_ensemble changes
            super_ensemble[super_ensemble.nonzero()] = 1  # Position of pixels to inclue in the spread computation

        se_mean = diags(mean, 0).dot(super_ensemble)  # matrix with mean[i] at each non-zero element of line i of super_ensemble
        X = super_ensemble.dot(data)-se_mean  # # M.diag(obs)-diag(e).M
        sd = X.multiply(X).multiply(pond).sum(axis=1).getA1()
        sd = sd / weight
        sd = np.sqrt(sd)
        sd = np.nan_to_num(sd)

        return sd

    def ref_field(self, field, date):

        smoothobs = uniform_filter(self.parameters.sel({'time':date}).mu, size=20)  # numpy array
        #smoothobs = uniform_filter(parameters.sel({'time':date}).mu, size=10)  # numpy array
        smoothobs = xr.DataArray(
            name   = 'rr',
            data   = smoothobs,
            dims   = ["lat", "lon"],
            coords = dict(lon=self.parameters.lon, lat=self.parameters.lat),
        )
        smoothobs = smoothobs.sel({'lat':field.lat, 'lon':field.lon})

        if self.plot:
            self.plot_array(smoothobs, smoothobs, 'Smoothed observation', f'{self.date_str}/Smooth_obs.pdf', cmap=plt.cm.YlGnBu)

        ref_field = field.data - smoothobs

        return ref_field

    def codistances(self, coords):
        """
        Solution pour le calcul des inter-distances trouvée sur : https://stackoverflow.com/questions/35296935/python-calculate-lots-of-distances-quickly
        """
        tree = cKDTree(coords)
        dist = tree.sparse_distance_matrix(tree, max_distance=self.max_dist, p=2, output_type='coo_matrix')
        dist = csr_matrix(dist)
        #TODO : utiliser une gaussienne plutot qu'une exponentielle décroissante
        dist[dist.nonzero()] = -dist[dist.nonzero()]/self.ld
        np.exp(dist.data, out=dist.data )
        return dist

    def plot_matrix(self, matrix, ref_field, label, outname, cmap=plt.cm.YlGnBu, vmin=None, vmax=None, domain=None):
        if domain is None:
            domain = self.domain
        #diag = np.array([matrix[i,i] for i in range(len(matrix))])
        diag = matrix.diagonal()
        field = xr.DataArray(
                name   = 'rr',
                data   = diag.reshape((len(ref_field.lat), len(ref_field.lon))),
                dims   = ["lat", "lon"],
                coords = dict(lon=ref_field.lon, lat=ref_field.lat),
                )

        # Reduce the data to the actual domain (remove the potential correlation length edge)
        latmin = domain_coords[domain]['latmin']
        latmax = domain_coords[domain]['latmax']
        lonmin = domain_coords[domain]['lonmin']
        lonmax = domain_coords[domain]['lonmax']
        sel_lat = np.round(np.arange(latmin, latmax, 0.01), 2)
        sel_lon = np.round(np.arange(lonmin, lonmax, 0.01), 2)

        # Add correlation area (TMP)
        #point = 0
        #corr = xr.DataArray(
        #        name   = 'correlation',
        #        data   = self.pond.getrow(point).toarray()[0].reshape((len(field.lat), len(field.lon))),
        #        dims   = ["lat", "lon"],
        #        coords = dict(lon=field.lon, lat=field.lat),
        #    )
        #corr = corr.sel({'lat':np.intersect1d(sel_lat, corr.lat.data), 'lon':np.intersect1d(sel_lon, corr.lon.data)})

        field = field.sel({'lat':np.intersect1d(sel_lat, field.lat.data), 'lon':np.intersect1d(sel_lon, field.lon.data)})

        fig,ax = plt.subplots(figsize=figsize[domain]['singleplot'])
        #fig,ax = plt.subplots(figsize=(10,12))  #Alps
        if vmin is None:
            vmin = np.nanmin(field)
        if vmax is None:
            vmax = np.nanmax(field)
        im = plot_field(field, ax, vmin, vmax, self.domain, cmap=cmap)

        # Add correlation area (TMP)
        #ax = plot_correlation(ax, field, corr, point=1200)

        add_boundaries(ax)
        finalize_fig(fig, im, label=label, outname=outname)

    def plot_array(self, array, ref_field, label, outname, cmap=plt.cm.YlGnBu, vmin=None, vmax=None, domain=None, add_landmarks=True, text1=None, text2=None):
        if domain is None:
            domain = self.domain
        field = xr.DataArray(
                name   = 'rr',
                data   = array,
                dims   = ["lat", "lon"],
                coords = dict(lon=ref_field.lon, lat=ref_field.lat),
            )

        # Reduce the data to the actual domain (remove the potential correlation length edge)
        latmin = domain_coords[domain]['latmin']
        latmax = domain_coords[domain]['latmax']
        lonmin = domain_coords[domain]['lonmin']
        lonmax = domain_coords[domain]['lonmax']
        sel_lat = np.round(np.arange(latmin, latmax, 0.01), 2)
        sel_lon = np.round(np.arange(lonmin, lonmax, 0.01), 2)

        field = field.sel({'lat':np.intersect1d(sel_lat, field.lat.data), 'lon':np.intersect1d(sel_lon, field.lon.data)})

        fig,ax = plt.subplots(figsize=figsize[domain]['singleplot'])
        #fig,ax = plt.subplots(figsize=(14,16))  # Alps
        if vmin is None:
            vmin = np.nanmin(array)
        if vmax is None:
            vmax=np.nanmax(array)
        im = plot_field(field, ax, vmin, vmax, self.domain, cmap=cmap)

        if add_landmarks:
            # Add landmarks
            if domain == 'GrandesRousses':
                for landmark, infos in landmarks.items():
                    plt.plot(infos['lon'], infos['lat'], marker=infos['marker'], color='red', markersize=10)
                    plt.annotate(landmark, (infos['lon']+0.003, infos['lat']+0.003), color='red', fontsize=20)

            latmin = domain_coords[domain]['latmin']
            latmax = domain_coords[domain]['latmax']
            lonmin = domain_coords[domain]['lonmin']
            lonmax = domain_coords[domain]['lonmax']
            add_cities(latmin, latmax, lonmin, lonmax)
            add_boundaries(ax)

        if text1 is not None:
            #ax.scatter(bias.lon.data, bias.lat.data, c=bias.data)
            for lon, lat, text in text1:
                if lon>=lonmin and lon<=lonmax and lat>=latmin and lat<=latmax:
                    text = f'{text:.2f}'
                    plt.plot(lon, lat, marker='.', color='k', linestyle='', markersize=1)
                    ax.text(lon, lat, text, fontsize=14)
        if text2 is not None:
            #ax.scatter(bias.lon.data, bias.lat.data, c=bias.data)
            for lon, lat, text in text2:
                if lon>=lonmin and lon<=lonmax and lat>=latmin and lat<=latmax:
                    text = f'{text:.2f}'
                    plt.plot(lon, lat, marker='.', color='red', markersize=1)
                    ax.text(lon, lat, text, fontsize=14, color='red')

        finalize_fig(fig, im, label=label, outname=outname)

    @speedtest
    def output(self, outfield):

        i = 0
        j = 0
        #for m in range(1, self.Ne+1):
        for m in range(0, self.Ne+1):
            outfield.loc[{'member':m}] = self.newlocalfield[m]  # self.newlocalfield is a numpy array

        if self.plot:
            plt.close('all')

        return outfield


class RandomSampling(Assimilation):

    def __init__(self, period, obs, nivometeo, plot, frequency, gridded, localisation, mask, debiasing, domain, likelyhood, Ne=16):

        self.period = period
        self.frequency = frequency

        self.radar = obs
        self.plot = plot

        self.gridded = gridded
        self.nivometeo = nivometeo
        self.localisation = localisation
        self.mask = mask
        self.debiasing = debiasing
        self.domain = domain
        self.likelyhood = likelyhood
        self.Ne = Ne  # Ouptut ensemble size

        # Parameters to compute Euclidian distance between all points in the domain
        self.ld = ld
        self.max_dist = max_dist

        self.arome_clim = None
        self.read_obs_auto()

    def read_obs_auto(self):
        datadir = '/home/vernaym/These/DATA'
        fic_score = os.path.join(datadir, f'obs_quotidienne_auto_RR_20211101_20220430.csv')
        obs_auto = pd.read_csv(fic_score, sep=';', parse_dates=['date'], dtype={'num_poste':int, 'poste':str, 'lat':float, 'lon':float, 'alti':int, 'rr':float, 'reseau_poste':int})
        #obs_auto.rename(columns={'dat':'date'}, inplace=True)
        #obs_auto = obs_auto.set_index(['num_poste', 'date'])
        obs_auto = obs_auto.set_index(['num_poste'])

        #obs_auto["rr"] = obs_auto["rr"].round(1)

        self.obs_auto = obs_auto

    def dynamic_error_estimation(self, parameters, obs_auto, var='mu', delta=0.1):
        """
        """

        if self.arome_clim is None:
            arome_clim = xr.open_dataset(os.path.join(datadir, 'CUMUL_AROME.nc'))
            self.arome_clim = arome_clim.sel(lat=parameters.lat, lon=parameters.lon)

        # 1. Select automatic stations
        obs_auto = Preprocessing_ANTILOPE.filter_gauges(obs_auto, parameters[var], delta=delta)

        new_ratio, new_error = Preprocessing_ANTILOPE.dynamic_error_estimation(parameters[var], obs_auto, self.arome_clim, delta=delta)

        return new_ratio, new_error, obs_auto

    @speedtest
    def run(self):
        """ 
        Main method that loop over the assimilation dates and grid points.
        TODO : compléter la doc sur la méthode
        """

        domain = self.domain

        # Initialisation of output fields
        if self.gridded:
            self.nlon, self.nlat = len(self.radar.lon), len(self.radar.lat)
            null  = np.empty((self.nlat, self.nlon, len(self.period)))  # 2D (lat/lon) field
        else:
            self.nposte = len(self.nivometeo.num_poste)
            null = np.empty((self.nposte, len(self.period)))

        null[:] = np.nan
        self.newlocalfield = {m:null.copy() for m in range(0, self.Ne+1)}
        self.error = null.copy()  # Initialisation of error data
        #self.newlocalfield = {m:dict() for m in range(1, self.Ne+1)}  # Used only for ponctual assimilation

        actual_parameters = self.parameters
        #actual_parameters = np.round(self.parameters, 1)

        codistances = os.path.join('/home/vernaym/These/DATA', f'codistance_max_dist_{self.max_dist}_{domain}.npz')
#            if not os.path.exists(codistances):
#                # Compute inter-distances
#                coords=[(lon,lat) for lat in actual_parameters.lat.data for lon in actual_parameters.lon.data]
#                self.pond = self.codistances(coords)
#            else:
#                self.pond = scipy.sparse.load_npz(codistances)
        coords=[(lon,lat) for lat in actual_parameters.lat.data for lon in actual_parameters.lon.data]
        #self.pond = self.codistances(coords)
        self.pond = Preprocessing_ANTILOPE.codistances(coords, self.domain)

        for idd, date in enumerate(self.period):
            print(date)
            self.date_str = date.strftime('%Y%m%d%H')
            if self.plot:
                if not os.path.exists(self.date_str):
                    os.makedirs(self.date_str)

            parameters = actual_parameters.sel({'time':date}).compute()

            obs_auto = self.obs_auto[self.obs_auto.date==date]  # Select date
            var = 'mu'
            delta = 1
            rat, err, obs_auto = self.dynamic_error_estimation(parameters, obs_auto, var=var, delta=delta)

            if self.plot:
                # Reduce the data to the actual domain (remove the potential correlation length edge)
                latmin = domain_coords[domain]['latmin']
                latmax = domain_coords[domain]['latmax']
                lonmin = domain_coords[domain]['lonmin']
                lonmax = domain_coords[domain]['lonmax']
                sel_lat = np.round(np.arange(latmin, latmax, 0.01), 2)
                sel_lon = np.round(np.arange(lonmin, lonmax, 0.01), 2)
                #sel_lat = rat.lat.data  # TODO : TMP !!!
                #sel_lon = rat.lon.data  # TODO : TMP !!!

                self.plot_array(rat.data, rat, 'ratio', f'{self.date_str}/Ratio_{self.domain}.pdf', cmap=palettable.colorbrewer.diverging.RdBu_7_r.mpl_colormap, vmin=0.2, vmax=1.8, domain=self.domain)
                self.plot_array(err.data, err, 'error (mm)', f'{self.date_str}/Error_dyn_{self.domain}.pdf', cmap=plt.cm.YlOrBr, domain=self.domain, vmin=0)
                self.plot_array(err.data+parameters.sigma, err, 'error (mm)', f'{self.date_str}/Error_{self.domain}.pdf', cmap=plt.cm.YlOrBr, domain=self.domain, vmin=0)
                #fig, ax = plt.subplots(figsize=figsize[self.domain]['singleplot'])
                #tmp = rat.sel({'lat':np.intersect1d(sel_lat, rat.lat.data), 'lon':np.intersect1d(sel_lon, rat.lon.data)})
                #im = plot_field(tmp, ax, 0.5, 1.5, self.domain, cmap=palettable.colorbrewer.diverging.RdBu_7_r.mpl_colormap)
                #im = make_mask.plot_field(fig, ax, tmp, cmap=palettable.colorbrewer.diverging.RdBu_7_r.mpl_colormap, vmin=0.5, vmax=1.5, scores=obs_auto)
                #fig.savefig(f'{self.date_str}/Ratio_{self.domain}.pdf', format='pdf')
                #fig, ax = plt.subplots(figsize=figsize[self.domain]['singleplot'])
                #tmp = err.sel({'lat':np.intersect1d(sel_lat, err.lat.data), 'lon':np.intersect1d(sel_lon, err.lon.data)})
                #im = make_mask.plot_field(fig, ax, tmp, cmap=plt.cm.YlOrBr, scores=obs_auto)
                #im = plot_field(tmp, ax, 0, 20, self.domain, cmap=plt.cm.YlOrBr)
                #fig.savefig(f'{self.date_str}/Error_{self.domain}.pdf', format='pdf')
                #plt.close('all')

            # Fill parameters Dataset with dynamic fields
            parameters['error'] = err
            parameters['ratio'] = rat
            parameters['db'] = (parameters[var]+delta) / parameters['ratio'] - delta  # Add delta to introduce precipitation in "missed precipitation" pixels
            #parameters['db'] = parameters[var] / parameters['ratio']
            mask = parameters[var].data > 0
            parameters['db'].data[mask] = parameters[var].data[mask] / parameters['ratio'].data[mask]

            ####################  TMP  #####################
            # Plot distributions before / after conversion
#            if self.plot:
#                fig, ax = plt.subplots()
#                R1 = parameters.mu.data.flatten()
#                R2 = np.sqrt(R1)
#                ax.hist(R2, density=True, bins=np.arange(np.floor(np.nanmin(R2))-0.1, np.ceil(np.nanmax(R2)) + 0.1, 0.1), label='R* (mm^1/2)', alpha=0.5)
#                mu = np.nanmean(R2)
#                sd = np.sum((R2-mu)**2)/len(R2)
#                ax = plot_distribution(ax, mu, sd, color='blue')
#                ax.hist(R1, density=True, bins=np.arange(np.floor(np.nanmin(R1))-0.1, np.ceil(np.nanmax(R1)) + 0.1, 0.1), label='R (mm)', alpha=0.5)
#                ax.legend()
#                ax.set_xlim(right=13)
#                if not os.path.exists(f'{self.date_str}/distributions'):
#                    os.makedirs(f'{self.date_str}/distributions')
#                fig.savefig(f'{self.date_str}/distributions/conversion_rr.pdf')
#                plt.close(fig)
            ####################  END  #####################

            # Change variable R --> R^(1/2) to bring the distributions closer to a Normal one
            #parameters.mu.data = np.sqrt(parameters.mu.data)
            #parameters.rr.data = np.sqrt(parameters.rr.data)

            if self.gridded:
                self.gridded_random_draw(date, idd, parameters, domain, obs_auto=obs_auto)
            else:
                self.ponctual_random_draw(date, idd, parameters)

    def save_corrected_field(self, extract_period, num_poste):
        corrected_field = self.newlocalfield[0]  # self.newlocalfield is a numpy array
        error           = self.error
#        if args.gridded:
#            out = xr.DataArray(
#                    name   = 'rr',
#                    data   = corrected_field,
#                    dims   = ["lat", "lon", "time"],
#                    coords = dict(lon=lon, lat=lat, time=extract_period),
#                    attrs  = dict(description="24 hour precipitation",units="mm"),
#                )
#        else:
        out = xr.Dataset(
                data_vars = dict(
                    rr    = (["num_poste","date"], corrected_field),
                    error = (["num_poste","date"], error),
                ),
                coords    = dict(
                    num_poste = (["num_poste"], num_poste),
                    date      = (["date"], extract_period),
                ),
                attrs     = dict(
                    description="Corrected ANTILOPE"
                ),
            )
        outname = f"ANTILOPEQ_{args.datebegin.strftime('%Y%m%d%H')}_{args.dateend.strftime('%Y%m%d%H')}_{args.domain}_corrected"
        # WARNING : encode(utf-8) nécessaire si outname contient un entier formatté en string
        out.to_netcdf(f"{outname}.nc")
        #out.to_netcdf(os.path.join('/home/vernaym/These/DATA', f"{outname}.nc"))

    @speedtest
    def gridded_random_draw(self, date, idd, parameters, domain, nmembers=16, obs_auto=None):

        # TODO : Add plots of various fields

        self.rrmin = 0.
        self.rrmax = min(80, max(
                #np.nanmax(np.square(parameters.obs.data)),
                #np.nanmax(np.square(parameters.rr.data)),
                #np.nanmax(np.square(parameters.mu.data)),
                np.nanmax(parameters.rr.data),
                np.nanmax(parameters.mu.data),
                ))

        R, Rstat, Rdyn, updated_obs = self.observation_ECM_new(parameters, date)
        Y = updated_obs.data  # Observation vector. WARNING : Use mu to take debiasing into account !

        analysis = xr.DataArray(
            name   = 'rr',
            dims   = ["member", "lat", "lon"],
            coords = dict(lon=parameters.lon, lat=parameters.lat, member=range(0, nmembers+1)),
        )
        obs = Y.reshape((len(parameters.lat), len(parameters.lon)))  # Get observation field
        obs = np.round(obs, 1)  # Round precipitation <0.1 at 0 (different distribution used in this case) TODO : convertir dans l'espace r^1/2

        # Fill first member with corrected observation
        analysis.loc[{'member':0}] = obs
        #analysis.loc[{'member':0}] = np.square(obs)

        # Extract reference points and corresponding values
        nivometeo = self.nivometeo.sel({'date':date}).dropna(dim='num_poste').drop('date')

        df = nivometeo.to_dataframe().rename(columns={'nom':'poste', 'obs':'rr'})
        obs_auto.drop(columns=['date', 'reseau_poste'], inplace=True)
        allobs = pd.concat([obs_auto, df])
        allobs = allobs[~np.isnan(allobs.rr)]

        # Get data over evaluation points and compute errors
        evaluation_points = analysis.sel(member=0, lat=xr.DataArray(allobs.lat.values, dims="poste"), lon=xr.DataArray(allobs.lon.values, dims="poste"), method='nearest')
        bias  = evaluation_points - allobs.rr.values
        ratio = (evaluation_points+0.01) / (allobs.rr.values+0.01)
        # Kriging of reference values to get a reference field
        kriging = False
        #if len(nivometeo.num_poste) > 1:
        if len(allobs) > 1 and kriging:
            kriging = True
            y    = allobs.lat.values
            x    = allobs.lon.values
            rr   = allobs.rr.values
            kriging = UniversalKriging(x, y, rr, variogram_model='exponential')
            rr_ref, ss = kriging.execute('grid', parameters.lon.data, parameters.lat.data)
            reference_field = xr.DataArray(
                name   = 'reference',
                data   = rr_ref,
                dims   = ["lat", "lon"],
                coords = dict(lon=parameters.lon, lat=parameters.lat),
            )

        sd1 = Rstat.diagonal().reshape((len(parameters.lat), len(parameters.lon)))  # Get standard deviation field
        #sd1 = uniform_filter(sd1, 3)
        sd2 = Rdyn.diagonal().reshape((len(parameters.lat), len(parameters.lon)))
        #sd2 = uniform_filter(sd2, 3)
        sd = R.diagonal().reshape((len(parameters.lat), len(parameters.lon)))  # Get standard deviation field
        #sd = sd1 + sd2
        # If no precipitation have been introduced by the WMA we are confident that there is actually no precipitation
        # Avoid small dispersion around 0mm and flatten the rank histogram
        sd[obs==0] = 0
        sd1[obs==0] = 0
        sd2[obs==0] = 0

        error = xr.DataArray(
            name   = 'error',
            #data   = np.square(sd),
            data   = sd,
            dims   = ["lat", "lon"],
            coords = dict(lon=parameters.lon, lat=parameters.lat),
        )

        if self.plot:
            text = zip(bias.lon.values, bias.lat.values, bias.values)
            #text = zip(ratio.lon.data, ratio.lat.data, ratio.data)
            self.plot_array(analysis.sel(member=0), parameters.rr, 'Corrected field', f'{self.date_str}/Corrected_field_{self.date_str}_{self.domain}.pdf', vmin=0, vmax=self.rrmax, cmap=plt.cm.YlGnBu, text1=text)
            self.plot_array(error, parameters.rr, 'Error (mm)', f'{self.date_str}/ERROR_{self.domain}.pdf', vmin=0, vmax=np.max(error), cmap=plt.cm.Reds, text1=text)
            #self.plot_array(reference_field, parameters.rr, 'Precipitation (mm)', f'{self.date_str}/Reference_{self.date_str}_{self.domain}.pdf', vmin=0, vmax=self.rrmax, cmap=plt.cm.YlGnBu, bias=nivometeo.obs)
            if kriging:
                #obs_auto = obs_auto[(obs_auto.lon<=np.max(parameters.lon.data)) & (obs_auto.lon>=np.min(parameters.lon.data)) & (obs_auto.lat<=np.max(parameters.lat.data)) & (obs_auto.lat>=np.min(parameters.lat.data))]
                text1 = zip(obs_auto.lon.values, obs_auto.lat.values, obs_auto.rr.values)
                text2 = zip(df.lon.values, df.lat.values, df.rr.values)  # nivometeo observations
                self.plot_array(reference_field, parameters, 'Precipitation (mm)', f'{self.date_str}/Reference_{self.date_str}_{self.domain}.pdf', vmin=0, vmax=self.rrmax, cmap=plt.cm.YlGnBu, text1=text1, text2=text2)

            npoints = len(evaluation_points)
            if npoints <=3:
                nrow = 1
                ncol = npoints
            elif npoints <= 8:
                nrow = 2
                ncol = math.ceil(npoints/2)
            elif npoints <= 12:
                nrow = 3
                ncol = math.ceil(npoints/3)
            elif npoints <= 16:
                nrow = 4
                ncol = 4
            elif npoints <= 21:
                nrow = 7
                ncol = 3
            elif npoints <= 24:
                nrow = 6
                ncol = 4
            elif npoints <= 28:
                nrow = 7
                ncol = 4
            elif npoints <= 32:
                nrow = 8
                ncol = 4
            elif npoints <= 35:
                nrow = 7
                ncol = 5
            else:
                nrow = 7
                ncol = 7
            #point = np.where(Y==np.nanmax(Y))  # max observation (plot only)
            # TODO : plot distributions for reference points and add reference
            #point = np.where(sd==np.nanmax(sd))  # max error (plot only)
            # plot distributions
            fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(10*ncol,10*nrow))
            if npoints == 1:
                ax = np.array([[ax]])
            elif npoints <=3:
                ax = np.array([ax])
            i = 0
            j = 0
            ymax = list()
            for poste in nivometeo.num_poste.data:
                ref = nivometeo.sel(num_poste=poste)
                lat = ref.lat.data
                lon = ref.lon.data
                reference = ref.obs.data
                ax[i,j].bar(reference, 1, width=0.3, label='Reference observation', color='Green', alpha=1)
                original_obs = parameters.sel(lat=lat, lon=lon, method='nearest').rr.data
                #original_obs = parameters.rr.data[point]
                ax[i,j].bar(original_obs, 1, width=0.3, label='Original observation', color='k', alpha=0.5)
                #obs = Y[point][0]
                new_obs = analysis.sel(lat=lat, lon=lon, member=0, method='nearest').data
                ax[i,j].bar(new_obs, 1, width=0.3, color='red', alpha=1, label='Corrected observation')
                #sd = sd[point][0]
                std = error.sel(lat=lat, lon=lon, method='nearest').data
                #ax = plot_distribution(ax, np.square(obs), np.square(sd), label=f'New observation distribution (sd={np.square(sd)})', color='red')
                #ax[i,j] = plot_distribution(ax[i,j], new_obs, std, label=f'New observation distribution (std={np.round(std, 2)})', color='red')  # sigma --> sigma² dans loi normale
                ymax.append(max(reference, original_obs, new_obs))
                j = j + 1
                if j==ncol:
                    j = 0
                    i = i + 1

            fig1,ax1 = plt.subplots(nrows=4, ncols=4, figsize=figsize[domain]['ensembleplot'])
            #fig2,ax2 = plt.subplots(nrows=2, ncols=8, figsize=(16,10))
            i = 0
            j = 0

        #std = np.abs(parameters.sigma.data)
        #pond = self.pond.dot(diags(1/std.flatten(), 0))

        for member in analysis.member.data:
            #ana = Preprocessing_ANTILOPE.random_draw(obs, sd, distribution='gamma')
            #ana = Preprocessing_ANTILOPE.random_draw(obs, sd1, distribution='gamma')
            ana = Preprocessing_ANTILOPE.random_draw(obs, sd1, sd2=sd2, distribution='gamma')
            #ana = Preprocessing_ANTILOPE.random_draw(obs, sd, distribution='normal')
            analysis.loc[{'member':member}] = ana

            self.newlocalfield[member][:,:,idd] = analysis.sel({'member':member}).data

            #print('!!!! WARNING : TMP !!!!')
            #ana, toto, tutu = Preprocessing_ANTILOPE.dynamic_correction(ana.flatten(), pond)
            #analysis.loc[{'member':member}] = ana.reshape((len(parameters.lat), len(parameters.lon)))


            if self.plot and member>0:
                # TODO : Add reference values
                im1 = plot_field(analysis.loc[{'member':member}], ax1[i,j], self.rrmin, self.rrmax, self.domain)
                #scatter(d['lons'], d['lats'], c=d[f'{score}'], cmap=cmap, norm=norm, marker=marker, s=150, edgecolors='black')
                ax1[i,j].set_title(None)
                j = j + 1
                if j==4:
                    j = 0
                    i = i + 1

        if self.plot:
            # Plot analysis distribution
            i = 0
            j = 0
            for poste in nivometeo.num_poste.data:
                ref = nivometeo.sel(num_poste=poste)
                lat = ref.lat.data
                lon = ref.lon.data
                #ens = analysis.isel(lat=point[0], lon=point[1]).data.flatten()
                ens = analysis.sel(lat=ref.lat.data, lon=ref.lon.data, method='nearest').data.flatten()
                mu = np.mean(ens)
                #std = np.sqrt(np.sum((ens-mu)**2)/16)
                std = np.sqrt(np.mean((ens-mu)**2))
                ax[i,j] = plot_distribution(ax[i,j], mu, std, ensemble=ens, label='Analysis', color='blue', linewidth=1)
                #ax = plot_distribution(ax, mu, np.square(sd[point]), label='Analysis theoretical PDF', color='k', linewidth=1)
                std = error.sel(lat=ref.lat.data, lon=ref.lon.data, method='nearest').data
                #ax[i,j] = plot_distribution(ax[i,j], mu, std, label='Analysis theoretical PDF', color='k', linewidth=1)
                ax[i,j].legend()
                #ax[i,j].set_xlim(left=0, right=np.max(ens)*2)
                ax[i,j].set_xlim(left=0, right=80)
                #ax[i,j].set_ylim(top=norm.pdf(mu, loc=mu, scale=std)*1.2)
                ax[i,j].set_ylim(top=0.06)
                ax[i,j].set_xlabel('Precipitation (mm/24h)')
                ymax[i+j] = max(ymax[i+j], np.max(ens))*1.1
                j = j + 1
                if j==ncol:
                    j = 0
                    i = i + 1
            if not os.path.exists(f'{self.date_str}/distributions'):
                os.makedirs(f'{self.date_str}/distributions')
            fig.savefig(f'{self.date_str}/distributions/DISTRIBUTION_ANALYSE_{domain}.pdf')
            plt.close(fig)

            #ECM_max = np.square(np.nanmax(R))
            #ECM_max = np.nanmax(R.toarray())
            ECM_max = max(np.nanmax(Rdyn.toarray()), np.nanmax(Rstat.toarray()))
            self.plot_matrix(R, parameters.rr, 'Observation_ECM', f'{self.date_str}/Observation_ECM_{domain}.pdf', vmin=0, vmax=ECM_max, cmap=plt.cm.Reds)
            self.plot_matrix(Rdyn, parameters.rr, 'Observation_ECM', f'{self.date_str}/Observation_dyn_ECM_{domain}.pdf', vmin=0, vmax=ECM_max, cmap=plt.cm.Reds)
            self.plot_matrix(Rstat, parameters.rr, 'Observation_ECM', f'{self.date_str}/Observation_stat_ECM_{domain}.pdf', vmin=0, vmax=ECM_max, cmap=plt.cm.Reds)

            #parameters.mu.data = np.square(parameters.mu.data)
            #parameters.rr.data = np.square(parameters.rr.data)

            evaluation_points = parameters.rr.sel(lat=xr.DataArray(allobs.lat.values, dims="poste"), lon=xr.DataArray(allobs.lon.values, dims="poste"), method='nearest')
            bias  = evaluation_points - allobs.rr.values
            text = zip(bias.lon.data, bias.lat.data, bias.data)  # Raw observation error
            #text = zip(ratio.lon.data, ratio.lat.data, ratio.data)  # Raw observation ratio
            self.plot_obs(parameters, domain=domain, text1=text)

            if self.debiasing:
                # Compute error against debiased field
                evaluation_points = parameters.mu.sel(lat=xr.DataArray(allobs.lat.values, dims="poste"), lon=xr.DataArray(allobs.lon.values, dims="poste"), method='nearest')
                bias  = evaluation_points - allobs.rr.values
                text = zip(bias.lon.data, bias.lat.data, bias.data)
                #text = zip(ratio.lon.data, ratio.lat.data, ratio.data)
                self.plot_obs(parameters, var='mu', domain=domain, text1=text)
                evaluation_points = parameters.db.sel(lat=xr.DataArray(allobs.lat.values, dims="poste"), lon=xr.DataArray(allobs.lon.values, dims="poste"), method='nearest')
                bias  = evaluation_points - allobs.rr.values
                text = zip(bias.lon.data, bias.lat.data, bias.data)
                self.plot_obs(parameters, var='db', domain=domain, text1=text)
                #self.plot_obs(parameters, var='obs', domain=domain)
                #parameters['diff'] = parameters.obs-parameters.mu
                #parameters['diff'] = parameters.obs-parameters.rr  # !! TODO : TMP !!
                #self.plot_obs(parameters, var='diff', domain=domain)

            mean = self.ensemble_mean(analysis)
            self.plot_array(mean, parameters.rr, 'Mean precipitation (mm)', f'{self.date_str}/Analysis_mean_{self.domain}.pdf', cmap=plt.cm.YlGnBu, vmin=self.rrmin, vmax=self.rrmax)

            disp = self.ensemble_dispersion(analysis)
            disp = xr.DataArray(
                name   = 'spread',
                #data   = np.square(sd),
                data   = disp,
                dims   = ["lat", "lon"],
                coords = dict(lon=parameters.lon, lat=parameters.lat),
            )
            evaluation_points = disp.sel(lat=xr.DataArray(allobs.lat.values, dims="poste"), lon=xr.DataArray(allobs.lon.values, dims="poste"), method='nearest')
            text = zip(evaluation_points.lon.data, evaluation_points.lat.data, evaluation_points.data)
            self.plot_array(disp, parameters.rr, 'Dispersion (mm)', f'{self.date_str}/Analysis_dispersion_{self.domain}.pdf', vmin=0, vmax=np.nanmax(disp), cmap=plt.cm.YlGnBu, text1=text)

            finalize_fig(fig1, im1, label='24-hour precipitation (mm)', outname=f'{self.date_str}/ANALYSIS_{self.date_str}_{self.domain}.pdf')

    @speedtest
    def ponctual_random_draw(self, date, idd, parameters, nmembers=16):

        evaluation_points = zip(self.nivometeo.num_poste.data, np.nanmax(self.nivometeo.lat, axis=1).data, np.nanmax(self.nivometeo.lon, axis=1).data)
        for idp, (num_poste, lat, lon) in enumerate(evaluation_points):
            print(num_poste, idp)
            # Extract rectangle around evaluation point
            nearest_lat = nearest(parameters.lat, lat)
            nearest_lon = nearest(parameters.lon, lon)
#            try:
            sel_lat = np.round(np.arange(nearest_lat-self.max_dist, nearest_lat+self.max_dist, 0.01), 2)
            sel_lon = np.round(np.arange(nearest_lon-self.max_dist, nearest_lon+self.max_dist, 0.01), 2)
            # Compute inter-distances

            parameters_loc = parameters.sel({'lat':np.intersect1d(sel_lat, parameters.lat), 'lon':np.intersect1d(sel_lon, parameters.lon)})
            coords=[(lon,lat) for lat in parameters_loc.lat for lon in parameters_loc.lon]
            self.pond = Preprocessing_ANTILOPE.codistances(coords, self.domain)  # TODO : ameliorer les perf

            #if int(num_poste) == 74033400:
            #if int(num_poste) == 38191400:
            #if int(num_poste) == 5133400:
            #if int(num_poste) == 38253400:
            if int(num_poste) == 73176400:
                R, Rstat, Rdyn, updated_obs = self.observation_ECM_new(parameters_loc, date, plot=dict(lat=nearest_lat, lon=nearest_lon, date=date, num_poste=num_poste))
            else:
                R, Rstat, Rdyn, updated_obs = self.observation_ECM_new(parameters_loc, date)
            Y = updated_obs.data  # Observation vector
            obs = Y.reshape((len(parameters_loc.lat), len(parameters_loc.lon)))  # Get observation field
            obs = np.round(obs, 1)  # Round precipitation <0.1 at 0 (different distribution used in this case)

            # Initialisation of ensemble output field
            analysis = xr.DataArray(
                name   = 'rr',
                dims   = ["member", "lat", "lon"],
                coords = dict(lon=parameters_loc.lon, lat=parameters_loc.lat, member=range(0, nmembers+1)),
            )

            # Fill first member with corrected observation
            analysis.loc[{'member':0}] = obs
            self.newlocalfield[0][idp,idd] = analysis.sel({'lat':nearest_lat, 'lon':nearest_lon, 'member':0}).data

            sd1 = Rstat.diagonal().reshape((len(parameters_loc.lat), len(parameters_loc.lon)))  # Get standard deviation field
            #sd1 = uniform_filter(sd1, 3)
            sd2 = Rdyn.diagonal().reshape((len(parameters_loc.lat), len(parameters_loc.lon)))
            #sd2 = uniform_filter(sd2, 3)
            sd = R.diagonal().reshape((len(parameters_loc.lat), len(parameters_loc.lon)))  # Get standard deviation field
            #sd = sd1 + sd2
            # If no precipitation have been introduced by the WMA we are confident that there is actually no precipitation
            # Avoid small dispersion around 0mm and flatten the rank histogram
            sd[obs==0] = 0
            sd1[obs==0] = 0
            sd2[obs==0] = 0

            error = xr.DataArray(
                name   = 'error',
                #data   = np.square(sd),
                data   = sd,
                dims   = ["lat", "lon"],
                coords = dict(lon=parameters_loc.lon, lat=parameters_loc.lat),
            )
            self.error[idp, idd] = error.sel({'lat':nearest_lat, 'lon':nearest_lon})

            # Fill other members with random draw arround the corrected observation
            for member in range(1, nmembers+1):
                #ana = Preprocessing_ANTILOPE.random_draw(obs, sd, distribution='gamma')
                #ana = Preprocessing_ANTILOPE.random_draw(obs, sd1, distribution='gamma')
                ana = Preprocessing_ANTILOPE.random_draw(obs, sd1, sd2=sd2, distribution='gamma')
                #ana = Preprocessing_ANTILOPE.random_draw(obs, sd, distribution='normal')
                analysis.loc[{'member':member}] = ana

                self.newlocalfield[member][idp,idd] = analysis.sel({'lat':nearest_lat, 'lon':nearest_lon, 'member':member}).data

#            except KeyError:
#                print(f'Dropping station number {num_poste} (too close from the edge of the domain)')
##                # TODO : enlever les postes concernés du fichier de sortie pour ne pas dégrader les scores artificiellement !


class EnsembleKalmanFilter(Assimilation):

    def __init__(self, period, obs, ensemble, nivometeo, plot, frequency, gridded, localisation, mask, debiasing, domain, likelyhood):

        super(EnsembleKalmanFilter, self).__init__(period, obs, ensemble, nivometeo, plot, frequency, gridded, localisation, mask, debiasing, domain, likelyhood)

        # Parameters to compute Euclidian distance between all points in the domain
        self.ld = ld
        self.max_dist = max_dist


    @speedtest
    def run(self, covariance=False):
        """ 
        Main method that loop over the assimilation dates and grid points.
        TODO : compléter la doc sur la méthode
        """

        domain = self.domain

        # Initialisation of output fields
        if self.gridded:
            self.nlon, self.nlat = len(self.radar.lon), len(self.radar.lat)
            null  = np.empty((self.nlat, self.nlon, len(self.period)))  # 2D (lat/lon) field
        else:
            self.nposte = len(self.nivometeo.num_poste)
            null = np.empty((self.nposte, len(self.period)))
            #self.nb_out_raw = np.zeros(self.nposte)  # Count number of obs outside raw ensemble
            # Extract evalution points

        null[:] = np.nan
        self.newlocalfield = {m:null.copy() for m in range(self.Ne+1)}  # Used only for ponctual assimilation
        #self.newlocalfield = {m:dict() for m in range(1, self.Ne+1)}  # Used only for ponctual assimilation

        actual_ensemble = self.ensemble
        actual_parameters = self.parameters

        if covariance:
            codistances = os.path.join('/home/vernaym/These/DATA', f'codistance_max_dist_{self.max_dist}_{domain}.npz')
#            if not os.path.exists(codistances):
#                # Compute inter-distances
#                coords=[(lon,lat) for lat in actual_parameters.lat.data for lon in actual_parameters.lon.data]
#                self.pond = self.codistances(coords)
#            else:
#                self.pond = scipy.sparse.load_npz(codistances)
            coords=[(lon,lat) for lat in actual_parameters.lat.data for lon in actual_parameters.lon.data]
            #self.pond = self.codistances(coords)
            self.pond = Preprocessing_ANTILOPE.codistances(coords, self.domain)
        else:
            self.pond = scipy.sparse.eye(len(actual_parameters.lat)*len(actual_parameters.lon))
            self.pond = csr_matrix(self.pond)

#        print('DBUG0')
#        tmp=0.0000001*scipy.sparse.identity(np.shape(self.pond)[0])
#        tmp=csc_matrix(self.pond+tmp)
#        #tmp=np.linalg.cholesky(tmp.toarray())
#        import pdb
#        pdb.set_trace()
#        tmp=sparse_cholesky(tmp)
#        #from cholespy import CholeskySolverF
#        #solver = CholeskySolverF(np.shape(self.pond)[0], , cols, self.pond.nonzero(), scipy.sparse.csr.csr_matrix)
#        tmp=np.round(tmp, 2)
#        self.C = sparse_cholesky(csc_matrix(tmp))
#        #self.C = np.round(sparse_cholesky(csc_matrix(self.pond+0.0000001*scipy.sparse.identity(np.shape(self.pond)[0]))), 2)
#        print('DBUG1')

        for idd, date in enumerate(self.period):
            print(date)
            self.date_str = date.strftime('%Y%m%d%H')
            if self.plot:
                if not os.path.exists(self.date_str):
                    os.makedirs(self.date_str)

            ensemble   = actual_ensemble.sel({'time':date}).compute()  # Load data into memory now
            parameters = actual_parameters.sel({'time':date}).compute()

            ####################  TMP  #####################
            # Plot distributions before / after conversion
#            if self.plot:
#                fig, ax = plt.subplots()
#                R1 = parameters.mu.data.flatten()
#                R2 = np.sqrt(R1)
#                ax.hist(R2, density=True, bins=np.arange(np.floor(np.nanmin(R2))-0.1, np.ceil(np.nanmax(R2)) + 0.1, 0.1), label='R* (mm^1/2)', alpha=0.5)
#                mu = np.nanmean(R2)
#                sd = np.sum((R2-mu)**2)/len(R2)
#                ax = plot_distribution(ax, mu, sd, color='blue')
#                ax.hist(R1, density=True, bins=np.arange(np.floor(np.nanmin(R1))-0.1, np.ceil(np.nanmax(R1)) + 0.1, 0.1), label='R (mm)', alpha=0.5)
#                ax.legend()
#                ax.set_xlim(right=13)
#                if not os.path.exists(f'{self.date_str}/distributions'):
#                    os.makedirs(f'{self.date_str}/distributions')
#                fig.savefig(f'{self.date_str}/distributions/conversion_rr.pdf')
#                plt.close(fig)
            ####################  END  #####################

            # Change variable R --> R^(1/2) to bring the distributions closer to a Normal one
            #ensemble.rr.data = np.sqrt(ensemble.rr.data)
            #parameters.mu.data = np.sqrt(parameters.mu.data)
            #parameters.rr.data = np.sqrt(parameters.rr.data)

            if self.gridded:
                self.gridded_analysis(date, idd,  ensemble, parameters, domain)
            else:
                self.ponctual_analysis(date, idd, ensemble, parameters, covariance=covariance)

    @speedtest
    def ponctual_analysis(self, date, idd, ensemble, parameters, covariance=False, nmembers=16):

        evaluation_points = zip(self.nivometeo.num_poste.data, np.nanmax(self.nivometeo.lat, axis=1).data, np.nanmax(self.nivometeo.lon, axis=1).data)
        for idp, (num_poste, lat, lon) in enumerate(evaluation_points):
            print(num_poste, idp)
            # Extract rectangle around evaluation point
            nearest_lat = nearest(parameters.lat, lat)
            nearest_lon = nearest(parameters.lon, lon)
            #try:
            if covariance:
                sel_lat = np.round(np.arange(nearest_lat-self.max_dist, nearest_lat+self.max_dist, 0.01), 2)
                sel_lon = np.round(np.arange(nearest_lon-self.max_dist, nearest_lon+self.max_dist, 0.01), 2)
                ensemble_loc = ensemble.sel({'lat':np.intersect1d(sel_lat, ensemble.lat), 'lon':np.intersect1d(sel_lon, ensemble.lon)})
                parameters_loc = parameters.sel({'lat':np.intersect1d(sel_lat, parameters.lat), 'lon':np.intersect1d(sel_lon, parameters.lon)})
                # Compute inter-distances
                coords=[(lon,lat) for lat in parameters_loc.lat for lon in parameters_loc.lon]
                #self.pond = self.codistances(coords)
                self.pond = Preprocessing_ANTILOPE.codistances(coords, self.domain)
            else:  # Verrue !
                ensemble_loc = ensemble.sel({'lat':np.round([nearest_lat], 2), 'lon':np.round([nearest_lon], 2)})
                parameters_loc = parameters.sel({'lat':np.round([nearest_lat], 2), 'lon':np.round([nearest_lon], 2)})
                self.pond = scipy.sparse.eye(1)

            R, Rstat, Rdyn, updated_obs = self.observation_ECM_new(parameters_loc, date)  # WARNING : prameters loc est un point unique !
            Y = updated_obs.data  # Observation vector. WARNING : Use mu to take debiasing into account !
            #R = R + diags(Y.flatten(), 0)*0.3  # Increase observation error by 30% of the observation value to match RS
            R = R + diags((Y+0.1).flatten(), 0)*0.3  # Increase observation error by 30% of the observation value to match RS + avoid singular matrix
            B, updated_ensemble = self.background_error_covariance_new(ensemble_loc, updated_obs, R)  # Localisation
            ensemble_loc = updated_ensemble

            # TODO Ajouter une étape de comparaison des distribution d'ébauche et d'obs (augmentation de l'erreur d'ébauche
            # si distribution disjointes : on fait plus confiance à l'obs dans ce cas)

            Y = Y.flatten()

            analysis = xr.DataArray(
                name   = 'rr',
                dims   = ["member", "lat", "lon"],
                coords = dict(lon=parameters_loc.lon, lat=parameters_loc.lat, member=range(nmembers+1)),
            )

            analysis.loc[{'member':0}] = Y.reshape((len(parameters_loc.lat), len(parameters_loc.lon)))
            self.newlocalfield[0][idp,idd] = analysis.sel({'lat':nearest_lat, 'lon':nearest_lon, 'member':0}).data

            for member in ensemble_loc.member.data:
                raw = ensemble_loc.sel({'member':member}).rr
                X = raw.data.flatten()  # Ensemble member vector
                Z = spsolve(B+R, Y-X)
                A = X+B.dot(Z)
                #A = X + K.dot(Y-X)
#            Solution to avoid the "B+R" matrix inversion :
#            1. solve (B+R).Z=Y-X
#            --> the matrix is already "band diagonal" but could be converted using a
#            reverse_cuthill_mckee algorithm
#            --> use "spsolve" method for sparse matrices (solveh_banded for dense
#            matrices)
#            2. compute anlaysis as A=X+BZ

                # On peut maintenant extraire les vrais domaines (on a plus besoind e la marge sur les bords)
                analysis.loc[{'member':member}] = A.reshape((len(raw.lat), len(raw.lon)))

                self.newlocalfield[member][idp,idd] = analysis.sel({'lat':nearest_lat, 'lon':nearest_lon, 'member':member}).data

#            except KeyError:
#                print(f'Dropping station number {num_poste} (too close from the edge of the domain)')
#                # TODO : enlever les postes concernés du fichier de sortie pour ne pas dégrader les scores artificiellement !

    @speedtest
    def gridded_analysis(self, date, idd, ensemble, parameters, domain, nmembers=16):

        R, Rstat, Rdyn, updated_obs = self.observation_ECM_new(parameters, date)
        Y = updated_obs.data  # Observation vector. WARNING : Use mu to take debiasing into account !
        #R = R + diags(Y.flatten(), 0)*0.3  # Increase observation error by 30% of the observation value to match RS
        R = R + diags((Y+0.1).flatten(), 0)*0.3  # Increase observation error by 30% of the observation value to match RS + avoid singular matrix
        parameters = parameters.update({'obs':updated_obs})
        B, updated_ensemble = self.background_error_covariance_new(ensemble, updated_obs, R)  # Background error covariance matrix
        ensemble = updated_ensemble

        if self.plot:
            point = np.where(Y==np.nanmax(Y))  # max observation (plot only)
            #point = np.where(Y==np.nanmin(Y))  # min observation (plot only)
            original_ens = ensemble.isel(lat=point[0], lon=point[1]).rr.data.flatten()  # (plot only)

            # plot distributions
            fig, ax = plt.subplots()
            original_obs = parameters.rr.data[point]
            plt.bar(original_obs, 1, width=0.03, label='Original observation', color='red', alpha=0.5)
            obs = Y[point][0]
            plt.bar(obs, 1, width=0.03, color='red')
            std = R.diagonal().reshape((len(parameters.lat), len(parameters.lon)))[point][0]
            ax = plot_distribution(ax, obs, std, label='Observation', color='red')
            mu = np.mean(original_ens)
            std = np.sum((original_ens-mu)**2)/len(original_ens)
            ax = plot_distribution(ax, mu, std, ensemble=original_ens, label='Raw ensemble', color='grey')
            ens = ensemble.isel(lat=point[0], lon=point[1]).rr.data.flatten()
            mu = np.mean(ens)
            #std = B.diagonal().reshape((len(parameters.lat), len(parameters.lon)))[point][0]
            std = np.sum((ens-mu)**2)/len(ens)
            ax = plot_distribution(ax, mu, std, ensemble=ens, label='Background', color='k')

        # TODO : reconvertir en précipitation (R --> R^2) avant de plotter !
        self.rrmin = 0.
        self.rrmax = min(80, max(
                #np.nanmax(np.square(ensemble.raw.data)),
                #np.nanmax(np.square(ensemble.rr.data)),
                np.nanmax(parameters.obs.data),
                np.nanmax(parameters.rr.data),
                np.nanmax(parameters.mu.data),
                ))

        # TODO Ajouter une étape de comparaison des distribution d'ébauche et d'obs (augmentation de l'erreur d'ébauche
        # si distribution disjointes : on fait plus confiance à l'obs dans ce cas)

        if self.plot:
            if not os.path.exists(f'{self.date_str}/RAW_{self.date_str}_{self.domain}.pdf'):
                fig1,ax1 = plt.subplots(nrows=4, ncols=4, figsize=figsize[domain]['ensembleplot'])
            if not os.path.exists(f'{self.date_str}/BACKGROUND_{self.date_str}_{self.domain}.pdf'):
                fig4,ax4 = plt.subplots(nrows=4, ncols=4, figsize=figsize[domain]['ensembleplot'])
            fig2,ax2 = plt.subplots(nrows=4, ncols=4, figsize=figsize[domain]['ensembleplot'])
            fig3,ax3 = plt.subplots(nrows=4, ncols=4, figsize=figsize[domain]['ensembleplot'])
            #fig2,ax2 = plt.subplots(nrows=2, ncols=8, figsize=(16,10))
            i = 0
            j = 0

        Y = Y.flatten()
        analysis = xr.DataArray(
            name   = 'rr',
            dims   = ["member", "lat", "lon"],
            coords = dict(lon=ensemble.lon, lat=ensemble.lat, member=range(nmembers+1)),
        )
        for member in ensemble.member.data:

            ##############################  TMP  ###############################
            # To compare the analysis with a random field generation
            #random = np.random.normal(loc=0.0, scale=1.0, size=1)[0]
            #sd = R.diagonal().reshape((len(ensemble.lat), len(ensemble.lon)))
            #obs = Y.reshape((len(ensemble.lat), len(ensemble.lon)))
            #analysis.loc[{'member':member}] = np.square(obs+random*sd/5)
            ##############################  END  ###############################

            raw = ensemble.sel({'member':member}).raw
            background = ensemble.sel({'member':member}).rr
            X = background.data.flatten()  # Ensemble member vector

            #A = X + K.dot(Y-X)
#            Solution to avoid the "B+R" matrix inversion :
#            1. solve (B+R).Z=Y-X
#            --> the matrix is already "band diagonal" but could be converted using a
#            reverse_cuthill_mckee algorithm
#            --> use "spsolve" method for sparse matrices (solveh_banded for dense
#            matrices)
#            2. compute anlaysis as A=X+BZ
            Z = spsolve(B+R, Y-X)
            A = X+B.dot(Z)

            # On peut maintenant extraire les vrais domaines (on a plus besoind e la marge sur les bords)
            analysis.loc[{'member':member}] = A.reshape((len(ensemble.lat), len(ensemble.lon)))  # Go back in the real precipitation space

            # Reduction du domain en enlevant la marge en bordure
            # Inutile : la domain est réduit au moment de plotter les champs
#            raw = raw.sel({'lat':np.intersect1d(sel_lat, raw.lat.data), 'lon':np.intersect1d(sel_lon, raw.lon.data)})

            if self.plot:
                if not os.path.exists(f'{self.date_str}/RAW_{self.date_str}_{self.domain}.pdf'):
                    im1 = plot_field(raw, ax1[i,j], self.rrmin, self.rrmax, self.domain)
                    ax1[i,j].set_title(None)

                    #######################  TMP  ################
#                    if member == 8:
#                        circle = plt.Circle((6.27, 45.07), max_dist, color='red', fill=False, linewidth=4)
#                        ax1[i,j].add_artist(circle)
##                        pt = 1200
##                        corr = xr.DataArray(
##                                name   = 'correlation',
##                                data   = self.pond.getrow(pt).toarray()[0].reshape((len(raw.lat), len(raw.lon))),
##                                dims   = ["lat", "lon"],
##                                coords = dict(lon=raw.lon, lat=raw.lat),
##                            )
##
###                        field = field.sel({'lat':np.intersect1d(sel_lat, field.lat.data), 'lon':np.intersect1d(sel_lon, field.lon.data)})
###                        corr = corr.sel({'lat':np.intersect1d(sel_lat, corr.lat.data), 'lon':np.intersect1d(sel_lon, corr.lon.data)})
##                        plot_correlation(ax1[i,j], raw, corr, point=pt)
                    #######################  TMP  ################

                if not os.path.exists(f'{self.date_str}/BACKGROUND_{self.date_str}_{self.domain}.pdf'):
                    im4 = plot_field(background, ax4[i,j], self.rrmin, self.rrmax, self.domain)
                    ax4[i,j].set_title(None)
                im2 = plot_field(analysis.loc[{'member':member}], ax2[i,j], self.rrmin, self.rrmax, self.domain)
                ax2[i,j].set_title(None)
                im3 = plot_field(parameters.mu-raw, ax3[i,j], np.nanmin(parameters.mu.data-raw.data), np.nanmax(parameters.mu.data-raw.data), self.domain)
                j = j + 1
                if j==4:
                    j = 0
                    i = i + 1

        if self.plot:

            # Plot analysis distribution
            ens = analysis.isel(lat=point[0], lon=point[1]).data.flatten()
            mu = np.mean(ens)
            std = np.sqrt(np.sum((ens-mu)**2)/16)
            ax = plot_distribution(ax, mu, std, ensemble=ens, label='Analysis', color='blue')
            ax.legend()
            ax.set_xlim(right=10)
            ax.set_ylim(top=1)
            ax.set_xlabel('Precipitation (mm)')
            if not os.path.exists(f'{self.date_str}/distributions'):
                os.makedirs(f'{self.date_str}/distributions')
            fig.savefig(f'{self.date_str}/distributions/DISTRIBUTION_ANALYSE.pdf')
            plt.close(fig)

#            if domain != 'alp' and self.localisation is None:
            # Compute and plot K
            # Working inversion of large sparse matrix
            A = B+R
#            A = A + 0.001*scipy.sparse.eye(A.shape[0])
#            K = B.dot(scipy.sparse.linalg.inv(A))
            K = B/A  # since A is diagonal !
#            K = B.dot(np.linalg.inv((B+R).toarray()))

            # Plot matrices
            self.plot_matrix(K, parameters.rr, 'Kalman_Gain', f'{self.date_str}/Kalman_Gain_{domain}.pdf', cmap=plt.cm.coolwarm, vmin=0, vmax=1)
            ECM_max = max(np.nanmax(R.diagonal()), np.nanmax(B.diagonal()))
            self.plot_matrix(B, parameters.rr, 'Background_ECM', f'{self.date_str}/Background_ECM_{domain}.pdf', vmin=0, vmax=ECM_max, cmap=plt.cm.viridis)
            #self.plot_matrix(B, parameters.rr, 'Background_ECM', f'{self.date_str}/Background_ECM_{domain}.pdf', vmin=0, cmap=plt.cm.viridis)
            self.plot_matrix(R, parameters.rr, 'Observation_ECM', f'{self.date_str}/Observation_ECM_{domain}.pdf', vmin=0, vmax=ECM_max, cmap=plt.cm.viridis)
            #self.plot_matrix(R, parameters.rr, 'Observation_ECM', f'{self.date_str}/Observation_ECM_{domain}.pdf', vmin=0, cmap=plt.cm.viridis)
            #self.plot_matrix(Rdyn, parameters.rr, 'Observation_ECM', f'{self.date_str}/Observation_dyn_ECM_{domain}.pdf', vmin=0, vmax=ECM_max, cmap=plt.cm.viridis)
            #self.plot_matrix(Rdyn, parameters.rr, 'Observation_ECM', f'{self.date_str}/Observation_dyn_ECM_{domain}.pdf', vmin=0, cmap=plt.cm.viridis)
            #self.plot_matrix(Rstat, parameters.rr, 'Observation_ECM', f'{self.date_str}/Observation_stat_ECM_{domain}.pdf', vmin=0, vmax=ECM_max, cmap=plt.cm.viridis)
            #self.plot_matrix(Rstat, parameters.rr, 'Observation_ECM', f'{self.date_str}/Observation_stat_ECM_{domain}.pdf', vmin=0, cmap=plt.cm.viridis)

#            ensemble.rr.data = ensemble.rr.data
#            ensemble.raw.data = ensemble.raw.data
#            parameters.mu.data = parameters.mu.data
#            parameters.rr.data = parameters.rr.data
#            parameters.obs.data = parameters.obs.data

            self.plot_obs(parameters, domain=domain)
            if self.debiasing:
                self.plot_obs(parameters, var='mu', domain=domain)
                self.plot_obs(parameters, var='obs', domain=domain)
                parameters['diff'] = parameters.obs-parameters.mu
                #parameters['diff'] = parameters.obs-parameters.rr  # !! TODO : TMP !!
                self.plot_obs(parameters, var='diff', domain=domain)

            mean = self.ensemble_mean(analysis)
            self.plot_array(mean, parameters.rr, 'Mean precipitation (mm)', f'{self.date_str}/Analysis_mean_{self.domain}.pdf', cmap=plt.cm.YlGnBu, vmin=self.rrmin, vmax=self.rrmax)
            # TODO : comprendre pourquoi la dispersion est plus importante sur les bords du domaine (distance de coorélation moins impactante ?
            disp = self.ensemble_dispersion(analysis)
            self.plot_array(disp, parameters.rr, 'Dispersion (mm)', f'{self.date_str}/Analysis_dispersion_{self.domain}.pdf', vmin=0, cmap=plt.cm.YlGnBu)

            rawmean = self.ensemble_mean(ensemble.raw)
            rawdisp = self.ensemble_dispersion(ensemble.raw)
            self.plot_array(rawmean, parameters.rr, 'Mean precipitation (mm)', f'{self.date_str}/Raw_mean_{self.domain}.pdf', cmap=plt.cm.YlGnBu, vmin=self.rrmin, vmax=self.rrmax)
            self.plot_array(rawdisp, parameters.rr, 'Dispersion (mm)', f'{self.date_str}/Raw_dispersion_{self.domain}.pdf', vmin=0, cmap=plt.cm.YlGnBu)

            backgroundmean = self.ensemble_mean(ensemble.rr)
            backgrounddisp = self.ensemble_dispersion(ensemble.rr)
            self.plot_array(backgroundmean, parameters.rr, 'Mean precipitation (mm)', f'{self.date_str}/Background_mean_{self.domain}.pdf', cmap=plt.cm.YlGnBu, vmin=self.rrmin, vmax=self.rrmax)
            self.plot_array(backgrounddisp, parameters.rr, 'Dispersion (mm)', f'{self.date_str}/Background_dispersion_{self.domain}.pdf', vmin=0, cmap=plt.cm.YlGnBu)

            if not os.path.exists(f'{self.date_str}/RAW_{self.date_str}_{self.domain}.pdf'):
                finalize_fig(fig1, im1, label='24-hour precipitation (mm)', outname=f'{self.date_str}/RAW_{self.date_str}_{self.domain}.pdf')
            if not os.path.exists(f'{self.date_str}/BACKGROUND_{self.date_str}_{self.domain}.pdf'):
                finalize_fig(fig4, im4, label='24-hour precipitation (mm)', outname=f'{self.date_str}/BACKGROUND_{self.date_str}_{self.domain}.pdf')
            finalize_fig(fig2, im2, label='24-hour precipitation (mm)', outname=f'{self.date_str}/ANALYSIS_{self.date_str}_{self.domain}.pdf')
            finalize_fig(fig3, im3, label='24-hour precipitation difference (mm)', outname=f'{self.date_str}/INNOVATION_{self.date_str}_{self.domain}.pdf')

            plt.close('all')


class ParticleFilter(Assimilation):

    def __init__(self, period, obs, ensemble, nivometeo, plot, frequency, gridded, localisation, mask, debiasing, domain, likelyhood):

        super(ParticleFilter, self).__init__(period, obs, ensemble, nivometeo, plot, frequency, gridded, localisation, mask, debiasing, domain, likelyhood)

        # Parameters to compute Euclidian distance between all points in the domain
        self.ld = ld
        self.max_dist = max_dist

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
            t1 = time.time()
            selected_particles.append(int(np.apply_along_axis(lambda a: a.searchsorted(rdm), axis=0, arr=cumulated_weights)))  # add index value (int)
            t2 = time.time()
            #print(f'resampling took {(t2-t1)*1000.}ms')
            # Go 1 step forward and start again
            rdm += step
        return selected_particles

#    @speedtest
    def weighting(self, x, mu, sigma, obs, plot_distribution=False, **kw):

        if self.likelyhood == 'normal':
            draw = self.normal_dist(x, mu, sigma)
        elif self.likelyhood == 'gamma':
            draw = self.gamma_dist(x, mu, sigma)

        if plot_distribution:
            dy = 0.01
            num = np.nanmax((mu+15*sigma)/dy).astype(int)
            y = np.linspace(0, mu+15*sigma, num=num)
            if self.likelyhood == 'normal':
                normal = self.normal_dist(y, mu, sigma)
            elif self.likelyhood == 'gamma':
                gamma = self.gamma_dist(y, mu, sigma)

            fig,ax = plt.subplots()
            color = next(ax._get_lines.prop_cycler)['color']
            # Plot over a smaller range for better lisibility
            ymin = mu-3*sigma
            ymax = mu+3*sigma
            ax.plot(x, draw, linestyle='', marker='+', markersize=10.)
            if self.likelyhood == 'normal':
                ax.plot(y[(y>=0) & (y<ymax)], normal[(y>=0) & (y<ymax)],
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

        R, Rstat, Rdyn, updated_obs = self.observation_ECM_new(parameters_date, date)
        Y = updated_obs.data  # Observation vector. WARNING : Use mu to take debiasing into account !
        parameters_date = parameters_date.update({'obs':updated_obs})
        R = R + diags(Y.flatten(), 0)*0.3  # Increase observation error by 30% of the observation value to match RS
        sigma = xr.DataArray(
            data   = R.diagonal().reshape((len(parameters_date.lat), len(parameters_date.lon))),
            name   = 'obs_error',
            dims   = ["lat", "lon"],
            coords = dict(lon=parameters_date.lon, lat=parameters_date.lat)
        )
        parameters_date = parameters_date.update({'obs_error':sigma})

        #B, updated_ensemble = self.background_error_covariance_new(ensemble, updated_obs, R)  # Background error covariance matrix unsued in PF

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
            fig1,ax1 = plt.subplots(nrows=4, ncols=4, figsize=figsize[self.domain]['ensembleplot'])
            fig2,ax2 = plt.subplots(nrows=4, ncols=4, figsize=figsize[self.domain]['ensembleplot'])
            i = 0
            j = 0
            raw = localized_period.sel({'time':date})
            rrmin = 0.
            rrmax = max(
                    #np.nanmax(raw.rr.data),
                    np.nanmax(parameters_date.rr.data),
                    np.nanmax(parameters_date.mu.data),
                    np.nanmax(parameters_date.obs.data),
                    )
            fig, ax = plt.subplots(figsize=figsize[self.domain]['singleplot'])
            im = plot_field(parameters_date.rr, ax, rrmin, rrmax, self.domain)
            finalize_fig(fig, im, label='24-hour precipitation (mm)', outname=f'{self.date_str}/RAW_observation_{self.date_str}.pdf')

            fig, ax = plt.subplots(figsize=figsize[self.domain]['singleplot'])
            im = plot_field(parameters_date.obs, ax, rrmin, rrmax, self.domain)
            finalize_fig(fig, im, label='24-hour precipitation (mm)', outname=f'{self.date_str}/ASSIMILATED_observation_{self.date_str}.pdf')

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
                im1 = plot_field(raw.sel({'member':member}).rr, ax1[i,j], rrmin, rrmax, self.domain)
                #assim.plot(ax=ax2[i,j], cmap=plt.cm.YlGnBu)
                im2 = plot_field(assim, ax2[i,j], rrmin, rrmax, self.domain)
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

        R, Rstat, Rdyn, updated_obs = self.observation_ECM_new(parameters_date, date)
        Y = updated_obs.data  # Observation vector. WARNING : Use mu to take debiasing into account !
        parameters_date = parameters_date.update({'obs':updated_obs})
        R = R + diags(Y.flatten(), 0)*0.3  # Increase observation error by 30% of the observation value to match RS
        sigma = xr.DataArray(
            data   = R.diagonal().reshape((len(parameters_date.lat), len(parameters_date.lon))),
            name   = 'obs_error',
            dims   = ["lat", "lon"],
            coords = dict(lon=parameters_date.lon, lat=parameters_date.lat)
        )
        parameters_date = parameters_date.update({'obs_error':sigma})

        assimilation_points = zip(self.nivometeo.num_poste.data, np.max(self.nivometeo.lat, axis=1).data, np.max(self.nivometeo.lon, axis=1).data)
        for idp, (num_poste, lat, lon) in enumerate(assimilation_points):
            print(num_poste)
            t1 = time.time()
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
            #if self.frequency == 'hourly':
            raw_localized.compute()  # Load data now
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

            t2 = time.time()
            #print(f'reading data took {(t2-t1)*1000.}ms')
            new, inflation, sigma = self.assimilation(date, raw, raw_localized, parameters, lat, lon, idx, idy, num_poste=num_poste)

            t3 = time.time()
            #print(f'assimilation took {(t3-t2)*1000.}ms')

            self.erreur_obs[idp,idd] = sigma

            if inflation >= 2:
                print(f'{inflation} iterations for station {num_poste}')
            self.inflation[idp] += int(inflation)

            for member, field in self.newlocalfield.items():
                field[idp,idd]  = new[member-1]  # fill new member

#    @speedtest
    def assimilation(self, date, raw, raw_localized, parameters, lat, lon, idx, idy, num_poste=None):

        sigma = parameters.obs_error.data
        initial_obs = parameters.rr.data
        assimilated_obs = parameters.obs.data

        nb_new_member = 0

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

            t1 = time.time()
            # 2.b Weighting
            #-------------
            #weights = self.weighting(raw_localized, parameters.mu.data, sigma, plot_distribution=True)
            #weights = self.weighting(raw_localized, mu, sigma, obs, plot_distribution=True)
            weights = self.weighting(raw_localized, assimilated_obs, sigma, initial_obs)
            weights = weights / np.sum(weights)

            t2 = time.time()
            #print(f'weighting took {(t2-t1)*1000.}ms')
            # 3. Resampling
            #--------------
            # ECC is the order of members indicies (from 0 to 15 !) sorted by increasing precipitation
            selection_locale = self.resample(weights, self.Ne)  # Idicies of selected particles from the "super-ensemble"
            nb_new_member = len(np.unique(selection_locale))
            t3 = time.time()
            #print(f'resampling took {(t3-t2)*1000.}ms')

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

            t4 = time.time()
            #print(f'ECC took {(t4-t3)*1000.}ms')

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
            fig1, axes1 = plt.subplots(nrows=4, ncols=4, figsize=figsize[self.domain]['ensembleplot'])
            fig2, axes2 = plt.subplots(nrows=4, ncols=4, figsize=figsize[self.domain]['ensembleplot'])
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

        actual_parameters = self.parameters
        if self.localisation is not None:
            codistances = os.path.join('/home/vernaym/These/DATA', f'codistance_max_dist_{self.max_dist}_alp.npz')
#            if not os.path.exists(codistances):
#                # Compute inter-distances
#                coords=[(lon,lat) for lat in actual_parameters.lat.data for lon in actual_parameters.lon.data]
#                self.pond = self.codistances(coords)
#            else:
#                self.pond = scipy.sparse.load_npz(codistances)
            coords=[(lon,lat) for lat in actual_parameters.lat.data for lon in actual_parameters.lon.data]
            #self.pond = self.codistances(coords)
            self.pond = Preprocessing_ANTILOPE.codistances(coords, self.domain)
        else:
            self.pond = scipy.sparse.eye(len(actual_parameters.lat)*len(actual_parameters.lon))

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
    """
    Find element of "array" the closer to "value"
    "value" can either be an array of a float
    """
    if isinstance(value, np.ndarray):
        out = list()
        for val in value:
            out.append(float(array[np.abs(array - val).argmin()].data))
        return np.array(out)
    else:
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
    if not args.assimilation == 'rs':
        pearome = read_ensemble(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'), args.frequency, args.domain, antilope)

    nivometeo = read_nivometeo_obs(domain=args.domain)
    if args.gridded:
        localfields = xr.DataArray(
                name   = 'rr',
                dims   = ["lat", "lon", "time", "member"],
                coords = dict(lon=antilope.lon, lat=antilope.lat, time=extract_period, member=range(0,17)),
                attrs  = dict(description="24 hour precipitation",units="mm"),
            )
        globalfields = xr.DataArray(
                name   = 'rr',
                dims   = ["lat", "lon", "time", "member"],
                coords = dict(lon=antilope.lon, lat=antilope.lat, time=extract_period, member=range(1,17)),
                attrs  = dict(description="24 hour precipitation",units="mm"),
            )
    else:
        localfields = xr.DataArray(
                name   = 'rr',
                dims   = ["num_poste", "time", "member"],
                coords = dict(num_poste=nivometeo.num_poste.data, time=extract_period, member=range(0,17)),
                attrs  = dict(description="24 hour precipitation",units="mm"),
            )
        globalfields = None

    #interp_ensemble = pearome.interp(lon=antilope.lon, lat=antilope.lat).clip(0)  # Avoid <0 precipitation values


    # Assimilation
    # ------------

    if args.assimilation == 'pf':  # 1. Particle Filter

        #pf = ParticleFilter(extract_period, antilope, interp_ensemble, nivometeo, args.plot, args.frequency, args.gridded, args.localisation, args.mask, args.debiasing, args.domain, args.likelyhood)
        pf = ParticleFilter(extract_period, antilope, pearome, nivometeo, args.plot, args.frequency, args.gridded, args.localisation, args.mask, args.debiasing, args.domain, args.likelyhood)
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
        localfields.to_netcdf(f"{outname}.nc".encode('utf-8'))
        #globalfields.to_netcdf(f"Assimilation_globale_{args.datebegin.strftime('%Y%m%d%H')}_{args.dateend.strftime('%Y%m%d%H')}_{args.frequency}.nc")

    elif args.assimilation == 'enkf':  # 2. Ensemble Kalman Filter

        #enkf = EnsembleKalmanFilter(extract_period, antilope, interp_ensemble, nivometeo, args.plot, args.frequency, args.gridded, args.localisation, args.mask, args.debiasing, args.domain, args.likelyhood)
        enkf = EnsembleKalmanFilter(extract_period, antilope, pearome, nivometeo, args.plot, args.frequency, args.gridded, args.localisation, args.mask, args.debiasing, args.domain, args.likelyhood)
        mask = enkf.pdf_parameters()
        if args.localisation is not None:
            enkf.run(covariance=True)
        else:
            enkf.run()
        out = enkf.output(localfields)
        outname = f"EnKF_{args.datebegin.strftime('%Y%m%d%H')}_{args.dateend.strftime('%Y%m%d%H')}_{args.frequency}_{args.domain}"
        out.to_netcdf(f"{outname}.nc".encode('utf-8'))

    elif args.assimilation == 'rs':  # Random Sampling

        rs = RandomSampling(extract_period, antilope, nivometeo, args.plot, args.frequency, args.gridded, args.localisation, args.mask, args.debiasing, args.domain, args.likelyhood)
        mask = rs.pdf_parameters()
        rs.run()
        #if not args.gridded:
        #    rs.save_corrected_field(extract_period, nivometeo.num_poste.data)
        out = rs.output(localfields)
        outname = f"Random_Sampling_{args.datebegin.strftime('%Y%m%d%H')}_{args.dateend.strftime('%Y%m%d%H')}_{args.frequency}_{args.domain}"
        out.to_netcdf(f"{outname}.nc".encode('utf-8'))

    tfin = time.time()
    print(f'Total execution time : {(tfin-t0)/60.} minutes')
