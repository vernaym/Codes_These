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

import argparse

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # F401 unused import --> to ignore !
from matplotlib.text import Annotation
from matplotlib import offsetbox

from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.proj3d import proj_transform

#from sklearn.linear_model import LinearRegression, RANSACRegressor
#from sklearn.datasets import make_regression
#from sklearn.metrics import mean_squared_error, r2_score

#from scipy.stats import gaussian_kde

import statistics

#from snowtools.plots.maps import cartopy

##############################################################################################
##############################################################################################

datadir = '/home/vernaym/These/DATA'

# Domaine des Grandes Rousses
#extract_dom = dict(
latmax = 45.240
latmin = 44.990
lonmin = 6.010
lonmax = 6.490
#)

norm = plt.Normalize()

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
    parser.add_argument('-d', '--domain', help='Domain of the file', choices=['alp', 'pyr', 'cor', 'GrandesRousses'], default='GrandesRousses')
    parser.add_argument('-w', '--workdir', help='Runing directory', default='/home/vernaym/workdir/ASSIMILATION/XP00')
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


def read_ensemble(datebegin, dateend, domain='GrandesRousses'):
    pearome = dict()
    for member in range(1,17):
        filename = f'aspearome_{member:03d}_{datebegin}_{dateend}_{domain}.nc'
        filename = os.path.join(datadir, filename)
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

def finalize_fig(figure, imm, label, outname):
    figure.tight_layout()
    figure.subplots_adjust(right=0.85)
    cbar_ax = figure.add_axes([0.90, 0.15, 0.05, 0.7])
    figure.colorbar(imm, cax=cbar_ax, label=label)
    figure.savefig(outname, format='pdf')

def read_obs(args):
    filename = 'ANTILOPEQ_{0:s}_{1:s}_{2:s}.nc'.format(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'), args.domain)
    if not os.path.exists(filename):
        print(f'WARNING : file {filename} does not exist, looking for it under {datadir}')
        filename = os.path.join(datadir, filename)
    if os.path.exists(filename):
        antilope = xr.open_dataset(filename)
        antilope = antilope.transpose('lat', 'lon', 'time')  # TODO : Fix the dataset in the generation script
        # WARNING :  la commande suivante réduit sensibement le domaine, attention aux comparaisons entre figures (en particulier avec les CUMULS)
        antilope = antilope.where((antilope.lon>=lonmin) & (antilope.lon<=lonmax) & (antilope.lat>=latmin) & (antilope.lat<=latmax), drop=True)
    else:
        print(f'ERROR : file {filename} does not exist')
        sys.exit(1)
    return antilope


class ParticleFilter(object):

    def __init__(self, date, obs, ensemble):

        self.date = date
        self.date_str = date.strftime('%Y%m%d%H')
        if not os.path.exists(self.date_str):
            os.makedirs(self.date_str)
        #goto(self.date_str)
        self.nline, self.ncol = np.shape(obs.rr.data)
        self.radar = obs
        self.rrmin = 0.
        self.rrmax = max(
                np.nanmax([mod.rr for mod in ensemble.values()]),
                np.nanmax(radar.rr.data)
                )
        self.ensemble = ensemble
        self.Ne = len(self.ensemble)

        # Plot observation field
        self.plot_obs()

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
                lambda t: gamma_shape_PDF(t+delta, k=k, theta=theta) if t > 0 else 0, y)))
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

    def pdf_parameters(self):
        # I. Define PDF parameters
        #-------------
        parameters = self.radar.copy()  # TODO : vérifier si ce n'est pas trop couteux (de toutes façon l'bs n'a pas de raison d'être modifiée)
        parameters = parameters.rename({'rr':'mu'})  #  Mode=observation (WARNING : mu is NOT the mean) TODO : check if the ensemble after assimilation is not biased
        parameters['sigma'] = (0.261 + 0.263 * parameters['mu'])  # According to the linear regression of ANTILOPE RMSE vs ANTILOPE RR
        # shift of the gamma PDF ==> this defines the weight given to 0mm forecats (=0 if delta=0) !!
        parameters = parameters.assign(delta=lambda x: 10/x.mu)  # TODO : find a better shift than 10/mu
        parameters.delta.data[np.isinf(parameters.delta.data)] = 0  # Si l'obs est nulle on veut imposer une PDF exponentielle décroissante (k=1) sans translation
        #delta = np.zeros(np.shape(mu))
        #delta[np.where(mu>0)] = 1 / mu[np.where(mu>0)]
        #
        # TODO : Intégrer delta au calcul de k ?
        # ======================================
        parameters['k'] = (2*parameters.sigma**2+parameters.mu**2+np.sqrt((parameters.mu**2*(4*parameters.sigma**2+parameters.mu**2))))/(2*parameters.sigma**2)  # shape parameter of the Gamma PDF
        #k = (2*sigma**2+mu**2+np.sqrt((mu**2*(4*sigma**2+mu**2))))/(2*sigma**2)  # shape parameter of the Gamma PDF
        #parameters.k.where(parameters.mu==0).data = parameters.mu.where(parameters.mu==0).data / parameters.mu.where(parameters.mu==0).data   # k>1 if Y>0 else k==1
        parameters["k"] = xr.where(parameters.mu==0, 1, parameters.k)   # k>1 if Y>0 else k==1
        #k[np.where(mu==0)] = 1
        parameters['theta'] = parameters.mu / (parameters.k-1)  # Scale parameter of the Gamma PDF
        parameters['theta'] = xr.where(parameters.k==1, 3*parameters.k, parameters.theta)  # Set theta=3 by default if k=1. TODO : In this case the scale parameter could depend on neighboring observations
        #theta = mu / (k-1)
        self.parameters = parameters

        # TODO : plot PDF for some pixels
        # gamma_shape_PDF(Y, k=k, theta=theta)

        self.plot_parameters()

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

    def resample(self, weights):
        import random
        Ne = len(weights)
        delta = 1/Ne
        # Sort particules on [0,1[ according to their weight
        cumulated_weights = np.cumsum(weights, axis=0)
        #for i in range(1, Ne+1):
        selected_particles = list()
        # Random draw between [0, 1/Ne[
        rdm = random.uniform(0, delta)
        #rdm = np.random.random_sample(np.shape(cumulated_weights[0]))*delta
        while rdm <= 1:
            # Select particle in wich rdm falls
            #selected_particles.append(np.searchsorted(cumulated_weights, rdm)+1)
            selected_particles.append(np.apply_along_axis(lambda a: a.searchsorted(rdm), axis=0, arr=cumulated_weights)+1)
            # Go 1 step forward and start again
            rdm += delta
        return selected_particles

    def raw_weighting(self):

        # On profite de la loop sur les membres pour tracer les ensembles bruts avant et après interpolation
        fig1, axes1 = plt.subplots(nrows=4, ncols=4, figsize=(16, 8))
        fig2, axes2 = plt.subplots(nrows=4, ncols=4, figsize=(16, 8))
        i = 0
        j = 0
        self.weight = dict()
        self.wpixel = dict()
        for member, model in self.ensemble.items():
            # Weighting
            #----------
            self.wpixel[member] = self.SCGD_shape_PDF(model.rr, self.parameters.mu.data, k=self.parameters.k.data, theta=self.parameters.theta.data, delta=self.parameters.delta.data)
            self.wpixel[member].name = 'weight'
            # Plot pixel weights
            self.weight[member] = np.sum(self.wpixel[member].data)
            #---------------------------------------------------------------------------------------------------------

            # Plot interpolated model field
            im1 = plot_field(model.rr, axes1[i,j], self.rrmin, self.rrmax, title=f'member {member:03d}')
            # Plot raw model field  (do it now to avoid another loop)
            im2 = plot_field(raw_ensemble[member].rr, axes2[i,j], self.rrmin, self.rrmax, title=f'member {member:03d}')
            j = j + 1
            if j==4:
                j = 0
                i = i + 1

        finalize_fig(fig1, im1, label='24-hour precipitation (mm)', outname=f'{self.date_str}/MODEL_interp_{self.date_str}.pdf')
        finalize_fig(fig2, im2, label='24-hour precipitation (mm)', outname=f'{self.date_str}/MODEL_raw_{self.date_str}.pdf')

    def selection(self):
        self.global_selection()
        self.local_selection()

    def global_selection(self):
        # 1 Weighting
        total_global = sum(self.weight.values())
        global_weights =  {k: v/total_global for k, v in self.weight.items()}
        # 2 Resampling
        self.selection_globale = self.resample(list(global_weights.values()))

    def local_selection(self):
        # 1 Ensemble Copula Coupling (to produce fields matching the raw ones)
        self.ecc = np.argsort([field.rr.data for field in self.ensemble.values()], axis=0)

        # 2 Weighting
        #total_local = sum(wpixel.values()).data
        total_local = sum(self.wpixel.values())
        self.local_weights =  {k: v.data/total_local for k, v in self.wpixel.items()}
        # 3 Resampling
        self.selection_locale = self.resample(np.array(list(self.local_weights.values())))

        # 4.field reconstruction
        self.make_new_local_fields()

    def make_new_local_fields(self):
        # Initialisation of output fields
        null  = np.empty(np.shape(radar.rr))
        self.newlocalfield = {m:null.copy() for m in range(1, self.Ne+1)}
#        self.localfields = {m:xr.DataArray(
#                    data   = null.copy(),  # copy variable to avoid to erase the value at each iteration
#                    name   = 'rr',
#                    dims   = ["lat", "lon"],
#                    coords = dict(lon=self.radar.lon, lat=self.radar.lat),
#                    attrs  = dict(description="Total precipitation",units="mm"),
#                ) for m in range(1, self.Ne+1)}

        # Loop over the domain's pixels
        for i in range(self.nline):
            for j in range(self.ncol):
                free = list(self.ecc[:,i,j].copy()+1)  # Define the order in which the fields are to be filled
                selection = [self.selection_locale[idx][i,j] for idx in range(self.Ne)]  # local PF output
                # Loop over ordered raw members
                for idx in self.ecc[:,i,j]:
                    oldmember = idx + 1  # Convert the index into an ensemble member
                    value = self.ensemble[oldmember].rr.data[i,j]  # corresponding raw value
                    # Fill as many members as the raw member duplicates
                    for count in range(selection.count(oldmember)):
                        newmember = free.pop(0)  # select first member to fill in the "waiting list"
                        self.newlocalfield[newmember][i,j] = value  # fill new member
#                print('i,j=',i,j)
#                print('Obs = ',self.parameters.mu.data[i,j])
#                print('Raw = ',[rawfield.rr.data[i,j] for rawfield in self.ensemble.values()])
#                print('ECC = ',self.ecc[:,i,j])
#                print('Selection = ',selection)
#                print('Assim = ',[field.data[i,j] for field in self.newlocalfield.values()])

    def plot_one_pixel(self, x, y):
        model = [values.rr.data[x][y] for values in self.ensemble.values()]
        self.SCGD_shape_PDF(model, self.parameters.mu.data[x][y], k=self.parameters.k.data[x][y], theta=self.parameters.theta.data[x][y],delta=self.parameters.delta.data[x][y], plot_distribution=True)

    def output(self, localfields, globalfields):
        fig1, ax1 = plt.subplots(nrows=4, ncols=4, figsize=(16, 8))
        fig2, ax2 = plt.subplots(nrows=4, ncols=4, figsize=(16, 8))
        fig3, axes3 = plt.subplots(nrows=4, ncols=4, figsize=(16, 8))
        i = 0
        j = 0
        for m in range(1, self.Ne+1):
            localfields.loc[{'time':self.date, 'member':m}] = self.newlocalfield[m]  # self.newlocalfield is a numpy array
            globalfields.loc[{'time':self.date, 'member':m}] = self.ensemble[self.selection_globale[m-1]].rr.data  # self.ensemble is a DataArray

            # Plot local weight fields
            vmin = 0
            #vmax = 0.25  # TODO : vmax=f(sigma) [peut être renvoyé par la fonction de la PDF comme le max de probabilité]
            #vmax = np.max(gamma_shape_PDF(parameters.mu.data, k=parameters.k.data, theta=parameters.theta.data))
            vmax = max([np.max(ww.data) for ww in self.local_weights.values()])
            im3 = plot_field(self.local_weights[m], axes3[i,j], vmin, vmax, title=f'member {m:03d}, total weight={self.weight[m]:.3f}', cmap=plt.cm.Greys)

#            # Selection de valeur pixel par pixel
#            local = np.array([[model_interp[selection_locale[m-1][i,j]].rr.data[i,j] for j in range(ncol)] for i in range(nline)])
#            localfield[m] = xr.DataArray(
#                    data=local,
#                    name='rr',
#                    dims=["lat", "lon"],
#                    coords=dict(lon=radar.lon, lat=radar.lat),
#                    attrs=dict(description="Total precipitation",units="mm"),
#                ).to_dataset()
            im1 = plot_field(globalfields.loc[{'time':self.date, 'member':m}], ax1[i,j], self.rrmin, self.rrmax, title=f'Member {self.selection_globale[m-1]:03d}')
            im2 = plot_field(localfields.loc[{'time':self.date, 'member':m}], ax2[i,j], self.rrmin, self.rrmax, title=f'New member {m:03d}')
            j = j + 1
            if j==4:
                j = 0
                i = i + 1
        finalize_fig(fig1, im1, label='24-hour precipitation (mm)', outname=f'{self.date_str}/ASSIM_globale_{self.date_str}.pdf')
        finalize_fig(fig2, im2, label='24-hour precipitation (mm)', outname=f'{self.date_str}/ASSIM_locale_{self.date_str}.pdf')
        finalize_fig(fig3, im3, label='Weight', outname=f'{self.date_str}/WEIGHTS_{self.date_str}.pdf')

        plt.close('all')

        return localfields, globalfields

if __name__ == "__main__":
    args = parse_command_line()

    goto(args.workdir)
    extract_period = date_range(args.datebegin, args.dateend)

    antilope = read_obs(args)
    pearome = read_ensemble(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'))
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

    # TODO : on doit peut être pouvoir se passer de la boucle temporelle en utilisant les fonctionalités de xarray
    #date = args.datebegin
    #while date <= args.dateend:
    for date in extract_period:
        print(date)
        radar           = antilope.sel(time=date)
        raw_ensemble    = {k:v.sel(time=date) for k,v in pearome.items()}
        interp_ensemble = {k:v.interp(lon=radar.lon, lat=radar.lat) for k,v in raw_ensemble.items()}

        pf = ParticleFilter(date, radar, interp_ensemble)
        pf.pdf_parameters()
        pf.raw_weighting()  # Compute weight fields before normalisation + plot raw/interp fields
        pf.selection()  # Global and local selections

        localfields, globalfields = pf.output(localfields, globalfields)

        # TODO : SAVE all assimilated fields for an evaluation of the performance over the period

    evaluation = Evaluation()

#        date = date + timedelta(days=1)

#        import pdb
#        pdb.set_trace()
