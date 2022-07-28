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
extract_dom = dict(
    latmax = 45.240,
    latmin = 44.990,
    lonmin = 6.010,
    lonmax = 6.490,
)

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
def gamma_shape_PDF(x, k=3., theta=1.):
    #import math  # math.gamma does not work with arrays
    from scipy.special import gamma

    return x**(k-1)*np.exp(-x/theta)/(theta**k*gamma(k))

#    if isinstance(x, np.ndarray):
#        return list(map(lambda t: t**(k-1)*np.exp(-t/theta)/(theta**k*gamma(k)) if t > 0 else None, x))
#    else:
#        return x**(k-1)*np.exp(-x/theta)/(theta**k*gamma(k)) if x > 0 else None

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
    plt.savefig(f'OBS_3D_{date}.pdf', format='pdf')

def resample(weights):
    import random
    Ne = len(weights)
    delta = 1/Ne
    cumulated_weights = np.cumsum(weights)
    for i in range(1, Ne+1):
        selected_particles = list()
        u = random.uniform(0, delta)
        while u <= 1:
            #print(u)
            selected_particles.append(np.searchsorted(cumulated_weights, u)+1)
            u += delta
    return selected_particles

if __name__ == "__main__":
    args = parse_command_line()

    extract_period = date_range(args.datebegin, args.dateend)
    filename = 'ANTILOPEQ_{0:s}_{1:s}_{2:s}.nc'.format(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'), args.domain)
    if not os.path.exists(filename):
        print(f'WARNING : file {filename} does not exist, looking for it under {datadir}')
        filename = os.path.join(datadir, filename)
    if os.path.exists(filename):
        antilope = xr.open_dataset(filename)
    else:
        print(f'ERROR : file {filename} does not exist')
        sys.exit(1)
    pearome = read_ensemble(args.datebegin.strftime('%Y%m%d%H'), args.dateend.strftime('%Y%m%d%H'))
    #mnt = xr.open_dataset(os.path.join(datadir, "MNT_GrandesRousses.nc"))
    mnt = xr.open_dataset('/home/vernaym/QGIS/MNT/DEM_ALPES_WGS84_250m_bilinear.nc')  # Pour tracer sur toutes les Alpes
    # Extract Grandes Rousses domain :
    latmin = extract_dom['latmin']
    lonmin = extract_dom['lonmin']
    latmax = extract_dom['latmax']
    lonmax = extract_dom['lonmax']

    date = args.datebegin
    # TODO : on doit même pouvoir se passer de la boucle temporelle !
    while date <= args.dateend:
        date_str = date.strftime('%Y%m%d%H')
        radar = antilope.sel(time=date)
        # WARNING :  la commande suivante réduit sensibement le domaine, attention aux comparaisons entre figures (en particulier avec les CUMULS)
        radar = radar.where((radar.lon>=lonmin) & (radar.lon<=lonmax) & (radar.lat>=latmin) & (radar.lat<=latmax), drop=True)
        # Save min/max values to set common colorbar
        rrmin = np.nanmin(radar.rr.data) * 0.9
        rrmax = np.nanmax(radar.rr.data) * 1.1
        obs = radar['rr'].data
        radar_lat = radar.lat.data
        radar_lon = radar.lon.data
        # Plot ANTILOPE precipitation field
        fig = plt.figure(figsize=(18,8))
        radar.transpose('lat', 'lon').rr.plot(vmin=rrmin, vmax=rrmax, cbar_kwargs={'label': "24 hour precipitation (mm)"})  # quadmesh object
        # Add landmarks
        for landmark, infos in landmarks.items():
            plt.plot(infos['lon'], infos['lat'], marker=infos['marker'], color='red', markersize=10)
            plt.annotate(landmark, (infos['lon']+0.003, infos['lat']+0.003), color='red', fontsize=20)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_title(f'Date {date_str}', fontsize=20)
        ax.axes.get_xaxis().get_label().set_visible(False)
        ax.axes.get_yaxis().get_label().set_visible(False)
        fig.tight_layout()
        fig.savefig(f'OBS_{date_str}.pdf', format='pdf')

        # Plot 3D ANTILOPE precipitation field
        tmp = mnt.interp(lon=radar.lon, lat=radar.lat, method='nearest')  # Pour interpoller le MNT sur la grille ANTILOPE
        # WARNING :  la commande suivante réduit sensibement le domaine, attention aux comparaisons entre figures (en particulier avec les CUMULS)
        #tmp = tmp.where((tmp.lon>=lonmin) & (tmp.lon<=lonmax) & (tmp.lat>=latmin) & (tmp.lat<=latmax), drop=True)
        X, Y = np.meshgrid(tmp['lon'].values, tmp['lat'].values)
        Z = np.nan_to_num(tmp['Band1'].values)
        # define pixel colors
        colors = plt.cm.coolwarm(norm(np.nan_to_num(radar.transpose('lat', 'lon').rr.data)))
        plot3D(X, Y, Z, colors, date_str)

        # Define PDF parameters
        Y     = obs
        mu    = Y 
        delta = 0
        sigma = 5  # constant over the domain
        k = (2*sigma**2+mu**2+np.sqrt((mu**2*(4*sigma**2+mu**2))))/(2*sigma**2)  # condition : k > 1
        if not np.any(k>1):
            print('ERROR : shape parameter k must be >0')
        theta = mu / (k-1)
        gamma_shape_PDF(Y, k=k, theta=theta)
        #SCGD_shape_PDF(Y, k=k, theta=theta, delta=delta, plot_parameters=False)
        ##################################################################################################

        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 8))
        fig2, axes2 = plt.subplots(nrows=4, ncols=4, figsize=(16, 8))
        fig3, axes3 = plt.subplots(nrows=4, ncols=4, figsize=(16, 8))
        i = 0
        j = 0
        weight = dict()
        for member,model in pearome.items():
            model = model.sel(time=date)
            # WARNING :  la commande suivante réduit sensibement le domaine, attention aux comparaisons entre figures (en particulier avec les CUMULS)
            #model = model.where((model.lon>=lonmin) & (model.lon<=lonmax) & (model.lat>=latmin) & (model.lat<=latmax), drop=True)
            # TODO : transpose data before creating netcdf model files
            #im2 = model.transpose('lat', 'lon').rr.plot(ax=axes2[i,j], add_colorbar=False, vmin=rrmin, vmax=rrmax)
            im2 = model.rr.plot(ax=axes2[i,j], add_colorbar=False, vmin=rrmin, vmax=rrmax)
            #model_interp = model.transpose('lat', 'lon').interp(lon=radar.lon, lat=radar.lat)
            model_interp = model.interp(lon=radar.lon, lat=radar.lat)
            # WARNING :  la commande suivante réduit sensibement le domaine, attention aux comparaisons entre figures (en particulier avec les CUMULS)
            model_interp = model_interp.where((model_interp.lon>=lonmin) & (model_interp.lon<=lonmax) & (model_interp.lat>=latmin) & (model_interp.lat<=latmax), drop=True)

            w = gamma_shape_PDF(model_interp.rr, k=k, theta=theta)
            w.name='weight'
            im3 = w.plot(ax=axes3[i,j], add_colorbar=False, vmin=0, vmax=0.1)
            weight[member] = np.sum(w.data)

            ########################################################################################################################
            #im = model_interp.transpose('lat', 'lon').rr.plot(ax=axes[i,j], add_colorbar=False, vmin=rrmin, vmax=rrmax)
            im = model_interp.rr.plot(ax=axes[i,j], add_colorbar=False, vmin=rrmin, vmax=rrmax)
            for ax in [axes[i,j], axes2[i,j], axes3[i,j]]:
                for landmark, infos in landmarks.items():
                    ax.plot(infos['lon'], infos['lat'], marker=infos['marker'], color='red', markersize=4)
                ax.set_aspect('equal')
                ax.axis('off')
                if ax == axes3[i,j]:
                    ax.set_title(f'member {member:03d}, total weight={weight[member]:.3f}')
                else:
                    ax.set_title(f'member {member:03d}')
            j = j + 1
            if j==4:
                j = 0
                i = i + 1
            ########################################################################################################################

        def finalize_fig(figure):
            figure.tight_layout()
            figure.subplots_adjust(right=0.85)
            cbar_ax = figure.add_axes([0.90, 0.15, 0.05, 0.7])
            return cbar_ax

        for figure in [fig, fig2, fig3]:
            cbar_ax = finalize_fig(figure)
            if figure == fig:
                figure.colorbar(im, cax=cbar_ax, label='24-hour precipitation (mm)')
            elif figure == fig2:
                figure.colorbar(im2, cax=cbar_ax, label='24-hour precipitation (mm)')
            else:
                figure.colorbar(im3, cax=cbar_ax, label='Weight')
        fig.savefig(f'MODEL_interp_{date_str}.pdf', format='pdf')
        fig2.savefig(f'MODEL_raw_{date_str}.pdf', format='pdf')
        fig3.savefig(f'WEIGHTS_{date_str}.pdf', format='pdf')

        total = sum(weight.values())
        weights =  {k: v/total for k, v in weight.items()}
        selection = resample(list(weights.values()))

        import pdb
        pdb.set_trace()




        date = date + timedelta(days=1)


    x = list()
    y = list()
    alti = list()
    for station in np.unique(df['num_poste']):
        workdf = df.loc[df['num_poste'] == station]
        elevation = int(workdf['elevation'].mean())
        nb_obs = len(workdf)
        if nb_obs > 10:
            snow = workdf.loc[workdf['elevation'] > workdf['LPNX']]
            rain = workdf.loc[workdf['elevation'] <= workdf['LPNX']]

#            plot(workdf, rain, snow, elevation)

            mu0 = workdf['bias'].mean()
            sigma = statistics.stdev(workdf['bias'].values)
            x.append(mu0)
            y.append(sigma)
            alti.append(elevation)

            fig1 = plt.figure()
            for Y in np.arange(0, 5, 0.1):
                mu1 = Y - mu0
                #delta = delta0*np.exp(-mu)
                #mu = mu - delta
                delta = -2*np.exp(-Y)
                mu = mu1 - delta
                k = (2*sigma**2+mu**2+np.sqrt((mu**2*(4*sigma**2+mu**2))))/(2*sigma**2)  # condition : k > 1
                theta = mu / (k-1)
                SCGD_shape_PDF(Y, k=float(k), theta=float(theta), delta=float(delta), plot_parameters=True)
            fig1.savefig(f"parameters/station_{station}.svg", format='svg')


#    fig0,ax0 = plt.subplots()
#    sc = ax0.scatter(x, y, c=alti, marker="^")
#    plt.colorbar(sc, label='Elevation (m)')
#    ax0.set_xlabel('Mean ANTILOPE / rain gauge deviation (mm)')
#    ax0.set_ylabel('ANTILOPE / rain gauge standard deviation (mm)')
#    fig0.savefig('Mean_std_scatterplot.svg', format='svg')
#    plt.close(fig0)
