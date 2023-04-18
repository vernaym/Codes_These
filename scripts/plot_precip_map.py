#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 15/04/2023

# Script pour générer une carte contenant l'ensemble des informations disponibles
# sur les précipitations en 24h d'une date donnée.

import os, sys
import datetime
import numpy as np
import xarray as xr
import geopandas as gpd  # To install
import plotly.express as px
from plotly.offline import plot
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from pyproj import Proj, transform

######   DOC UTILE PLOTLY  ######
# https://zacks.one/python-plotly/
# https://plotly.com/python/scatter-plots-on-maps/  --> precipitation map

# TODO : trouver un moyen de virer le blanc de la colorbar
# TODO : ajouter un filtre par altitude
# TODO : Ajouter SAFRAN

def usage():
    print("USAGE plot_precip_map.py date")
    print("format de la date : YYYYMMDD (précipitations de YYYYMMD-1 6h à YYYYMMDD6H")
    sys.exit(1)

try:
    date = datetime.datetime.strptime(sys.argv[1], '%Y%m%d')
except Exception as e:
    usage()
    raise e


token = open("/home/vernaym/.mapbox/token").read() # Token from mapbox account

config = dict({'scrollZoom': True})  # plotly image configuration

datadir = '/home/vernaym/workdir/visualisation'

domain = 'alp'

domain_coords = dict(
        GrandesRousses = dict(latmax=45.240, latmin=44.990, lonmin=6.010, lonmax = 6.490),
        NorthernAlps   = dict(lonmin=6.0, lonmax=6.9, latmin=45.6, latmax=46.35),
        CentralAlps    = dict(lonmin=5.6, lonmax=7.0, latmin=45.0, latmax=45.6),
        SouthernAlps   = dict(lonmin=5.7, lonmax=7.0, latmin=44.2, latmax=45.0),
        HauteSavoie    = dict(lonmin=6.45, lonmax=6.95, latmin=45.67, latmax=46.35),
        MontBlanc      = dict(lonmin=6.45, lonmax=7.1, latmin=45.65, latmax=46.1),
        Savoie         = dict(lonmin=6.06, lonmax=7.06, latmin=45.15, latmax=45.65),
        Isere          = dict(lonmin=5.54, lonmax=6.19, latmin=44.89, latmax=45.16),
        Brianconnais   = dict(lonmin=6.48, lonmax=6.95, latmin=44.67, latmax=44.95),
        HautesAlpes    = dict(lonmin=5.90, lonmax=6.36, latmin=44.58, latmax=44.81),
        AlpesSud       = dict(lonmin=6.56, lonmax=6.92, latmin=44.18, latmax=40.49),
        alp            = dict(latmax=46.450, latmin=44.100, lonmin=5.400, lonmax=7.200),
)
coords = domain_coords[domain]


def plot(antilope=None, antilope_error=None, safran=None, nivometeo=None, auto=None):
    """ 
    Plot from shapfile
    https://stackoverflow.com/questions/71780189/how-to-show-only-boundaries-no-fill-of-a-shapefile-in-python-plotly-express
    """

    fig = go.Figure(go.Scattermapbox())

    # 1. Plot ANTILOPE as dot scatterplot
    x,y = np.meshgrid(antilope.lon.data, antilope.lat.data)
    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
                lon  = x.flatten(),
                lat  = y.flatten(),
                mode = 'markers',
                name = 'ANTILOPE',
                text = antilope.rr.data.flatten(),
                marker = dict(
                    color = antilope.rr.data.flatten(),
                    size  = np.nan_to_num(error),
                    #opacity=0.5,
                    colorscale = 'YlGnBu',
                    #symbol='square',  # Impossible to change if color is definied : https://stackoverflow.com/questions/59628536/option-symbol-in-scattermapbox-is-not-working
                    #cmin = 100,
                    #cmax = 1200,
                    colorbar_title = "Precipitation(mm)",
                    colorbar = dict(
                        titleside = "right",
                        #outlinecolor = "rgba(68, 68, 68, 0)",
                        ticks = "outside",
                        #showticksuffix = "last",
                        #dtick = 0.1
                    ),
                ),
            )
        )


    # 2. Add ponctual rain gauges (red for nivometeo, black for other networks)
    def add_ponctual_obs(df, color='black', name='Unknown'):
        fig.add_trace(go.Scattermapbox(
                    lon  = df.lon.round(3),
                    lat  = df.lat.round(3),
                    text = df.rr.round(1).astype('string'),  # WARNING : working only with token mapbox :  https://plotly.com/python/mapbox-layers/
                    mode = 'text',
                    name = name,
                    textfont = dict(size=16, family='Arial', color=color),
                    textposition = 'middle center',
                    hovertext=df.num_poste,
                    # TODO : formater le texte flottant : "Nom (num_poste)"
                    #hovertemplate = '',
                )
            )
    # Nivométéo observations
    if nivometeo is not None:
        add_ponctual_obs(nivometeo, color='red', name='Nivometeo')

    # Automatic observations
    if auto is not None:
        add_ponctual_obs(auto, color='black', name='Automatic stations')

    # Plot background map with SAFRAN massifs
    massifs = "/home/vernaym/safran/ctes/shapefiles/massifs_safran.shp"
    massifs = gpd.read_file(massifs)
    fig.update_layout(
        #coloraxis_showscale=False,
        title = 'Total ANTILOPE precipitation between 2021080106 and 2022070106 (mm) + mean ratio',
        #margin = dict(l=0, t=0, r=1, b=0, pad=0),
        mapbox = dict(
            accesstoken = token,
            style = "outdoors",
#        mapbox = dict(
#            style = "stamen-terrain",  # https://plotly.com/python/mapbox-layers/
#            #style = "open-street-map",  # https://plotly.com/python/reference/layout/mapbox/
#            #style = "basic",  # https://plotly.com/python/reference/layout/mapbox/
            layers =  [
                {
                    "source": massifs["geometry"].__geo_interface__,
                    "type": "line",
                    "color": "black",
                    #"below":"traces",
                    #"opacity":0.5,
                },
            ],
            center = go.layout.mapbox.Center(
                lat=45.2,
                lon=6.0,
            ),
            #pitch = 0,
            zoom = 7,
        ),
    )
    fig.show()


def interactive_layer_choice():
    """
    https://plotly.com/python/custom-buttons/  (+3D map)
    """
    pass

def get_antilope():
    # TODO : appliquer le pré-processing pour plotter ANTILOPE corrigé
    filename = os.path.join(datadir, f'ANTILOPE.nc')
    #if os.path.exists(filename):
    try:
        antilope = xr.open_dataset(os.path.join(datadir, filename))
        # TODO : gérer le changement d'heure !
        antilope = antilope.where((antilope.time>np.datetime64(datebegin)) & (antilope.time<=np.datetime64(dateend)), drop=True).sum('time')
        return antilope
    except:
        return None

def get_antilope_error(full_array):

    try:
        error = xr.open_dataset(os.path.join(datadir, 'Observation_error.nc'))
        full_array['error']=error.ratio  # Fill missing data between ANTILOPE field and the error field
        # normalisation de l'erreur entre low and high
        low  = 1
        high = 20
        def nan_ptp(a):
            return np.ptp(a[np.isfinite(a)])
        error = np.abs(full_array.error.data.flatten())
        error = low + (error - np.nanmin(error))/(nan_ptp(error)/high)
        error = 10 + high/error

        return np.nan_to_num(error, nan=10)
    except:
        return None

def get_nivometeo():
    try:
        fic_score = os.path.join(datadir, 'nivometeo.data')
        nivometeo = pd.read_csv(fic_score, sep=';', parse_dates=['date'],
                dtype={'num_poste':int, 'nom':str, 'alti':int, 'lat':float, 'lon':float, 'massif':int, 'rr': float, 'reseau_poste': int}, na_values=['--'])
        nivometeo = nivometeo.loc[nivometeo["date"]==np.datetime64(date)]
        return nivometeo
    except:
        return None

def get_obs_auto():
    try:
        fic_score = os.path.join(datadir, f'auto.data')
        auto = pd.read_csv(fic_score, sep=';', parse_dates=['date'],
                dtype={'num_poste':int, 'nom':str, 'lat':float, 'lon':float, 'alti':int, 'rr': float, 'reseau_poste': int}, na_values=['--'])
        auto=auto.loc[(auto["date"]>np.datetime64(datebegin)) & (auto["date"]<=dateend)]
        auto = auto[~np.isnan(auto['rr'])]

        #df = auto.set_index(['num_poste', 'lat', 'lon', 'alti', 'nom', 'reseau_poste', 'date']).sort_index()
        #df = df.assign(date=newdates).drop(columns=['date'])  # Replace date column
        auto = auto.set_index(['num_poste', 'lat', 'lon', 'alti', 'nom', 'reseau_poste', 'date']).sort_index()
        auto=auto.groupby(['num_poste', 'lat', 'lon', 'alti', 'reseau_poste']).sum()
        auto = auto.reset_index()

        return auto
    except:
        raise
        return None

def get_safran():
    # TODO : appliquer le pré-processing pour plotter ANTILOPE corrigé
    filename = os.path.join(datadir, f'SAFRAN.nc')
    try:
        safran = xr.open_dataset(os.path.join(datadir, filename))
        import pdb
        pdb.set_trace()
        return safran
    except:
        return None


datebegin = date.replace(hour=6)
dateend   = datebegin+datetime.timedelta(days=1)

# 1. Read ANTILOPE data
antilope = get_antilope()  # xarray.Dataset
error = get_antilope_error(antilope)  # flatten array "read to use"

# 2. Read nivometeo observations
nivometeo = get_nivometeo()

# 3. Read automatic observations
auto = get_obs_auto()

# 4. read SAFRAN
#safran = get_safran()

plot(antilope=antilope, antilope_error=error, nivometeo=nivometeo, auto=auto)
#plot(antilope=antilope, antilope_error=error, nivometeo=nivometeo)

