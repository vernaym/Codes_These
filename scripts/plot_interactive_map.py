#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 15/04/2023

import os
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


config = dict({'scrollZoom': True})  # plotly image configuration

datadir = '/home/vernaym/These/DATA'

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


def basemap():
    """ 
    Plot from shapfile
    https://stackoverflow.com/questions/71780189/how-to-show-only-boundaries-no-fill-of-a-shapefile-in-python-plotly-express
    """

    fig = go.Figure(go.Scattermapbox())

    filename =f'CUMUL_ANTILOPEH_alp_2021103000_2022060200.nc'
    antilope = xr.open_dataset(os.path.join(datadir, filename))
    # Add errors
    error = xr.open_dataset(os.path.join(datadir, 'mask', 'Observation_error_0.2_2_alp_ref.nc'))
    #error = 10*np.nan_to_num((1/np.abs(error.ratio))/np.max(1/np.abs(error.ratio))).flatten()

    # normalisation de l'erreur entre low and high
    # TODO : revoir la formule de normalisation
    low  = 1
    high = 15
    def nan_ptp(a):
        return np.ptp(a[np.isfinite(a)])
    error = np.abs(error.ratio.data.flatten())
    error = low + (error - np.nanmin(error))/(nan_ptp(error)/high)
    #error = 1/np.abs(error.ratio.data.flatten())
    error = 5 + high/error

    x,y = np.meshgrid(antilope.lon.data, antilope.lat.data)
    massifs = "/home/vernaym/safran/ctes/shapefiles/massifs_safran.shp"
    massifs = gpd.read_file(massifs)

    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
                lon=x.flatten(),
                lat=y.flatten(),
                mode='markers',
                text=antilope.rr_cumul.data.flatten(),
                marker=dict(
                #marker=go.scattermapbox.Marker(
                    color=antilope.rr_cumul.data.flatten(),
#                    color=np.nan_to_num(error),
                    #size=[1]*len(antilope.rr_cumul.data.flatten()),
                    #size=np.nan_to_num(antilope.rr_cumul.data.flatten()),
#                    size=10,
                    size=np.nan_to_num(error),
                    #opacity=0.5,
                    colorscale='YlGnBu',
                    #colorscale='deep',
                    #colorbar=None,  # TODO : define the colobar
                    #symbol='square',  # Impossible to change if color is definied : https://stackoverflow.com/questions/59628536/option-symbol-in-scattermapbox-is-not-working
                )
            )
        )

    fig.update_layout(
        coloraxis_showscale=False,
        mapbox={
            "style":"stamen-terrain",
            #"style":"open-street-map",  # https://plotly.com/python/reference/layout/mapbox/
            #"style":"basic",  # https://plotly.com/python/reference/layout/mapbox/
            "layers": [
                {
                    "source": massifs["geometry"].__geo_interface__,
                    "type": "line",
                    "color": "black",
                    #"below":"traces",
                    #"opacity":0.5,
                },
            ],
        },
    )
    fig.show()

    # Add layer switch
#    fig.update_layout(
#        updatemenus=[
#            {
#                "buttons":
#                [
#                    {
#                        "label": '{k}',
#                        "method": "update",
#                        "args":
#                        [
#                            {'x': [df[k]]},
#                            {'xaxis':{'title':k}},
#                            {"visible": k},
#                        ],
#                    }
#                    for k in cols
#                ]
#            }
#    ).update_traces(visible=True, selector=0)


def interactive_layer_choice():
    """
    https://plotly.com/python/custom-buttons/  (+3D map)
    """

def plot_antilope(fig):
    filename =f'CUMUL_ANTILOPEH_alp_2021103000_2022060200.nc'
    antilope = xr.open_dataset(os.path.join(datadir, filename))
    #fig.imshow(antilope.rr_cumul.data, color_continuous_scale='YlGnBu', origin='lower')  # https://plotly.com/python/2D-Histogram/
    #fig = fig.add_contour(x=antilope.lon,y=antilope.lat, z=antilope.rr_cumul.data, colorscale='YlGnBu', opacity=0.5, visible=True)
    #fig1 = go.Contour(x=antilope.lon,y=antilope.lat, z=antilope.rr_cumul, colorscale='YlGnBu')
    #fig = fig.add_trace(go.Contour(x=antilope.lon,y=antilope.lat, z=antilope.rr_cumul, colorscale='YlGnBu'))
    fig1=px.imshow(antilope.rr_cumul.data, color_continuous_scale='YlGnBu', origin='lower')
    fig.add_traces(fig1.data).update_layout()
    return fig

fig = basemap()
import pdb
pdb.set_trace()



figure.show(config=config)

