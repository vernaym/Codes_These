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
import json
import plotly.express as px
from plotly.offline import plot
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from pyproj import Proj, transform

import scipy
from scipy.sparse import csr_matrix, diags
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree


######   DOC UTILE PLOTLY  ######
# https://zacks.one/python-plotly/
# https://plotly.com/python/scatter-plots-on-maps/  --> precipitation map

# TODO : trouver un moyen de virer le blanc de la colorbar
# TODO : ajouter un filtre par altitude : https://plotly.com/python/v3/selection-events/?_gl=1*pup8cg*_ga*MTI0ODI4NTA5Ni4xNjgxODAwMDUx*_ga_6G7EE0JNSC*MTY4MTg4OTIwMy45LjEuMTY4MTg5MTk4OS4wLjAuMA
# TODO : Une seule colorbar pour toutes les couches associées


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

ld = 0.05
max_dist = ld*3

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


def update_axes(xaxis, yaxis):
    scatter = f.data[0]
    scatter.x = df[xaxis]
    scatter.y = df[yaxis]
    with f.batch_update():
        f.layout.xaxis.title = xaxis
        f.layout.yaxis.title = yaxis
        scatter.x = scatter.x + np.random.rand(N)/10 *(df[xaxis].max() - df[xaxis].min())
        scatter.y = scatter.y + np.random.rand(N)/10 *(df[yaxis].max() - df[yaxis].min())

def plot(antilope=None, antilope_error=None, safran=None, nivometeo=None, auto=None, var='obs'):
    """ 
    Plot from shapfile
    https://stackoverflow.com/questions/71780189/how-to-show-only-boundaries-no-fill-of-a-shapefile-in-python-plotly-express
    """

    fig = go.Figure()

    # 1. Plot ANTILOPE as multiple dot scatterplot (depending on the elevation)
    # read_relief
    mnt = xr.open_dataset(os.path.join('/home/vernaym/QGIS/MNT', "DEM_ALPES_WGS84_250m_bilinear.nc"))
    # WARNING : update of xarray necessary !
    data = antilope.interp(lat=mnt.lat.data, lon=mnt.lon.data)
    data["elevation"] = mnt.Band1

    #for elevation in [0, 600, 1200, 2000]:
    x = dict()
    y = dict()
    z = dict()
    for i,elevation in enumerate([0, 2000]):
        # Make all data visible by default
        if elevation == 0:
            x[i],y[i] = np.meshgrid(antilope.lon.data, antilope.lat.data)
            z[i]   = antilope[var].data.flatten(),  # Corrected obs
            visible = True
            name = 'ANTILOPE'
        else:
            tmp = data.where(data.elevation>=elevation, drop=True)
            x[i],y[i] = np.meshgrid(tmp.lon.data, tmp.lat.data)
            z[i]   = tmp[var].data.flatten(),  # Corrected obs above elevation
            visible = False
            name = f'ANTILOPE>{elevation:d}m'

        fig.add_trace(go.Scattermapbox(
                    lon  = x[i].flatten(),
                    lat  = y[i].flatten(),
                    mode = 'markers',
                    name = name,
                    #text = antilope.rr.data.flatten(),  # Raw obs
                    text = z[i],  # obs=corrected obs, rr=raw obs
                    visible = visible,
                    marker = dict(
                        #color = antilope.rr.data.flatten(),
                        color = z[i],
                        cmin  = 0,
                        cmax  = np.nanmax(antilope.rr.data.flatten()),
                        size  = np.nan_to_num(error),
                        #opacity=0.5,
                        colorscale = 'YlGnBu',
                        #symbol='square',  # Impossible to change if color is definied : https://stackoverflow.com/questions/59628536/option-symbol-in-scattermapbox-is-not-working
                        #cmin = 100,
                        #cmax = 1200,
                        colorbar_title = "Precipitation(mm)",
                        colorbar = dict(
                            titleside = "right",
                            ticks = "outside",
                            # Move the colorbar away from the legend :
                            yanchor="top",
                            y=1,
                            x=-0.1,
                            #showticksuffix = "last",
                            #dtick = 0.1
            #coloraxis_colorbar = dict(yanchor="top", y=1, x=0),  # Move the colorbar away from the legend
                        ),
                    ),
                )
            )

    updatemenus = [{
                'active':1,
                'buttons': [{'method': 'update',
                             'label': 'Elevation',
                             'args': [
                                      # 1. updates to the traces
                                      {
                                       'lon':x[0],
                                       'lat':y[0],
                                       'marker': {'color':antilope[var].data.flatten()},
                                       #'name':['sin', 'sin - 1'],
                                       'visible': True}, 
                                      # 2. updates to the layout
                                      #{'title':'Sine'},
                                      # 3. which traces are affected 
                                      [-1, -2],
                                      ],
                             'args2': [
                                      # 1. updates to the traces
                                      {
                                       'lon':x[1],
                                       'lat':y[1],
                                       'marker': {'color':tmp[var].data.flatten()},
                                       #'name':['sin', 'sin - 1'],
                                       'visible': True}, 
                                      # 2. updates to the layout
                                      #{'title':'Sine'},
                                      # 3. which traces are affected 
                                      [-1, -2],
                                      ],
                              },
                            ],
#                'type':'buttons',
                'type':'dropdown',
                'direction': 'down',
                'showactive': True,}
            ]

    # 2. Add ponctual rain gauges (red for nivometeo, black for other networks)
    def add_ponctual_obs(df, color='black', name='Unknown'):
        fig.add_trace(go.Scattermapbox(
                    #df,
                    lon  = df.lon.round(3),
                    lat  = df.lat.round(3),
                    text = df.rr.round(1).astype('string'),  # WARNING : working only with token mapbox :  https://plotly.com/python/mapbox-layers/
                    mode = 'text',
                    name = name,
                    textfont = dict(size=16, family='Arial', color=color),
                    textposition = 'middle center',
                    hoverinfo = 'text',
                    #hover_data=[df.num_poste, df.nom],
                    #hovertext = [df.num_poste, df.nom],
                    customdata = np.stack((df.num_poste, df.nom, df.alti, df.reseau_poste), axis=-1),
                    #customdata = [df.num_poste, df.nom],
                    #hovertemplate="<br>".join([
                    #    f"Num poste: : %{customdata[0]}",
                    #    f"Nom : %{customdata[1]}",
                    #]),
                    hovertemplate =
                        '<b>Num poste</b>: %{customdata[0]:d}<br>'+
                        '<b>Nom</b>: %{customdata[1]}<br>'+
                        '<b>Altitude</b>: %{customdata[2]}m<br>'+
                        '<b>Réseau</b>: %{customdata[3]}<br>'
                    ,
                    #hovertext=df.num_poste,
                    #hovertext=[df.num_poste, df.nom],
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

    # 3. Plot background map with SAFRAN massifs and fill them
    # Method from : https://community.plotly.com/t/plot-a-shapefile-shp-in-a-choropleth-chart/27850
    # a. read massif shapefile
    massifs = "/home/vernaym/safran/ctes/shapefiles/massifs_safran.shp"
    massifs = gpd.read_file(massifs)

    # b. convert it to a plotly-readable GeoJSON file
    massifs_json = os.path.join(datadir, "massifs_safran.json")
    if not os.path.exists(massifs_json):
        massifs.to_file(massifs_json, driver = "GeoJSON")
    with open(massifs_json) as geofile:
        j_file = json.load(geofile)
        i = 0
        for feature in j_file["features"]:
            feature['id'] = massifs.code[i]
            i += 1
        plotsafran = safran.where(safran.ZS==1500., drop=True)

        fig.add_trace(go.Choroplethmapbox(
            geojson = j_file,
            locations = plotsafran.massif_number,
            z = plotsafran.rr.data,  #TODO : add elevation choice
            colorscale = 'YlGnBu',
            name = 'SAFRAN',
            zmin = 0,
            #zmax = np.nanmax(antilope.rr.data.flatten()),
            zmax = np.nanmax(antilope[var].data.flatten()),
            visible = True,
            uid = 4,
            uirevision = True,
            showscale = False,  # Same scale as ANTILOPE data (à vérifier !)
            showlegend = True,
            #opacity=0.5,
        ))

    # 4. Update figure layout
    fig.update_layout(
        #coloraxis_showscale=False,
        title = f'24h precipitation (mm) between {datebegin} and {dateend}',
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
        updatemenus = updatemenus,
    )

#    fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
#                                      legendgroup = newnames[t.name],
#                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
#                                     )
# from : https://stackoverflow.com/questions/64371174/how-to-change-variable-label-names-for-the-legend-in-a-plotly-express-line-chart

    fig.show()

def update_menu():
    """"
    https://stackoverflow.com/questions/68894919/how-to-set-the-values-of-args-and-args2-in-plotlys-buttons-in-updatemenus
    """

    # Stragtegy to plot data by elevation band :

    # 1. Split ANTILOPE data into elevation-based clusters
    # 2. PLot each cluster independently and make them all visible by default
    # 3. Use updatemenus tool to mask some clusters


    lowelevation = [dict(type="circle",
                            xref="x", yref="y",
                            x0=min(x0), y0=min(y0),
                            x1=max(x0), y1=max(y0),
                            line=dict(color="DarkOrange"))]
    #midelevation = ...
    #highelevation = ...



    return updatemenus

def interactive_layer_choice():
    """
    https://plotly.com/python/custom-buttons/  (+3D map)
    https://stackoverflow.com/questions/66414456/update-visibility-of-traces-with-fig-update-layout-plotly
    """
    pass

def get_antilope():
    # TODO : appliquer le pré-processing pour plotter ANTILOPE corrigé
    filename = os.path.join(datadir, f'ANTILOPE.nc')
    #if os.path.exists(filename):
    antilope = xr.open_dataset(os.path.join(datadir, filename))
    # TODO : gérer le changement d'heure !
    antilope = antilope.where((antilope.time>np.datetime64(datebegin)) & (antilope.time<=np.datetime64(dateend)), drop=True).sum('time')

    # Static de-biasing :
    mask = xr.open_dataset(os.path.join(datadir, f"Estimated_ratio.nc"))

    antilope["ratio"]=mask.ratio  # Fill missing point with NaNs
    antilope["rr_debiaise"] = (antilope.rr/antilope.ratio).fillna(antilope.rr)  # Fill NaN values with the original ANTILOPE value

    # Dynamic correction (localisation)
    error = xr.open_dataset(os.path.join(datadir, 'Observation_error.nc'))
    std = np.abs(error.ratio.data)

    filename = os.path.join('/home/vernaym/These/DATA', f'codistance_max_dist_{max_dist}_{domain}.npz')
    if not os.path.exists(filename):
        # Compute inter-distances
        coords=[(lon,lat) for lat in error.lat.data for lon in error.lon.data]
        pond = codistances(coords)
        scipy.sparse.save_npz(filename, pond, compressed=False)
    else:
        pond = scipy.sparse.load_npz(filename)

    pond = pond.dot(diags(np.exp(-std).flatten(), 0))
    obs = antilope.rr_debiaise.sel(({'lat':np.intersect1d(error.lat.data, antilope.lat.data), 'lon':np.intersect1d(error.lon.data, antilope.lon.data)})).data.flatten()

    new = update_obs(obs, pond, replacement_strategy='toward_mean')  # Update obs
    antilope['obs'] = xr.DataArray(
            data   = new.reshape((len(mask.lat), len(mask.lon))),
            dims   = ["lat", "lon"],
            coords = dict(lon=mask.lon, lat=mask.lat)
        )
    antilope['obs'] = antilope['obs'].fillna(antilope.rr)

    return antilope

def update_obs(field, pond, weight=None, super_ensemble=None, replacement_strategy='keep'):

    field[np.isnan(field)] = 0.0
    initial_field = field.flatten()
    X = diags(field.flatten(), 0)

    # 1. Calcul de la moyenne pondérée par la distance ET l'erreur statique
    if weight is None:
        pond.data[np.isnan(pond.data)] = 0.0
        weight = pond.sum(axis=1).getA1()  # The sum of the weights (axis=1 <==> sum over rows)

    mean = pond.dot(X).sum(axis=1).getA1()  # getA1 transforms the 1*N matrix object into a 1D np.array
    mean = mean / weight
    pixel_weight = pond.diagonal()  # = exp(-erreur_statique) pour l'obs et =likelyhood du pixel pour les membres de l'ensemble
    sums = pond.sum(axis=1).A1
    nb_nonzero = (pond != 0).sum(0).getA1()  # Count non zero elements of each row
    meanweight = sums / nb_nonzero
    newfield = (initial_field * pixel_weight + mean * meanweight) / (pixel_weight + meanweight)
    #sd = self.get_std(X, newfield, pond, weight=weight, super_ensemble=super_ensemble)
    return newfield


def codistances(coords):
    """
    Solution pour le calcul des inter-distances trouvée sur : https://stackoverflow.com/questions/35296935/python-calculate-lots-of-distances-quickly
    """
    tree = cKDTree(coords)
    dist = tree.sparse_distance_matrix(tree, max_distance=max_dist, p=2, output_type='coo_matrix')
    dist = csr_matrix(dist)
    #TODO : utiliser une gaussienne plutot qu'une exponentielle décroissante
    dist[dist.nonzero()] = -dist[dist.nonzero()]/ld
    np.exp(dist.data, out=dist.data )
    return dist

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
        auto=auto.groupby(['num_poste', 'nom', 'lat', 'lon', 'alti', 'reseau_poste']).sum()
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
        #safran = safran.where(safran.ZS==1500., drop=True)
        safran['rr'] = (safran['Rainf']+safran['Snowf'])*3600.
        dates = pd.date_range(datebegin+datetime.timedelta(hours=1), dateend, freq='1H')
        safran = safran.loc[{'time':dates}]
        #safran['rr'] = safran.where((safran.time>np.datetime64(datebegin)) & (safran.time<=np.datetime64(dateend)), drop=True)  # This adds time dimension to ZS variable
        safran['rr'] = safran['rr'].sum('time')
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
safran = get_safran()

plot(antilope=antilope, antilope_error=error, nivometeo=nivometeo, auto=auto, safran=safran, var='obs')
#plot(antilope=antilope, antilope_error=error, nivometeo=nivometeo, auto=auto, safran=safran, var='rr')
#plot(antilope=antilope, antilope_error=error, nivometeo=nivometeo)

