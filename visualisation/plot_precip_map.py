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

# TODO : réduire la taille du fichier html (éviter de plotter des points en double et réduire le domaine du fond de carte)
# TODO : ajouter un filtre par altitude : https://plotly.com/python/v3/selection-events/?_gl=1*pup8cg*_ga*MTI0ODI4NTA5Ni4xNjgxODAwMDUx*_ga_6G7EE0JNSC*MTY4MTg4OTIwMy45LjEuMTY4MTg5MTk4OS4wLjAuMA
# TODO : Une seule colorbar pour toutes les couches associées
# TODO : Append various files every day with last-day data and let the user select the date to plot : use dash
#   - https://dash.plotly.com/basic-callbacks
#   - https://dash.plotly.com/advanced-callbacks
#
# TODO : Avoid plot data multiple time (use button/dash) --> ABSOLUTE PRIORITY !!
# Plotting the full 1-km ANTILOPE domain takes about 12M memory...
# https://plotly.com/python/v3/selection-events/

#def usage():
#    print("USAGE plot_precip_map.py date")
#    print("format de la date : YYYYMMDD (précipitations de YYYYMMD-1 6h à YYYYMMDD6H")
#    sys.exit(1)
#
#try:
#    date = datetime.datetime.strptime(sys.argv[1], '%Y%m%d')
#except Exception as e:
#    usage()
#    raise e

ld = 0.05
max_dist = ld*3
#datadir = '/home/vernaym/workdir/visualisation'
rootdir = '/d0/intra-cen/ANTILOPE'
datadir = '.'
token = open("/home/vernaym/.mapbox/token").read() # Token from mapbox account

class PrecipitationAnalysis(object):


    def __init__(self, date, domain='alp', antilope=None, safran=None, nivometeo=None, auto=None, var='obs'):
        self.antilope = antilope
        self.var = var
        self.safran = safran
        self.nivometeo = nivometeo
        self.auto = auto
        self.date = date
        self.datebegin = date.replace(hour=6)
        self.dateend   = self.datebegin+datetime.timedelta(days=1)
        self.domain = domain
        self.massifs = None

        self.fig = go.Figure()  # Figure initialisation

    def plot(self):
        """ 
        Plot from shapfile
        https://stackoverflow.com/questions/71780189/how-to-show-only-boundaries-no-fill-of-a-shapefile-in-python-plotly-express
        """

        # 1. ANTILOPE
        if self.antilope is not None:
            self.plot_antilope()

        # 2. Nivométéo observations
        self.add_ponctual_obs(self.nivometeo, color='red', name='Nivometeo')

        # 3. Automatic observations
        self.add_ponctual_obs(self.auto, color='black', name='Automatic stations')

        # 4. SAFRAN
        if self.safran is not None:
            self.plot_safran()

        self.update_figure()

        #self.fig.show()
        #fig.write_json('test.json')
        self.fig.write_html(os.path.join(rootdir, self.domain, f"precipitation_{self.date.strftime('%Y%m%d')}.html"))

    def plot_antilope(self):
        """
        Plot ANTILOPE as multiple dot scatterplot (depending on the elevation)
        """
        # 1. read_relief (TODO : add to ANTILOPE pre-processing ?)
        mnt = xr.open_dataset(os.path.join(datadir, "DEM_ALPES_WGS84_250m_bilinear.nc"))
        # WARNING : update of xarray necessary !
        #data = antilope.interp(lat=mnt.lat.data, lon=mnt.lon.data)
        tmp = mnt.interp(lat=self.antilope.lat.data, lon=self.antilope.lon.data)
        self.antilope["elevation"] = tmp.Band1

        x,y  = np.meshgrid(self.antilope.lon.data, self.antilope.lat.data)
        x = x.flatten()
        y = y.flatten()
        rr = self.antilope[self.var].data.flatten()  # Corrected obs
        error = self.antilope['error'].data.flatten()
        #    # normalisation de l'erreur entre low and high
        # TODO : revoir la méthode pour convertir l'erreur en une dimension de marker plotly
        low  = 5
        high = 30
        def nan_ptp(a):
            return np.ptp(a[np.isfinite(a)])
        error = np.abs(error)
        error = low + (error - np.nanmin(error))/(nan_ptp(error)/high)
        error = low + high/error

        alti = self.antilope['elevation'].data.flatten()
        df = pd.DataFrame(
                data    = np.transpose([x, y, rr, error, alti]),
                columns = ['lon', 'lat', 'rr', 'error', 'alti'],
                index   = range(len(rr)),
            )
        select = dict()
        for i,elevation in enumerate(range(0, 3500, 1000)):
        # TODO : Avoid plot data multiple time (use button ?) --> ABSOLUTE PRIORITY !!
        # Plotting the full 1-km ANTILOPE domain takes about 12M memory...
        # https://plotly.com/python/v3/selection-events/
            if elevation == 0:
                # Make all data visible by default
                visible = True
                name = 'ANTILOPE'
                select[i] = np.where((rr>0.1) & (~np.isnan(error)))
                showscale = True
            else:
                select[i]  = np.where((rr>0.1) & (alti>elevation) & (~np.isnan(error)))
                visible ='legendonly'  # Does not work as intended : https://community.plotly.com/t/legendonly-doesnt-work-anymore-in-scattermapbox/72822
                visible = True
                name    = f'ANTILOPE>{elevation:d}m'
                showscale = False

            self.fig.add_trace(self.add_antilope_scatter(df, name, select[i], showscale=showscale, uncertainty=True))
            #if i == 0:
                # TODO : useless (same data ==> use a button !)
            #    self.fig = self.fig.add_trace(self.add_antilope_scatter(df, name, select[i], uncertainty=False))

    def add_antilope_scatter(self, df, name, mask, showscale=False,  uncertainty=True, visible=True):
        return go.Scattermapbox(
                    lon  = df.lon.values[mask],
                    lat  = df.lat.values[mask],
                    #selectedpoints = select,  # TODO : use this footprint to set updatemenus buttons DOES NOT WORK (because it refers to user selected points)
                    mode = 'markers',
                    name = name,
                    #text = antilope.rr.data.flatten(),  # Raw obs
                    text = df.rr.values[mask],  # obs=corrected obs, rr=raw obs
                    visible = visible,
                    showlegend = True,
                    #selected = go.scattermapbox.Selected(marker={"size":50}),
                    customdata = np.stack((df.alti.values[mask], df.rr.values[mask], df.error.values[mask]), axis=-1),
                    hovertemplate =
                        '<b>Altitude</b>: %{customdata[0]:d}m<br>'+
                        '<b>Precipitation</b>: %{customdata[1]:f}<br>'+
                        '<b>Incertitude</b>: %{customdata[2]:f}<br>',
                    marker = dict(
                        #color = antilope.rr.data.flatten(),
                        color = df.rr.values[mask],
                        cmin  = 0,
                        cmax  = np.nanmax(self.antilope.rr.data.flatten()),
                        # TODO : revoir la conversion erreur --> taille + ajouter une fourchette dans le text flottan (genre "10mm d'incertitude")
                        size  = np.nan_to_num(df.error.values[mask], nan=5) if uncertainty else 10,
                        #opacity=0.5,
                        #colorscale = 'YlGnBu',
                        colorscale = 'dense',
                        #symbol='square',  # Impossible to change if color is definied : https://stackoverflow.com/questions/59628536/option-symbol-in-scattermapbox-is-not-working
                        #cmin = 100,
                        #cmax = 1200,
                        colorbar_title = "Precipitation(mm)",
                        showscale = showscale,  # Plot only one colorscale
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

    def add_button(self):
        """
        NOT IMPLEMENTED YET
        Add button to activate / deactivate marker size depending on the uncertainty
        DOEST NOT WORK : try with dash ?
        https://stackoverflow.com/questions/68894919/how-to-set-the-values-of-args-and-args2-in-plotlys-buttons-in-updatemenus
        """
        updatemenus = [{
                    'active':1,
                    'buttons': [{'method': 'update',  # whether the button changes the plot, the layout or both
                                 'label': 'Incertitude',  # what is written on the button / label
                                 'args':[  # what happens when the button is clicked
                                        # 1. updates to the traces
                                        dict(marker = dict(
                                            #color = antilope.rr.data.flatten(),
                                            color = rr[select[0]],
                                            cmin  = 0,
                                            cmax  = np.nanmax(antilope.rr.data.flatten()),
                                            size  = 20,  # TODO : revoir la taille minimale
                                            colorscale = 'dense',
                                            colorbar_title = "Precipitation(mm)",
                                            colorbar = dict(
                                                titleside = "right",
                                                ticks = "outside",
                                                yanchor="top",
                                                y=1,
                                                x=-0.1,
                                            ),
                                         ),
                                         ),
                                         # 2. updates to the layout
                                         #{'title':'Sine'},
                                         {},
                                         # 3. which traces are affected 
                                         [0],
                                         #[trace for trace in range(-(i+1)*2, 0)],
                                         ],
                                 'args2':[  # what happens when it’s unclicked
                                        # 1. updates to the traces
                                        dict(marker = dict(
                                            #color = antilope.rr.data.flatten(),
                                            color = rr[select[0]],
                                            cmin  = 0,
                                            cmax  = np.nanmax(antilope.rr.data.flatten()),
                                            size  = np.nan_to_num(error[select[i]], nan=5),  # TODO : revoir la taille minimale
                                            colorscale = 'dense',
                                            colorbar_title = "Precipitation(mm)",
                                            colorbar = dict(
                                                titleside = "right",
                                                ticks = "outside",
                                                yanchor="top",
                                                y=1,
                                                x=-0.1,
                                            ),
                                         ),
                                         ),
                                         #'name':['sin', 'sin - 1'],
                                         #'visible': True}, 
                                         # 2. updates to the layout
                                         {},
                                         #{'title':'Sine'},
                                         # 3. which traces are affected 
                                         [0],
                                         #[trace for trace in range(-(i+1)*2, 0)],
                                         ],
                                  },
                                ],
                    'type':'buttons',
#                'type':'dropdown',
#                'direction': 'down',
                    'showactive': True,}
                ]

    def add_ponctual_obs(self, df, color='black', name='Unknown'):
        """
        Add ponctual rain gauges (red for nivometeo, black for other networks)
        """
        if df is not None:
            self.fig.add_trace(go.Scattermapbox(
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
                            '<b>Réseau</b>: %{customdata[3]}<br>',
                        #hovertext=df.num_poste,
                        #hovertext=[df.num_poste, df.nom],
                        # TODO : formater le texte flottant : "Nom (num_poste)"
                        #hovertemplate = '',
                    )
                )

    def read_safran_massifs(self, massifs_json):
        # 1. read massif shapefile
        massifs = "/home/vernaym/safran/ctes/shapefiles/massifs_safran.shp"
        self.massifs = gpd.read_file(massifs)

        if not os.path.exists(massifs_json):
            # 1. convert it to a plotly-readable GeoJSON file
            self.massifs.to_file(massifs_json, driver = "GeoJSON")

    def plot_safran(self):
        """
        Plot background map with SAFRAN massifs and fill them
        Method from : https://community.plotly.com/t/plot-a-shapefile-shp-in-a-choropleth-chart/27850
        """
        massifs_json = os.path.join(datadir, "massifs_safran.json")
        self.read_safran_massifs(massifs_json)

        with open(massifs_json) as geofile:
            j_file = json.load(geofile)
            i = 0
            for feature in j_file["features"]:
                feature['id'] = self.massifs.code[i]
                i += 1
            # TODO : add dynamic elevation choice
            plotsafran = self.safran.where(self.safran.ZS==2100., drop=True)

            self.fig.add_trace(go.Choroplethmapbox(
                geojson = j_file,
                locations = plotsafran.massif_number,
                z = plotsafran.rr.data,  #TODO : add elevation choice
                #colorscale = 'YlGnBu',
                colorscale = 'dense',
                name = 'SAFRAN',
                zmin = 0,
                #zmax = np.nanmax(antilope.rr.data.flatten()),
                zmax = np.nanmax(self.antilope[self.var].data.flatten()),
                visible = True,
                uid = 4,
                uirevision = True,
                showscale = False,  # Same scale as ANTILOPE data (à vérifier !)
                showlegend = True,
                #opacity=0.5,
            ))

    def update_figure(self):
        """
        Update figure layout
        """

        self.fig.update_layout(
            #coloraxis_showscale=False,
            title = f'24h precipitation (mm) between {self.datebegin} and {self.dateend}',
            margin = dict(l=1, t=40, r=1, b=0, pad=0),
            #mapbox_bounds={"west": 2, "east": 11, "south": 42, "north": 48},
            mapbox = dict(
                accesstoken = token,
                style = "outdoors",
#        mapbox = dict
#            style = "stamen-terrain",  # https://plotly.com/python/mapbox-layers/
#            #style = "open-street-map",  # https://plotly.com/python/reference/layout/mapbox/
#            #style = "basic",  # https://plotly.com/python/reference/layout/mapbox/
                layers =  [
                    {
                        "source": self.massifs["geometry"].__geo_interface__,
                        "type": "line",
                        "color": "black",
                        #"below":"traces",
                        #"opacity":0.5,
                    } if self.massifs is not None else {},
                ],
                center = go.layout.mapbox.Center(  # TODO : à adapter selon le domain)
                    lat=45.2,
                    lon=6.0,
                ),
                #pitch = 0,
                zoom = 7,
            ),
            #updatemenus = self.updatemenus,  # To activate updatemenu
        )

#    fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
#                                      legendgroup = newnames[t.name],
#                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
#                                     )
# from : https://stackoverflow.com/questions/64371174/how-to-change-variable-label-names-for-the-legend-in-a-plotly-express-line-chart

    def update_menu(self):
        """"
        NOT IMPLEMENTED YET
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

    def interactive_layer_choice(self):
        """
        NOT IMPLEMENTED YET
        https://plotly.com/python/custom-buttons/  (+3D map)
        https://stackoverflow.com/questions/66414456/update-visibility-of-traces-with-fig-update-layout-plotly
        """
        pass

#def get_antilope():
#    # TODO : appliquer le pré-processing pour plotter ANTILOPE corrigé
#    filename = os.path.join(datadir, f'ANTILOPE.nc')
#    #if os.path.exists(filename):
#    antilope = xr.open_dataset(os.path.join(datadir, filename))
#
#    if 'analysis' in antilope.variables.keys():
#        # File already pre-processed
#        return antilope
#
#    else:
#        # TODO : gérer le changement d'heure !
#        antilope = antilope.where((antilope.time>np.datetime64(datebegin)) & (antilope.time<=np.datetime64(dateend)), drop=True).sum('time')
#
#        # Static de-biasing :
#        mask = xr.open_dataset(os.path.join(datadir, f"Estimated_ratio.nc"))
#
#        antilope["ratio"]=mask.ratio  # Fill missing point with NaNs
#        antilope["rr_debiaise"] = (antilope.rr/antilope.ratio).fillna(antilope.rr)  # Fill NaN values with the original ANTILOPE value
#
#        # Dynamic correction (localisation)
#        error = xr.open_dataset(os.path.join(datadir, 'Observation_error.nc'))
#        std = np.abs(error.ratio.data)
#
#        filename = os.path.join('/home/vernaym/These/DATA', f'codistance_max_dist_{max_dist}_{self.domain}.npz')
#        if not os.path.exists(filename):
#            # Compute inter-distances
#            coords=[(lon,lat) for lat in error.lat.data for lon in error.lon.data]
#            pond = codistances(coords)
#            #scipy.sparse.save_npz(filename, pond, compressed=False)
#        else:
#            pond = scipy.sparse.load_npz(filename)
#
#        pond = pond.dot(diags(np.exp(-std).flatten(), 0))
#        obs = antilope.rr_debiaise.sel(({'lat':np.intersect1d(error.lat.data, antilope.lat.data), 'lon':np.intersect1d(error.lon.data, antilope.lon.data)})).data.flatten()
#
#        new = update_obs(obs, pond, replacement_strategy='toward_mean')  # Update obs
#        antilope['obs'] = xr.DataArray(
#                data   = new.reshape((len(mask.lat), len(mask.lon))),
#                dims   = ["lat", "lon"],
#                coords = dict(lon=mask.lon, lat=mask.lat)
#            )
#        antilope['obs'] = antilope['obs'].fillna(antilope.rr)
#
#        # TODO : ecrire le fichier pour ne pas refaire les calculs à chaque fois
#
#        return antilope
#
#def update_obs(field, pond, weight=None, super_ensemble=None, replacement_strategy='keep'):
#
#    field[np.isnan(field)] = 0.0
#    initial_field = field.flatten()
#    X = diags(field.flatten(), 0)
#
#    # 1. Calcul de la moyenne pondérée par la distance ET l'erreur statique
#    if weight is None:
#        pond.data[np.isnan(pond.data)] = 0.0
#        weight = pond.sum(axis=1).getA1()  # The sum of the weights (axis=1 <==> sum over rows)
#
#    mean = pond.dot(X).sum(axis=1).getA1()  # getA1 transforms the 1*N matrix object into a 1D np.array
#    mean = mean / weight
#    pixel_weight = pond.diagonal()  # = exp(-erreur_statique) pour l'obs et =likelyhood du pixel pour les membres de l'ensemble
#    sums = pond.sum(axis=1).A1
#    nb_nonzero = (pond != 0).sum(0).getA1()  # Count non zero elements of each row
#    meanweight = sums / nb_nonzero
#    newfield = (initial_field * pixel_weight + mean * meanweight) / (pixel_weight + meanweight)
#    #sd = self.get_std(X, newfield, pond, weight=weight, super_ensemble=super_ensemble)
#    return newfield
#
#
#def codistances(coords):
#    """
#    Solution pour le calcul des inter-distances trouvée sur : https://stackoverflow.com/questions/35296935/python-calculate-lots-of-distances-quickly
#    """
#    tree = cKDTree(coords)
#    dist = tree.sparse_distance_matrix(tree, max_distance=max_dist, p=2, output_type='coo_matrix')
#    dist = csr_matrix(dist)
#    #TODO : utiliser une gaussienne plutot qu'une exponentielle décroissante
#    dist[dist.nonzero()] = -dist[dist.nonzero()]/ld
#    np.exp(dist.data, out=dist.data )
#    return dist
#
#def add_antilope_error(antilope):
#
#    error = xr.open_dataset(os.path.join(datadir, 'Observation_error.nc'))
#    antilope['error'] = error.ratio  # Fill missing data between ANTILOPE field and the error field
#    # normalisation de l'erreur entre low and high
#    # TODO : revoir la méthode pour convertir l'erreur en une dimension de marker plotly
#    low  = 5
#    high = 30
#    def nan_ptp(a):
#        return np.ptp(a[np.isfinite(a)])
#    #antilope["error"] = np.abs(antilope.error.data.flatten())
#    antilope["error"].data = np.abs(antilope.error.data)
#    antilope["error"].data = low + (antilope["error"].data - np.nanmin(antilope["error"].data))/(nan_ptp(antilope["error"].data)/high)
#    antilope["error"].data = low + high/antilope["error"].data
#
#    return antilope
#
#def get_nivometeo():
#    try:
#        fic_score = os.path.join(datadir, 'nivometeo.data')
#        nivometeo = pd.read_csv(fic_score, sep=';', parse_dates=['date'],
#                dtype={'num_poste':int, 'nom':str, 'alti':int, 'lat':float, 'lon':float, 'massif':int, 'rr': float, 'reseau_poste': int}, na_values=['--'])
#        nivometeo = nivometeo.loc[nivometeo["date"]==np.datetime64(date)]
#        return nivometeo
#    except:
#        return None
#
#def get_obs_auto():
#    try:
#        fic_score = os.path.join(datadir, f'auto.data')
#        auto = pd.read_csv(fic_score, sep=';', parse_dates=['date'],
#                dtype={'num_poste':int, 'nom':str, 'lat':float, 'lon':float, 'alti':int, 'rr': float, 'reseau_poste': int}, na_values=['--'])
#        auto=auto.loc[(auto["date"]>np.datetime64(datebegin)) & (auto["date"]<=dateend)]
#        auto = auto[~np.isnan(auto['rr'])]
#
#        #df = auto.set_index(['num_poste', 'lat', 'lon', 'alti', 'nom', 'reseau_poste', 'date']).sort_index()
#        #df = df.assign(date=newdates).drop(columns=['date'])  # Replace date column
#        auto = auto.set_index(['num_poste', 'lat', 'lon', 'alti', 'nom', 'reseau_poste', 'date']).sort_index()
#        auto=auto.groupby(['num_poste', 'nom', 'lat', 'lon', 'alti', 'reseau_poste']).sum()
#        auto = auto.reset_index()
#
#        return auto
#    except:
#        raise
#        return None
#
#def get_safran():
#    # TODO : appliquer le pré-processing pour plotter ANTILOPE corrigé
#    filename = os.path.join(datadir, f'SAFRAN.nc')
#    try:
#        safran = xr.open_dataset(os.path.join(datadir, filename))
#        #safran = safran.where(safran.ZS==1500., drop=True)
#        safran['rr'] = (safran['Rainf']+safran['Snowf'])*3600.
#        dates = pd.date_range(datebegin+datetime.timedelta(hours=1), dateend, freq='1H')
#        safran = safran.loc[{'time':dates}]
#        #safran['rr'] = safran.where((safran.time>np.datetime64(datebegin)) & (safran.time<=np.datetime64(dateend)), drop=True)  # This adds time dimension to ZS variable
#        safran['rr'] = safran['rr'].sum('time')
#        return safran
#    except:
#        return None
#
#
#datebegin = date.replace(hour=6)
#dateend   = datebegin+datetime.timedelta(days=1)
#
## 1. Read ANTILOPE data
#antilope = get_antilope()  # xarray.Dataset
#antilope = add_antilope_error(antilope)  # flatten array "read to use"
#
## 2. Read nivometeo observations
#nivometeo = get_nivometeo()
#
## 3. Read automatic observations
#auto = get_obs_auto()
#
## 4. read SAFRAN
#safran = get_safran()
#
#myplot = PrecipitationAnalysis(date, 'alp', antilope=antilope, nivometeo=nivometeo, auto=auto, safran=safran, var='analysis')
#myplot.plot()
#plot(antilope=antilope, nivometeo=nivometeo, auto=auto, safran=safran, var='analysis')
#plot(antilope=antilope, nivometeo=nivometeo, auto=auto, safran=safran, var='obs')
#plot(antilope=antilope, nivometeo=nivometeo, auto=auto, safran=safran, var='rr')

