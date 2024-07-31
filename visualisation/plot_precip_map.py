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

from netrc import netrc


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

ld = 0.1
max_dist = ld*2
#datadir = '/home/vernaym/workdir/visualisation'
rootdir = '/d0/intra-cen/ANTILOPE'  # On sxcen
if not os.path.exists(rootdir):
    rootdir = '.'
datadir = '.'
token = open("/home/vernaym/.mapbox/token").read() # Token from mapbox account

class PrecipitationAnalysis(object):


    def __init__(self, date, antilope=None, safran=None, nivometeo=None, auto=None, var='obs'):
        self.antilope = antilope
        self.rrmax = min([max([np.nanmax(antilope.rr.data.flatten()) for antilope in self.antilope.values() if antilope is not None]), 80])
        #self.rrmax = max([np.nanmax(antilope.analysis.data.flatten()) for antilope in self.antilope.values() if antilope is not None])
        self.var = var
        self.safran = safran
        self.nivometeo = nivometeo
        # TODO : concatener les df des obs auto
        self.auto = pd.concat([df for df in auto.values()])
        self.date = date
        self.datebegin = date.replace(hour=6)
        self.dateend   = self.datebegin+datetime.timedelta(days=1)
        #self.domain = domain
        self.massifs = None

        self.fig = go.Figure()  # Figure initialisation

    def plot(self):
        """ 
        Plot from shapfile
        https://stackoverflow.com/questions/71780189/how-to-show-only-boundaries-no-fill-of-a-shapefile-in-python-plotly-express
        """

        self.ntrace = 0

        # 1. SAFRAN (plot first to be bellow other layers)
        if self.safran is not None:
            self.plot_safran()
            self.ntrace += 1
        else:
            self.read_safran_massifs()

        # 2. ANTILOPE
        if self.antilope is not None:
            self.plot_antilope()

        # 3. Nivométéo observations
        self.add_ponctual_obs(self.nivometeo, color='red', name='Nivometeo')
        self.ntrace += 1

        # 4. Automatic observations
        self.add_ponctual_obs(self.auto, color='black', name='Automatic stations')
        self.ntrace += 1

        self.update_figure()

    def show(self):
        self.fig.show()

    def save(self):
        self.jsonfile = os.path.join(rootdir, 'figures', "as_antilope.json")
        self.fig.write_json(self.jsonfile)
        if self.var == 'analysis':
            self.fig.write_html(os.path.join(rootdir, 'figures', f"precipitation_{self.date.strftime('%Y%m%d')}.html"))
        else:
            self.fig.write_html(os.path.join(rootdir, 'figures', f"precipitation_{self.var}_{self.date.strftime('%Y%m%d')}.html"))

    def put_ftp(self):
        import pysftp
        host = "coalpm31-sidev.meteo.fr"
        username, account, password = netrc().authenticators(host)
        with pysftp.Connection(host=host, username=username, password=password) as sftp:
            with sftp.cd('/home/users/coalp-adm/external/antilope'):  # temporarily chdir to public
                sftp.put(self.jsonfile)  # put json file

    def plot_antilope(self):
        """
        Plot ANTILOPE as multiple dot scatterplot (depending on the elevation)
        """

        x = np.array([])
        y = np.array([])
        rr = np.array([])
        error = np.array([])
        alti = np.array([])
        for domain in self.antilope.keys():
            # 1. read_relief (TODO : add to ANTILOPE pre-processing ?)
            #mnt = xr.open_dataset(f"/home/vernaym/These/DATA/DEM_{domain.upper()}_WGS84_1km.nc")
            mnt = xr.open_dataset(f"DEM_{domain.upper()}_WGS84_1km.nc")
            # WARNING : update of xarray necessary !
            #data = antilope.interp(lat=mnt.lat.data, lon=mnt.lon.data)
            tmp = mnt.interp(lat=self.antilope[domain].lat.data, lon=self.antilope[domain].lon.data)
            self.antilope[domain]["elevation"] = tmp.elevation

            X,Y  = np.meshgrid(self.antilope[domain].lon.data, self.antilope[domain].lat.data)
            x = np.concatenate((x, X.flatten()))
            y = np.concatenate((y, Y.flatten()))
            rr = np.concatenate((rr, self.antilope[domain][self.var].data.flatten()))  # Corrected obs
            error = np.concatenate((error, self.antilope[domain]['error'].data.flatten()))
            alti = np.concatenate((alti, self.antilope[domain]['elevation'].data.flatten()))

        self.df = pd.DataFrame(
                data    = np.transpose([x, y, rr, error, alti]),
                columns = ['lon', 'lat', 'rr', 'error', 'alti'],
                index   = range(len(rr)),
            )
        select = dict()
        visible ='legendonly'  # Does not work as intended : https://community.plotly.com/t/legendonly-doesnt-work-anymore-in-scattermapbox/72822
        visible = True

        #mask  = np.where((rr>=0.1) & (~np.isnan(error)))
        self.df = self.df[self.df['rr']>0]
        name = 'AS-ANTILOPE'
        self.scaletrace = self.ntrace
        self.fig.add_trace(self.add_antilope_scatter(name, showscale=True, uncertainty=True))
#
#        elevation_range = [0, 500, 1000, 1500, 2000, 2500, 3000]
#        for i,elevation in enumerate(elevation_range):
#        # Avoid to plot data multiple time --> plot by elevation bands
#        # Plotting the full 1-km ANTILOPE domain takes about 12M memory...
#        # https://plotly.com/python/v3/selection-events/
#            if elevation == 0:
#                showscale = True
#                self.scaletrace = self.ntrace.copy()
#            else:
#                showscale = False
#            z0 = elevation_range[i]
#            if i+1 < len(elevation_range):
#                z1 = elevation_range[i+1]
#                mask  = np.where((rr>0.05) & (alti>=z0) & (alti<z1) & (~np.isnan(error)))
#            else:
#                mask  = np.where((rr>0.05) & (alti>=z0) & (~np.isnan(error)))
#            name    = f'ANTILOPE>{z0:d}m'
#
#            self.fig.add_trace(self.add_antilope_scatter(df, name, mask, showscale=showscale, uncertainty=True))
#            self.ntrace += 1
#            #if i == 0:
#                # TODO : useless (same data ==> use a button !)
#            #    self.fig = self.fig.add_trace(self.add_antilope_scatter(df, name, select[i], uncertainty=False))

    def add_antilope_scatter(self, name, df=None, showscale=False, uncertainty=True, visible=True):
        # Normalisation de l'erreur entre low and high
        # Linear decrease between high and low marker size values
        low  = 3
        high = 15
        def nan_ptp(a):
            return np.ptp(a[np.isfinite(a)])
        rr = self.df.rr.values
        error = self.df.error.values
        self.errorsize = low + (high-low)*np.exp(-error/(rr+0.01))
        self.errorsize[self.errorsize<low] = low
        #error = low + (error - np.nanmin(error))/(nan_ptp(error)/high)
        #error = low + high/error

        if df is None:
            df = self.df


        return go.Scattermapbox(
                    lon  = df.lon.values,
                    lat  = df.lat.values,
                    #selectedpoints = select,  # TODO : use this footprint to set updatemenus buttons DOES NOT WORK (because it refers to user selected points)
                    mode = 'markers',
                    name = name,
                    #text = antilope.rr.data.flatten(),  # obs=corrected obs, rr=raw obs
                    text = df.rr.values,
                    visible = visible,
                    showlegend = True,
                    #selected = go.scattermapbox.Selected(marker={"size":50}),
                    customdata = np.stack((df.alti.values, df.rr.values, df.error.values), axis=-1),
                    hovertemplate =
                        '<b>Altitude (m)</b>: %{customdata[0]:d}m<br>'+
                        '<b>Precipitation (mm)</b>: %{customdata[1]:.2f}mm<br>'+
                        '<b>Incertitude (mm)</b>: %{customdata[2]:.2f}mm<br>',
                    marker = dict(
                        #color = antilope.rr.data.flatten(),
                        color = df.rr.values,
                        cmin  = 0,
                        #cmax  = np.nanmax(self.antilope.rr.data.flatten()),
                        cmax  = self.rrmax,
                        size  = np.nan_to_num(self.errorsize, nan=5) if uncertainty else 10,
                        #opacity=0.5,
                        #colorscale = 'YlGnBu',
                        #colorscale = 'dense',
                        colorscale = 'Jet',
                        #symbol='square',  # Impossible to change if color is defined : https://stackoverflow.com/questions/59628536/option-symbol-in-scattermapbox-is-not-working
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

    def add_ponctual_obs(self, data, color='black', name='Unknown'):
        """
        Add ponctual rain gauges (red for nivometeo, black for other networks)
        """

        if data is not None:
            self.fig.add_trace(go.Scattermapbox(
                        #data,
                        lon  = data.lon.values,
                        lat  = data.lat.values,
                        text = data.rr.round(1).astype('string'),  # WARNING : working only with token mapbox :  https://plotly.com/python/mapbox-layers/
                        #text = data.rr.values.round(1),  # WARNING : working only with token mapbox :  https://plotly.com/python/mapbox-layers/
                        mode = 'text',
                        name = name,
                        textfont = dict(size=16, family='Arial', color=color),
                        textposition = 'middle center',
                        hoverinfo = 'text',
                        #hover_data=[data.num_poste, data.nom],
                        #hovertext = [data.num_poste, data.nom],
                        customdata = np.stack((data.num_poste, data.nom, data.alti, data.reseau_poste), axis=-1),
                        #customdata = [data.num_poste, data.nom],
                        #hovertemplate="<br>".join([
                        #    f"Num poste: : %{customdata[0]}",
                        #    f"Nom : %{customdata[1]}",
                        #]),
                        hovertemplate =
                            '<b>Num poste</b>: %{customdata[0]:d}<br>'+
                            '<b>Nom</b>: %{customdata[1]}<br>'+
                            '<b>Altitude</b>: %{customdata[2]}m<br>'+
                            '<b>Réseau</b>: %{customdata[3]}<br>',
                        #hovertext=data.num_poste,
                        #hovertext=[data.num_poste, data.nom],
                        # TODO : formater le texte flottant : "Nom (num_poste)"
                        #hovertemplate = '',
                    )
                )

    def read_safran_massifs(self):
        # 1. read massif shapefile
        massifs_json = os.path.join(datadir, "massifs_safran.json")
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
        self.read_safran_massifs()

        with open(massifs_json) as geofile:
            j_file = json.load(geofile)
            i = 0
            for feature in j_file["features"]:
                feature['id'] = self.massifs.code[i]
                i += 1

            massif_number = np.array([])
            rr = np.array([])
            for domain in self.safran.keys():
                # TODO : add dynamic elevation choice
                if self.safran[domain] is not None:
                    safran = self.safran[domain].where(self.safran[domain].ZS==1800., drop=True)
                    massif_number = np.concatenate((massif_number, safran.massif_number))
                    rr = np.concatenate((rr, safran.rr.data))

            self.fig.add_trace(go.Choroplethmapbox(
                geojson = j_file,
                locations = massif_number,
                z = rr,  #TODO : add elevation choice
                #colorscale = 'YlGnBu',
                colorscale = 'dense',
                name = 'SAFRAN (1800m)',
                zmin = 0,
                #zmax = np.nanmax(antilope.rr.data.flatten()),
                zmax = self.rrmax,
                visible = False,
                uid = 4,
                uirevision = True,
                showscale = False,  # Same scale as ANTILOPE data (à vérifier !)
                showlegend = True,
                #opacity=0.5,
            ))

    def marker_prop(self, colorscale=None, size=None):

        if colorscale is None:
            colorscale = 'dense'
        if size is None:
            size = np.nan_to_num(self.errorsize, nan=5)

        marker = dict(
                color = self.df.rr.values,
                #autocolorscale = False,
                colorscale = colorscale,
                showscale = True,
                cmin  = 0,
                #cmax  = np.nanmax(self.antilope.analysis.data.flatten()),
                cmax  = self.rrmax,
                size  = size,
                #symbol ='square',  # Impossible to change if color is defined : https://stackoverflow.com/questions/59628536/option-symbol-in-scattermapbox-is-not-working
                colorbar_title = "Precipitation(mm)",
                colorbar = dict(
                    titleside = "right",
                    ticks = "outside",
                    # Move the colorbar away from the legend :
                    yanchor="top",
                    y=1,
                    x=-0.1,
                )
            )
        return marker

    def update_figure(self):
        """
        Update figure layout
        """


        # 1. Button to set colorscale
        buttons = list()
        # See https://plotly.com/python/reference/scattermapbox/ for possible colorscales
        #for colorscale in ['Dense', 'Rainbow', 'dense', 'viridis', 'Viridis', 'HSV', px.colors.sequential.dense]:
        for colorscale in ['Blackbody','Bluered','Blues','Cividis','dense','Earth','Electric','Greens','Greys','Hot','Jet','Picnic','Portland','Rainbow','RdBu','Reds','Viridis','YlGnBu','YlOrRd']:
            buttons.append(
                    dict(
                        # See https://stackoverflow.com/questions/73435977/change-colorscale-of-marker-with-update-menu-without-repeating-data
                        args = [{'marker.colorscale':colorscale}, [self.scaletrace]],
                        label = f"{colorscale}",
                        #method = "relayout",
                        method = "restyle",
                        #method="update",
                    )
                )

        updatecolorbar = dict(
            buttons = buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.03,
            xanchor="left",
            y=0.45,
            yanchor="top"
        )

        # 2. Button to reverse colorscale
        buttons=list([
            dict(
                args=[{'marker.reversescale':False}],
                args2=[{'marker.reversescale':True}],
                label="Reverse colorscale",
                method="restyle"
            ),
        ])

        updatecolorscaledirection = dict(
            type='buttons',
            buttons = buttons,
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.03,
            xanchor="left",
            y=0.4,
            yanchor="top"
        )

        # 3. Button to switch on/off error dependent marker size
        buttons = list([
                dict(
                    args = [{'marker.size':20}, [self.scaletrace]],
                    args2 = [{'marker.size':np.nan_to_num(self.errorsize, nan=5)}, [self.scaletrace]],
                    label  = 'Uncertainty',
                    method = 'restyle',
                    #method = 'update',
                    ),
            ])

        updateuncertainty = dict(
            type='buttons',
            buttons = buttons,
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.03,
            xanchor="left",
            y=0.3,
            yanchor="top"
        )

        # 4. Button to filter data by elevation
        buttons = list()
        elevation_range = [0, 500, 1000, 1500, 2000, 2500, 3000]
        for i,elevation in enumerate(elevation_range):
            df = self.df[self.df["alti"]>=elevation]
            mask = np.where(self.df["alti"]>=elevation)
            buttons.append(
                    dict(
                        label  = f'>{elevation}m',
                        #method = "restyle",  # modify data or data attributes
                        #method = "relayout",  # modify layout attributes
                        method = "update",  # modify data and layout attributes; combination of "restyle" and "relayout"
                        args = [dict(selectedpoints=mask, unselected=dict(marker=dict(opacity=0))), [self.scaletrace]],
                        #args = [dict(selectedpoints=mask, unselected=dict(marker=dict(opacity=0)))],
                    )
                )

        # TODO : https://stackoverflow.com/questions/61556618/plotly-how-to-display-and-filter-a-dataframe-with-multiple-dropdowns
        elevationfilter = dict(
                type="buttons",
                buttons=buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.03,
                xanchor="left",
                y=0.8,
                yanchor="top"
            )

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
            #updatemenus = [updatecolorbar],
            updatemenus = [elevationfilter, updatecolorbar, updateuncertainty, updatecolorscaledirection],
            #updatemenus = [elevationfilter, updatecolorbar],
        )

        self.fig.update_layout(
            annotations=[
                dict(text="Elevation filter :", x=1.03, xref="paper", y=0.82, xanchor="left",
                                     yanchor="top", align="center", yref="paper", showarrow=False),
                dict(text="colorscale :", x=1.03, xref="paper", y=0.47, yref="paper", xanchor="left",
                                     yanchor="top", align="center", showarrow=False),
                dict(text="Marker size :", x=1.03, xref="paper", y=0.32, yref="paper", xanchor="left",
                                     yanchor="top", align="center", showarrow=False)
            ])


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

