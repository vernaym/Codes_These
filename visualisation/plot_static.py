#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import xarray as xr
import plotly.graph_objects as go

from plot_precip_map import PrecipitationAnalysis

datadir = '/home/vernaym/workdir/ASSIMILATION/mask/alp'

def add_radar(fig):
    text = ['Le Moucherotte', 'Colombis', 'La Dole']
    fig.add_trace(
        go.Scattermapbox(
            lon    = [5.63933, 6.21729, 6.10001],
            lat    = [45.14776, 44.49664, 46.42565],
            mode   = 'markers+text',
            text   = text,
            marker = dict(
                size   = 12,
                symbol = 'triangle',
                color  = 'black',
            )
        )
    )


#ratio = os.path.join(datadir, 'Estimated_ratio.nc')
#ds = xr.open_dataset(ratio)
#df = ds.to_dataframe().reset_index()
#myplot = PrecipitationAnalysis()
#newtrace = myplot.add_scatter('Ratio', colorbar_title='Estimated ratio', df=df, var='Ratio', colorscale='RdBu_r')
#myplot.fig.add_trace(newtrace)
#add_radar(myplot.fig)
#myplot.update_figure(title=False)
#myplot.save(savename='Ratio.html', json=False)

error = os.path.join(datadir, 'Observation_uncertainty.nc')
ds = xr.open_dataset(error)
df = ds.to_dataframe().reset_index()
myplot = PrecipitationAnalysis()
newtrace = myplot.add_scatter('Uncertainty', colorbar_title='Estimated uncertainty', df=df,
        var='Uncertainty', colorscale='YlOrBr')
myplot.fig.add_trace(newtrace)
add_radar(myplot.fig)
myplot.update_figure(title=False)
myplot.save(savename='Uncertainty.html', json=False)
