#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 11/09/2023

import os, sys
from datetime import datetime,timedelta
import pandas as pd  # Version 0.25.3
import xarray as xr
import numpy as np

import scipy
from scipy.sparse import csr_matrix, csc_matrix, diags
from scipy.spatial.distance import cdist

from These.radar import Preprocessing_ANTILOPE

datadir = '/home/vernaym/These/DATA'

latmin = 45.24
lonmin = 6.01
latmax = 44.99
lonmax = 6.49

margin = 0.2  # Margin on the domain's borders to account for spatial correlations

if __name__ == "__main__":

    # Read raw ANTILOPE file
    filename = 'ANTILOPEQ_2017073106_2020080106_ange.nc'
    antilope = xr.open_dataset(os.path.join(datadir, filename))
    antilope["rr_corrected"] = antilope.rr.copy()

    # Read ratio/error files
    ratio = xr.open_dataarray(os.path.join(datadir, f"Estimated_ratio.nc"))
    ratio = ratio.sel(({'lat':np.intersect1d(ratio.lat.data, antilope.lat.data), 'lon':np.intersect1d(ratio.lon.data, antilope.lon.data)}))
    error = xr.open_dataarray(os.path.join(datadir, 'Observation_error.nc'))
    error = ratio.sel(({'lat':np.intersect1d(error.lat.data, antilope.lat.data), 'lon':np.intersect1d(error.lon.data, antilope.lon.data)}))

    # Read correlation matrix
    codist = os.path.join(datadir, f'codistance_max_dist_20_ange.npz')
    if not os.path.exists(codist):
        # Compute inter-distances
        coords=[(lon,lat) for lat in error.lat.data for lon in error.lon.data]
        pond = Preprocessing_ANTILOPE.codistances(coords)
        scipy.sparse.save_npz(codist, pond, compressed=False)  # TODO comprendre pourquoi ca ne marche pas pour éviter de recalculer les codistances à chaque fois
    else:
        pond = scipy.sparse.load_npz(codist)
    pond = pond.dot(diags(1/error.data.flatten(), 0))  # std is in [1, inf[

    for date in antilope.time.data:
        tmp = antilope.sel(time=date)

        # 1. de-biasing
        tmp["rr_debiaise"] = (tmp.rr/ratio).fillna(tmp.rr)  # Fill NaN values with the original ANTILOPE value

        # 2. Dynamic correction (localisation)
        new, mean, sd = Preprocessing_ANTILOPE.dynamic_correction(tmp.rr_debiaise.data.flatten(), pond)  # Update obs
        tmp['obs'] = xr.DataArray(
                data   = new.reshape((len(tmp.lat), len(tmp.lon))),
                dims   = ["lat", "lon"],
                coords = dict(lon=tmp.lon, lat=tmp.lat)
            )
        #antilope['obs'] = antilope['obs'].fillna(antilope.rr)
        #antilope = antilope.rename({'obs':'analysis'})

        # Save corrected field
        antilope.sel(time=date)['rr_corrected'] = tmp.rr_corrected

    antilope.to_netcdf(os.path.join(datadir, 'ANTILOPEQ_2017073106_2020080106_corrected_ange.nc'))
