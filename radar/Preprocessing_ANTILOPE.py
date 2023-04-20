#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 20/04/2023

# Script de pre-processing des champs de précipitation en 24h estimés par ANTILOPE
# 1. débiaisage statique avec le champs de ratio estimé
# 2. Localisation
# 3. [Assimilation des obs nivométéo par un filtre de Kalman] (si obs disponibles)

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
from scipy.sparse import csr_matrix, csc_matrix, diags
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.sparse.linalg import inv, spsolve


def usage():
    print("USAGE Preprocessing_ANTILOPE.py date")
    print("format de la date : YYYYMMDD (précipitations de YYYYMMD-1 6h à YYYYMMDD6H")
    sys.exit(1)

try:
    date = datetime.datetime.strptime(sys.argv[1], '%Y%m%d')
except Exception as e:
    usage()
    raise e

datebegin = date.replace(hour=6)
dateend   = datebegin+datetime.timedelta(days=1)

datadir = '/home/vernaym/workdir/visualisation'
domain = 'alp'
ld = 0.05
max_dist = ld*3

def get_antilope():
    # TODO : appliquer le pré-processing pour plotter ANTILOPE corrigé
    filename = os.path.join(datadir, f'ANTILOPEH_2023041306_2023041606_alp.nc')
    #if os.path.exists(filename):
    antilope = xr.open_dataset(os.path.join(datadir, filename))
    # TODO : gérer le changement d'heure !

    if 'analysis' not in antilope.variables.keys():
        antilope = antilope.where((antilope.time>np.datetime64(datebegin)) & (antilope.time<=np.datetime64(dateend)), drop=True).sum('time')

        # 1. Static de-biasing :
        mask = xr.open_dataset(os.path.join(datadir, f"Estimated_ratio.nc"))

        antilope["ratio"]=mask.ratio  # Fill missing point with NaNs
        antilope["rr_debiaise"] = (antilope.rr/antilope.ratio).fillna(antilope.rr)  # Fill NaN values with the original ANTILOPE value

        # 2. Dynamic correction (localisation)
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

    # 3. Nivometeo Assimilation
    nivometeo_assimilation(antilope)

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

def get_nivometeo():
    fic_score = os.path.join(datadir, 'nivometeo.data')
    nivometeo = pd.read_csv(fic_score, sep=';', parse_dates=['date'],
            dtype={'num_poste':int, 'nom':str, 'alti':int, 'lat':float, 'lon':float, 'massif':int, 'rr': float, 'reseau_poste': int}, na_values=['--'])
    nivometeo = nivometeo.loc[nivometeo["date"]==np.datetime64(date)]
    return nivometeo

def nivometeo_assimilation(antilope):
    """
    Use Kalman Filter : a = x + BH'(HBH'+R)⁻¹(y-Hx)
    x = ANTILOPE (Background)
    B = Backroud ECM (static ANTILOPE error)
    y = obs nivométéo
    R = Observation error (TODO : à définir (erreur fixe incluant erreur de représentativité et incertitude sur la mesure ?)
    H = Observation operator
    """

    # Read ANTILOPE error (=Background error !)
    error = xr.open_dataset(os.path.join(datadir, 'Observation_error.nc'))

    # Background
    # WARNING : on ne peut appliquer l'analyse que sur le domaine où l'erreur d'ANTILOPE a été estimée !!!
    x = antilope.obs.sel({'lat':error.lat, 'lon':error.lon}).data.flatten()

    #Read nivometeo observations
    nivometeo = get_nivometeo()
    y = nivometeo.rr.to_numpy()

    # Construction of the observation operator
    # - Put nivometeo observations on the ANTILOPE grid
    nivometeo.lat = nivometeo.lat.round(2)
    nivometeo.lon = nivometeo.lon.round(2)
    Hx = antilope.obs.sel(lat=nivometeo.lat.to_xarray(), lon=nivometeo.lon.to_xarray(), method = 'nearest').data

    X,Y = np.meshgrid(error.lon.data, error.lat.data)
    X = X.flatten()
    Y = Y.flatten()
    #coords = [elem for elem in zip(X,Y)]
    #points = [elem for elem in zip(nivometeo.lon, nivometeo.lat)]
    H = np.zeros((len(y), len(x)))  # n*k matrix
    # Update H matrix with 1 where an observation is present
    for i,idx in enumerate(nivometeo.index):
        lon = nivometeo.loc[idx, 'lon']
        lat = nivometeo.loc[idx, 'lat']
        H[i, np.where((X==lon) & (Y==lat))[0]] = 1  # update H matrix
    Ht = np.transpose(H)
    H  = csr_matrix(H)
    Ht = csr_matrix(Ht)


    # Correlation matrix
    filename = os.path.join('/home/vernaym/These/DATA', f'codistance_max_dist_{max_dist}_{domain}.npz')
    if not os.path.exists(filename):
        # Compute inter-distances
        coords=[(lon,lat) for lat in error.lat.data for lon in error.lon.data]
        codist = codistances(coords)
        scipy.sparse.save_npz(filename, codist, compressed=False)
    else:
        codist = scipy.sparse.load_npz(filename)

    # Background (ANTILOPE) ECM
    std = diags(error.ratio.data.flatten(), 0)
    B = std.dot(codist.dot(std))
    B = csr_matrix(B)

    # Observation ECM
    R = diags(0.1+y/100)  # Uniform and uncorrelated 1% + 0.1 mm error for nivometeo observations

    # Analysis
    HB = H.dot(B)
    HBHt = HB.dot(Ht)
    K = B.dot(Ht).dot(scipy.sparse.linalg.inv(csc_matrix(HBHt+R)))
    A = x + K.dot(y-Hx)

#   Solution to avoid the "B+R" matrix inversion :
#   1. solve (B+R).Z=Y-X
#   --> the matrix is already "band diagonal" but could be converted using a
#   reverse_cuthill_mckee algorithm
#   --> use "spsolve" method for sparse matrices (solveh_banded for dense
#   matrices)
#    Z = spsolve(HBHt+R, y-Hx)
#   2. compute anlaysis as A=X+BZ
#    A = X+HB.dot(Z)

    antilope['analysis'] = xr.DataArray(
            data   = A.reshape((len(error.lat), len(error.lon))),
            dims   = ["lat", "lon"],
            coords = dict(lon=error.lon, lat=error.lat)
        )

    antilope['analysis'] = antilope.analysis.fillna(antilope.rr)  # Fill Nan values with previous ones

    filename = os.path.join(datadir, f'ANTILOPE.nc')
    antilope.to_netcdf(filename)

    #TODO : Update Error covariance matrix

get_antilope()
