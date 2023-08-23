#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 20/04/2023

# Script de pre-processing des champs de précipitation en 24h estimés par ANTILOPE
# 1. débiaisage statique avec le champs de ratio estimé
# 2. Localisation
# 3. [Assimilation des obs nivométéo par un filtre de Kalman] (si obs disponibles)

import os, sys
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

import matplotlib
import matplotlib.pyplot as plt

import These.scripts.cas_test as ct

import scipy
from scipy.sparse import csr_matrix, csc_matrix, diags
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.sparse.linalg import inv, spsolve

import vortex
from bronx.stdtypes.date import Date, Period

#def usage():
#    print("USAGE Preprocessing_ANTILOPE.py date")
#    print("format de la date : YYYYMMDD (précipitations de YYYYMMD-1 6h à YYYYMMDD6H")
#    sys.exit(1)
#
#try:
#    date = datetime.datetime.strptime(sys.argv[1], '%Y%m%d')
#except Exception as e:
#    usage()
#    raise e

#datadir = '/home/vernaym/workdir/visualisation'
datadir = '/home/vernaym/extraction_obs'  # On sxcen
workdir = '.'  # On sxcen

domain = 'alp'
ld = 0.05
max_dist = ld*3


def dynamic_correction(field, pond, weight=None, super_ensemble=None, plot=False):
    """
    * field          : 2D (n*k) array containing the field to modify
    * pond           : (nk*nk) sparse ponderation matrix (each line gives the correlation between the corresponding pixel
                        and every other pixel of the domain). It is defined using the confidenc of each pixel and the
                       distance between the pixels
    * weight         : nk vector of the sum of the weights in the neighborhood of each pixel
    * super_ensemble : nk*nk mask matrix ("M") defining neighbor pixels to include in the computation
    """

    field[np.isnan(field)] = 0.0
    initial_field = field.flatten()
    X = diags(field.flatten(), 0)

    # 1. Calcul de la moyenne pondérée par la distance ET l'erreur statique
    if super_ensemble is None:
        super_ensemble = pond.copy()  # WARNING : make a copy or pond will change when super_ensemble changes
        super_ensemble[super_ensemble.nonzero()] = 1  # Position of pixels to inclue in the spread computation
    if weight is None:
        pond.data[np.isnan(pond.data)] = 0.0
        weight = pond.sum(axis=1).A1  # The sum of the weights (axis=1 <==> sum over rows)

    mean = pond.dot(X).sum(axis=1).A1  # getA1 transforms the 1*N matrix object into a 1D np.array
    mean = mean / weight
    pixel_weight = pond.diagonal()  # = "exp(-err)" ou "1/err" pour l'obs et "likelyhood" du pixel pour les membres de l'ensemble
    #weight = pond.sum(axis=1).A1
    nb_nonzero = (pond != 0).sum(0).A1  # Count non zero elements of each row
    meanweight = weight / nb_nonzero
    # pixel_weight is in ]0, 1]
    # Do not try to preserve values of pixels with low errors on average : the dynamic correction must account
    # for temporary failures as well as uncertainties due to the error estimation method
    #newfield = (initial_field * pixel_weight + mean * meanweight) / (pixel_weight + meanweight)  # Stay closer to the original value (spatial structures can still be visible)
    newfield = (initial_field * pixel_weight + mean * meanweight/pixel_weight) / (pixel_weight + meanweight/pixel_weight)  # Smoother fields --> underestimation of extreme values
    #newfield = (initial_field * pixel_weight + mean * meanweight/(meanweight+pixel_weight)) / (pixel_weight + meanweight/(meanweight+pixel_weight))
    #newfield = (initial_field * pixel_weight + mean * meanweight/(meanweight+pixel_weight)) / (pixel_weight + meanweight/(meanweight+pixel_weight))
    newfield = np.round(newfield, 1)

    # Plot correction coefficient
    if plot:
        correction_coefficient = (meanweight/pixel_weight * 1 / (pixel_weight + meanweight/pixel_weight)).reshape(np.shape(field))
        filename = 'Correction_weight.pdf'
        ct.plot_field(correction_coefficient, filename, label='Correction coefficient', cmap=plt.cm.viridis, vmin=0, vmax=1, add_circle=True)

        # Plot original value coefficient
        original_value_coefficient = (pixel_weight / (pixel_weight + meanweight/pixel_weight)).reshape(np.shape(field))
        filename = 'Original_value_weight.pdf'
        ct.plot_field(original_value_coefficient, filename, label='Original value coefficient', cmap=plt.cm.viridis, vmin=0, vmax=1, add_circle=True)

    #sd = get_std(X, newfield, pond, weight=weight, super_ensemble=super_ensemble)  # Dispersion of the super ensemble
    # TODO : Objective evaluation of formulas 1 and 2
    sd1 = get_std(X, initial_field, pond, weight=weight, super_ensemble=super_ensemble)  # 1. Dispersion of the super ensemble around the initial field --> More dispersion on high error pixels (--> spatial structures)
    sd2 = get_std(X, mean, pond, weight=weight, super_ensemble=super_ensemble)  # 2. Dispersion of the super ensemble around the mean --> Smoother fields
    #sd3 = np.abs(initial_field-newfield)  # Obs displacment  --> Apparition of spatial structures

    #sd = sd + 1  # Add 1 to ensure that the error is >1 (mm or mm^(1/2)). --> Dispersion too large
    #sd = (sd1+sd2)/2
    #sd = sd1/2+sd2
    sd = sd2

    return newfield, mean, sd

def get_std(data, mean, pond, weight=None, super_ensemble=None):
    """
    INPUT
    -----
    * mean is the Nk*Nk matrix
    * pond is the Nk*Nk ponderation matrix (defined by the confidence of each pixel and the distance between the pixels)
    * weight is the Nk vector of the sum of the weights in the neighborhood of each pixel
    * super_ensemble is the mask Nk*Nk matrix ("M") defining neighbor pixels to include in the computation

    OUTPUT
    ------
    * sd  : Nk vector of the dispersion of the neighborhood of each pixel of the domain (in mm)

    """

    se_mean = diags(mean, 0).dot(super_ensemble)  # matrix with mean[i] at each non-zero element of line i of super_ensemble
    X = super_ensemble.dot(data)-se_mean  # M.diag(obs)-diag(mean).M  --> difference between each neighbor value and the neighborhood mean
    sd = X.multiply(X).multiply(pond).sum(axis=1).getA1()  # Ponderation of the squared difference by the confidence (pond) + sum over all neighbor values
    sd = sd / weight  # Normalisation with the total weight in the neighborhood
    sd = np.sqrt(sd)
    sd = np.nan_to_num(sd)  # replace nan values by 0

    return sd

def codistances(coords, ld=0.07):
    """
    Solution pour le calcul des inter-distances trouvée sur : https://stackoverflow.com/questions/35296935/python-calculate-lots-of-distances-quickly
    """

    #max_dist = ld*3  # exp(-2)=0.14, exp(-3)=0.05 ==> facteur 3 pour ignorer les pixels avec un poid < 5%
    max_dist = ld*2  # exp(-2^2)=0.018 ==> facteur 2 pour ignorer les pixels avec un poid < 2%
    #max_dist = ld
    tree = cKDTree(coords)
    dist = tree.sparse_distance_matrix(tree, max_distance=max_dist, p=2, output_type='coo_matrix')
    dist = csr_matrix(dist)
    #TODO : utiliser une gaussienne plutot qu'une exponentielle décroissante ?
    dist[dist.nonzero()] = dist[dist.nonzero()]/ld
    #dist[dist.nonzero()] = 1/dist[dist.nonzero()]
    #np.exp(-dist.data, out=dist.data )
    np.exp(-dist.data**2/2, out=dist.data )

    return dist

def random_draw(obs, sd):
    gauss = np.random.normal(loc=0.0, scale=1.0, size=1)[0]  # Draw random element from normal distribution
    exp = np.random.default_rng().exponential(scale=1)  # TODO : set scale parameter using the density of pixels at 0mm in the vicinity ?

    # Ensure that RR are >=0
    # ==> Draw from gama distribution ? ==> Not a good idea since the conversion to square root precipitation aims at
    # normalising the distribution
    #ana = obs+gauss*sd/5
    #sd = np.sqrt(sd)  # sigma --> sigma² dans la formulation de la loi normale
    # TODO : comprendre pourquoi la conversion R^1/2 --> R disperse autant l'ensemble
    #ana = np.square(obs)+gauss*sd  # Gaussian perturbation around >0 obs
    ana = obs+gauss*sd  # Gaussian perturbation around >0 obs
    #ana[ana<0] = exp*sd[ana<0]  # Avoid "mass accumulation" in 0. !! WARNING : the analysis distribution is not Normal anymore !!
    # TODO : different treatment of values <1 and >1 since conversion RR^1/2 --> RR completly changes the behavior
    ana[ana<0] = 0  # WARNING : "mass accumulation" in 0 (analysis distribution not normal anymore)
    ana[obs==0] = obs[obs==0]+exp*sd[obs==0]  # Exponential perturbation arround 0. TODO : arround 0, use the density
    ana = np.round(ana, 1)
    # of pixels at 0mm in the vicinity instead of sd ?
    #ana[ana<0] = 0

    return ana
    #return np.square(ana)


class AntilopePreprocessing(object):

    def __init__(self, date, domain, filename):
        self.date = date
        self.datebegin = date.replace(hour=7)
        self.dateend   = self.datebegin+Period(hours=23)
        #self.dateend   = self.datebegin+datetime.timedelta(days=1)  # TODO : extract only up to 6h
        self.domain = domain
        self.filename = filename
        self.run()

    def run(self):
        #filename = os.path.join(datadir, f'ANTILOPEH_{self.datebegin.strftime("%Y%m%d%H")}_{self.dateend.strftime("%Y%m%d%H")}_{self.domain}.nc')  # TODO : extract only up to 6h
        #if os.path.exists(filename):
        antilope = xr.open_dataset(self.filename)
        # TODO : gérer le changement d'heure !

        if 'analysis' in antilope.variables.keys():
            antilope = antilope.rename({'analysis':'analysis_save'})  # !!!!! TODO : TMP !!!!!!!
            antilope = antilope.rename({'rr':'analysis'})

        if 'analysis' not in antilope.variables.keys():  # File already pre-processed
            if 'time' in antilope.coords:
                antilope = antilope.where((antilope.time>np.datetime64(self.datebegin)) & (antilope.time<=np.datetime64(self.dateend)), drop=True).sum('time')  # Security ?

            # TODO : séparer les méthodes plus clairement pour pouvoir les appliquer indépendament les unes des autres
            # TODO : commencer par la correction dynamique puis appliquer le débiaisage (facteur à modifier pour prendre en compte le biais moyen après correction ?)

            # 1. Static de-biasing :
            antilope = self.debiasing(antilope)
            mask = xr.open_dataset(os.path.join(workdir, f"Estimated_ratio.nc"))
            #mask = xr.open_dataset(os.path.join("/home/vernaym/workdir/ASSIMILATION/mask/alp", f"Estimated_ratio.nc"))  # !!!!! TODO : TMP !!!!!

            antilope["ratio"] = mask.ratio  # Fill missing point with NaNs
            antilope["rr_debiaise"] = (antilope.rr/antilope.ratio).fillna(antilope.rr)  # Fill NaN values with the original ANTILOPE value

            # 2. Dynamic correction (localisation)
            #antilope['error'] = xr.open_dataset(os.path.join(datadir, 'Observation_error.nc'))
            error = xr.open_dataarray(os.path.join(workdir, 'Observation_error.nc'))
            #error = xr.open_dataarray(os.path.join("/home/vernaym/workdir/ASSIMILATION/mask/alp", 'Observation_error.nc'))
            std = error.data
            antilope['error'] = np.abs(error)
            codist = os.path.join(workdir, f'codistance_max_dist_{max_dist:.2f}_{domain}.npz')
            if not os.path.exists(codist):
                # Compute inter-distances
                coords=[(lon,lat) for lat in error.lat.data for lon in error.lon.data]
                pond = codistances(coords)
                scipy.sparse.save_npz(codist, pond, compressed=False)  # TODO comprendre pourquoi ca ne marche pas pour éviter de recalculer les codistances à chaque fois
            else:
                pond = scipy.sparse.load_npz(codist)
            pond = pond.dot(diags(np.exp(-(std-1)).flatten(), 0))  # std is in [1, inf[
            obs = antilope.rr_debiaise.sel(({'lat':np.intersect1d(error.lat.data, antilope.lat.data), 'lon':np.intersect1d(error.lon.data, antilope.lon.data)})).data.flatten()

            new = dynamic_correction(obs, pond)  # Update obs
            antilope['obs'] = xr.DataArray(
                    data   = new.reshape((len(mask.lat), len(mask.lon))),
                    dims   = ["lat", "lon"],
                    coords = dict(lon=mask.lon, lat=mask.lat)
                )
            antilope['obs'] = antilope['obs'].fillna(antilope.rr)

#            antilope = antilope.rename({'rr_debiaise':'analysis'})

            # 3. Nivometeo Assimilation
            antilope = self.nivometeo_assimilation(antilope, pond)


            #antilope.to_netcdf(self.filename)  # WARNING : overwrite the initial file !!  TMP !


        return antilope

    def get_nivometeo(self):
        fic_score = os.path.join(datadir, f'obs_nivometeo_daily_RR_{self.datebegin.ymd}_{self.dateend.ymd}.csv')
        if os.path.exists(fic_score):
            nivometeo = pd.read_csv(fic_score, sep=';', parse_dates=['date'],
                    dtype={'num_poste':int, 'nom':str, 'alti':int, 'lat':float, 'lon':float, 'massif':int, 'rr': float, 'reseau_poste': int}, na_values=['--'])
            if len(nivometeo)>0:
                nivometeo = nivometeo.loc[nivometeo['date']==np.datetime64(date)]
                return nivometeo
            else:
                return None
        else:
            return None

    def nivometeo_assimilation(self, antilope, pond):
        """
        Use Kalman Filter : a = x + BH'(HBH'+R)⁻¹(y-Hx)
        x = ANTILOPE (Background)
        B = Backroud ECM (static ANTILOPE error)
        y = obs nivométéo
        R = Observation error (TODO : à définir (erreur fixe incluant erreur de représentativité et incertitude sur la mesure ?)
        H = Observation operator
        """

        # Read ANTILOPE error (=Background error !)
        error = xr.open_dataset(os.path.join(workdir, 'Observation_error.nc'))

        # Background
        # WARNING : on ne peut appliquer l'analyse que sur le domaine où l'erreur d'ANTILOPE a été estimée !!!
        x = antilope.obs.sel({'lat':error.lat, 'lon':error.lon}).data.flatten()

        #Read nivometeo observations
        nivometeo = self.get_nivometeo()
        if nivometeo is None:
            antilope = antilope.rename({'obs':'analysis'})
        else:
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

            # Background (ANTILOPE) ECM
            std = diags(error.ratio.data.flatten(), 0)
            B = std.dot(pond.dot(std))
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


            #TODO : Update Error covariance matrix
        return antilope

#pp = AntilopePreprocessing(date, domain)
#antilope = pp.run()
