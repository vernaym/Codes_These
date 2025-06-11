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
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import scipy
from scipy.ndimage import uniform_filter
from scipy.sparse import csr_matrix, csc_matrix, diags
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.sparse.linalg import inv, spsolve

import vortex
from vortex import toolbox
from bronx.stdtypes.date import Date, Period

from snowtools.tools.xarray_backend import CENBackendEntrypoint

from These.scripts import tools

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
workdir = '/cnrm/cen/users/NO_SAVE/vernaym/ANTILOPE/workdir'  # On sxcen
datadir = '/home/vernaym/These/DATA'

domain = 'alp'
ld = 0.1
max_dist = ld*2
d0 = max_dist


def dynamic_correction(field, pond, weight=None, super_ensemble=None, plot=False, qq_adjustment=False, gradient=None, uncertainty=None):
    """
    * field          : 2D (n*k) array containing the field to modify
    * pond           : (nk*nk) sparse ponderation matrix (each line gives the correlation between the corresponding pixel
                        and every other pixel of the domain). It is defined using the confidenc of each pixel and the
                       distance between the pixels
    * weight         : nk vector of the sum of the weights in the neighborhood of each pixel
    * super_ensemble : nk*nk mask matrix ("M") defining neighbor pixels to include in the computation
    """

    field[np.isnan(field)] = 0.0
    #initial_field = field.flatten().compute()  # For PF experiments
    initial_field = np.array(field.flatten())
    X = diags(initial_field, 0)

    # 1. Calcul de la moyenne pondérée par la distance ET l'erreur statique
    if super_ensemble is None:
        super_ensemble = pond.copy()  # WARNING : make a copy or pond will change when super_ensemble changes
        super_ensemble[super_ensemble.nonzero()] = 1  # Position of pixels to inclue in the spread computation
    if weight is None:
        pond.data[np.isnan(pond.data)] = 0.0
        weight = pond.sum(axis=1).A1  # The sum of the weights (axis=1 <==> sum over rows)

    # TODO : try sd=weight (Measures the confidence in both the pixel and its neighbours values)

    # multiply field by AROME gradient
    if gradient is not None:
        C = diags(gradient, 0)  # Matrice de cumul AROME
        C0 = super_ensemble.dot(C)  # 1 line = all values of a given window
        C0.data = 1 / C0.data
        C1 = C.dot(super_ensemble)  # 1 line = value of the central point
        #C1.data = 1 / C1.data
        G = C0.multiply(C1)  # Gradient local
        #G.data = 1 / G.data
        pond = pond.multiply(G)

    mean = pond.dot(X).sum(axis=1).A1  # getA1 transforms the 1*N matrix object into a 1D np.array
    mean = mean / weight

    sd1 = get_std(X, initial_field, pond, weight=weight, super_ensemble=super_ensemble)  # 1. Dispersion of the super ensemble around the initial field --> More dispersion on high error pixels (--> spatial structures)
    sd2 = get_std(X, mean, pond, weight=weight, super_ensemble=super_ensemble)  # 2. Dispersion of the super ensemble around the mean --> Smoother fields

    # TODO : include sd in the field modification algorithm ?

    pixel_weight = pond.diagonal()  # = "exp(-err)" ou "1/err" pour l'obs et "likelyhood" du pixel pour les membres de l'ensemble
    #weight = pond.sum(axis=1).A1
    nb_nonzero = (pond != 0).sum(0).A1  # Count non zero elements of each row
    meanweight = weight / nb_nonzero
    # pixel_weight is in ]0, 1]
    # Do not try to preserve values of pixels with low errors on average : the dynamic correction must account
    # for temporary failures as well as uncertainties due to the error estimation method
    #newfield = (initial_field * pixel_weight + mean * meanweight) / (pixel_weight + meanweight)  # Stay closer to the original value (spatial structures can still be visible)
    #newfield = (initial_field * pixel_weight + mean * meanweight/pixel_weight) / (pixel_weight + meanweight/pixel_weight)  # Smoother fields --> underestimation of extreme values
    newfield = (initial_field * pixel_weight + mean * weight/pixel_weight) / (pixel_weight + weight/pixel_weight)  # Smoother fields --> underestimation of extreme values
    #newfield = (initial_field * pixel_weight + mean * meanweight/(meanweight+pixel_weight)) / (pixel_weight + meanweight/(meanweight+pixel_weight))
    #newfield = np.round(newfield, 1)

#    if gradient is not None:
#        # TODO : apply AROME vertical gradient only for pixels with large uncertainties to avoid to introduce underestimaiton in valleys
#        uncertainty = uncertainty.reshape(np.shape(field))
#        w1 = uncertainty / (newfield + uncertainty)
#        w1[newfield==0] = 1
#        w0 = 1 - w1
#        newfield = newfield * (1 * w0 + gradient * w1)
#        sd2 = sd2 * (1 * w0 + gradient * w1)  # Increase error proportionnally

    if qq_adjustment:
        # Try to match extreme values with original field (quantile-quantile like method)
        # The goal is to bring the field distribution closer to the debiased one
        # Le but est de rapprocher la pente de la régression linéaire du champs corrigé vs le champs initial de 1
        # WARNINGs :
        # 1. cette méthode débiaise le champs corrigé --> utiliser une loi normale pour les perturbations ?
        # 2. le champs produit n'est plus aussi lisse

        # METHODE 1
#        rmax = np.max(initial_field)/np.max(newfield)
#        rmin = np.min(initial_field)/np.min(newfield)
#        if (np.min(initial_field) == 0) or (np.min(newfield) == 0):
#            rmin = 1
#        #slope = (rmax-rmin)/(np.max(newfield)-np.min(newfield))
#        #if np.max(newfield) == np.min(newfield) : slope = 1
#        #intersect = rmin - slope * np.min(newfield)
#        rmean = np.mean(initial_field)/np.mean(newfield)  # Pour éviter le cas ou le min initial est 0 et le nouveau min est >0
#        if (np.mean(initial_field) == 0) or (np.mean(newfield) == 0):
#            rmean = 1
#        slope = (rmax-rmean)/(np.max(newfield)-np.mean(newfield))
#        if np.max(newfield) == np.mean(newfield) : slope = 1
#        intersect = rmean - slope * np.mean(newfield)
#        ratio = slope * newfield + intersect
#        newfield = newfield * ratio
#        sd2 = sd2 * (1+slope)  # Increase spread !
        # * QQ correction does not outperform the dynamic correction alone
        # Need to draw from a normal law to produce an ensemble

        # METHODE 2 --> BEST method so far ! (XP6, XP9)
        # Try to match extreme values with original field
        rmax = np.max(initial_field)/np.max(newfield)
        rmin = np.min(initial_field)/np.min(newfield)
        #a = (rmax-rmin)/(np.max(newfield)-np.min(newfield))
        #b = rmin-a*np.min(newfield)
        #ratio = rmax/(np.max(newfield)-np.min(newfield))
        slope = (rmax-rmin)/(np.max(newfield)-np.min(newfield))
        if np.isnan(slope): slope=0
        #print('Slope=',slope)
        newfield = newfield*(1+slope)
        sd2 = sd2 * (1+slope)  # Increase spread !

        # METHODE 3
        # Seems good, does not work :
#        slope = (np.max(initial_field)-np.min(initial_field))/(np.max(newfield)-np.min(newfield))  # Slope of the regression to match min and max values
#        if np.max(newfield) == np.min(newfield) : slope = 1
#        intersect = np.min(initial_field) - slope * np.min(newfield)
#        newfield = slope*newfield+intersect  # --> can lean to large errors !
#        sd2 = sd2 * (1+slope)

    # TODO : there is still a probleme for low precipitation fields (artefacts)

    sd3 = np.abs(mean-newfield)  # Difference between the new value and the mean value in the neigborhood
    # Plot correction coefficient
    if plot:
        import These.scripts.cas_test as ct
        correction_coefficient = (meanweight/pixel_weight * 1 / (pixel_weight + meanweight/pixel_weight)).reshape(np.shape(field))
        filename = 'Correction_weight.pdf'
        ct.plot_field(correction_coefficient, filename, label='Correction coefficient', cmap=plt.cm.viridis, vmin=0, vmax=1, add_circle=True)

        # Plot original value coefficient
        original_value_coefficient = (pixel_weight / (pixel_weight + meanweight/pixel_weight)).reshape(np.shape(field))
        filename = 'Original_value_weight.pdf'
        ct.plot_field(original_value_coefficient, filename, label='Original value coefficient', cmap=plt.cm.viridis, vmin=0, vmax=1, add_circle=True)

        ct.plot_field(sd1.reshape(np.shape(field)), 'sd1', cmap=plt.cm.YlGnBu)
        ct.plot_field(sd2.reshape(np.shape(field)), 'sd2', cmap=plt.cm.YlGnBu)
        ct.plot_field(sd3.reshape(np.shape(field)), 'sd3', cmap=plt.cm.YlGnBu)
        ct.plot_field(np.sqrt(sd3*sd2).reshape(np.shape(field)), 'sqrt(sd1_times_sd2)', cmap=plt.cm.YlGnBu)
        ct.plot_field(np.sqrt(sd3*sd2).reshape(np.shape(field)), 'sqrt(sd3_times_sd2)', cmap=plt.cm.YlGnBu)
        ct.plot_field((sd2*sd1/sd2).reshape(np.shape(field)), 'sqrt(sd2_times_sd1_over_sd2)', cmap=plt.cm.YlGnBu)
        ct.plot_field((sd2+sd1/sd2).reshape(np.shape(field)), 'sqrt(sd2_plus_sd1_over_sd2)', cmap=plt.cm.YlGnBu)

    #sd = sd + 1  # Add 1 to ensure that the error is >1 (mm or mm^(1/2)). --> Dispersion too large
    #sd = (sd1+sd2)/2
    #sd = sd1/2+sd2
    sd = sd2
    #sd = sd2 + newfield*0.2  # Add a 20% error to ensure a minimum error
    #sd = np.sqrt(sd1*sd2)  # --> Increase spread (overdispersif in cas_test)
    #sd = sd2*sd1/sd2  # --> Increase spread (overdispersif in cas_test)
    #sd = np.sqrt(sd2*newfield)  # --> Increase spread (overdispersif in cas_test)
    #sd = sd1+sd2  # --> Increase spread and add spatial variability
    #sd = sd1  # Add spatial variability

    #sd = sd * (1+np.abs(ratio))  # Allow to increase spread in case of underdispersion

    return newfield, mean, sd

def filter_gauges(df, antilope, delta=0.1):
    """
    Filter gauges observations and compute corresponding ANTILOPE ratio.
    """
    latmax = np.max(antilope.lat.data)
    latmin = np.min(antilope.lat.data)
    lonmax = np.max(antilope.lon.data)
    lonmin = np.min(antilope.lon.data)
    df = df[(df.lat>=latmin) & (df.lat<=latmax) & (df.lon>=lonmin) & (df.lon<=lonmax)]
    # Columns 'lats'/'lons' used by method make_mask.plot_field
    # columns 'lat'/'lon' used for concatenation with nivometeo observations before kriging in 'gridded_random_draw"
    df['lats'] = df['lat']
    df['lons'] = df['lon']
    # Remove unreliable EDF obervations :
    mask = df['nom'].str.contains('EDF')
    df = df[~mask]
    # Extract corresponding antilope values :
    df["rr_antilope"] = antilope.sel(lat=xr.DataArray(df.lats.values, dims='poste'), lon=xr.DataArray(df.lons.values, dims='poste'), method='nearest').data
    # Compute ratio and error :
    # TODO : Try de-commenting the following line
    #df = df[df["rr"] >= 1]  # No precipitation in reference ==> keep ANTILOPE (gauge obstructed ?) --> solve missed precipitation over ridges. (ex : Savoie 20220110)
    df["delta"] = delta
    mask = (df["rr"]>1)
    df["delta"][mask] = 0
    df["ratio"] = (df["rr_antilope"]+df["delta"]) / (df["rr"]+df["delta"])  # Add delta to avoid division by 0 issues  --> large modification of the ratio for low precipitation events !
    df["error"] = df["rr_antilope"] - df["rr"]
    # Remove observations with unrealistic ratios :
    mask = (df['ratio']<1.2) & (df['ratio']>0.8)  # Gauges with larger intial_errors must have been rejected by the ANTILOPE algorithm --> we trust it
    df = df[mask]
    #df["ratio"][df["rr"] == 0] = 1  # No precipitation in reference ==> keep ANTILOPE (gauge obstructed ?). Most of these situations are filtered out by the condition 0.1<ratio<1.9

    return df

def dynamic_error_estimation(antilope, obs_auto, arome_clim=None, delta=0.1):

    lons, lats = np.meshgrid(antilope.lon.data, antilope.lat.data)
    weights = list()
    ratios  = list()
    errors  = list()
    for i,poste in enumerate(obs_auto.index):
        tmp = obs_auto.loc[poste]
        rr_antilope = antilope.sel(lat=tmp.lats, lon=tmp.lons, method='nearest').data
        #antilope_ratio = (antilope.data+0.1) / (rr_antilope+0.1)  # Goal : estimlate ratio for ridges pixels with no precipitation detected by ANTILOPE
#            if rr_antilope > 1 and tmp["rr"] > 1:
#                mask = np.where(antilope.data < 1)
#                # Compute a ratio if there is ANTILOLPE precipitation at reference point but not at target point (--> missed precipitation over ridges ? ex : Savoie 20220110)
#                delta2 = np.zeros(np.shape(antilope.data))
#                delta2[mask] = delta
#                #delta2 = 0
#            elif rr_antilope > 0 and tmp["rr"] > 0:
#                delta2 = np.zeros(np.shape(antilope.data))
#                mask = np.where(antilope.data > 0)
#                delta2[mask] = 1
        if rr_antilope > 1:
            delta2 = 0
        else:
            delta2 = delta
        antilope_ratio = (antilope.data+delta2) / (rr_antilope+delta2)  # Goal : estimlate ratio for ridges pixels with no precipitation detected by ANTILOE
        #antilope_ratio[antilope.data==0] = 1  # Do not introduce precipitation on pixel with no precipitation and no reference
        #if tmp["rr"] == 0:
        #    antilope_ratio[rr_antilope==0] = 1  # Not enough information available to estimate a ratio --> apply only AROME vertical gradient
        antilope_ratio[antilope.data==0] = 1  # Do not introduce precipitation on pixel with no precipitation and no reference
        if arome_clim is not None:
            arome_cumul = arome_clim.sel(lat=tmp.lats, lon=tmp.lons, method='nearest')
            ratio_arome = arome_clim.rr_cumul.data / arome_cumul.rr_cumul.data
            ratio_arome[antilope.data==0] = 1  # Do not introduce precipitation on pixel with no precipitation and no reference
        else:
            ratio_arome = 1
#                if tmp["rr"] < 1:
#                    #antilope_ratio[rr_antilope==0] = 1  # Not enough information available to estimate a ratio --> apply only AROME vertical gradient
#                    mask = (antilope.data == 0)
#                    antilope_ratio[mask] = 1  # Not enough information available to estimate a ratio --> apply only AROME vertical gradient
#                    ratio_arome[mask] = 1  # Do not introduce precipitation on pixel with no precipitation and no reference

        dist = np.sqrt((lats-tmp.lats)**2+(lons-tmp.lons)**2)  # Euclidian horizontal distance
        #w = 1/(0.01+dist)**2  # Distance weighting  --> adding 0.01 instead of 1 ensures that the value at the reference point is preserved
        #w = 1 - dist / d0  # Distance weighting
        #w[w<0] = 0
        #dist[dist==0] = 0.001
        #w = 1 / dist
        w = d0 / (d0+dist)  # IDW
        w = 1 / (0.01+dist)**2  # Idem XP25.7

        #ratio = (rr_antilope.rr.data+0.1) / (tmp.rr+0.1)  # WARNING : division by 0
        #error = rr_antilope.rr.data - tmp.rr
        ratio = tmp.ratio  # ratio is well defined (+0.1)
        weights.append(w)
        #ratios.append(ratio*antilope_ratio)
        rat = ratio*antilope_ratio/ratio_arome
        ratios.append(rat)
        error = np.abs(rat - 1) * ((antilope.data + delta2) / rat - delta2)
        errors.append(error)

    weights = np.array(weights)
    W = np.sum(weights, axis=0)
    Wm = np.mean(weights, axis=0)
    Wd = np.sqrt(np.mean((weights-Wm)**2, axis=0))

    mean_ratio = np.divide(np.sum(weights*ratios, axis=0), W)
    mean_ratio[W==0] = 1
    D = np.sqrt(np.sum(weights*(ratios-mean_ratio)**2, axis=0)/W)
    D[W==0] = 0
    ratios = np.array(ratios)

    # Compute estimated ratio by weighting between 1 and the mean estimated ratio
    # Decrease the weight for pixels with large ratio dispersion (more uncertainty !)
    # Ensure that estimated ratio for pixels with no information around (W=0) stay at 1
    # Rules :
    # * w0+w1=1  (Keep ratio ODG)
    # * W=0 ==> w1=0  (Ensure that estimated ratio for pixels with no information around (W=0) stay at 1)
    # * D=0 ==> w1=W/(W+1)
    # * D-->inf ==> w1-->0  (choix : D=1 ==> w1=1/2)
    # * W-->inf ==> w1-->1
    # w1 can be seen as a measure of the confidence in the method
    #X = 1/(2*Wm-1)  # Facteur pour assurer la condition D=1 ==> w1=1/2. WARNING : W=1/2 valeur singulière
    #K = Wm * (1 - D / (D + X))
    #K[Wm==0.5] = 0.5  # W=1/2 valeur singulière de X
    #K[Wm==0] = 0
###################################
# XP25.7:
#    X = 1/(2*W-1)  # Facteur pour assurer la condition D=1 ==> w1=1/2. WARNING : W=1/2 valeur singulière
#    K = W * (1 - D / (D + X))
#    K[W==0.5] = 0.5  # W=1/2 valeur singulière de X
#    K[W==0] = 0
#    w1 = np.exp(-1/K)
#    w0 = 1 - w1
    #w1 = W / (1 + W)
    #w0 = 1 - w1
###################################

    w1 = np.exp(-D/(np.abs(mean_ratio-1)/np.log(10)))
    w0 = 1 - w1
    new_ratio = 1 * w0 + w1 * mean_ratio
    new_ratio[mean_ratio==1] = 1
    #estimated_ratio[np.isnan(estimated_ratio)] = 1

    errors = np.array(errors)
    errors[np.isnan(errors)] = 0  # Security
    errors[np.isinf(errors)] = 0  # No precipitation in reference
    new_error = np.divide(np.sum(weights*errors, axis=0), W)
    new_error[W==0] = antilope.data[W==0] * 0.3  # Default error ~ 30%

    # Conversion to xarray
    new_ratio = tools.to_xarray(new_ratio, antilope, varname='Ratio')
    new_error = tools.to_xarray(np.abs(new_error), antilope, varname='Error (mm)')

    return new_ratio, new_error

def get_std(data, mean, pond, weight=None, super_ensemble=None):
    """
    Compute the weighted variance of the weighted ensemble :

    sd = var(x) = np.sqrt(1/sum(weight)*sum((xi-xmean)²))

    INPUT
    -----
    * mean is the Nk*Nk matrix
    * pond is the Nk*Nk ponderation matrix (defined by the confidence of each pixel and the distance between the pixels)
    * weight is the Nk vector of the sum of the weights in the neighborhood of each pixel
    * super_ensemble is the mask Nk*Nk matrix ("M") defining neighbor pixels to include in the computation (1 if pixel in the
      window, else 0)

    OUTPUT
    ------
    * sd  : Nk vector of the dispersion of the neighborhood of each pixel of the domain (in mm)

    """

    se_mean = diags(mean, 0).dot(super_ensemble)  # matrix with mean[i] at each non-zero element of line i of super_ensemble
    X = super_ensemble.dot(data)-se_mean  # M.diag(obs)-diag(mean).M  --> difference between each neighbor value and the neighborhood mean
    #TODO : récupérer le nombre de 0 dans chaque voisinage pour la génération de l'ensemble
    sd = X.multiply(X).multiply(pond).sum(axis=1).getA1()  # Ponderation of the squared difference by the confidence (pond) + sum over all neighbor values
    sd = sd / weight  # Normalisation with the total weight in the neighborhood
    sd = np.sqrt(sd)
    sd = np.nan_to_num(sd)  # replace nan values by 0

    return sd

def codistances(coords, domain='alp', ld=0.2, Zdist=False):  # TMP for illustration. TODO : test different correlation distances
    """
    Solution pour le calcul des inter-distances trouvée sur : https://stackoverflow.com/questions/35296935/python-calculate-lots-of-distances-quickly
    """

    # 1. Horizontal inter-dtances
    #max_dist = ld*3  # exp(-2)=0.14, exp(-3)=0.05 ==> facteur 3 pour ignorer les pixels avec un poid < 5%
    #max_dist = ld*2  # exp(-2^2)=0.018 ==> facteur 2 pour ignorer les pixels avec un poid < 2%
    #max_dist = ld
    filename = f'codistance_max_dist_{ld}_{domain}.npz'
    if os.path.exists(filename):
        dist = scipy.sparse.load_npz(filename)
    else:
        tree = cKDTree(coords)
        dist = tree.sparse_distance_matrix(tree, max_distance=max_dist, p=2, output_type='coo_matrix')
        dist = csr_matrix(dist)
        #dist.data = d0 / (d0+dist.data)  # IDW
        #dist.data = 1 / (1 + dist.data)  # IDW
        dist.data = 1 - dist.data / ld  # Pondération de Franke-Little
        dist.data[dist.data<0] = 0
        #dist.data = (max_dist-dist.data)/(max_dist*dist.data)^2  # Modified Shepard's ponderation
        #dist.data=1/(1+dist.data)**2  # IDW
        #dist.data=1/(0.01+dist.data)**2  # IDW
        #dist.data=1/(0.1+dist.data)**2  # IDW
        #dist.data=1/(0.5+dist.data)**2  # IDW
        #dist[dist.nonzero()] = dist[dist.nonzero()]/ld
        #np.exp(-dist.data, out=dist.data )
        #np.exp(-dist.data**2/2, out=dist.data )
        #np.exp(1/(1+dist.data), out=dist.data )

        # 2. Elevation inter-distance
        if Zdist:
            #mnt1km = xr.open_dataset('/home/vernaym/These/DATA/DEM_ALPESFR_WGS84_1km.nc')  # Open 1km DEM
            #mnt250m = xr.open_dataset('/home/vernaym/QGIS/MNT/DEM_FRANCE_L93_250m_bilinear.nc')  # Open 1km DEM
            #mnt1km = xr.open_dataset(f'/home/vernaym/These/DATA/DEM_{domain.upper()}_WGS84_1km.nc')  # Open 1km DEM
            try:
                mnt1km = xr.open_dataset(os.path.join(datadir, f'DEM_{domain.upper()}_WGS84_1km.nc'))  # Open 1km DEM
            except:
                mnt1km = xr.open_dataset(os.path.join(datadir, f'DEM_ALP_WGS84_1km.nc'))  # Default
            lons = np.unique([coord[0] for coord in coords])
            lats = np.unique([coord[1] for coord in coords])
            mnt1km = mnt1km.sel({'lat':np.intersect1d(lats, mnt1km.lat), 'lon':np.intersect1d(lons, mnt1km.lon)})
            Z = mnt1km.elevation.data.flatten()
            # WARNING : dZ=0 not taken into account ==> ponderation at central point = 0 !!!!
            # Solution : avoid to have exactly 0 at central point ==> Add 1m difference
            Z1 = diags(Z, 0)
            Z2 = diags(Z+1, 0)
            tmppond = dist.copy()
            tmppond[tmppond.nonzero()] = 1
            dZ = tmppond.dot(Z2) - Z1.dot(tmppond)  # Compute elevation inter-distance
            dZ = np.abs(dZ)
            #dZ.data=1/(1000+dZ.data)**2
            #np.exp(-dZ.data/1000, out=dZ.data)
            dZ.data = 1/(1+dZ.data/500)

            dist = dist.multiply(dZ)
            #dist=dZ
        scipy.sparse.save_npz(filename, dist, compressed=False)

    return dist


def random_draw(distribution='gamma', members=16, sort=True):
    """
    Return a sorted array (size=*members*) of randomly draw values from *distribution*
    """

    if distribution == 'normal':
        draw = np.random.normal(loc=0.0, scale=1.0, size=members)  # Draw random element from normal distribution
    elif distribution == 'gamma':
        # TODO : essayer de faire dependre k de l'obs
        # PROBLEME : en tirant 1 valeur / pixel on perd la cohérence spatiale
        k = 2  # k must be >1. TODO : fixer k de façon automatique --> + forte asymétrie
        theta = np.sqrt(1 / k)  # Ensure a variance of 1 (var=k*theta^2)
        shift = k * theta  # shift = mean  --> no bias introduction
        # Draw random element from gamma distribution (>0 only ==> shift necessary to convert into perturbations)
        draw = np.random.gamma(k, scale=theta, size=members) - shift
    else:
        print('Error : unknown distribution')
        return None

    if sort:
        return np.sort(draw)
    else:
        return draw


def perturb(obs, sd, perturbation1, perturbation2, ratio=None, sd2=None, frac=0.5):
    """
    Perturb an *obs* field with a previously randomly dranw value *perturbation* and an estimated error *sd*.
    """

    # Add 2 perturbations terms:
    # - 1 gamma distributed proportionnal to the precipitation intensity
    # - normal distributed around the estimated error --> especially important for error for small prexipitation values
    # This 2 step perturbation reduces the dispersion but introduces spatial variability in the analysis fields
    # WARNING : the small ensemble size (16) lead to a large variability
    # of the ensemble mean but this algorithm ensures that on average the ensemble
    # mean is centered on the corrected observation

    # WARNING : uniform filter issues near the border of the domain
    # --> need for normalized convolution
    # source : https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html
    mask    = np.where(np.isfinite(obs), np.ones(obs.shape), 0)
    weights = uniform_filter(mask, size=20, mode='constant')
    smooth  = uniform_filter(np.where(np.isfinite(obs), obs, 0), 20, mode='constant')
    smooth  = np.where(mask == 1, smooth / weights, np.nan)

    ana = obs + smooth * frac * perturbation1 + sd * perturbation2

    if sd2 is not None:
        gamma = random_draw(distribution='gamma', members=1)[0]
        ana = ana + sd2 * gamma

    ana[ana < 0] = 0  # WARNING : "mass accumulation" in 0 (analysis distribution not normal anymore)

    return ana


class AntilopePreprocessing(object):

    def __init__(self, domain, filename, datebegin=None, dateend=None, date=None):
        if datebegin is not None:
            self.datebegin = datebegin
        else:
            if date is not None:
                self.datebegin = date.replace(hour=7)
            else:
                print('ERROR : either "date" or "datebegin" / "dateend"  must be provided')
        if dateend is not None:
            self.dateend = dateend
        else:
            self.dateend = self.datebegin + Period(hours=23)
        self.domain = domain
        self.filename = filename

    def run(self, obs_auto=None, nivometeo=None):
        #filename = os.path.join(datadir, f'ANTILOPEH_{self.datebegin.strftime("%Y%m%d%H")}_{self.dateend.strftime("%Y%m%d%H")}_{self.domain}.nc')  # TODO : extract only up to 6h
        #if os.path.exists(filename):
        antilope = xr.open_dataset(self.filename, engine='cen')
        # TODO : gérer le changement d'heure !

        if 'analysis' in antilope.variables.keys():  # File already pre-processed
            antilope = antilope.rename({'analysis':'analysis_save'})  # !!!!! TODO : TMP !!!!!!!
            #antilope = antilope.rename({'rr':'analysis'})

        else:
            if 'time' in antilope.coords:
                antilope = antilope.where((antilope.time>np.datetime64(self.datebegin)) & (antilope.time<=np.datetime64(self.dateend)), drop=True).sum('time')  # Security ?

            # TODO : séparer les méthodes plus clairement pour pouvoir les appliquer indépendament les unes des autres
            # TODO : commencer par la correction dynamique puis appliquer le débiaisage (facteur à modifier pour prendre en compte le biais moyen après correction ?)

            # 1. Static de-biasing :
            if self.datebegin.month in [12, 1, 2, 3]:
                toolbox.input(
                    genv    = 'uenv:edelweiss.3@vernaym',
                    gvar    = f'WINTER_GRADIENT_{self.domain.upper()}',
                    local   = f'Estimated_winter_gradient_{self.domain}.nc',
                    unknown = True,
                )
                toolbox.input(
                    genv    = 'uenv:edelweiss.3@vernaym',
                    gvar    = f'WINTER_RATIO_{self.domain.upper()}',
                    local   = f'Estimated_winter_ratio_{self.domain}.nc',
                    unknown = True,
                )
                clim_ratio = xr.open_dataarray(f"Estimated_winter_ratio_{self.domain}.nc", engine='cen')
                clim_gradient = xr.open_dataarray(f"Estimated_winter_gradient_{self.domain}.nc", engine='cen')
            else:
                toolbox.input(
                    genv    = 'uenv:edelweiss.3@vernaym',
                    gvar    = f'SUMMER_GRADIENT_{self.domain.upper()}',
                    local   = f'Estimated_summer_gradient_{self.domain}.nc',
                    unknown = True,
                )
                toolbox.input(
                    genv    = 'uenv:edelweiss.3@vernaym',
                    gvar    = f'SUMMER_RATIO_{self.domain.upper()}',
                    local   = f'Estimated_summer_ratio_{self.domain}.nc',
                    unknown = True,
                )
                clim_ratio = xr.open_dataarray(f"Estimated_summer_ratio_{self.domain}.nc", engine='cen')
                clim_gradient = xr.open_dataarray(f"Estimated_summer_gradient_{self.domain}.nc", engine='cen')

            tmp = antilope.sel({'xx': clim_ratio.xx.data, 'yy': clim_ratio.yy.data}, method='nearest')

            smooth = uniform_filter(tmp.Precipitation.data, 20)
            smooth = np.where(smooth > 0, np.round(smooth, 1), 0)
            # Try to detect "missed" precipitation
            rr = xr.where((tmp.Precipitation == 0) & (smooth > 0), 0.1, tmp.Precipitation)
            dyn_gradient = xr.where((rr.data > 0) & (smooth > 0), rr / smooth, clim_ratio)
            dyn_ratio = dyn_gradient / clim_gradient

            actual_ratio = (clim_ratio * np.maximum(0.5 - abs(clim_gradient - dyn_gradient), 0) +
                    dyn_ratio * np.minimum(abs(clim_gradient - dyn_gradient), 0.5)) / 0.5

            # antilope['analysis'] = rr / actual_ratio
            # antilope['error'] = abs(antilope['analysis'] - antilope['rr']) * 0.3 + antilope['analysis'] * 0.5
            antilope['debiasing'] = rr / actual_ratio
            smooth = antilope['debiasing'].where(antilope['debiasing'].notnull(), drop=True)
            smooth.data = uniform_filter(smooth, 20)
            smooth = xr.align(smooth, antilope, join='outer')[0]
            error = abs(antilope['debiasing'] - antilope['Precipitation']) * 0.3 + smooth * 0.5
            #antilope['error'] = abs(antilope['debiasing'] - antilope['Precipitation']) * 0.3 + smooth * 0.5

            # 2. Nivometeo Assimilation
            # Use a relative error in case of missed precipitation
            if nivometeo is not None:
                relative_error = xr.where(antilope['debiasing'] > error, error / antilope['debiasing'], 1)
                antilope['analysis'] = self.nivometeo_assimilation(antilope['debiasing'], relative_error, nivometeo.copy())
            else:
                antilope['analysis'] = antilope['debiasing']

            smooth = antilope['analysis'].where(antilope['analysis'].notnull(), drop=True)
            smooth.data = uniform_filter(smooth, 20)
            smooth = xr.align(smooth, antilope, join='outer')[0]
            antilope['error'] = abs(antilope['analysis'] - antilope['Precipitation']) * 0.3 + smooth * 0.5

            # antilope.to_netcdf(self.filename)  # WARNING : overwrite the initial file !!  TMP !

        return antilope

    def get_nivometeo(self):
        fic_score = os.path.join(datadir, f'obs_nivometeo_daily_RR_{self.datebegin.ymd}_{self.dateend.ymd}.csv')
        if os.path.exists(fic_score):
            nivometeo = pd.read_csv(fic_score, sep=';', parse_dates=['date'],
                    dtype={'num_poste': int, 'nom': str, 'alti': int, 'lat': float, 'lon': float, 'massif': int,
                        'rr': float, 'reseau_poste': int}, na_values=['--'])
            if len(nivometeo) > 0:
                nivometeo = nivometeo.loc[nivometeo['date'] == np.datetime64(self.datebegin)]
                return nivometeo
            else:
                return None
        else:
            return None

    def nivometeo_assimilation(self, background, error, obs, obs_error=0.2):
        """
        Use Kalman Filter : a = x + BH'(HBH'+R)⁻¹(y-Hx)
        x = ANTILOPE (Background)
        B = Backroud ECM (ANTILOPE error)
        y = obs nivométéo
        R = Observation error (TODO : à définir (erreur fixe incluant erreur de représentativité et incertitude sur la mesure ?)
        H = Observation operator
        """

        if obs is None:
            print('No nivometeo observation')
            out = background
        else:
            x = background.data.flatten()
            y = obs.rr.to_numpy()
            # Construction of the observation operator
            # - Put observations on the ANTILOPE grid
            obs.lat = obs.lat.round(2)
            obs.lon = obs.lon.round(2)
            Hx = background.sel(lat=obs.lat.to_xarray(), lon=obs.lon.to_xarray(), method = 'nearest').data

            X,Y = np.meshgrid(error.lon.data, error.lat.data)
            X = X.flatten()
            Y = Y.flatten()
            #coords = [elem for elem in zip(X,Y)]
            #points = [elem for elem in zip(obs.lon, obs.lat)]
            H = np.zeros((len(y), len(x)))  # n*k matrix
            # Update H matrix with 1 where an observation is present
            for i,idx in enumerate(obs.index):
                lon = obs.loc[idx, 'lon']
                lat = obs.loc[idx, 'lat']
                H[i, np.where((X==lon) & (Y==lat))[0]] = 1  # update H matrix
            Ht = np.transpose(H)
            H  = csr_matrix(H)
            Ht = csr_matrix(Ht)

            # Background (ANTILOPE) ECM
            std = diags(error.data.flatten(), 0)
            coords = [(lon,lat) for lat in background.lat for lon in background.lon]
            pond = codistances(coords)
            B = std.dot(pond.dot(std))
            B = csr_matrix(B)

            # Observation ECM
            #R = diags(obs_error * (y + 1))  # Uniform and uncorrelated 50%
            # Use a relative error in case of missed precipitation
            R = diags(obs_error * (y + 0.1) / (y + 0.1))  # Uniform and uncorrelated 50%

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

            out = xr.DataArray(
                    data   = A.reshape((len(background.lat), len(background.lon))),
                    dims   = ["lat", "lon"],
                    coords = dict(lon=background.lon, lat=background.lat)
                )

            out = out.fillna(background)  # Fill Nan values with previous ones

            #TODO : Update Error covariance matrix

        return out

#pp = AntilopePreprocessing(date, domain)
#antilope = pp.run()
