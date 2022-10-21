#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os, sys
from datetime import datetime,timedelta
import numpy as np
import xarray as xr
import pandas as pd

import argparse

import matplotlib as mpl
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

#from snowtools.scores.ensemble import EnsembleScores


##############################################################################################
##############################################################################################

datadir = '/home/vernaym/These/DATA'

latmax = 45.240
latmin = 44.990
lonmin = 6.010
lonmax = 6.490

# Liste des coordonnées des domaines connus lat_max, lat_min, lon_max, lon_min
coords = dict(
    alp = ['46875', '43125', '4500', '8500'],
    pyr = ['43500', '42000', '-2000', '3500'],
    cor = ['43000', '41000', '8000', '10500'],
    GrandesRousses = ['45250', '44750', '6000', '6500'],
    ange = ['45240', '44990', '6010', '6490']
)

domain = 'GrandesRousses'

savedir = "/home/vernaym/workdir/ASSIMILATION/XP00"

class Evaluation(object):

    def __init__(self, ensemble):
        self.ensemble = ensemble  # DataArray(lat,lon,time,member)
        self.Ne = len(self.ensemble.member)
        self.read_nivometeo_obs()  # Read observation --> self.obs

    def ensemble_attributes(self):
        disp = self.dispersion()

    @property
    def mean(self):
        return  self.ensemble.mean(axis=3).rr.data

    def mean_error(self, simu, obs):
        return simu.mean() - obs

    def median_error(self, simu, obs):
        return simu.median() - obs

    def dispersion(self):
        """
        spread over all dates (and pixels ?)
        """
        disp = np.sqrt(np.mean([np.nanmean((self.ensemble.loc[{'member':m}].rr.data-self.mean)**2) for m in self.ensemble.member.data]))
        print('Dispersion = ', disp)

        return disp

    def rmse(self, simu, obs, num_poste):
        rmse = np.sqrt(np.mean(np.square(simu.mean() - obs)))
        #print(f'RMSE for station {num_poste} :',rmse)
        print(f'RMSE=',rmse)
        return rmse

    def brier_skill_score(self, simu, ref, obs, threshold):
        """  BSS = 1 - BS / BSref  """
        return 1 - self.brier_score(simu, obs, threshold=threshold) / self.brier_score(ref, obs, threshold=threshold)

    def brier_score(self, simu, obs, threshold):
        pens  = np.count_nonzero(simu>=threshold, axis=0) / self.Ne
        fobs   = np.where(obs>=threshold, 1, 0)
        brier = np.mean((pens-fobs)**2)
        print('Brier=',brier)
        return brier

    def brier_decomposition(self, simu, obs, threshold, nb_cat=17):

        # TODO : la décomposition du score de Brier devrait donner le même résultat
        # que le calcul direct (BS=BSfiab-BSres+BSunc), mais ce n'est pas le cas...

        catsize  = list()
        freq_occ = list()
        proba    = list()
        for Nm in range(nb_cat):
            Ni = np.count_nonzero(np.count_nonzero(simu>=threshold, axis=0)==Nm)
            if Ni > 0:
                proba.append(Nm/self.Ne)
                catsize.append(Ni)
                freq_occ.append(np.count_nonzero(obs[np.count_nonzero(simu>=threshold, axis=0)==Nm]>=threshold)/Ni)
        ndays = len(obs)
        fiability   = self.fiability(proba, catsize, freq_occ, ndays)
        global_freq_obs = np.count_nonzero(obs[obs>=threshold]) / ndays
        resolution  = self.resolution(proba, catsize, freq_occ, global_freq_obs, ndays)
        uncertainty = self.uncertainty(global_freq_obs)
        import pdb
        pdb.set_trace()

        return fiability, resolution, uncertainty

    def fiability(self, proba, catsize, freq_occ, ndays):
        fiability = 1/ndays*np.sum([Ni*(proba-focc)**2 for (proba, Ni, focc) in zip(proba, catsize, freq_occ)])
        print('Fiability=',fiability)
        return fiability

    def resolution(self, proba, catsize, freq_occ, global_freq_obs, ndays):
        resolution = 1/ndays*np.sum([Ni*(focc-global_freq_obs)**2 for (proba, Ni, focc) in zip(proba, catsize, freq_occ)])
        print('Resolution=',resolution)
        return resolution

    def uncertainty(self, global_freq_obs):
        uncertainty =  global_freq_obs*(1-global_freq_obs)**2
        print('Uncertainty=',uncertainty)
        return uncertainty

    def read_nivometeo_obs(self):
        nivometeo = pd.read_csv(os.path.join(datadir, 'obs_nivometeo_daily_RR_20211201_20220430.csv'), sep=';', parse_dates=['Q.dat'],
            dtype={'Q.num_poste':int, 'poste_nivo.nom_usuel':str, 'poste_nivo.alti':int, 'poste_nivo.lat_dg':float, 'poste_nivo.lon_dg':float, 'poste_nivo.massif_nivo':int, 'Q.rr':float})

        latmax, latmin, lonmin, lonmax = np.array(coords[domain]).astype(float)/1000.
        self.obs = nivometeo.loc[(nivometeo['poste_nivo.lat_dg']>=latmin) & (nivometeo['poste_nivo.lat_dg']<=latmax) & (nivometeo['poste_nivo.lon_dg']>=lonmin) & (nivometeo['poste_nivo.lon_dg']<=lonmax)]

    def read_nivometeo_coords(self, domain):
        metadata = pd.read_csv(os.path.join(datadir, 'postes_nivometeo.csv'), sep=';')
        latmax, latmin, lonmin, lonmax = np.array(coords[domain]).astype(float)/1000.
        subdata = metadata[(metadata['poste_nivo.lat_dg']>=latmin) & (metadata['poste_nivo.lat_dg']<=latmax) & (metadata['poste_nivo.lon_dg']>=lonmin) & (metadata['poste_nivo.lon_dg']<=lonmax)]
        return dict(zip(np.array(subdata['poste_nivo.num_poste']), zip(np.array(subdata['poste_nivo.lat_dg']), np.array(subdata['poste_nivo.lon_dg']))))

    def evaluate(self):

        def nearest(array, value):
            # Find element of "array" the closer to "value"
            return float(array[np.abs(array - value).argmin()].data)

        threshold = 1

        # To Extract specific values where evaluation data (obs nivometeo) is available
        pos = 1
        dates = dict()
        simu  = dict()
        obs   = dict()
        brier = dict()
        for num_poste, (lat, lon) in self.read_nivometeo_coords(domain).items():
            nearest_lat = nearest(self.ensemble.lat, lat)
            nearest_lon = nearest(self.ensemble.lon, lon)
            tmp  = self.obs.loc[self.obs["Q.num_poste"]==num_poste]
            if len(tmp)>0:
                print(f'Poste {num_poste} ({len(tmp)} data)')
                dates[num_poste] = np.array(tmp["Q.dat"] + timedelta(hours=30))  # Date yyyymmdd is the cumul between yyyymmdd06 and yyyymm(d+1)06
                obs[num_poste]  = np.array(tmp['Q.rr'])
                # Select corresponding simulations
                simu[num_poste] = np.transpose(self.ensemble.sel({'lat':nearest_lat, 'lon':nearest_lon}).loc[{'time':dates[num_poste]}].rr.data)
                self.rmse(simu[num_poste], obs[num_poste], num_poste)
                if num_poste == 5064403:
                    fiability, resolution, uncertainty = self.brier_decomposition(simu[num_poste], obs[num_poste], threshold)
                    print('BSfiab+BSres+BSunc=',fiability-resolution+uncertainty)
                    brier       = self.brier_score(simu[num_poste], obs[num_poste], threshold)
                #self.temporal_plot(time, simu, obs, num_poste)
                #plt.violinplot(self.mean_error(simu, obs), positions=[pos])
                pos = pos + 1
        brier_global = self.brier_score(np.concat(simu.values()), np.concat(obs.values()), threshold)
        #plt.show()
        return tmp

    def temporal_plot(self, time, simu, obs, num_poste):

        # TODO : add ANTILOPE value and raw ensemble
        fig = plt.figure(figsize=(14,6))
        plt.plot(time, obs, marker='.', linestyle='', color='k')
        plt.violinplot(simu, positions=mpl.dates.date2num(time))
        fig.savefig(f'{savedir}/{num_poste}.pdf', formatout='pdf',  bbox_inches='tight')
        #plt.show()

    def eval_simu(self):
        pass


if __name__ == "__main__":

    data = xr.open_dataset(os.path.join(datadir, 'Assimilation_locale_2021080106_2022070106.nc'))
    evaluation = Evaluation(data)
    evaluation.ensemble_attributes()
    evaluation.evaluate()
