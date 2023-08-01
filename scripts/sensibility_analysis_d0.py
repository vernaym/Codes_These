#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os, sys
import glob
import time
from datetime import datetime,timedelta
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import rankdata
import CRPS.CRPS as pscore

import argparse

import matplotlib as mpl
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def nearest(array, value):
    """ Find element of "array" the closer to 'value' """
    # Security to ensure that the station is within the simulated domain.
    if np.abs(array - value).data.min() < 0.1:
        return float(array[np.abs(array - value).argmin()].data)
    else:
        print(f'ERROR : no corresponding pixel found for value {value}')

class Evaluation(object):

    def __init__(self, threshold):
        self.scores = None
        self.threshold = threshold  # threshold to use as event detection in the Brier Score
        self.lpn = None
        self.obs_error = None
        self.thresholds = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]

    def bias(self, simu, obs, *args, **kw):

        simu = simu[~np.isnan(obs)]
        obs = obs[~np.isnan(obs)]

        if np.shape(simu) == np.shape(obs):  # "Simulation" déterministe
            bias = simu - obs
        else:  # Simulation s'ensmble
            bias = self.mean_error(simu, obs)

        return np.nanmean(bias)

    def rmse(self, simu, obs, *args, **kw):

        simu = simu[~np.isnan(obs)]
        obs = obs[~np.isnan(obs)]

        if np.shape(simu) == np.shape(obs):  # "Simulation" déterministe
            rmse = np.sqrt(np.nanmean(np.square(simu-obs))) if np.nanmean(np.square(simu-obs)) > 0 else np.nan
        else:  # Simulation d'ensemble
            rmse = np.sqrt(np.nanmean(np.square(np.median(simu, axis=1) - obs)))
        return rmse

    def brier(self, simu, obs, Ne=16, threshold=10, *args):

        # TODO : verifier le calcul du score de brier
        simu = simu[~np.isnan(obs)]
        obs = obs[~np.isnan(obs)]

        if np.shape(simu) == np.shape(obs):  # "Simulation" déterministe
            psimu = np.where(simu>=threshold, 1, 0)
        else:  # Simulation d'ensemble
            psimu  = np.count_nonzero(simu>=threshold, axis=1) / Ne
            #psimu  = (np.count_nonzero(simu>=self.threshold, axis=1)+ 2/3) / (self.Ne+4/3)  # Tukey's plotting position
        fobs   = np.where(obs>=threshold, 1, 0)
        brier = np.nanmean((psimu-fobs)**2)
        #print('Brier=',brier)

        return brier

    def CRPS(self, simu, obs, *args):

        crps = list()
        for i in range(len(obs)):
            if len(np.shape(simu)) == 1:
                crps.append(pscore([simu[i]], obs[i]).compute()[0])
            else:
                crps.append(pscore(simu[i], obs[i]).compute()[0])

        return np.nanmean(np.array(crps))

    def read_nivometeo_obs(self):
#        nivometeo = pd.read_csv(os.path.join(datadir, 'obs_nivometeo_daily_RR_20211201_20220430.csv'), sep=';', parse_dates=['Q.dat'],
#            dtype={'Q.num_poste':int, 'poste_nivo.nom_usuel':str, 'poste_nivo.alti':int, 'poste_nivo.lat_dg':float, 'poste_nivo.lon_dg':float, 'poste_nivo.massif_nivo':int, 'Q.rr':float},)
#            rename={'Q.date':'date', 'Q.num_poste':'num_poste', 'poste_nivo.nom_usuel':'nom', 'poste_nivo.alti':'alti', 'poste_nivo.lat_dg':'lat', 'poste_nivo.lon_dg':'lon', 'Q.rr':'obs'})

            #Q.dat;Q.num_poste;poste_nivo.nom_usuel;poste_nivo.alti;poste_nivo.lat_dg;poste_nivo.lon_dg;poste_nivo.massif_nivo;Q.rr;hist_reseau_poste.reseau_poste

        nivometeo = pd.read_csv(os.path.join(datadir, 'obs_nivometeo_daily_RR_20211201_20220430.csv'), sep=';', parse_dates=['date'], header=0,
                names=['date', 'num_poste', 'nom', 'alti', 'lat', 'lon', 'massif', 'obs', 'unused'],
                usecols=['date', 'num_poste', 'nom', 'alti', 'lat', 'lon', 'obs'],
                dtype={'num_poste':int, 'nom':str, 'alti':int, 'lat':float, 'lon':float, 'obs':float},
            )

        latmax, latmin, lonmin, lonmax = np.array(coords[domain]).astype(float)/1000.
        nivometeo = nivometeo.loc[(nivometeo['lat']>=latmin) & (nivometeo['lat']<=latmax) & (nivometeo['lon']>=lonmin) & (nivometeo['lon']<=lonmax)]  # Select area
        #nivometeo = nivometeo[nivometeo['num_poste'].isin(indep)]  # Select evaluation stations
        nivometeo.date = nivometeo.date + pd.Timedelta("1d6h")   #BDClim extraction for date ymd is the observation from ymd6h to ym(d+1)6h
        #nivometeo.groupby('num_poste')['nom', 'lat', 'lon', 'alti'].agg(set)
        #nivometeo = nivometeo.set_index(['num_poste', 'lat', 'lon', 'nom', 'alti', 'date'])  # Utilité de passer en index ?
        nivometeo.set_index(['num_poste','date'], inplace=True)

        return nivometeo.to_xarray()

    def read_obs_clim(self):

        obs = pd.read_csv(os.path.join(datadir, "obs_quotidienne_clim_RR.data"), sep=';', parse_dates=['date'], header=0,
                names = ['num_poste', 'lat', 'lon', 'alti', 'nom', 'reseau_poste', 'date', 'obs'],
                usecols=['num_poste', 'lat', 'lon', 'alti', 'nom', 'date', 'obs'],
                dtype={'num_poste':int, 'nom':str, 'alti':int, 'lat':float, 'lon':float, 'obs':float},
                )
        latmax, latmin, lonmin, lonmax = np.array(coords[domain]).astype(float)/1000.
        obs = obs.loc[(obs['lat']>=latmin) & (obs['lat']<=latmax) & (obs['lon']>=lonmin) & (obs['lon']<=lonmax)]  # Select area
        obs.set_index(['num_poste','date'], inplace=True)

        return obs.to_xarray()

    def read_antilope(self):
        #filename = 'ANTILOPEQ_2021073106_2022070106_GrandesRousses.nc'
        filename = 'ANTILOPEH_2021103000_2022060200_alp.nc'
        antilope = xr.open_dataset(os.path.join(datadir, filename))
        if filename.startswith('ANTILOPEH'):
            # Convert hourly precipitation into 24h precipitation between 6h J-1 and 6h J
            # Problem : the xarray tools to do that allows only accumulations between
            # 0h and 23h.
            # solution : shift time serie by 7h, compute 24h accumulations and
            # shift back !
            antilope['time'] = antilope.time-np.timedelta64(7, 'h')
            antilope = antilope.resample(time='1D').sum(dim='time')  # !!! VERY SLOW !!! WARNING : does not work with pandas>=2.0.0
            antilope['time'] = antilope.time+np.timedelta64(30, 'h')

        return antilope

    def evaluate(self):

        # To Extract specific values where evaluation data (obs nivometeo) is available
        pos = 1

        reference = self.read_nivometeo_obs()  # Read observation --> self.obs
        # Remove time dimension from metadata :
        reference['lon']=np.max(reference, axis=1)
        reference['lat']=np.max(reference, axis=1)
        reference['alti']=np.max(reference, axis=1)
        reference['nom']=np.max(reference, axis=1)
        dates_obs = reference.date

        antilope = self.read_antilope()
        dates_antilope = antilope.time.data

        dates = np.intersect1d(dates_obs, dates_antilope)

        antilope = antilope.loc[{'time':dates}]
        reference = reference.loc[{'date':dates}]

        scores = dict(antilope=list())

        ratio = dict()
        obs_error=dict()
        for d0 in ['0.1', '0.15', '0.2', '0.25', '0.3']:
            ratio[d0]  = xr.open_dataset(os.path.join("/home/vernaym/workdir/ASSIMILATION/mask/alp/analyse_sensibilite_d0", f"Estimated_ratio_alp_{d0}.nc")).ratio
            obs_error[d0] = xr.open_dataset(os.path.join("/home/vernaym/workdir/ASSIMILATION/mask/alp/analyse_sensibilite_d0", f"Observation_error_alp_{d0}.nc")).error
            scores[d0] = list()

        def debiaise(ds):
            return ds / ratio
        def to_ensemble(ds):
            ds1 = ds + ds * error
            ds2 = ds - ds * error
#            ds1 = ds * (1 + 0.263) + error
#            ds2 = ds * (1 - 0.263) - error
#            ds1 = ds + error
#            ds2 = ds - error
            return xr.concat([ds, ds1, ds2], 'member')


        liste_postes = np.array([])
        for idx, num_poste in enumerate(self.data.num_poste.data):
            print(f'Station {idx+1}/{len(self.data.num_poste.data)}')
            t1 = time.time()
            tmp = reference.loc[{'num_poste':num_poste}]
            lat = tmp.lat
            lon = tmp.lon
            obs = tmp.obs.data
            alti = tmp.alti.data.max()  # Altitude du poste
            lat = tmp.lat.data.max()  # Latitude du poste
            lon = tmp.lon.data.max()  # Longitude du poste
            t2 = time.time()
            print(f'Reading obs informations took {(t2-t1)*1000.}ms')
            if len(obs[~np.isnan(obs)]) >= 100:  # Filter stations with too few observations
                #...
                #TODO : continuer ici !

                liste_postes = np.append(liste_postes, num_poste)
                data['antilope'].append(antilope.sel({'lat':nearest(antilope.lat, lat), 'lon':nearest(antilope.lon, lon)}).rr.data)
                if 'antiloped' in data.keys():
                    data['antiloped'].append(antiloped.sel({'lat':nearest(antiloped.lat, lat), 'lon':nearest(antiloped.lon, lon)}).rr.data)
                if 'antiloper' in data.keys():
                    data['antiloper'].append(antiloper.sel({'lat':nearest(antiloper.lat, lat), 'lon':nearest(antiloper.lon, lon)}).rr.data)
                t3 = time.time()
                print(f'Reading antilope informations took {(t3-t2)*1000.}ms')
                data['raw'].append(raw.sel({'lat':nearest(raw.lat, lat), 'lon':nearest(raw.lon, lon)}).rr.data)
                for xpid,filename in experiments.items():
                    if xpid not in data.keys():
                        data[xpid] = list()
                    if domain == 'GrandesRousses':  # gridded data
                        data[xpid].append(simus[xpid].sel({'lat':nearest(simus[xpid].lat, lat), 'lon':nearest(simus[xpid].lon, lon)}).rr.data)
                    else:
                        data[xpid].append(simus[xpid].sel({'num_poste':num_poste}).rr.data)
                for product in data.keys():
                    if product not in scores.keys():
                        scores[product] = {score:list() for score in scores_list}
                    for score_name, score in scores[product].items():
                        if score_name.startswith('brier'):
                            threshold = float(score_name.split('_')[-1])
                            if product == 'antilope':
                                score.append(getattr(self, 'brier')(data[product][-1][~np.isnan(obs)], obs[~np.isnan(obs)], Ne=1, threshold=threshold))
                            elif product in ['antiloped', 'antiloper']:
                                score.append(getattr(self, 'brier')(data[product][-1][~np.isnan(obs)], obs[~np.isnan(obs)], Ne=3, threshold=threshold))
                            else:
                                score.append(getattr(self, 'brier')(data[product][-1][~np.isnan(obs)], obs[~np.isnan(obs)], threshold=threshold))
                        else:
                            score.append(getattr(self, score_name)(data[product][-1][~np.isnan(obs)], obs[~np.isnan(obs)]))
                    t7 = time.time()
                    #print(f'Computing score for simulation {xpid} took {(t7-t6)*1000.}ms')
                t7 = time.time()
            else:
                self.data = self.data.where(self.data.num_poste!=num_poste, drop=True)  # Drop station

        # TODO : optimiser le calcul des scores !

        self.data['antilope'] = (('num_poste', 'date'), data['antilope'])
        if 'antiloper' in data.keys():
            self.data['antiloper'] = (('num_poste', 'date', 'pseudo_member'), data['antiloper'])
        if 'antiloped' in data.keys():
            self.data['antiloped'] = (('num_poste', 'date', 'pseudo_member'), data['antiloped'])
        self.data['raw'] = (('num_poste', 'date', 'member'), data['raw'])
        for xpid in experiments.keys():
            self.data[xpid] = (('num_poste', 'date', 'member'), data[xpid])
        t8 = time.time()
        #print(f'Filling self.data took {(t8-t7)*1000.}ms')

        ###################################################################################
        #antiloped = antilope.groupby('date').apply(debiaise)
        antiloped = antilope.apply(debiaise)
        #antiloped = antilope.apply(to_ensemble)
        # Génération d'un ensemble de 3 membres prenant en compte l'erreur d'observation
        antiloped = antiloped.expand_dims('member')
        antiloped = antiloped.apply(to_ensemble)
        antiloped = antiloped.clip(0)
        antiloped = antiloped.transpose('lat', 'lon', 'time', 'member')

        antiloper = antilope.expand_dims('member')
        antiloper = antiloper.apply(to_ensemble)
        antiloper = antiloper.clip(0)
        antiloper = antiloper.transpose('lat', 'lon', 'time', 'member')

        data = dict(antilope=list(), raw=list())
        simus = dict()
        for xpid,filename in experiments.items():
            simus[xpid] = self.read_simu(os.path.join(workdir, filename)).loc[{'time':dates}]

#        scores_list = ['reliability', 'resolution', 'uncertainty', 'rmse', 'bias', 'brier']
        scores_list = ['rmse', 'bias'] + [f'brier_{threshold}' for threshold in self.thresholds] + ['CRPS']
        scores = dict()
        #dates = dates[:10]

if __name__ == "__main__":

    evaluation = Evaluation(threshold=10)
    #evaluation.ensemble_attributes()
    evaluation.plot_scores()

