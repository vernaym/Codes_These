#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auteur: Matthieu Vernay
# Date : 02/02/2022

import os, sys
import glob
from datetime import datetime,timedelta
import numpy as np
import xarray as xr
import pandas as pd

import argparse

import matplotlib as mpl
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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
    GrandesRousses = ['45240', '44990', '6010', '6490'],
    ange = ['45240', '44990', '6010', '6490']
)

domain = 'GrandesRousses'

savedir = "/home/vernaym/These/figures/evaluation"

experiments = dict(
        LD0      = 'Assimilation_locale_2021073106_2022070106_daily.nc',
        LDM      = 'Assimilation_locale_2021073106_2022070106_daily_avec_masque.nc',
        LH0      = 'Assimilation_locale_2021073106_2022070106_hourly.nc',
        #LHM      = 'Assimilation_locale_2021073106_2022070106_hourly_avec_masque.nc',
    )

xpid_label = dict(
        LD0      = 'Daily assimilation without mask',
        LDM      = 'Daily assimilation with mask',
        LH0      = 'Hourly assimilation without mask',
        #LHM      = 'Hourly assimilation with mask',
    )


class Evaluation(object):

    def __init__(self, threshold):
        #self.data = xr.Dataset()
#        self.ensemble = ensemble  # DataArray(lat,lon,time,member)
        self.Ne = 16  # TODO : à definir dynamiquement
        self.scores = None
        self.threshold = threshold  # threshold to use as event detenction in the breier score

    def ensemble_attributes(self):
        disp = self.dispersion()

    @property
    def mean(self):
        return  self.ensemble.mean(axis=3).rr.data

    def mean_error(self, simu, obs):
        return simu.mean() - obs

    def median_error(self, simu, obs):
        return simu.median() - obs

    def bias(self, simu, obs, **kw):

        if np.shape(simu) == np.shape(obs):  # "Simulation" déterministe
            bias = simu - obs
        else:  # Simulation s'ensmble
            bias = self.mean_error(simu, obs)

        return np.nanmean(bias)

    def dispersion(self):
        """
        spread over all dates (and pixels ?)
        """
        disp = np.sqrt(np.mean([np.nanmean((self.ensemble.loc[{'member':m}].rr.data-self.mean)**2) for m in self.ensemble.member.data]))
        print('Dispersion = ', disp)

        return disp

    def rmse(self, simu, obs, **kw):

        if np.shape(simu) == np.shape(obs):  # "Simulation" déterministe
            rmse = np.sqrt(np.nanmean(simu-obs))
        else:  # Simulation d'ensemble
            rmse = np.sqrt(np.nanmean(np.square(simu.mean() - obs)))
        return rmse

    def brier_skill_score(self, simu, ref, obs):
        """  BSS = 1 - BS / BSref  """
        return 1 - self.brier_score(simu, obs) / self.brier_score(ref, obs)

    def brier(self, simu, obs):

        if np.shape(simu) == np.shape(obs):  # "Simulation" déterministe
            psimu = np.where(simu>=self.threshold, 1, 0)
        else:  # Simulation d'ensemble
            psimu  = np.count_nonzero(simu>=self.threshold, axis=1) / self.Ne
        fobs   = np.where(obs>=self.threshold, 1, 0)
        brier = np.nanmean((psimu-fobs)**2)
        #print('Brier=',brier)

        return brier

    def ROC(self, simu, obs):
        """ Here "probability" is the forecasted probability above which the
        event is considered well forecasted by the ensemble.
        We built the contingency table :
            - a = forecasted and observed
            - b = forecasted but not observed
            - c = observed but not forecasted
            - d = Not forecasted and not observed

        Then the success rate is a/(a=c) and the false alarm rate is b/(b+d)
        """
        #for threshold in [1, 5, 10, 20]:
        succes_rate = list()
        false_alarm = list()
        for seuil in range(1, self.Ne+1):
            # Pour un dépassement de seuil :
#                a = np.count_nonzero(np.where((obs>=threshold) & (np.count_nonzero(simu>=threshold, axis=0)>=seuil)))
#                b = np.count_nonzero(np.where((obs<threshold) & (np.count_nonzero(simu>=threshold, axis=0)>=seuil)))
#                c = np.count_nonzero(np.where((obs>=threshold) & (np.count_nonzero(simu>=threshold, axis=0)<seuil)))
#                d = np.count_nonzero(np.where((obs<threshold) & (np.count_nonzero(simu>=threshold, axis=0)<seuil)))
            # Pour un intervalle :
            a = np.count_nonzero(np.where((obs>=1) & (obs<5) & (np.count_nonzero((simu>=1) & (simu<5), axis=0)>=seuil)))
            b = np.count_nonzero(np.where(((obs<1) | (obs>=5)) & (np.count_nonzero((simu>=1) & (simu<5), axis=0)>=seuil)))
            c = np.count_nonzero(np.where((obs>=1) & (obs<5) & (np.count_nonzero((simu>=1) & (simu<5), axis=0)<seuil)))
            d = np.count_nonzero(np.where(((obs<1) | (obs>=5)) & (np.count_nonzero((simu>=1) & (simu<5), axis=0)<seuil)))
            #print(a,b,c,d)
            succes_rate.append(a/(a+c) if a>0 else 0)
            false_alarm.append(b/(b+d) if b>0 else 0)
            #print(a/(a+c) if a>0 else 0)
            #print(b/(b+d) if b>0 else 0)
            #plt.plot(false_alarm, succes_rate, label=f"Threshold={threshold}mm")
        plt.plot(false_alarm, succes_rate)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.legend()
        plt.show()

    def brier_decomposition(self, simu, obs,nb_cat=17):

        # TODO : la décomposition du score de Brier devrait donner le même résultat
        # que le calcul direct (BS=BSfiab-BSres+BSunc), mais ce n'est pas le cas...

        catsize  = list()
        freq_occ = list()
        proba    = list()
        for Nm in range(nb_cat):
            Ni = np.count_nonzero(np.count_nonzero(simu>=self.threshold, axis=0)==Nm)
            if Ni > 0:
                proba.append(Nm/self.Ne)
                catsize.append(Ni)
                freq_occ.append(np.count_nonzero(obs[np.count_nonzero(simu>=self.threshold, axis=0)==Nm]>=self.threshold)/Ni)
        ndays = len(obs)
        fiability   = self.fiability(proba, catsize, freq_occ, ndays)
        global_freq_obs = np.count_nonzero(obs[obs>=self.threshold]) / ndays
        resolution  = self.resolution(proba, catsize, freq_occ, global_freq_obs, ndays)
        uncertainty = self.uncertainty(global_freq_obs)

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
        nivometeo.date = nivometeo.date + pd.Timedelta("1d6h")   #BDClim extraction for date ymd is the observation from ymd6h to ym(d+1)6h
        #nivometeo.groupby('num_poste')['nom', 'lat', 'lon', 'alti'].agg(set)
        #nivometeo = nivometeo.set_index(['num_poste', 'lat', 'lon', 'nom', 'alti', 'date'])  # Utilité de passer en index ?
        nivometeo.set_index(['num_poste','date'], inplace=True)

        return nivometeo.to_xarray()

#    def read_nivometeo_coords(self, domain):
#        metadata = pd.read_csv(os.path.join(datadir, 'postes_nivometeo.csv'), sep=';')
#        latmax, latmin, lonmin, lonmax = np.array(coords[domain]).astype(float)/1000.
#        subdata = metadata[(metadata['poste_nivo.lat_dg']>=latmin) & (metadata['poste_nivo.lat_dg']<=latmax) & (metadata['poste_nivo.lon_dg']>=lonmin) & (metadata['poste_nivo.lon_dg']<=lonmax)]
#        return dict(zip(np.array(subdata['poste_nivo.num_poste']), zip(np.array(subdata['poste_nivo.lat_dg']), np.array(subdata['poste_nivo.lon_dg']))))
#
    def read_antilope(self):
        filename = 'ANTILOPEQ_2021073106_2022070106_GrandesRousses.nc'
        antilope = xr.open_dataset(os.path.join(datadir, filename))
        return antilope

    def read_raw_ensemble(self):
        filenames = [os.path.join(datadir, f'aspearome_{mb:03d}_2021073106_2022070106_GrandesRousses_daily.nc') for mb in range(1,17)]
        raw = xr.open_mfdataset(filenames, combine='nested', concat_dim='member').compute()
        raw['member']=np.arange(1,17)
        raw = raw.transpose('lat', 'lon', 'time', 'member')  # transpose data to put dimension in the same order as assimilated fields

        return raw

    def read_simu(self, filename):
        if not os.path.exists(filename):
            print(f'WARNING : file {filename} does not exist, looking for it under {datadir}')
            filename = os.path.join(datadir, filename)

        if os.path.exists(filename):
            simulation =  xr.open_dataset(os.path.join(datadir, filename))
        else:
            print(f'ERROR : file {filename} does not exist')
            sys.exit(1)

        if 'hourly' in filename:
            # Convert hourly precipitation into 24h precipitation between 6h J-1 and 6h J
            # Problem : the xarray tools to do that allows only accumulations between
            # 0h and 23h.
            # solution : shift time serie by 7h, compute 24h accumulations and
            # shift back !
            #print(simulation.time.data[127*24-1])
            #print(simulation.time.data[126*24])
            simulation['time'] = simulation.time-np.timedelta64(7, 'h')
            toto=simulation.loc[{'lat':44.99, 'lon':6.01, 'member':1}].rr.data
            simulation = simulation.resample(time='D').sum(dim='time')  # !!! VERY SLOW !!!
            tata=simulation.loc[{'lat':44.99, 'lon':6.01, 'member':1}].rr.data
            #print(toto[126*24:127*24-1])
            #print(np.sum(toto[126*24:127*24-1]), tata[126])
            simulation['time'] = simulation.time+np.timedelta64(30, 'h')
            #print(simulation.time.data[126])
            #import pdb
            #pdb.set_trace()

        return simulation

    def evaluate(self):

#        if os.path.exists(os.path.join(datadir, 'scores.nc')):
#            self.scores = xr.open_dataset(os.path.join(datadir, 'scores.nc'))
#            return

        def nearest(array, value):
            # Find element of "array" the closer to "value"
            return float(array[np.abs(array - value).argmin()].data)

        # To Extract specific values where evaluation data (obs nivometeo) is available
        pos = 1

        # TODO : gerer les données avec une DataFrame ou un DataSet
#        simu  = dict()
#        obs   = dict()
#        brier = dict()
#
#        dates  = list()
#        postes = list()
#        obs    = list()
#        data   = dict()
#        liste_poste = self.read_nivometeo_coords(domain).keys()
#        latmax, latmin, lonmin, lonmax = np.array(coords[domain]).astype(float)/1000.

        self.data = self.read_nivometeo_obs()  # Read observation --> self.obs
        # Remove time dimension from metadata :
        self.data['lon']=np.max(self.data.lon, axis=1)
        self.data['lat']=np.max(self.data.lat, axis=1)
        self.data['alti']=np.max(self.data.alti, axis=1)
        self.data['nom']=np.max(self.data.nom, axis=1)
#        data = self.obs.loc[self.obs["Q.num_poste"].isin(liste_poste)]  # TODO a adapter
#        self.stations = self.data[['num_poste', 'nom', 'lat', 'lon', 'alti']].drop_duplicates()

        antilope = self.read_antilope()

        raw = self.read_raw_ensemble()
        self.data['member'] = np.arange(1,17)

        data = dict(antilope=list(), raw=list())
        simus = dict()
        for xpid,filename in experiments.items():
            simus[xpid] = self.read_simu(os.path.join(datadir, filename))



        scores_list = ['rmse', 'bias', 'brier']
        scores = dict()
        dates = self.data.date
        #dates = dates[:10]
        for num_poste in self.data.num_poste.data:
#            num_poste = row['num_poste']
#            lat       = row['lat']
#            lon       = row['lon']
#            dates     = self.data[self.data['num_poste']==num_poste]['date'].values
            lat = self.data.loc[{'num_poste':num_poste}].lat
            lon = self.data.loc[{'num_poste':num_poste}].lon
            obs = self.data.loc[{'num_poste':num_poste}].obs.data
            #obs = obs[:10]
            data['antilope'].append(antilope.sel({'lat':nearest(antilope.lat, lat), 'lon':nearest(antilope.lon, lon)}).loc[{'time':dates}].rr.data)
            data['raw'].append(raw.sel({'lat':nearest(raw.lat, lat), 'lon':nearest(raw.lon, lon)}).loc[{'time':dates}].rr.data)
            for xpid,filename in experiments.items():
                if xpid not in data.keys():
                    data[xpid] = list()
                data[xpid].append(simus[xpid].sel({'lat':nearest(simus[xpid].lat, lat), 'lon':nearest(simus[xpid].lon, lon)}).loc[{'time':dates}].rr.data)
            #self.temporal_plot(dates, data['LH0'][-1], obs, num_poste, raw=data['raw'][-1], antilope=data['antilope'][-1], simu2=data['LD0'][-1])
            self.temporal_plot(dates, data['LH0'][-1], obs, num_poste, raw=data['raw'][-1], antilope=data['antilope'][-1], simu2=data['LDM'][-1])
            idx=10
            #self.plot_assimilation(data['raw'][-1][:idx], data['LD0'][-1][:idx], data['antilope'][-1][:idx], 'LD0', num_poste, np.datetime_as_string(dates.data[:idx], unit='D'))

            for product in data.keys():
                if product not in scores.keys():
                    scores[product] = {score:list() for score in scores_list}
                for score_name, score in scores[product].items():
                    score.append(getattr(self, score_name)(data[product][0], obs))

        self.data['antilope'] = (('num_poste', 'date'), data['antilope'])
        self.data['raw'] = (('num_poste', 'date', 'member'), data['raw'])
        for xpid in experiments.keys():
            self.data[xpid] = (('num_poste', 'date', 'member'), data[xpid])

        tmp = {"score":{"dims": ("score"), "data":scores_list}, "num_poste":{"dims": ("num_poste"), "data":self.data.num_poste.data}}
        tmp.update({key:{"dims": ("score", "num_poste"), "data":[value[score] for score in scores_list]} for key,value in scores.items()})
        self.scores = xr.Dataset.from_dict(tmp)
        self.scores.to_netcdf(os.path.join(datadir, 'scores.nc'))

#        self.rmse(simu[num_poste], obs[num_poste], num_poste)
#        fiability, resolution, uncertainty = self.brier_decomposition(simu[num_poste], obs[num_poste], threshold)
#        print('BSfiab+BSres+BSunc=',fiability-resolution+uncertainty)
#        brier       = self.brier_score(simu[num_poste], obs[num_poste], threshold)
#                nearest_lat = nearest(self.ensemble.lat, lat)
#                nearest_lon = nearest(self.ensemble.lon, lon)
#                print(f'Poste {num_poste} ({len(tmp)} data)')
#                dates[num_poste] = np.array(tmp["Q.dat"] + timedelta(hours=30))  # Date yyyymmdd is the cumul between yyyymmdd06 and yyyymm(d+1)06
#                obs[num_poste]  = np.array(tmp['Q.rr'])
#                # Select corresponding simulations
#                simu[num_poste] = np.transpose(self.ensemble.sel({'lat':nearest_lat, 'lon':nearest_lon}).loc[{'time':dates[num_poste]}].rr.data)
#                self.rmse(simu[num_poste], obs[num_poste], num_poste)
#                fiability, resolution, uncertainty = self.brier_decomposition(simu[num_poste], obs[num_poste], threshold)
#                print('BSfiab+BSres+BSunc=',fiability-resolution+uncertainty)
#                brier       = self.brier_score(simu[num_poste], obs[num_poste], threshold)
#                #self.temporal_plot(time, simu, obs, num_poste)
#                #plt.violinplot(self.mean_error(simu, obs), positions=[pos])
#                pos = pos + 1
#        import pdb
#        pdb.set_trace()
#        xr.Dataset(data_vars=dict(obs=(["num_poste","time"],obs)),coords=dict(num_poste=(["num_poste"],postes),time=(["num_poste", "time"],dates)),attrs=dict(description="Evaluation dataset"),)
#        self.ROC(np.concatenate([array for array in simu.values()], axis=1), np.concatenate([array for array in obs.values()]), threshold)
#        brier_global = self.brier_score(np.concatenate([array for array in simu.values()], axis=1), np.concatenate([array for array in obs.values()]), threshold)
#        #plt.show()
#        return tmp

    def plot_scores(self):
        if self.scores is None:
            self.evaluate()
        for score in self.scores.score.data:
            fig,ax = plt.subplots()
            pos = 1
            products = [var for var in self.scores.data_vars]
            print(products)
            for product in products:
                x = self.scores.loc[{'score':score}][product].data
                plt.violinplot(x[~np.isnan(x)], showmeans=True, positions=[pos])
                pos += 1
            ax.set_xticklabels([''] + products)
            if score == 'brier':
                fig.savefig(f'{savedir}/{score}_{self.threshold}.pdf', formatout='pdf',  bbox_inches='tight')
            else:
                fig.savefig(f'{savedir}/{score}.pdf', formatout='pdf',  bbox_inches='tight')

    def temporal_plot(self, time, simu, obs, num_poste, raw=None, antilope=None, simu2=None):
        # TODO : add flexibility in the number and oreder of simulations (use dict !)

        def add_label(violin, label, color=None):
            import matplotlib.patches as mpatches
            if color is None:
                color = violin["bodies"][0].get_facecolor().flatten()
            else:
                violin["bodies"][0].set_facecolor(color)
                violin["bodies"][0].set_edgecolor(color)
            labels.append((mpatches.Patch(color=color), label))


        labels = []
        fig, ax = plt.subplots(figsize=(14,9))
        ref, = plt.plot(time, obs, marker='.', linestyle='', color='k')
        labels.append((ref, 'Nivometeo reference'))
        if antilope is not None:
            antpe, = plt.plot(time, antilope, marker='+', linestyle='', color='red')
            labels.append((antpe, 'Antilope'))
        positions = mpl.dates.date2num(time)
        if raw is not None:
            #add_label(plt.violinplot(np.transpose(raw), positions=positions), 'Raw ensemble', color='sandybrown')
            add_label(plt.violinplot(np.transpose(raw), positions=positions), 'Raw ensemble')
        add_label(plt.violinplot(np.transpose(simu), positions=positions), 'Hourly assimilation')
        #add_label(plt.violinplot(np.transpose(simu), positions=positions), 'Hourly assimilation', color='limegreen')
        if simu2 is not None:
            #add_label(plt.violinplot(np.transpose(simu2), positions=positions), 'Daily assimilation', color='skyblue')
            add_label(plt.violinplot(np.transpose(simu2), positions=positions), 'Daily assimilation')
        ax.set_xlabel('Date')
        ax.set_ylabel('24 hour precipitation (mm)')
        plt.legend(*zip(*labels))
        fig.savefig(f'{savedir}/{num_poste}.pdf', formatout='pdf',  bbox_inches='tight')
        plt.close()
        #plt.show()

    def plot_assimilation(self, raw, assim, obs, xpid, num_poste, date, ref=None):
        """ References :
        https://stackoverflow.com/questions/64646449/how-to-create-asymmetric-violin-plot-in-python-using-matplotlib
        https://seaborn.pydata.org/generated/seaborn.violinplot.html
        """
        fig, ax = plt.subplots()
        data = pd.DataFrame({'raw':raw, 'assim':assim})
        data = data.melt()
        data['dummy'] = 0
        sns.violinplot(data=data, split=True, y='value', hue='variable', x='dummy', inner="stick", palette=['sandybrown', 'skyblue'])
        plt.plot(obs, marker='_', markersize=30, markeredgewidth=3, color='red')
        if ref is not None:
            plt.plot(obs, marker='_', markersize=30, markeredgewidth=3, color='dark')
        fig.savefig(f'{savedir}/assim_{xpid}_{num_poste}_{date}.pdf', formatout='pdf',  bbox_inches='tight')

        #sns.violinplot(data=data, y='24 hour precipitation (mm)', split=True, hue='Simulation')

    def eval_simu(self):
        pass

if __name__ == "__main__":

    evaluation = Evaluation(threshold=10)
    #evaluation.ensemble_attributes()
    evaluation.plot_scores()

