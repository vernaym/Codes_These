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


def ensemble_attributes():
    disp = dispersion()

@property
def mean(ensemble):
    return ensemble.mean(axis=3).rr.data

def mean_error(simu, obs):
    return simu.mean(axis=1) - obs

def median_error(simu, obs):
    return simu.median(axis=1) - obs

def bias(simu, obs, *args, **kw):

    simu = simu[~np.isnan(obs)]
    obs = obs[~np.isnan(obs)]

    if np.shape(simu) == np.shape(obs):  # "Simulation" déterministe
        bias = simu - obs
    else:  # Simulation s'ensmble
        bias = mean_error(simu, obs)

    return np.nanmean(bias)

def ratio(self, simu, obs, *args, **kw):

    if np.shape(simu) == np.shape(obs):  # "Simulation" déterministe
        mean = simu
    else:
        mean = simu.mean(axis=1)
    mask = np.where((mean>1) & (obs>1))
    simu = simu[mask]
    obs = obs[mask]

    if np.shape(simu) == np.shape(obs):  # "Simulation" déterministe
        ratio = simu / obs
    else:  # Simulation s'ensmble
        ratio =  simu.mean(axis=1)/ obs

    return np.nanmean(ratio)

def error_frequency(simu, obs, threshold=0.2, *args, **kw):

    simu = simu[~np.isnan(obs)]
    obs = obs[~np.isnan(obs)]

    if np.shape(simu) == np.shape(obs):  # "Simulation" déterministe
        error = np.where((simu>obs*(1+threshold)) | (simu<obs*(1-threshold)))
    else:  # Simulation d'ensmble
        error = np.where( (np.max(simu, axis=1)<obs) | (np.min(simu, axis=1)>obs) )

    #freq_error = (np.count_nonzero(error_above_treshold) / len(error_above_treshold) * 100
    freq_error = (len(error[0]) / len(obs)) * 100

    return freq_error

def dispersion(ensemble):
    """
    return spread over all dates
    """
    #disp = np.sqrt(np.mean([np.nanmean((ensemble.loc[{'member':m}].rr.data-mean(ensemble))**2) for m in ensemble.member.data]))
    N, Ne = np.shape(ensemble)
    disp = np.sqrt((ensemble-mean(ensemble, axis=1))**2/Ne)
    print('Dispersion = ', disp)

    return disp

def spread_skill(simu, obs, *args, **kw):

    simu = simu[~np.isnan(obs)]
    obs = obs[~np.isnan(obs)]
    if np.shape(simu) == np.shape(obs):  # "Simulation" déterministe
        mean = simu
    else:
        mean = simu.mean(axis=1)
    mask = np.where((mean>1) & (obs>1))
    simu = simu[mask]
    obs = obs[mask]
    spread = dispersion(simu)
    error = np.abs(simu-obs)

    fig, ax = plt.subplots()
    ax.plot(error, spread)
    fig.savefig(os.path.join(savedir, "spread_skill.pdf"), format='pdf')

def rmse(simu, obs, *args, **kw):

    simu = simu[~np.isnan(obs)]
    obs = obs[~np.isnan(obs)]

    if np.shape(simu) == np.shape(obs):  # "Simulation" déterministe
        rmse = np.sqrt(np.nanmean(np.square(simu-obs))) if np.nanmean(np.square(simu-obs)) > 0 else np.nan
    else:  # Simulation d'ensemble
        rmse = np.sqrt(np.nanmean(np.square(np.median(simu, axis=1) - obs)))
    return rmse

def brier_skill_score(simu, obs, ref, threshold=10):
    """  BSS = 1 - BS / BSref  """
    return 1 - brier(simu, obs, threshold) / brier(ref, obs, threshold)

def brier(simu, obs, Ne=16, threshold=10, *args):
    """
    * simu : 2D numpy array of shape(N*Ne)
    * obs  : 1D numpy array of size N
    N = Number of events
    Ne = number of ensemble members
    """
    simu = simu[~np.isnan(obs)]
    obs = obs[~np.isnan(obs)]

    if np.shape(simu) == np.shape(obs):  # "Simulation" déterministe
        psimu = np.where(simu>=threshold, 1, 0)
    else:  # Simulation d'ensemble
        N, Ne = np.shape(simu)
        psimu  = np.count_nonzero(simu>=threshold, axis=1) / Ne
        #psimu  = (np.count_nonzero(simu>=threshold, axis=1)+ 2/3) / (Ne+4/3)  # Tukey's plotting position
    fobs   = np.where(obs>=threshold, 1, 0)
    brier = np.nanmean((psimu-fobs)**2)
    #print('Brier=',brier)

    return brier

def CRPS(simu, obs, *args):

    crps = list()
    for i in range(len(obs)):
        if len(np.shape(simu)) == 1:
            crps.append(pscore([simu[i]], obs[i]).compute()[0])
        else:
            crps.append(pscore(simu[i], obs[i]).compute()[0])

    #return np.nanmean(np.array(crps))
    return np.array(crps)

def ROC(simu, obs, product, ax, Ne=16, threshold=10):
    """ 
    Here "probability" is the forecasted probability above which the
    event is considered well forecasted by the ensemble.
    We built the contingency table :
        - a = forecasted and observed
        - b = forecasted but not observed
        - c = observed but not forecasted
        - d = Not forecasted and not observed

    Then the success rate is a/(a+c) and the false alarm rate is b/(b+d)
    """
    simu = simu[~np.isnan(obs)]
    obs  = obs[~np.isnan(obs)]

    #linestyle_map = {1:':', 10:'-', 20:'--'}
    #color = next(ax._get_lines.prop_cycler)['color']

    # TODO : Use Tukey's plotting probabilities 

    succes_rate = list()
    false_alarm = list()
    if np.shape(simu) == np.shape(obs):
        a = np.count_nonzero(np.where((obs>threshold) & (simu>threshold)))
        b = np.count_nonzero(np.where((obs<=threshold) & (simu>threshold)))
        c = np.count_nonzero(np.where((obs>threshold) & (simu<=threshold)))
        d = np.count_nonzero(np.where((obs<=threshold) & (simu<=threshold)))
        succes_rate.append(a/(a+c) if a>0 else 0)
        false_alarm.append(b/(b+d) if b>0 else 0)
    else:
        for seuil in range(1, Ne+1):
            # Pour un dépassement de seuil :
            a = len(np.where((obs>threshold) & (np.count_nonzero(simu>threshold, axis=1)>=seuil))[0])
            b = len(np.where((obs<=threshold) & (np.count_nonzero(simu>threshold, axis=1)>=seuil))[0])
            c = len(np.where((obs>threshold) & (np.count_nonzero(simu>threshold, axis=1)<seuil))[0])
            d = len(np.where((obs<=threshold) & (np.count_nonzero(simu>threshold, axis=1)<seuil))[0])
#            # Pour un intervalle :
#            a = np.count_nonzero(np.where((obs>=1) & (obs<5) & (np.count_nonzero((simu>=1) & (simu<5), axis=1)>=seuil)))
#            b = np.count_nonzero(np.where(((obs<1) | (obs>=5)) & (np.count_nonzero((simu>=1) & (simu<5), axis=1)>=seuil)))
#            c = np.count_nonzero(np.where((obs>=1) & (obs<5) & (np.count_nonzero((simu>=1) & (simu<5), axis=1)<seuil)))
#            d = np.count_nonzero(np.where(((obs<1) | (obs>=5)) & (np.count_nonzero((simu>=1) & (simu<5), axis=1)<seuil)))

            succes_rate.append(a/(a+c) if a+c>0 else np.nan)  # a+c=0 if the event is never observed
            false_alarm.append(b/(b+d) if b+d>0 else np.nan)  # b+d= 0 if the event is always observed

    #plt.plot(false_alarm, succes_rate, label=product, linestyle=linestyle_map[threshold])
    if np.shape(simu) == np.shape(obs):
        #ax.plot(false_alarm, succes_rate, marker = '+', markersize=12, linestyle='', label=product, color='k')
        ax.plot(false_alarm, succes_rate, marker = '+', markersize=12, linestyle='', label=product)
    else:
        ax.plot(false_alarm, succes_rate, label=product)

    return (false_alarm, succes_rate)

def rank_histogram(ensemble, obs, ax, *args):
    """
    Inspired from : https://github.com/oliverangelil/rankhistogram/blob/master/ranky.py
    When two or more forecasts have same value (most commonly 0), random selection is made for which bin receives the count.
    """
#        maxsim = np.amax(simu, 1)
    # WARNING : la condition obs>0 réduit PLUS le nombre de cas.
    # Le choix de la condition est très important car il fait apparaitre ou disparaitre
    # une énorme majorité des situations où tout est à 0 sauf 1 membre (grand pic à gauche de l'histogramme)
#        simu = simu[obs>0]
#        obs = obs[obs>0]
#        simu = simu[(~np.isnan(obs)) & ((obs>0) | (maxsim>0))]
#        obs = obs[(~np.isnan(obs)) & ((obs>0) | (maxsim>0))]
#        position = np.array([])
#        for idx, obs in enumerate(obs):
#            position = np.append(position, np.searchsorted(np.sort(simu[idx]), obs))
#        ax.hist(position, bins=range(np.shape(simu)[1]))

    ensemble = np.transpose(ensemble)  # Shape (Nmember, Ndates)
    ensemble = ensemble[:,~np.isnan(obs)]
    obs = obs[~np.isnan(obs)]
    combined = np.vstack((obs[np.newaxis], ensemble))

    # Computing ranks
    ranks = np.apply_along_axis(lambda x: rankdata(x, method='min'), 0, combined)

    # Computing ties')
    ties = np.sum(ranks[0]==ranks[1:], axis=0)
    ranks = ranks[0]
    tie = np.unique(ties)

    for i in range(1,len(tie)):
        index = ranks[ties==tie[i]]
        # print('randomizing tied ranks for ' + str(len(index)) + ' instances where there is ' + str(tie[i]) + ' tie/s. ' + str(len(tie)-i-1) + ' more to go')
        ranks[ties==tie[i]] = [np.random.randint(index[j], index[j]+tie[i]+1, tie[i])[0] for j in range(len(index))]

    #return np.histogram(ranks, bins=np.linspace(0.5, combined.shape[0]+0.5, combined.shape[0]+1))
    #ax.hist(ranks, bins=range(np.shape(ensemble)[0]))
    ax.hist(ranks, bins=np.linspace(0.5, combined.shape[0]+0.5, combined.shape[0]+1))

def reliability_diagram(simu, obs, product, ax, Ne=16, threshold=10):
    ndays = len(obs)
    simu = simu[~np.isnan(obs)]
    obs = obs[~np.isnan(obs)]
    proba, catsize, freq_occ, global_freq_occ = probability_classes(simu, obs, Ne=Ne)
    # TODO : taille du marker proportionelle au nombre de prevision dans une categorie
    ax.plot(proba, freq_occ, marker=None, linestyle='-', label=f'{product}')
    ax.scatter(proba, freq_occ, catsize/np.mean(catsize)*100)
    ax.plot([0,1], [0,1], linestyle=':', color='k')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Forecast Probability')
    ax.set_ylabel('Observed Frequency')

def probability_classes(simu, obs, threshold=10, Ne=16, nb_cat=17):

    # TODO : la décomposition du score de Brier devrait donner le même résultat
    # que le calcul direct (BS=BSfiab-BSres+BSunc), mais ce n'est pas le cas...

    catsize  = list()
    freq_occ = list()
    proba    = list()
    for Nm in range(nb_cat):
        Ni = np.count_nonzero(np.count_nonzero(simu>=threshold, axis=1)==Nm)
        if Ni > 0:
            proba.append(Nm/Ne)
            catsize.append(Ni)
            # TODO : problème avec les dimensions de "simu" lorsque simu est un ensemble...
            freq_occ.append(np.count_nonzero(obs[np.count_nonzero(simu>=threshold, axis=1)==Nm]>=threshold)/Ni)
    ndays = len(obs)
    global_freq_obs = np.count_nonzero(obs[obs>=threshold]) / ndays

    return np.array(proba), np.array(catsize), np.array(freq_occ), global_freq_obs

def reliability(simu, obs, *args):
    simu = simu[~np.isnan(obs)]
    obs = obs[~np.isnan(obs)]
    ndays = len(obs)
    proba, catsize, freq_occ, global_freq_occ = probability_classes(simu, obs)
    reliability = 1/ndays*np.sum(catsize*(proba-freq_occ)**2)
    #reliability = 1/ndays*np.sum([Ni*(proba-focc)**2 for (proba, Ni, focc) in zip(proba, catsize, freq_occ)])  # Equivalent
    print('reliability=',reliability)
    return reliability

def resolution(simu, obs, *args):
    simu = simu[~np.isnan(obs)]
    obs = obs[~np.isnan(obs)]
    ndays = len(obs)
    proba, catsize, freq_occ, global_freq_obs = probability_classes(simu, obs)
    resolution = 1/ndays*np.sum(catsize*(freq_occ-global_freq_obs)**2)
    #resolution = 1/ndays*np.sum([Ni*(focc-global_freq_obs)**2 for (Ni, focc) in zip(catsize, freq_occ)])  # Equivalent
    print('Resolution=',resolution)
    return resolution

def uncertainty(simu, obs, *args):
    simu = simu[~np.isnan(obs)]
    obs = obs[~np.isnan(obs)]
    proba, catsize, freq_occ, global_freq_obs = probability_classes(simu, obs)
    uncertainty =  global_freq_obs*(1-global_freq_obs)**2
    print('Uncertainty=',uncertainty)
    return uncertainty

def violinplot(ax, position, data, label, color=None, addbar=False):
    """ Customize violinplot by adding a label"""
    import matplotlib.patches as mpatches

    if addbar:
        ax.plot([position]*len(data), data, linestyle='', marker='_', markersize='20', color='k')
    violin = ax.violinplot(data, positions=[position], showmeans=True)
    ax.axhline(color='k')
    if color is None:
        color = violin["bodies"][0].get_facecolor().flatten()
    else:
        violin["bodies"][0].set_facecolor(color)
        violin["bodies"][0].set_edgecolor(color)


    #return (mpatches.Patch(color=color), label)
    return ax


