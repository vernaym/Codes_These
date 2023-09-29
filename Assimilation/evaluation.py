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

from These.scripts import scores, tools

import argparse

import matplotlib as mpl
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

#from snowtools.scores.ensemble import EnsembleScores


##############################################################################################
# USAGE : p evaluation.py $xpid [$domain]
##############################################################################################

# TODO : lire https://www.researchgate.net/publication/238024585_A_New_Verification_Method_to_Ensure_Consistent_Ensemble_Forecasts_through_Calibrated_Precipitation_Downscaling_Models
# pour voir si la méthode de vérification peut s'appliquer
# TODO : Use the "Tukey’s plotting positions", which avoids probabilities 0 and 1 : P(t) = (n+2/3) / (M+4/3)
# --> see https://www.ecmwf.int/sites/default/files/elibrary/2010/10725-diagnosis-ensemble-forecasting-systems.pdf
# TODO : Réorganiser le code avec un module "score" séparé qui puisse être appellé par différents scirpts

#domain = 'GrandesRousses'
if len(sys.argv) == 1:
    domain = 'alp'  # default value
    xpid = 'reference'
if len(sys.argv) == 2:
    domain = 'alp'  # default value
    xpid = sys.argv[1]
elif len(sys.argv) == 3:
    domain = sys.argv[2]
    xpid = sys.argv[1]

# Random selection of station for evaluation
# --> a coordonner avec make_mask.py
#import random
#draw=random.sample(range(1, 64), 22)
#indep = [ 5139405, 73034400, 38006400,  5026400,  5096402,  5114402,
#        38020400, 74014402,  6120400, 74136400, 73257400, 73157400,
#            73232400, 73024400, 74056416,  5085403, 74191406,  5064403,
#            38253400, 73173400, 38548400, 73227400]  # Random draw of stations to use for evaluation
indep = [38253400,  4019404, 73318400, 73004400, 74063405, 38006400,
            38186400, 73040400, 74056416, 38527400,  6120400,  5064403,
            73322401,  5001400, 38020400, 73034400, 73232400, 38548400,
             5133400, 73257400,  5098402,  5079400, 73307400,  4073400,
             4006400, 73227400, 74134400, 74136400, 73054401, 73024400,
             5114402,  5085403]
indep = [74286400, 74063405, 74191406, 74014402, 731223402, 73132400, 73054401, 73232400, 73150400,
        73304404, 73047401, 73144404, 73257400, 73227400, 73206400, 73235400, 73318400, 73306403, 73173400,
        73307400, 73322401, 38191400, 38020400, 38527400, 38375400, 38186400, 5079400, 5085403, 5101400,
        5110400, 5061400, 5098402, 4073400, 4006400, 6073405]

datadir = '/home/vernaym/These/DATA'
workdir = '/home/vernaym/workdir/ASSIMILATION/'

latmax = 45.240
latmin = 44.990
lonmin = 6.010
lonmax = 6.490

# Liste des coordonnées des domaines connus lat_max, lat_min, lon_max, lon_min
coords = dict(
    alp = ['46450', '44100', '5400', '7200'],
    pyr = ['43500', '42000', '-2000', '3500'],
    cor = ['43000', '41000', '8000', '10500'],
    #GrandesRousses = ['45240', '44990', '6010', '6490'],
    GrandesRousses = ['45210', '45020', '6040', '6460'],  # TODO : modifier quand les bords du domaines seront inclus dans la localisation
    ange = ['45240', '44990', '6010', '6490']
)

all_experiments = dict(
        #GD0      = 'Assimilation_globale_2021073106_2022070106_daily.nc',
        LD0         = 'XP00_assimilation_quotidienne_sans_masque_sans_localisation/Assimilation_locale_2021120106_2022050106_daily_alp.nc',
        LD0G        = 'XP24_assimilation_quotidienne_loi_gamma/Assimilation_locale_2021120106_2022050106_daily_alp.nc',
        LDM4        = 'XP02_assimilation_quotidienne_avec_masque/Assimilation_locale_2021120106_2022050106_daily_alp_mask4.nc',
        LDM3        = 'XP10_assimilation_quotidienne_avec_masque3/Assimilation_locale_2021120106_2022050106_daily_alp_mask3.nc',
        LDM1        = 'XP11_assimilation_quotidienne_avec_masque1/Assimilation_locale_2021120106_2022050106_daily_alp_mask1.nc',
        LDM2        = 'XP12_assimilation_quotidienne_avec_masque2/Assimilation_locale_2021120106_2022050106_daily_alp_mask2.nc',
        LDD0        = 'XP21_assimilation_quotidienne_avec_debiaisage_uniforme/Assimilation_locale_2021120106_2022050106_daily_alp_debiasing0.nc',
        LDD1        = 'XP22_assimilation_quotidienne_avec_debiaisage1/Assimilation_locale_2021120106_2022050106_daily_alp_debiasing1.nc',
        LDD2        = 'XP20_assimilation_quotidienne_avec_debiaisage2/Assimilation_locale_2021120106_2022050106_daily_alp_debiasing2.nc',
        LDM4L       = 'XP05_assimilation_quotidienne_avec_masque_et_localisation/Assimilation_locale_2021120106_2022050106_daily_alp_localisation_mask4.nc',
        LDM4D       = 'XP06_assimilation_quotidienne_avec_masque_et_debiaisage/Assimilation_locale_2021120106_2022050106_daily_alp_mask4_debiasing.nc',
        LDM4D_BIS   = 'XP09_assimilation_quotidienne_avec_masque_et_debiaisage_ratio_moyen/Assimilation_locale_2021120106_2022050106_daily_alp_mask4_debiasing.nc',
        LDM4LD      = 'XP08_assimilation_quotidienne_avec_masque_localisation_et_debiaisage/Assimilation_locale_2021120106_2022050106_daily_alp_localisation_mask4_debiasing.nc',
        LH0         = 'XP01_assimilation_horaire_sans_masque_sans_localisation/Assimilation_locale_2021120106_2022050106_hourly_alp.nc',
        LH0G        = 'XP26_assimilation_horaire_loi_gamma/Assimilation_locale_2021120106_2022050106_hourly_alp.nc',
        LHM4D       = 'XP03_assimilation_horaire_avec_masque_et_debiaisage/Assimilation_locale_2021120106_2022050106_hourly_alp_mask4_debiasing.nc',
        LHM4DL      = 'XP07_assimilation_horaire_avec_masque_localisation_et_debiaisage/Assimilation_locale_2021120106_2022050106_hourly_alp_localisation_mask4_debiasing.nc',
        LDM5D2      = 'XP15_assimilation_quotidienne_avec_masque5_et_debiaisage2/Assimilation_locale_2021120106_2022050106_daily_alp_mask5_debiasing2.nc',
        LHM5D2      = 'XP16_assimilation_horaire_avec_masque5_et_debiaisage2/Assimilation_locale_2021120106_2022050106_hourly_alp_mask5_debiasing2.nc',
        LDM5        = 'XP17_assimilation_quotidienne_avec_masque5/Assimilation_locale_2021120106_2022050106_daily_alp_mask5.nc',
        LDM5D2L5    = 'XP18_assimilation_quotidienne_avec_masque5_et_debiaisage2_et_localisation5/Assimilation_locale_2021120106_2022050106_daily_alp_localisation5_mask5_debiasing2.nc',
        LGDM5D2L5   = 'XP25_assimilation_quotidienne_loi_gamma_mask5_localisation5_debiaising2/Assimilation_locale_2021120106_2022050106_daily_alp_localisation5_mask5_debiasing2.nc',
        LHM5D2L5    = 'XP19_assimilation_horaire_avec_masque5_debiaisage2_localisation5/Assimilation_locale_2021120106_2022050106_hourly_alp_localisation5_mask5_debiasing2.nc',
        LGHM5D2L5    = 'XP27_assimilation_horaire_mask5_localization5_debiasing2_gamma_likelyhood/Assimilation_locale_2021120106_2022050106_hourly_alp_localisation5_mask5_debiasing2.nc',
        LHM4DL20    = 'XP13_assimilation_horaire_avec_masque_debiaisage_et_localisation20/Assimilation_locale_2021120106_2022050106_hourly_alp_localisation20_mask4_debiasing.nc',
        LHM4DL50    = 'XP14_assimilation_horaire_avec_masque_debiaisage_et_localisation50/Assimilation_locale_2021120106_2022050106_hourly_alp_localisation50_mask4_debiasing.nc',
        LHM4DLS20T6 = 'XP23_assimilation_horaire_avec_masque4_debiaisage1_localisation_20-6/Assimilation_locale_2021120106_2022050106_hourly_alp_localisation20_mask4_debiasing1.nc',
        KD0         = 'EnsembleKalmanFilter/XP00_Rstat/EnKF_2021120106_2022050106_daily_alp.nc',
        KD1         = 'EnsembleKalmanFilter/XP01_Rstat_Rdyn/EnKF_2021120106_2022050106_daily_alp.nc',
        KD2         = 'EnsembleKalmanFilter/XP02_Rdyn/EnKF_2021120106_2022050106_daily_alp.nc',
        KD4         = 'EnsembleKalmanFilter/XP03_sans_normalisation/EnKF_2021120106_2022050106_daily_alp.nc',
        KDM6D3      = 'EnsembleKalmanFilter/XP07_mask6_debiaisage3/EnKF_2021120106_2022050106_daily_alp.nc',
        KDM8D3      = 'EnsembleKalmanFilter/XP05_debiaisage/EnKF_2021120106_2022050106_daily_alp.nc',
        KDM8        = 'EnsembleKalmanFilter/XP06_mask8/EnKF_2021120106_2022050106_daily_alp.nc',
    )

mask_experiments = dict(
        #GD0      = 'Assimilation_globale_2021073106_2022070106_daily.nc',
        #LD0      = 'XP00_assimilation_quotidienne_sans_masque_sans_localisation/Assimilation_locale_2021120106_2022050106_daily_alp.nc',
        #LDM4     = 'XP02_assimilation_quotidienne_avec_masque/Assimilation_locale_2021120106_2022050106_daily_alp_mask4.nc',
        #LDM3     = 'XP10_assimilation_quotidienne_avec_masque3/Assimilation_locale_2021120106_2022050106_daily_alp_mask3.nc',
        #LDM1     = 'XP11_assimilation_quotidienne_avec_masque1/Assimilation_locale_2021120106_2022050106_daily_alp_mask1.nc',
        #LDM2     = 'XP12_assimilation_quotidienne_avec_masque2/Assimilation_locale_2021120106_2022050106_daily_alp_mask2.nc',
        #LDM5     = 'XP17_assimilation_quotidienne_avec_masque5/Assimilation_locale_2021120106_2022050106_daily_alp_mask5.nc',
        #LDM5D2    = 'XP15_assimilation_quotidienne_avec_masque5_et_debiaisage2/Assimilation_locale_2021120106_2022050106_daily_alp_mask5_debiasing2.nc',
        KD09      = 'EnsembleKalmanFilter/XP09/EnKF_2021120106_2022050106_daily_alp.nc',
        KD28      = 'EnsembleKalmanFilter/XP28/EnKF_2021120106_2022050106_daily_alp.nc',
    )

daily_experiments = dict(
        LD0       = 'XP00_assimilation_quotidienne_sans_masque_sans_localisation/Assimilation_locale_2021120106_2022050106_daily_alp.nc',
#        LDM4      = 'XP02_assimilation_quotidienne_avec_masque/Assimilation_locale_2021120106_2022050106_daily_alp_mask4.nc',
        LDM4D     = 'XP06_assimilation_quotidienne_avec_masque_et_debiaisage/Assimilation_locale_2021120106_2022050106_daily_alp_mask4_debiasing.nc',
        #LDD2      = 'XP20_assimilation_quotidienne_avec_debiaisage2/Assimilation_locale_2021120106_2022050106_daily_alp_debiasing2.nc',
#        LDM4D_BIS = 'XP09_assimilation_quotidienne_avec_masque_et_debiaisage_ratio_moyen/Assimilation_locale_2021120106_2022050106_daily_alp_mask4_debiasing.nc',
#        LDM5D2    = 'XP15_assimilation_quotidienne_avec_masque5_et_debiaisage2/Assimilation_locale_2021120106_2022050106_daily_alp_mask5_debiasing2.nc',
#        LDM4L     = 'XP05_assimilation_quotidienne_avec_masque_et_localisation/Assimilation_locale_2021120106_2022050106_daily_alp_localisation_mask4.nc',
        LDM4LD    = 'XP08_assimilation_quotidienne_avec_masque_localisation_et_debiaisage/Assimilation_locale_2021120106_2022050106_daily_alp_localisation_mask4_debiasing.nc',
#        LDM5D2L5  = 'XP18_assimilation_quotidienne_avec_masque5_et_debiaisage2_et_localisation5/Assimilation_locale_2021120106_2022050106_daily_alp_localisation5_mask5_debiasing2.nc',  # WARNING : erreur_obs * 10
        LGDM5D2L5   = 'XP25_assimilation_quotidienne_loi_gamma_mask5_localisation5_debiaising2/Assimilation_locale_2021120106_2022050106_daily_alp_localisation5_mask5_debiasing2.nc',

    )

hourly_experiments = dict(
        LH0         = 'XP01_assimilation_horaire_sans_masque_sans_localisation/Assimilation_locale_2021120106_2022050106_hourly_alp.nc',
        LH0G        = 'XP26_assimilation_horaire_loi_gamma/Assimilation_locale_2021120106_2022050106_hourly_alp.nc',
        LHM4D       = 'XP03_assimilation_horaire_avec_masque_et_debiaisage/Assimilation_locale_2021120106_2022050106_hourly_alp_mask4_debiasing.nc',
        LHM4DL      = 'XP07_assimilation_horaire_avec_masque_localisation_et_debiaisage/Assimilation_locale_2021120106_2022050106_hourly_alp_localisation_mask4_debiasing.nc',
        LHM5D2      = 'XP16_assimilation_horaire_avec_masque5_et_debiaisage2/Assimilation_locale_2021120106_2022050106_hourly_alp_mask5_debiasing2.nc',
        LHM5D2L5    = 'XP19_assimilation_horaire_avec_masque5_debiaisage2_localisation5/Assimilation_locale_2021120106_2022050106_hourly_alp_localisation5_mask5_debiasing2.nc',
        LGHM5D2L5    = 'XP27_assimilation_horaire_mask5_localization5_debiasing2_gamma_likelyhood/Assimilation_locale_2021120106_2022050106_hourly_alp_localisation5_mask5_debiasing2.nc',
        LHM4DLS20T6 = 'XP23_assimilation_horaire_avec_masque4_debiaisage1_localisation_20-6/Assimilation_locale_2021120106_2022050106_hourly_alp_localisation20_mask4_debiasing1.nc',
    )

localisation_experiments = dict(
        LHM4DL   = 'XP07_assimilation_horaire_avec_masque_localisation_et_debiaisage/Assimilation_locale_2021120106_2022050106_hourly_alp_localisation_mask4_debiasing.nc',
        LHM4DL20 = 'XP13_assimilation_horaire_avec_masque_debiaisage_et_localisation20/Assimilation_locale_2021120106_2022050106_hourly_alp_localisation20_mask4_debiasing.nc',
        LHM4DL50 = 'XP14_assimilation_horaire_avec_masque_debiaisage_et_localisation50/Assimilation_locale_2021120106_2022050106_hourly_alp_localisation50_mask4_debiasing.nc',
    )


debiaising_experiments = dict(
#        LD0      = 'XP00_assimilation_quotidienne_sans_masque_sans_localisation/Assimilation_locale_2021120106_2022050106_daily_alp.nc',
#        LDD0      = 'XP21_assimilation_quotidienne_avec_debiaisage_uniforme/Assimilation_locale_2021120106_2022050106_daily_alp_debiasing0.nc',
#        LDD1      = 'XP22_assimilation_quotidienne_avec_debiaisage1/Assimilation_locale_2021120106_2022050106_daily_alp_debiasing1.nc',
#        LDD2      = 'XP20_assimilation_quotidienne_avec_debiaisage2/Assimilation_locale_2021120106_2022050106_daily_alp_debiasing2.nc',
#        KD14         = 'EnsembleKalmanFilter/XP14/EnKF_2021120106_2022050106_daily_alp.nc',
#        KD15         = 'EnsembleKalmanFilter/XP15/EnKF_2021120106_2022050106_daily_alp.nc',
#        KD16         = 'EnsembleKalmanFilter/XP16/EnKF_2021120106_2022050106_daily_alp.nc',
#        KD18         = 'EnsembleKalmanFilter/XP18/EnKF_2021120106_2022050106_daily_alp.nc',
        KD30          = 'EnsembleKalmanFilter/XP30/EnKF_2021120106_2022050106_daily_alp.nc',
        KD31          = 'EnsembleKalmanFilter/XP31/EnKF_2021120106_2022050106_daily_alp.nc',
    )

tmp = dict(
        LD0       = 'XP00_assimilation_quotidienne_sans_masque_sans_localisation/Assimilation_locale_2021120106_2022050106_daily_alp.nc',
        LDM4D_BIS = 'XP09_assimilation_quotidienne_avec_masque_et_debiaisage_ratio_moyen/Assimilation_locale_2021120106_2022050106_daily_alp_mask4_debiasing.nc',
        LDM5D2    = 'XP15_assimilation_quotidienne_avec_masque5_et_debiaisage2/Assimilation_locale_2021120106_2022050106_daily_alp_mask5_debiasing2.nc',
        LDM5D2L5  = 'XP18_assimilation_quotidienne_avec_masque5_et_debiaisage2_et_localisation5/Assimilation_locale_2021120106_2022050106_daily_alp_localisation5_mask5_debiasing2.nc',
        LHM5D2L5  = 'XP19_assimilation_horaire_avec_masque5_debiaisage2_localisation5/Assimilation_locale_2021120106_2022050106_hourly_alp_localisation5_mask5_debiasing2.nc',
    )

likelyhood_experiments = dict(
        LD0         = 'XP00_assimilation_quotidienne_sans_masque_sans_localisation/Assimilation_locale_2021120106_2022050106_daily_alp.nc',
        LD0G        = 'XP24_assimilation_quotidienne_loi_gamma/Assimilation_locale_2021120106_2022050106_daily_alp.nc',
        LDM5D2L5    = 'XP18_assimilation_quotidienne_avec_masque5_et_debiaisage2_et_localisation5/Assimilation_locale_2021120106_2022050106_daily_alp_localisation5_mask5_debiasing2.nc',
        LGDM5D2L5   = 'XP25_assimilation_quotidienne_loi_gamma_mask5_localisation5_debiaising2/Assimilation_locale_2021120106_2022050106_daily_alp_localisation5_mask5_debiasing2.nc',
    )

basic = dict(
#        LD0         = 'XP00_assimilation_quotidienne_sans_masque_sans_localisation/Assimilation_locale_2021120106_2022050106_daily_alp.nc',
#        LDD0        = 'XP21_assimilation_quotidienne_avec_debiaisage_uniforme/Assimilation_locale_2021120106_2022050106_daily_alp_debiasing0.nc',
    )

algo = dict(
#        LD0         = 'XP00_assimilation_quotidienne_sans_masque_sans_localisation/Assimilation_locale_2021120106_2022050106_daily_alp.nc',
        #LDM5D2L5    = 'XP18_assimilation_quotidienne_avec_masque5_et_debiaisage2_et_localisation5/Assimilation_locale_2021120106_2022050106_daily_alp_localisation5_mask5_debiasing2.nc',
        #KD0         = 'EnsembleKalmanFilter/XP00_Rstat/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD1         = 'EnsembleKalmanFilter/XP01_Rstat_Rdyn/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD2         = 'EnsembleKalmanFilter/XP02_Rdyn/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD4         = 'EnsembleKalmanFilter/XP03_sans_normalisation/EnKF_2021120106_2022050106_daily_alp.nc',
        #KDM6D3      = 'EnsembleKalmanFilter/XP07_mask6_debiaisage3/EnKF_2021120106_2022050106_daily_alp.nc',
        #KDD3        = 'EnsembleKalmanFilter/XP05_debiaisage/EnKF_2021120106_2022050106_daily_alp.nc',
        #KDM8        = 'EnsembleKalmanFilter/XP06_mask8/EnKF_2021120106_2022050106_daily_alp.nc',
        #KDM9        = 'EnsembleKalmanFilter/XP08_mask9/EnKF_2021120106_2022050106_daily_alp.nc',
        #KDM9D3      = 'EnsembleKalmanFilter/XP09_mask9_debiaising3_Rstat.Y/EnKF_2021120106_2022050106_daily_alp.nc',
        #KDM9D3_bis  = 'EnsembleKalmanFilter/XP11_mask9_debiaising3_Rstat.Ydebiaise/EnKF_2021120106_2022050106_daily_alp.nc',
        #KDM8D3      = 'EnsembleKalmanFilter/XP05_debiaisage/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD01         = 'EnsembleKalmanFilter/XP01/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD02         = 'EnsembleKalmanFilter/XP02/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD03         = 'EnsembleKalmanFilter/XP03/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD04         = 'EnsembleKalmanFilter/XP04/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD05         = 'EnsembleKalmanFilter/XP05/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD06         = 'EnsembleKalmanFilter/XP06/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD07         = 'EnsembleKalmanFilter/XP07/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD08         = 'EnsembleKalmanFilter/XP08/EnKF_2021120106_2022050106_daily_alp.nc',
#        KD09         = 'EnsembleKalmanFilter/XP09/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD10         = 'EnsembleKalmanFilter/XP10/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD11         = 'EnsembleKalmanFilter/XP11/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD12         = 'EnsembleKalmanFilter/XP12/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD13         = 'EnsembleKalmanFilter/XP13/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD14         = 'EnsembleKalmanFilter/XP14/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD15         = 'EnsembleKalmanFilter/XP15/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD16         = 'EnsembleKalmanFilter/XP16/EnKF_2021120106_2022050106_daily_alp.nc',
#        LDM9D3        = 'XP28_mask9_debiaising3_0.1_2/Assimilation_locale_2021120106_2022050106_daily_alp_mask9_debiasing3.nc',
#        KD17         = 'EnsembleKalmanFilter/XP17/EnKF_2021120106_2022050106_daily_alp.nc',
#        KD18         = 'EnsembleKalmanFilter/XP18/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD19         = 'EnsembleKalmanFilter/XP19/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD20         = 'EnsembleKalmanFilter/XP20/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD21         = 'EnsembleKalmanFilter/XP21/EnKF_2021120106_2022050106_daily_alp.nc',
#        KD22         = 'EnsembleKalmanFilter/XP22/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD23         = 'EnsembleKalmanFilter/XP23/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD24         = 'EnsembleKalmanFilter/XP24/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD25         = 'EnsembleKalmanFilter/XP25/EnKF_2021120106_2022050106_daily_alp.nc',
        #KD26         = 'EnsembleKalmanFilter/XP26/EnKF_2021120106_2022050106_daily_alp.nc',
#        KD27         = 'EnsembleKalmanFilter/XP27/EnKF_2021120106_2022050106_daily_alp.nc',
#        KD29          = 'EnsembleKalmanFilter/XP29/EnKF_2021120106_2022050106_daily_alp.nc',
#        KD30          = 'EnsembleKalmanFilter/XP30/EnKF_2021120106_2022050106_daily_alp.nc',
#        RS00          = 'RandomSampling/XP00/Random_Sampling_2021120106_2022050106_daily_alp.nc',
#        RS01          = 'RandomSampling/XP01/Random_Sampling_2021120106_2022050106_daily_alp.nc',
#        RS02          = 'RandomSampling/XP02/Random_Sampling_2021120106_2022050106_daily_alp.nc',
#        RS03          = 'RandomSampling/XP03/Random_Sampling_2021120106_2022050106_daily_alp.nc',
        # IUGG experiments :
        #PF29          = 'XP29_mask_relief_AROME/Assimilation_locale_2021120106_2022050106_daily_alp.nc',
        #KD33          = 'EnsembleKalmanFilter/XP33/EnKF_2021120106_2022050106_daily_alp.nc',
#        RS04          = 'RandomSampling/XP04/Random_Sampling_2021120106_2022050106_daily_alp.nc'
        ####################
        #KD35          = 'EnsembleKalmanFilter/XP35/EnKF_2021120106_2022050106_daily_alp.nc',
        #RS07          = 'RandomSampling/XP07/Random_Sampling_2021120106_2022050106_daily_alp.nc',
        #RS08          = 'RandomSampling/XP08/Random_Sampling_2021120106_2022050106_daily_alp.nc',
        #RS09          = 'RandomSampling/XP09/Random_Sampling_2021120106_2022050106_daily_alp.nc',
        #RS10          = 'RandomSampling/XP10/Random_Sampling_2021120106_2022050106_daily_alp.nc',
        #RS11          = 'RandomSampling/XP11/Random_Sampling_2021120106_2022050106_daily_alp.nc',
        #RS12          = 'RandomSampling/XP12/Random_Sampling_2021120106_2022050106_daily_alp.nc',
        #RS13          = 'RandomSampling/XP13/Random_Sampling_2021120106_2022050106_daily_alp.nc',
        #RS14          = 'RandomSampling/XP14/Random_Sampling_2021120106_2022050106_daily_alp.nc',
        #RS15          = 'RandomSampling/XP15/Random_Sampling_2021120106_2022050106_daily_alp.nc',
        #RS16          = 'RandomSampling/XP16/Random_Sampling_2021120106_2022050106_daily_alp.nc',  --> WMA reference experiment
        #RS17          = 'RandomSampling/XP17/Random_Sampling_2021120106_2022050106_daily_alp.nc',
        #RS18          = 'RandomSampling/XP18/Random_Sampling_2021120106_2022050106_daily_alp.nc',
        #RS19          = 'RandomSampling/XP19/Random_Sampling_2021120106_2022050106_daily_alp.nc',
        #RS20          = 'RandomSampling/XP20/Random_Sampling_2021120106_2022050106_daily_alp.nc',
        #RS22          = 'RandomSampling/XP22/Random_Sampling_2021120106_2022050106_daily_alp.nc',
        #RS23          = 'RandomSampling/XP23/Random_Sampling_2021120106_2022050106_daily_alp.nc',  # commit f0d6615fe98865a0c87c88dca4dda7e70a1725a1
        RS23          = 'RandomSampling/XP23/Random_Sampling_2021120106_2022043006_daily_alp.nc',  # commit 
        RS24          = 'RandomSampling/XP24/Random_Sampling_2021120106_2022043006_daily_alp.nc',  # commit 
        RS25          = 'RandomSampling/XP25/Random_Sampling_2021120106_2022043006_daily_alp.nc',  # commit 
        ################################################################################################
        # PHD committee :
        RS21          = 'RandomSampling/XP21/Random_Sampling_2021120106_2022050106_daily_alp.nc',
        #PF31          = 'XP31/Assimilation_locale_2021120106_2022050106_daily_alp_mask9_debiasing3.nc',
        #KD35          = 'EnsembleKalmanFilter/XP35/EnKF_2021120106_2022050106_daily_alp.nc',
        ################################################################################################
    )



experiments_map = dict(
    reference                  = dict(),  # Plot only ANTILOPE and RAW ensemble
    algo                       = algo,
    basic                      = basic,
    debiaising_experiments     = debiaising_experiments,
    tmp                        = tmp,
    mask_experiments           = mask_experiments,
    daily_experiments          = daily_experiments,
    hourly_experiments         = hourly_experiments,
    likelyhood_experiments     = likelyhood_experiments,
    localisation_experiments   = localisation_experiments,
)
experiments = experiments_map[xpid]

savedir = os.path.join(f"/home/vernaym/These/figures/evaluation/{domain}", xpid)
if not os.path.exists(savedir):
    os.makedirs(savedir)

xpid_label = dict(
        antilope      = 'Raw ANTILOPE',
        #antilope      = 'Raw field',
        wma           = 'WMA',
        antiloper     = 'De-biased ANTILOPE + error',
        #antiloped     = 'ANTILOPE + error + debiaisage',
        antiloped     = 'De-biased ANTILOPE',
        #antilopec     = 'ANTILOPE + debiaisage + correction',
        #antilopec     = 'De-biasing + WMA',
        antilopec     = 'Pre-processed ANTILOPE',
        #raw           = 'Raw PE-AROME ensemble',
        raw           = 'PE-AROME',
        GD0           = 'Global daily analysis',
        LD0           = 'Daily analysis with PF',
        LD0G          = 'Daily analysis with gamma likelyhood and no option',
        LDM4          = 'Daily analysis with mask4',
        LDM3          = 'Daily analysis with mask3',
        LDM1          = 'Daily analysis with mask1',
        LDM2          = 'Daily analysis with mask2',
        LDM5          = 'Daily analysis with mask5',
        LDD0          = 'Daily analysis with uniform debiasing',
        LDD1          = 'Daily analysis with debiasing1',
        LDD2          = 'Daily analysis with debiasing2',
        LDM5D2        = 'Daily analysis with mask5 and debiasing2',
        LDM4L         = 'Daily analysis with mask4 and localization',
        LDM4D         = 'Daily analysis with mask4 and debiasing1',
        LDM4D_BIS     = 'Daily analysis with mask4 and uniform debiasing1',
        LDM4LD        = 'Daily analysis with mask4 and localization and debiasing1',
        LDM5D2L5      = 'Daily analysis with mask5 and localization5 and debiasing2',
        LGDM5D2L5     = 'Daily analysis with mask5 and localization5 and debiasing2 and gamma likelyhood',
        LH0           = 'Hourly analysis with no option',
        LH0G          = 'Hourly analysis with gamma likelyhood and no option',
        LHM4D         = 'Hourly analysis with mask4 and debiasing1',
        LHM5D2        = 'Hourly analysis with mask5 and debiasing2',
        LHM4DL        = 'Hourly analysis with mask4 and localization7 and debiasing1',
        LHM4DL20      = 'Hourly analysis with mask4 and localization20 and debiasing1',
        LHM4DL50      = 'Hourly analysis with mask4 and localization50 and debiasing1',
        LHM5D2L5      = 'Hourly analysis with mask5 and localization5 and debiasing2',
        LGHM5D2L5     = 'Hourly analysis with gamma likelyhood, mask5 and localization5 and debiasing2',
        LHM4DLS20T6   = 'Hourly analysis with mask4 and localization (20,6) and debiasing1',
        KD0           = 'Daily analysis with EnKF (Rstat)',
        KD1           = 'Daily analysis with EnKF (Rstat+Rdyn)',
        KD2           = 'Daily analysis with EnKF (Rdyn)',
        KD4           = 'Daily analysis with EnKF without ECM normalisation (bis)',
        KDM6D3        = 'Daily analysis with EnKF with mask6 and debiasing3',
        KDM8D3        = 'Daily analysis with EnKF with debiasing3 and mask8',
        KDM8          = 'Daily analysis with EnKF with mask8',
        KDM9          = 'Daily analysis with EnKF with mask9',
        KDM9D3        = 'Daily analysis with EnKF with mask9 and debiassing3 R=Rstat*Y',
        KDM9D3_bis    = 'Daily analysis with EnKF with mask9 and debiassing3 R=Rstat*Ydebiaise',
        KD01          = 'EnKF, mask9=diff',
        KD02          = 'EnKF, mask9=diff+estimated_ratio',
        KD03          = 'EnKF, mask9=estimated_ratio',
        KD04          = 'EnKF, mask9=estimated_ratio+diff_25_0.1_2, debiaisage3=25_0.1_2',
        KD05          = 'EnKF, mask9=estimated_ratio+diff_15_0.15_2, debiaisage3=15_0.15_2',
        KD06          = 'EnKF, mask9=estimated_ratio+diff_25_0.15_2, debiaisage3=25_0.15_2',
        KD07          = 'EnKF, debiaisage3=15_0.15_2',
        KD08          = 'EnKF, mask9=(estimated_ratio+diff_25_0.1_2)*2, debiaisage3=25_0.1_2',
        KD09          = 'EnKF, mask9=estimated_ratio, debiaisage3=0.1_2',
        KD10          = 'EnKF, mask9=estimated_ratio, debiaisage3=0.15_2',
        KD11          = 'EnKF, mask9=smooth, debiaisage3=smooth15',
        #KD12          = 'EnKF, mask9=estimated_ratio, debiaisage3=0.15_1',
        #KD12          = 'EnKF, d0=0.15, c0=1',
        KD13          = 'EnKF, mask9=estimated_ratio, debiaisage3=0.5_1',
        KD14          = 'EnKF, d0=0.1, c0=1',
        KD15          = 'EnKF, d0=0.25, c0=1',
        KD16          = 'EnKF, d0=0.25, c0=2',
        KD17          = 'EnKF, mask9=estimated_ratio, debiaisage3=25_0.15_2',
        KD18          = 'EnKF, d0=0.1, c0=2',
        KD19          = 'EnKF, mask9=estimated_ratio, debiaisage3=10_0.1_2',
        KD20          = 'EnKF, mask9=estimated_ratio, debiaisage3=25_0.15_2, R=((Y+1)*std)²',
        KD22          = 'EnKF, mask9=estimated_ratio, debiaisage3=0.2_2, localisation=0.05',
        KD23          = 'EnKF, mask9=estimated_ratio, debiaisage3=0.2_2, localisation=0.06, Rstat*20',
        KD24          = 'EnKF, mask9=estimated_ratio, debiaisage3=0.2_2, localisation=0.06, Rstat=exp(std)',
        KD25          = 'EnKF, mask9=estimated_ratio, debiaisage3=0.2_2, localisation=0.06, Rstat*50',
        KD26          = 'EnKF, mask9=estimated_ratio, debiaisage3=0.2_2, localisation=0.06, Rstat*30, Rdyn=ref_field*std',
        KD27          = 'EnKF, mask9=estimated_ratio, debiaisage3=0.2_2, localisation=0.06, Rstat*100, Rdyn=Y*std',
        KD28          = 'EnKF, debiaisage=0.2_1_new, localisation=0.06, Rstat*20, Rdyn=Y*std',
        KD29          = 'EnKF, debiaisage=0.2_2, Rstat from eval, Bstat',
        KD30          = 'Daily analysis with Ensemble Kalman Filter and debiaising',
        KD31          = 'Daily analysis with Ensemble Kalman Filter and no debiasing',  # Idem KD30 mais sans débiaisage
        #KD33          = 'Ensemble Kalman Filter analysis',
        KD33          = 'EnKF',
        KD34          = 'Ensemble Kalman Filter analysis',
        KD35          = 'EnKF',
        PF29          = 'PF',
        #PF29          = 'Particle Filter analysis',
        PF30          = 'Particle Filter analysis',
        PF31          = 'PF',
        #LDM9D3        = 'PF, mask9=estimated_ratio, debiaisage3=0.1_2',
        LDM9D3        = 'PF, d0=0.1, c0=2',
        RS00          = 'Random Sampling',
        RS01          = 'Random Sampling without debiasing',
        RS02          = 'Random Sampling with increased dynamic dispersion',
        RS03          = 'Random Sampling with increased dynamic dispersion + static',
        RS04          = 'Random Sampling',
        RS05          = 'Random Sampling',
        RS07          = 'Random Sampling with dynamic correction only',
        RS08          = 'Random Sampling with dynamic correction and qq adjustment',
        RS09          = 'Random Sampling with dynamic correction only',
        RS10          = 'Random Sampling with dynamic correction only and sd=sd1+sd2',
        #RS11          = 'Random Sampling with dynamic correction only and normal distribution',
        RS11          = 'Random Sampling ref (dyn corr + normal dist)',
        RS12          = 'Random Sampling with new observation uncertainty formulation',
        RS13          = 'Random Sampling RS12 + increased observation error',
        RS14          = 'Random Sampling RS13 + gamma distribution',
        #RS15          = 'Random Sampling RS13 + gamma distribution + IDW instead of exp',
        RS15          = 'RS',  # --> Overdispersif
        RS16          = 'Random Sampling with WMA only',  # reference for WMA method evaluation
        RS17          = 'RS',  # Idem RS 15 mais avec erreur obs=sd2 seulement (moins surdispersif et un peu moins biaisé)
        RS18          = 'RS18',  # random perturbations = gamma*0.2*obs + gamma*sd
        RS19          = 'RS19',  # random perturbations = gamma*obs*0.4 (ou 0.3 ?) + gamma*sd
        RS20          = 'RS20',  # Estimated ratio only on mountain ridges
        RS21          = 'RS',  # PHD committee
        RS22          = 'RS22',  # Test qq adjustment
        RS23          = 'RS - dynamic error/ratio estimation',  # Test dynamic ratio/error estimation
        RS24          = 'RS24',
        RS25          = 'RS25',
    )

colors = dict(
    raw       = 'Grey',
    antilopec = 'orange',
    RS21      = 'red',
    RS23      = 'k',
    RS24      = 'maroon',
    RS25      = 'green',
    PF31      = 'green',
    KD35      = 'blue',
)

def nearest(array, value):
    """ Find element of "array" the closer to 'value' """
    # Security to ensure that the station is within the simulated domain.
    if np.abs(array - value).data.min() < 0.1:
        return float(array[np.abs(array - value).argmin()].data)
    else:
        print(f'ERROR : no corresponding pixel found for value {value}')

class Evaluation(object):

    def __init__(self):
        #self.data = xr.Dataset()
#        self.ensemble = ensemble  # DataArray(lat,lon,time,member)
        #self.Ne = 16  # TODO : à definir dynamiquement
        self.scores = None
        self.threshold = 10  # threshold to use as event detection in the Brier Score
        self.lpn = None
        self.obs_error = None
        self.thresholds = [x for x in range(1, 31)]
        #self.thresholds = [0.5] + [x for x in range(1, 31)]
        #self.thresholds = [x/10 for x in range(1,10)] + [x for x in range(1, 31)]

    def ensemble_attributes(self):
        disp = self.dispersion()

    @property
    def mean(self):
        return  self.ensemble.mean(axis=3).rr.data

    def mean_error(self, simu, obs):
        return simu.mean(axis=1) - obs

    def median_error(self, simu, obs):
        return simu.median(axis=1) - obs

    def bias(self, simu, obs, *args, **kw):

        simu = simu[~np.isnan(obs)]
        obs = obs[~np.isnan(obs)]
        simu = simu[obs>0]  # TODO : TMP !!!!!
        obs=obs[obs>0]  # TODO : TMP !!!!!

        if np.shape(simu) == np.shape(obs):  # "Simulation" déterministe
            bias = simu - obs
        else:  # Simulation s'ensmble
            bias = self.mean_error(simu, obs)

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

    def error_frequency(self, simu, obs, treshold=0.2, *args, **kw):

        simu = simu[~np.isnan(obs)]
        obs = obs[~np.isnan(obs)]

        if np.shape(simu) == np.shape(obs):  # "Simulation" déterministe
            bias = simu - obs
        else:  # Simulation s'ensmble
            bias = self.mean_error(simu, obs)

        error_above_treshold = np.where((simu>=obs*(1+treshold)) & (simu<=obs*(1-treshold)))
        #freq_error = (np.count_nonzero(error_above_treshold) / len(error_above_treshold) * 100
        freq_error = (len(error_above_treshold) / len(obs)) * 100

        return freq_error

    def spread(self, ensemble):
        """
        Mean spread over dates.
        Spread = sqrt(1/(N-1)*sum(X-M)**2)
        """
        #ensemble = ensemble[np.where(ensemble.mean(axis=1))>1]  # Only for dates
        ensemble = ensemble.transpose()
        spread = np.sqrt(np.mean(np.square(ensemble-np.mean(ensemble, axis=0)), axis=0))
        mean = np.mean(spread)
        var = np.sqrt(np.mean((spread-mean)**2))

        return mean, var

    def spread_skill(self, simu, obs, by_station=False, *args, **kw):

        simu = simu[~np.isnan(obs)]
        obs = obs[~np.isnan(obs)]
        mean = simu.mean(axis=1)
        mask = np.where((mean>1) & (obs>1))
        simu = simu[mask].transpose()
        obs = obs[mask]
        spread = np.sqrt(np.mean(np.square(simu-np.mean(simu, axis=0)), axis=0))
        error = np.abs(np.mean(simu, axis=0)-obs)
        if by_station:
            spread = np.mean(spread)
            error = np.sqrt(np.mean(np.square(error)))

        return spread, error, obs

    def rmse(self, simu, obs, *args, **kw):

        simu = simu[~np.isnan(obs)]
        obs = obs[~np.isnan(obs)]

        if np.shape(simu) == np.shape(obs):  # "Simulation" déterministe
            rmse = np.sqrt(np.nanmean(np.square(simu-obs))) if np.nanmean(np.square(simu-obs)) > 0 else np.nan
        else:  # Simulation d'ensemble
            rmse = np.sqrt(np.nanmean(np.square(np.median(simu, axis=1) - obs)))
        return rmse

    def brier_skill_score(self, simu, obs, ref, threshold=10):
        """  BSS = 1 - BS / BSref  """
        return 1 - self.brier(simu, obs, threshold) / self.brier(ref, obs, threshold)

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

    def ROC(self, simu, obs, product, ax, Ne=16, threshold=10):
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

    def rank_histogram(self, ensemble, obs, product, ax, onlypos=False, *args):
        """
        Inspired from : https://github.com/oliverangelil/rankhistogram/blob/master/ranky.py
        When two or more forecasts have same value (most commonly 0), random selection is made for which bin receives the count.
        """
        # WARNING : la condition obs>0 réduit PLUS le nombre de cas.
        # Le choix de la condition est très important car il fait apparaitre ou disparaitre
        # une énorme majorité des situations où tout est à 0 sauf 1 membre (grand pic à gauche de l'histogramme)
#        maxsim = np.amax(ensemble, 1)
#        ensemble = ensemble[obs>0]
#        obs = obs[obs>0]
#        ensemble = ensemble[(~np.isnan(obs)) & ((obs>0) | (maxsim>0))]
#        obs = obs[(~np.isnan(obs)) & ((obs>0) | (maxsim>0))]
#        position = np.array([])
#        for idx, obs in enumerate(obs):
#            position = np.append(position, np.searchsorted(np.sort(simu[idx]), obs))
#        ax.hist(position, bins=range(np.shape(simu)[1]))

        ensemble = np.transpose(ensemble)  # Shape (Nmember, Ndates)
        ensemble = ensemble[:,~np.isnan(obs)]
        obs = obs[~np.isnan(obs)]

        if onlypos:
            # Filter out situations where everything is 0mm --> already managed with random positionning of the observation in the ensemble in this case
            maxsim = np.amax(ensemble, 0)
            #ensemble = ensemble[:,(~np.isnan(obs)) & ((obs>0) | (maxsim>0))]
            #obs = obs[(~np.isnan(obs)) & ((obs>0) | (maxsim>0))]
            # Filter out situations where observation is 0mm
            #ensemble = ensemble[:,(~np.isnan(obs)) & (obs>0)]
            #obs = obs[(~np.isnan(obs)) & (obs>0)]
            mean = ensemble.mean(axis=0)
            #mask = np.where((obs>1) & (mean>1))
            #mask = np.where((obs>0) & (mean>0))
            #mask = np.where((obs>5) & (mean>5))
            #mask = np.where((mean>1))
            #mask = np.where((obs>3))
            mask = np.where((obs>0))
            #mask = np.where((obs>1))
            ensemble = ensemble[:, mask]
            ensemble = np.squeeze(ensemble, axis=1)  # TODO : comprendre pourquoi cette ligne est nécessaire
            obs = obs[mask]

        combined = np.vstack((obs[np.newaxis], ensemble))

        # Computing ranks
        ranks = np.apply_along_axis(lambda x: rankdata(x, method='min'), 0, combined)

        # Computing ties'
        ties = np.sum(ranks[0]==ranks[1:], axis=0)
        ranks = ranks[0]
        tie = np.unique(ties)

        for i in range(1,len(tie)):
            # Random positionning of "all 0s" cases
            index = ranks[ties==tie[i]]
            # print('randomizing tied ranks for ' + str(len(index)) + ' instances where there is ' + str(tie[i]) + ' tie/s. ' + str(len(tie)-i-1) + ' more to go')
            ranks[ties==tie[i]] = [np.random.randint(index[j], index[j]+tie[i]+1, tie[i])[0] for j in range(len(index))]

        #return np.histogram(ranks, bins=np.linspace(0.5, combined.shape[0]+0.5, combined.shape[0]+1))
        #ax.hist(ranks, bins=range(np.shape(ensemble)[0]))
        ax.hist(ranks, bins=np.linspace(0.5, combined.shape[0]+0.5, combined.shape[0]+1))
        ax.set_xlabel('Position of the observation in the ensemble')
        ax.set_ylabel('Number of occurences')

    def reliability_diagram(self, simu, obs, product, ax, Ne=16):
        ndays = len(obs)
        simu = simu[~np.isnan(obs)]
        obs = obs[~np.isnan(obs)]
        proba, catsize, freq_occ, global_freq_occ = self.probability_classes(simu, obs, Ne=Ne)
        # TODO : taille du marker proportionelle au nombre de prevision dans une categorie
        ax.plot(proba, freq_occ, marker=None, linestyle='-', label=f'{product}')
        ax.scatter(proba, freq_occ, catsize)

    def probability_classes(self, simu, obs, Ne=16, nb_cat=17):

        # TODO : la décomposition du score de Brier devrait donner le même résultat
        # que le calcul direct (BS=BSfiab-BSres+BSunc), mais ce n'est pas le cas...

        catsize  = list()
        freq_occ = list()
        proba    = list()
        for Nm in range(nb_cat):
            Ni = np.count_nonzero(np.count_nonzero(simu>=self.threshold, axis=1)==Nm)
            if Ni > 0:
                proba.append(Nm/Ne)
                catsize.append(Ni)
                # TODO : problème avec les dimensions de "simu" lorsque simu est un ensemble...
                freq_occ.append(np.count_nonzero(obs[np.count_nonzero(simu>=self.threshold, axis=1)==Nm]>=self.threshold)/Ni)
        ndays = len(obs)
        global_freq_obs = np.count_nonzero(obs[obs>=self.threshold]) / ndays

        return np.array(proba), np.array(catsize), np.array(freq_occ), global_freq_obs

    def reliability(self, simu, obs, *args):
        simu = simu[~np.isnan(obs)]
        obs = obs[~np.isnan(obs)]
        ndays = len(obs)
        proba, catsize, freq_occ, global_freq_occ = self.probability_classes(simu, obs)
        reliability = 1/ndays*np.sum(catsize*(proba-freq_occ)**2)
        #reliability = 1/ndays*np.sum([Ni*(proba-focc)**2 for (proba, Ni, focc) in zip(proba, catsize, freq_occ)])  # Equivalent
        print('reliability=',reliability)
        return reliability

    def resolution(self, simu, obs, *args):
        simu = simu[~np.isnan(obs)]
        obs = obs[~np.isnan(obs)]
        ndays = len(obs)
        proba, catsize, freq_occ, global_freq_obs = self.probability_classes(simu, obs)
        resolution = 1/ndays*np.sum(catsize*(freq_occ-global_freq_obs)**2)
        #resolution = 1/ndays*np.sum([Ni*(focc-global_freq_obs)**2 for (Ni, focc) in zip(catsize, freq_occ)])  # Equivalent
        print('Resolution=',resolution)
        return resolution

    def uncertainty(self, simu, obs, *args):
        simu = simu[~np.isnan(obs)]
        obs = obs[~np.isnan(obs)]
        proba, catsize, freq_occ, global_freq_obs = self.probability_classes(simu, obs)
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

    def read_lpn(self):
        lpn = pd.read_csv(os.path.join(datadir, 'LPN_nivometeo_20210801_20220801.csv'), sep=';', parse_dates=['H_NIVO.DAT'], dtype={'H_NIVO.NUM_POSTE':int, 'H_NIVO.ALTI_LPNX':int})
        lpn.rename(columns={'H_NIVO.ALTI_LPNX':'LPNX', 'H_NIVO.NUM_POSTE':'num_poste', 'H_NIVO.DAT':'date'}, inplace=True)

        return lpn

#    def read_nivometeo_coords(self, domain):
#        metadata = pd.read_csv(os.path.join(datadir, 'postes_nivometeo.csv'), sep=';')
#        latmax, latmin, lonmin, lonmax = np.array(coords[domain]).astype(float)/1000.
#        subdata = metadata[(metadata['poste_nivo.lat_dg']>=latmin) & (metadata['poste_nivo.lat_dg']<=latmax) & (metadata['poste_nivo.lon_dg']>=lonmin) & (metadata['poste_nivo.lon_dg']<=lonmax)]
#        return dict(zip(np.array(subdata['poste_nivo.num_poste']), zip(np.array(subdata['poste_nivo.lat_dg']), np.array(subdata['poste_nivo.lon_dg']))))
#
    def read_antilope(self, dates):
        fic = 'ANTILOPEQ_evaluation.nc'
        if not os.path.exists(os.path.join(datadir, fic)):
            #filename = 'ANTILOPEQ_2021073106_2022070106_GrandesRousses.nc'
            filename = 'ANTILOPEH_2021103000_2022060200_alp.nc'
            antilope = xr.open_dataset(os.path.join(datadir, filename))
            #antilope = antilope.loc[{'time':np.intersect1d(dates, antilope.time.data)}]
            if filename.startswith('ANTILOPEH'):
                # Convert hourly precipitation into 24h precipitation between 7h (6 UTC in winter) J-1 and 7h (6 UTC) J
                # Problem : the xarray tools to do that allows only accumulations between
                # 0h and 24h.
                # solution : shift time serie by 7h, compute 24h accumulations and
                # shift back !
                antilope['time'] = antilope.time-np.timedelta64(7, 'h')
                antilope = antilope.resample(time='1D').sum(dim='time')  # !!! VERY SLOW !!! WARNING : does not work with pandas>=2.0.0
                antilope['time'] = antilope.time+np.timedelta64(30, 'h')
            antilope.to_netcdf(os.path.join(datadir, fic))
        else:
            antilope = xr.open_dataset(os.path.join(datadir, fic))

        return antilope

    def read_corrected_antilope(self):
        filename = 'ANTILOPEQ_2021120106_2022050106_alp_corrected.nc'
        antilope = xr.open_dataset(os.path.join(datadir, filename))
        antilope = antilope.loc[{'member':0}]

        return antilope

    def read_raw_ensemble(self, dates):
        filename = 'RAW_pearome_alp_daily.nc'
        if not os.path.exists(os.path.join(datadir, filename)):
            #filenames = [os.path.join(datadir, f'aspearome_{mb:03d}_2021073106_2022070106_GrandesRousses_daily.nc') for mb in range(1,17)]
            filenames = [os.path.join(datadir, f'aspearome_{mb:03d}_2021102806_2022060206_alp_hourly.nc') for mb in range(1,17)]
            #raw = xr.open_mfdataset(filenames, combine='nested', concat_dim='member').compute().clip(0)
            raw = xr.open_mfdataset(filenames, combine='nested', concat_dim='member', chunks={'time': 24})  # Setting chunks is critical (read the doc !)
            raw['member']=np.arange(1,17)
            # Convert hourly precipitation into 24h precipitation between 6h J-1 and 6h J
            # Problem : the xarray tools to do that allows only accumulations between
            # 0h and 23h.
            # solution : shift time serie by 7h, compute 24h accumulations and
            # shift back !
            raw['time'] = raw.time-np.timedelta64(7, 'h')
            raw = raw.resample(time='D').sum(dim='time')  # !!! VERY SLOW !!!
            raw['time'] = raw.time+np.timedelta64(30, 'h')
            #raw = raw.compute().clip(0)  # TODO : try without computing (seems towork !)
            raw = raw.clip(0)  # TODO : try without computing (seems towork !)
            raw = raw.transpose('lat', 'lon', 'time', 'member')  # transpose data to put dimension in the same order as assimilated fields
            raw = raw.compute()
            raw.to_netcdf(os.path.join(datadir, filename))
        else:
            raw = xr.open_dataset(os.path.join(datadir, filename))
        raw = raw.loc[{'time':dates}]

        return raw

    def read_simu(self, filename):

#        if not os.path.exists(filename):
#            print(f'WARNING : file {filename} does not exist, looking for it under {workdir}')
#            filename = os.path.join(workdir, filename)

        if os.path.exists(filename):
            simulation =  xr.open_dataset(filename)
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
            #toto=simulation.loc[{'lat':44.99, 'lon':6.01, 'member':1}].rr.data
            simulation = simulation.resample(time='D').sum(dim='time')  # !!! VERY SLOW !!!
            #tata=simulation.loc[{'lat':44.99, 'lon':6.01, 'member':1}].rr.data
            #print(toto[126*24:127*24-1])
            #print(np.sum(toto[126*24:127*24-1]), tata[126])
            simulation['time'] = simulation.time+np.timedelta64(30, 'h')
            #print(simulation.time.data[126])
            #import pdb
            #pdb.set_trace()

        return simulation

    def evaluate(self):

        # TODO : store each XPI score in a file to reuse instead of recompute
        # ==> inverse loops on stations and XPIDs
#        if os.path.exists(os.path.join(datadir, 'scores.nc')):
#            self.scores = xr.open_dataset(os.path.join(datadir, 'scores.nc'))
#            return

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
#        self.data = self.read_obs_clim()  # Read observation --> self.obs

        # Remove time dimension from metadata :
        self.data['lon']=np.max(self.data.lon, axis=1)
        self.data['lat']=np.max(self.data.lat, axis=1)
        self.data['alti']=np.max(self.data.alti, axis=1)
        self.data['nom']=np.max(self.data.nom, axis=1)
#        data = self.obs.loc[self.obs["Q.num_poste"].isin(liste_poste)]  # TODO a adapter
#        self.stations = self.data[['num_poste', 'nom', 'lat', 'lon', 'alti']].drop_duplicates()

        dates_obs = self.data.date

        antilope = self.read_antilope(dates_obs)
        dates_antilope = antilope.time.data
        dstd = np.datetime64('2022-03-27')
        #dates_antilope = dates_antilope[dates_antilope<dstd]
        dates = np.intersect1d(dates_obs[:-1], dates_antilope)
        antilope = antilope.loc[{'time':dates}]
        self.data = self.data.loc[{'date':dates}]

#        data = dict(antilope=list(), wma=list(), antiloped=list(), antilopec=list())
        #data = dict(antilope=list(), raw=list(), antilopec=list())
        data = dict(antilopec=list(), raw=list())
#        data = dict(antilope=list(), antiloped=list(), antiloper=list())
        #data = dict(antilope=list(), raw=list(), antilopec=list())
        #data = dict(antilopec=list())
        #data = dict(antilope=list(), antiloped=list(), antilopec=list())
        #data = dict(antilope=list(), antiloped=list())

        #mask = xr.open_dataset(os.path.join(datadir, 'mask', f"Estimated_ratio.nc"))
        self.rat  = xr.open_dataset(os.path.join("/home/vernaym/workdir/ASSIMILATION/mask/alp", "Estimated_ratio.nc"))  # To test a new estimation
        self.obs_error = xr.open_dataset(os.path.join("/home/vernaym/workdir/ASSIMILATION/mask/alp", "Observation_error.nc"))  # To test a new estimation
        #self.obs_error = self.obs_error.rename({'Observation error (mm)':'error'})
        rat = self.rat.Ratio
        #error = self.obs_error['Observation error (mm)']
        error = self.obs_error.Uncertainty
        def debiaise(ds):
            return ds / rat
        def to_ensemble(ds):
            ds1 = ds + ds * error
            ds2 = ds - ds * error
#            ds1 = ds * (1 + 0.263) + error
#            ds2 = ds * (1 - 0.263) - error
#            ds1 = ds + error
#            ds2 = ds - error
            return xr.concat([ds, ds1, ds2], 'member')

        # ANTILOPE + debiasage
        #antiloped = antilope.groupby('date').apply(debiaise)
        antiloped = antilope.apply(debiaise)
        #antiloped = antilope.apply(to_ensemble)
        # Génération d'un ensemble de 3 membres prenant en compte l'erreur d'observation
#        antiloped = antiloped.expand_dims('member')
#        antiloped = antiloped.apply(to_ensemble)
#        antiloped = antiloped.clip(0)
#        antiloped = antiloped.transpose('lat', 'lon', 'time', 'member')

        # ANTILOPE + error
        antiloper = antiloped.expand_dims('member')
        antiloper = antiloper.apply(to_ensemble)
        antiloper = antiloper.clip(0)
        antiloper = antiloper.transpose('lat', 'lon', 'time', 'member')

        simus = dict()
        antc = False
        for xpid,filename in experiments.items():
            tmp = self.read_simu(os.path.join(workdir, filename)).loc[{'time':dates}]
            print(xpid)
            #if xpid.startswith('RS'):
            #if xpid == 'RS12':
            if xpid == 'RS25':
                antilopec = tmp.loc[{'member':0}]
                antilopec = antilopec.loc[{'time':dates}]
                antc = True
            tmp = tmp.loc[{'member':range(1,17)}]
            simus[xpid] = tmp

        if 'raw' in data.keys():
            raw = self.read_raw_ensemble(dates)
            self.data['member'] = np.arange(1,17)
            self.data['pseudo_member'] = np.arange(1,4)

        if 'antilopec' in data.keys() and not antc:
            antilopec = self.read_corrected_antilope()
            antilopec = antilopec.loc[{'time':dates}]
            #antilopec = antilopec.loc[{'time':dates}]

        if 'wma' in data.keys():
            filename = 'Random_Sampling_2021120106_2022050106_daily_alp.nc'
            wma = xr.open_dataset(os.path.join('/home/vernaym/workdir/ASSIMILATION/RandomSampling/XP16', filename))
            wma = wma.loc[{'member':0}]
            wma = wma.loc[{'time':dates}]

#        scores_list = ['reliability', 'resolution', 'uncertainty', 'rmse', 'bias', 'brier', 'error_frequency']
        scores_list = ['rmse', 'bias', 'ratio', 'spread'] + [f'brier_{int(threshold*10)}' for threshold in self.thresholds] + ['CRPS']
        scores_dict = dict()

        spreadvar = list()  # list of spread variances

        #dates = dates[:10]
        liste_postes = np.array([])
        for idx, num_poste in enumerate(self.data.num_poste.data):
        #for idx, num_poste in enumerate(indep):
            print(f'Station {idx+1}/{len(self.data.num_poste.data)}')
#            num_poste = row['num_poste']
#            lat       = row['lat']
#            lon       = row['lon']
#            dates     = self.data[self.data['num_poste']==num_poste]['date'].values
            t1 = time.time()
            tmp = self.data.loc[{'num_poste':num_poste}]
            lat = tmp.lat
            lon = tmp.lon
            obs = tmp.obs.data
#            if num_poste == 73015400:
#                import pdb
#                pdb.set_trace()
#                toto = rat.sel({'lat':nearest(rat.lat, lat), 'lon':nearest(rat.lon, lon)})
            alti = tmp.alti.data.max()  # Altitude du poste
            lat = tmp.lat.data.max()  # Latitude du poste
            lon = tmp.lon.data.max()  # Longitude du poste
            t2 = time.time()
            print(f'Reading obs informations took {(t2-t1)*1000.}ms')
            if len(obs[~np.isnan(obs)]) >= 100:  # Filter stations with too few observations
                liste_postes = np.append(liste_postes, num_poste)
                #obs = obs[:10]
                if 'antilope' in data.keys():
                    data['antilope'].append(antilope.sel({'lat':nearest(antilope.lat, lat), 'lon':nearest(antilope.lon, lon)}).rr.data)
                if 'antiloped' in data.keys():
                    data['antiloped'].append(antiloped.sel({'lat':nearest(antiloped.lat, lat), 'lon':nearest(antiloped.lon, lon)}).rr.data)
                if 'antiloper' in data.keys():
                    data['antiloper'].append(antiloper.sel({'lat':nearest(antiloper.lat, lat), 'lon':nearest(antiloper.lon, lon)}).rr.data)
                if 'antilopec' in data.keys():
                    data['antilopec'].append(antilopec.sel({'num_poste':num_poste}).rr.data)
                    #data['antilopec'].append(antilopec.sel({'lat':nearest(antilopec.lat, lat), 'lon':nearest(antilopec.lon, lon)}).rr.data)
                if 'wma' in data.keys():
                    data['wma'].append(wma.sel({'num_poste':num_poste}).rr.data)
                t3 = time.time()
                print(f'Reading antilope informations took {(t3-t2)*1000.}ms')
                if 'raw' in data.keys():
                    data['raw'].append(raw.sel({'lat':nearest(raw.lat, lat), 'lon':nearest(raw.lon, lon)}).rr.data)
                t4 = time.time()
                print(f'Reading raw ensemble took {(t4-t3)*1000.}ms')
                for xpid,filename in experiments.items():
                    if xpid not in data.keys():
                        data[xpid] = list()
                    if domain == 'GrandesRousses':  # gridded data
                        data[xpid].append(simus[xpid].sel({'lat':nearest(simus[xpid].lat, lat), 'lon':nearest(simus[xpid].lon, lon)}).rr.data)
                    else:
                        data[xpid].append(simus[xpid].sel({'num_poste':num_poste}).rr.data)
                    #t5 = time.time()
                    #print(f'Reading simulation {xpid} took {(t5-t4)*1000.}ms')
#                self.temporal_plot(dates, obs, num_poste, lat, lon, alti, antilope=data['antilope'][-1])
                #self.temporal_plot(dates, obs, num_poste, lat, lon, alti, antilope=data['antilope'][-1], corrected=data['antilopec'][-1])
                #self.temporal_plot(dates, obs, num_poste, lat, lon, alti, raw=data['raw'][-1], antilope=data['antilope'][-1])
                #self.temporal_plot(dates, data['LH0'][-1], obs, num_poste, raw=data['raw'][-1], antilope=data['antilope'][-1], simu2=data['LD0'][-1])
                #self.temporal_plot(dates, data['LDML'][-1], obs, num_poste, raw=data['raw'][-1], antilope=data['antilope'][-1], simu2=data['LDM'][-1])
                #self.temporal_plot(dates, data['LDMLD'][-1], obs, num_poste, alti, raw=data['raw'][-1], antilope=data['antilope'][-1])
                #if num_poste == 73194401:
#                if num_poste == 5001400:
#                    self.temporal_plot(dates, 'LDM5D2', data['LDM5D2'][-1], obs, num_poste, alti, raw=data['raw'][-1], antilope=data['antilope'][-1])
                t6 = time.time()
                #print(f'Temporal plot took {(t6-t5)*1000.}ms')
                #idx=10
                #self.plot_assimilation(data['raw'][-1][:idx], data['LD0'][-1][:idx], data['antilope'][-1][:idx], 'LD0', num_poste, np.datetime_as_string(dates.data[:idx], unit='D'))

#                if num_poste == 74134400:
#                    date=np.datetime64("2022-02-17T06:00")
#                    antilope.sel({'lat':nearest(antilope.lat, lat), 'lon':nearest(antilope.lon, lon)}).loc[{'time':np.datetime64("2022-02-17T06:00")}].rr.data

                #if not os.path.exists(os.path.join(datadir, 'scores.nc')):
                # Spread skill
                ensemble_products = ['raw'] + [xpid for xpid in experiments.keys()]
                for product in data.keys():
                    if product not in scores_dict.keys():
                        scores_dict[product] = {score:list() for score in scores_list}
                    for score_name, score in scores_dict[product].items():
                        if score_name.startswith('brier'):
                            threshold = float(score_name.split('_')[-1])/10.
                            if product == 'antilope':
                                score.append(getattr(self, 'brier')(data[product][-1][~np.isnan(obs)], obs[~np.isnan(obs)], Ne=1, threshold=threshold))
                            #elif product in ['antiloped', 'antiloper']:
                            elif product in ['antiloper']:
                                score.append(getattr(self, 'brier')(data[product][-1][~np.isnan(obs)], obs[~np.isnan(obs)], Ne=3, threshold=threshold))
                            else:
                                score.append(getattr(self, 'brier')(data[product][-1][~np.isnan(obs)], obs[~np.isnan(obs)], threshold=threshold))
                        elif score_name == 'spread':
                            if product in ensemble_products:
                                spread, var = getattr(self, 'spread')(data[product][-1][np.where((~np.isnan(obs))&(obs>1))])
                                score.append(spread)
                                spreadvar.append(var)
                        else:
                            print(score_name, product)
                            score.append(getattr(self, score_name)(data[product][-1][~np.isnan(obs)], obs[~np.isnan(obs)]))
                    t7 = time.time()
                    #print(f'Computing score for simulation {xpid} took {(t7-t6)*1000.}ms')
                t7 = time.time()
            else:
                self.data = self.data.where(self.data.num_poste!=num_poste, drop=True)  # Drop station

        for product in data.keys():
            if product in ensemble_products:
                rmse = np.array(scores_dict[product]['rmse'])
                spread = np.array(scores_dict[product]['spread'])
                spreadvar = np.array(spreadvar)
                print(product)
                tools.plot_scatter(rmse, spread, 'RMSE (mm)', 'Mean spread (mm)', f"spread_skill_{product}_by_station.pdf", savedir, addtext=liste_postes)
                rr = np.nanmean(self.data.obs.data, axis=1)
                tools.plot_scatter(rr, spread/rmse, 'Mean precipitation (mm)', 'Spread/RMSE', f"spread_skill_vs_rr_{product}_by_station.pdf", savedir, addtext=liste_postes)
            scores_dict[product].pop('spread')
        scores_list.remove('spread')

#        if os.path.exists(os.path.join(datadir, 'scores.nc')):
#            self.scores = xr.open_dataset(os.path.join(datadir, 'scores.nc'))
#            return

        # TODO : optimiser le calcul des scores !

        if 'antilope' in data.keys():
            self.data['antilope'] = (('num_poste', 'date'), data['antilope'])
        if 'antiloper' in data.keys():
            self.data['antiloper'] = (('num_poste', 'date', 'pseudo_member'), data['antiloper'])
        if 'antiloped' in data.keys():
            #self.data['antiloped'] = (('num_poste', 'date', 'pseudo_member'), data['antiloped'])
            self.data['antiloped'] = (('num_poste', 'date'), data['antiloped'])
        if 'antilopec' in data.keys():
            self.data['antilopec'] = (('num_poste', 'date'), data['antilopec'])
        if 'wma' in data.keys():
            self.data['wma'] = (('num_poste', 'date'), data['wma'])
        if 'raw' in data.keys():
            self.data['raw'] = (('num_poste', 'date', 'member'), data['raw'])
        for xpid in experiments.keys():
            self.data[xpid] = (('num_poste', 'date', 'member'), data[xpid])
        t8 = time.time()
        #print(f'Filling self.data took {(t8-t7)*1000.}ms')

        fig, ax = plt.subplots()
        thresholds = np.arange(0.1, 1.01, 0.1)
        obse = self.data.obs.data.flatten()
        for product in data.keys():
#            if product == 'antiloper':
#                import pdb
#                pdb.set_trace()
            if 'member' in self.data[product].dims or 'pseudo_member' in self.data[product].dims:
                simu = self.data[product].stack(points=["num_poste", "date"]).data.transpose()
                freq_error = scores.error_frequency(simu, obse)
                ax.axhline(freq_error, color=next(ax._get_lines.prop_cycler)['color'], label=xpid_label[product])
                #ax.axhline(freq_error, label=xpid_label[product])
            else:
                simu = self.data[product].data.flatten()
                freq_error = list()
                for threshold in thresholds:
                    freq_error.append(scores.error_frequency(simu, obse, threshold=threshold))
                ax.plot(thresholds*100, np.array(freq_error), label=xpid_label[product], color=next(ax._get_lines.prop_cycler)['color'])
        ax.set_xlabel('Error threshold (%)')
        ax.set_ylabel('Frequency of error above threshold (%)\nFrequency of observation outside the ensemble (%)')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(os.path.join(savedir, "error_frequency.pdf"), format='pdf')
        plt.close(fig)

        # Spread skill
        products = [xpid for xpid in experiments.keys()]
        if 'raw' in data.keys(): products = products + ['raw']
        for product in products:  # Only for ensemble simulations
            spread, error, rr = self.spread_skill(self.data[product].data.reshape(-1, 16), self.data.obs.data.flatten())
            tools.plot_scatter(error, spread, 'Error (mm)', 'Spread (mm)', f"spread_skill_{product}.pdf", savedir, color=rr)
            #tools.plot_scatter(rr, error, 'Precipitation (mm)', 'Error (mm)', f"error_vs_intensity_{product}.pdf", savedir)
            #tools.plot_scatter(rr, spread, 'Precipitation (mm)', 'Spread (mm)', f"spread_vs_intensity_{product}.pdf", savedir)
            #tools.plot_scatter(rr, spread/error, 'Precipitation (mm)', 'Spread / Error', f"spread_over_error_vs_intensity_{product}.pdf", savedir)
            #tools.plot_scatter(error, spread/error, 'Error (mm)', 'Spread / Error', f"spread_over_error_vs_error_{product}.pdf", savedir)
#            fig, ax = plt.subplots()
#            ax.scatter(error, spread)
#            vmax = max(np.max(spread), np.max(error))
#            ax.set_ylim(top=vmax)
#            ax.set_xlim(top=vmax)
#            fig.savefig(os.path.join(savedir, f"spread_skill_{product}.pdf"), format='pdf')
#            plt.close(fig)

        fig1,ax1 = plt.subplots()
        if 'raw' in data.keys():
            for product in ['raw'] + [xpid for xpid in experiments.keys()]:
                self.reliability_diagram(self.data[product].data.reshape(-1, 16), self.data.obs.data.flatten(), product, ax1)
                fig2,ax2 = plt.subplots()
                self.rank_histogram(self.data[product].data.reshape(-1, 16), self.data.obs.data.flatten(), product, ax2)
                ax2.set_ylim(top=1200)
                fig2.savefig(f'{savedir}/rank_histogram_{product}.pdf', format='pdf')
                plt.close(fig2)
                fig2,ax2 = plt.subplots()
                self.rank_histogram(self.data[product].data.reshape(-1, 16), self.data.obs.data.flatten(), product, ax2, onlypos=True)
                ax2.set_ylim(top=400)
                fig2.savefig(f'{savedir}/rank_histogram_onlypos_{product}.pdf', format='pdf')
                plt.close(fig2)
                if product.startswith('RS'):
                    for poste in self.data[product].num_poste.data:
                        simu = self.data[product].loc[{'num_poste':poste}].data
                        obs = self.data.obs.loc[{'num_poste':poste}].data
                        fig, ax = plt.subplots()
                        self.rank_histogram(simu, obs, product, ax, onlypos=True)
                        fig.savefig(os.path.join(savedir, 'hists', f'rank_histogram_onlypos_{product}_{poste}.pdf'), format='pdf')
                        #self.rank_histogram(simu, obs, product, ax)
                        #fig.savefig(os.path.join(savedir, 'hists', f'rank_histogram_{product}_{poste}.pdf'), format='pdf')
                        plt.close(fig)
        ax1.plot([0,1], [0,1], linestyle=':', color='k')
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_xlabel('Forecast Probability')
        ax1.set_ylabel('Observed Frequency')
        ax1.legend(fontsize=20)
        fig1.savefig(f'{savedir}/reliability_diagram_{self.threshold}.pdf', format='pdf')
        plt.close(fig1)

        for threshold in [0, 1, 10, 20]:
            fig,ax = plt.subplots()
            ax.set_title(f'Threshold={threshold}mm')
            if 'raw' in data.keys():
                for product in ['raw'] + [xpid for xpid in experiments.keys()]:
                    # TODO : vérifier les données (virer les dates où obs=nan,...)
                    self.ROC(self.data[product].data.reshape(-1, 16), self.data.obs.data.flatten(), xpid_label[product], ax, threshold=threshold)
            if 'antilope' in data.keys():
                self.ROC(self.data['antilope'].data.flatten(), self.data.obs.data.flatten(), 'antilope', ax, threshold=threshold)
            if 'antilopec' in data.keys():
                self.ROC(self.data['antilopec'].data.flatten(), self.data.obs.data.flatten(), 'Pre-processed ANTILOPE', ax, threshold=threshold)
            if 'antiloper' in data.keys():
                self.ROC(self.data['antiloper'].data.reshape(-1, 3), self.data.obs.data.flatten(), 'antiloper', ax, Ne=3, threshold=threshold)
            if 'antiloped' in data.keys():
                #self.ROC(self.data['antiloped'].data.reshape(-1, 3), self.data.obs.data.flatten(), 'antiloped', ax, Ne=3, threshold=threshold)
                self.ROC(self.data['antiloped'].data.flatten(), self.data.obs.data.flatten(), 'antiloped', ax, threshold=threshold)
            ax.set_xlim([0, 0.5])
            ax.set_ylim([0.5, 1])
            ax.set_xlabel('False alarm rate')
            ax.set_ylabel('Sucess rate')
            ax.legend(fontsize=14)
            fig.savefig(f'{savedir}/ROC_threshold_{threshold}mm.pdf', format='pdf')
        t9 = time.time()
        print(f'Ploting ROC curves took {(t9-t8)*1000.}ms')

        tmp = {"score":{"dims": ("score"), "data":scores_list}, "num_poste":{"dims": ("num_poste"), "data":liste_postes}}
        tmp.update({key:{"dims": ("score", "num_poste"), "data":[value[score] for score in scores_list]} for key,value in scores_dict.items()})
        self.scores = xr.Dataset.from_dict(tmp)
#        self.scores.to_netcdf(os.path.join(datadir, f'scores_{xpid}.nc'))
        t10 = time.time()
        print(f'Saving scores took {(t10-t9)*1000.}ms')

#        self.rmse(simu[num_poste], obs[num_poste], num_poste)
#        reliability, resolution, uncertainty = self.brier_decomposition(simu[num_poste], obs[num_poste], threshold)
#        print('BSfiab+BSres+BSunc=',reliability-resolution+uncertainty)
#        brier       = self.brier_score(simu[num_poste], obs[num_poste], threshold)
#                nearest_lat = nearest(self.ensemble.lat, lat)
#                nearest_lon = nearest(self.ensemble.lon, lon)
#                print(f'Poste {num_poste} ({len(tmp)} data)')
#                dates[num_poste] = np.array(tmp["Q.dat"] + timedelta(hours=30))  # Date yyyymmdd is the cumul between yyyymmdd06 and yyyymm(d+1)06
#                obs[num_poste]  = np.array(tmp['Q.rr'])
#                # Select corresponding simulations
#                simu[num_poste] = np.transpose(self.ensemble.sel({'lat':nearest_lat, 'lon':nearest_lon}).loc[{'time':dates[num_poste]}].rr.data)
#                self.rmse(simu[num_poste], obs[num_poste], num_poste)
#                reliability, resolution, uncertainty = self.brier_decomposition(simu[num_poste], obs[num_poste], threshold)
#                print('BSfiab+BSres+BSunc=',reliability-resolution+uncertainty)
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
        # TODO : do not plot individual brier scores
        for score in self.scores.score.data:
            products = [var for var in self.scores.data_vars]
            if len(products) <=2:
                fig,ax = plt.subplots(figsize=(12,13))
            else:
                fig,ax = plt.subplots(figsize=(12,10))
            if score.startswith('brier'):
                pass
                #threshold = int(score.split('_')[-1])
                #fig2,ax2 = plt.subplots(figsize=(22,18))

            pos = 1
            pos2 = 1
            labels = []
            labels2 = []

            def add_num_poste(axis, pos, liste_score):
                for idx, poste in enumerate(self.scores.num_poste.data):
                    if not np.isnan(liste_score[idx]):
                        if not np.isnan(liste_score[idx]):
                            # Plot station numbers :
                            #axis.text(pos, liste_score[idx], str(int(poste)), fontsize=6)
                            # Plot only horizontal lines :
                            axis.plot(pos, liste_score[idx], linestyle='', marker='_', markersize='20', color='k')
                    else:
                        print(f'{score} of product {product} not available for poste {str(int(poste))}')

            for product in products:
                x = self.scores.loc[{'score':score}][product].data
                add_num_poste(ax, pos, x)
                print(score, product)
                labels.append(self.add_label(ax.violinplot(x[~np.isnan(x)], showmeans=True, positions=[pos]), xpid_label[product], color=colors[product]))
                if score == 'bias':
                    ax.axhline(color='k')
                if score == 'ratio':
                    ax.axhline(1, color='k')
                if score.startswith('brier') and product not in ['antilope', 'antiloped', 'wma', 'antilopec', 'raw']:
                    pass
#                    ref = self.scores.loc[{'score':score}]['raw'].data
#                    bss = 1 - x / ref
#                    add_num_poste(ax2, pos2, bss)
#                    labels2.append(self.add_label(ax2.violinplot(bss[~np.isnan(bss)], showmeans=True, positions=[pos2]), xpid_label[product]))
#                    ax2.axhline(color='k')
#                    pos2 += 1
                if len(products) <= 2:
                    pos +=0.5
                else:
                    pos += 1

            if score.startswith('brier'):
                pass
                #ax.set_ylabel(f'{score}', fontsize=28)
                #ax2.set_ylabel('Brier Skill Score', fontsize=28)
                #ax2.set_xticklabels([''] + products[2:], fontsize=28)
                #ax2.set_xticks(range(len(products)))
                #ax2.legend(*zip(*labels2), fontsize=18)
            else:
                ax.set_ylabel(f'{score} (mm)', fontsize=16)

            #ax.set_xticklabels([''] + products, fontsize=28)
            #ax.set_xticks(range(len(products)+2))
            ax.yaxis.set_tick_params(labelsize=16)
            ax.legend(*zip(*labels), fontsize=16)

            if score.startswith('brier'):
                pass
                #fig.savefig(f'{savedir}/{score}.pdf', formatout='pdf',  bbox_inches='tight')
                #fig.savefig(f'{savedir}/{score}.pdf', format='pdf',  bbox_inches='tight')
                #fig2.savefig(f'{savedir}/brier_skill_score_{threshold}.pdf', formatout='pdf',  bbox_inches='tight')
                #fig2.savefig(f'{savedir}/brier_skill_score_{threshold}.pdf', format='pdf',  bbox_inches='tight')
            else:
                #fig.savefig(f'{savedir}/{score}.pdf', formatout='pdf',  bbox_inches='tight')
                fig.savefig(f'{savedir}/{score}.pdf', format='pdf',  bbox_inches='tight')
            plt.close('all')

        self.plot_brier_evolution()

    def plot_brier_evolution(self):
        fig,ax = plt.subplots(figsize=(12,10))
        products = [var for var in self.scores.data_vars]
        for product in products:
            brier = np.array([])
            for threshold in self.thresholds:
                brier = np.append(brier, np.mean(self.scores.loc[{'score':f'brier_{int(threshold*10)}'}][product].data))
            ax.plot(self.thresholds, brier, label=xpid_label[product], linewidth=2, color=colors[product])
            #ax.semilogx(self.thresholds, brier, label=xpid_label[product], linewidth=3)
        ax.legend(fontsize=18)
        ax.set_ylabel('Brier Score', fontsize=16)
        ax.set_xlabel('Threshold (mm)', fontsize=16)
        #ax.set_xticklabels(self.thresholds, fontsize=18)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        #ax.set_yticks(fontsize=18)

        fig.savefig(f'{savedir}/brier_evolution.pdf', format='pdf', bbox_inches='tight')

    def add_label(self, violin, label, color=None):
        """ Customize violinplot by adding a label"""
        import matplotlib.patches as mpatches
        if color is None:
            color = violin["bodies"][0].get_facecolor().flatten()
        else:
            violin["bodies"][0].set_color(color)
            violin['cmeans'].set_color(color)
            violin['cmaxes'].set_color(color)
            violin['cmins'].set_color(color)
            violin['cbars'].set_color(color)
            #for item in violin.keys():
            #    item.set_color(color)
            #violin["bodies"][0].set_facecolors(color)
            #violin["bodies"][0].set_facecolor(color)
            #violin["bodies"][0].set_edgecolors(color)
            #violin["bodies"][0].set_edgecolor(color)

        return (mpatches.Patch(color=color), label)

    def read_observation_error(self):
        self.obs_error = xr.open_dataset(os.path.join("/home/vernaym/workdir/ASSIMILATION/mask/alp", "Observation_error.nc"))  # To test a new estimation

    def read_ratio(self):
        self.rat = xr.open_dataset(os.path.join("/home/vernaym/workdir/ASSIMILATION/mask/alp", "Estimated_ratio.nc"))  # To test a new estimation

    def temporal_plot(self, time, obs, num_poste, lat, lon, alti, raw=None, antilope=None, corrected=None, xpid=None, simu=None, simu2=None):
        # TODO : add flexibility in the number and oreder of simulations (use dict !)

        if self.lpn is None:
            self.lpn = self.read_lpn()

        if self.obs_error is None:
            self.read_observation_error()
        #error = self.obs_error.loc[{'num_poste':num_poste, 'time':time}].erreur_obs.data
        error = self.obs_error.sel({'lat':nearest(self.obs_error.lat, lat), 'lon':nearest(self.obs_error.lon, lon)}).error.data

        if self.rat is None:
            self.read_rat()
        rat = self.rat.Ratio.sel({'lat':nearest(self.rat.lat, lat), 'lon':nearest(self.rat.lon, lon)}).data

        lpn = self.lpn.loc[self.lpn['num_poste']==num_poste]
        diff_alti_lpn = lpn.LPNX - alti

        self.labels = []
        fig, ax = plt.subplots(figsize=(100,9))
        ref, = plt.plot(time, obs, marker='.', linestyle='', color='k')
        self.labels.append((ref, 'Nivometeo reference'))
        positions = mpl.dates.date2num(time)
        if simu is not None:
            self.add_label(plt.violinplot(np.transpose(simu), positions=positions), xpid_label[xpid])
        if raw is not None:
            #add_label(plt.violinplot(np.transpose(raw), positions=positions), 'Raw ensemble', color='sandybrown')
            #add_label(plt.violinplot(np.transpose(raw), positions=positions), 'Raw ensemble')
            self.add_label(plt.violinplot(np.transpose(raw), positions=positions), 'Raw ensemble')
        #add_label(plt.violinplot(np.transpose(simu), positions=positions), 'Hourly assimilation', color='limegreen')
        if antilope is not None:
            #antpe, = plt.plot(time, antilope, marker='+', linestyle='', color='red')
            #antpe = plt.errorbar(positions, antilope, yerr=error+0.263*antilope, fmt="+", color='red', alpha=1)
            antpe = plt.errorbar(positions, antilope, yerr=antilope*error, fmt="+", color='red', alpha=1)
            self.labels.append((antpe, 'Antilope'))
            #antped = plt.errorbar(positions, antilope/rat, yerr=error+0.263*antilope/rat, fmt="+", color='blue', alpha=0.5)
            antped = plt.errorbar(positions, antilope/rat, yerr=error*antilope/rat, fmt="+", color='blue', alpha=0.5)
            self.labels.append((antped, 'Antilope debiaisé'))
        if corrected is not None:
            antpec, = plt.plot(positions, corrected, marker='+', linestyle='', color='green')
            #antpec = plt.errorbar(positions, corrected, yerr=corrected*error, fmt="+", color='green', alpha=0.5)
            self.labels.append((antpec, 'Antilope Corrected field'))
        if simu2 is not None:
            #add_label(plt.violinplot(np.transpose(simu2), positions=positions), 'Daily assimilation', color='skyblue')
            self.add_label(plt.violinplot(np.transpose(simu2), positions=positions), 'Daily assimilation')
        ax.set_xlabel('Date')
        ax.set_ylabel('24 hour precipitation (mm)')
        ax.legend(*zip(*self.labels), fontsize=32)
        ax.axhline(y=0, linewidth=1, color='k')
        #rrmax = int(np.ceil(max([np.nanmax(obs), np.nanmax(simu), np.nanmax(raw), np.nanmax(antilope)])))+10
        rrmax = int(np.ceil(max([np.nanmax(obs), np.nanmax(antilope)])))+10
        for rr in range(10, rrmax, 10):
            ax.axhline(y=rr, linewidth=0.1, color='k', linestyle='dotted')
        ax.set_ylim(-rrmax, rrmax)

        # make a plot with different y-axis using second axis object
        ax2=ax.twinx()
        ax2.plot(lpn.date, diff_alti_lpn, color="k", marker="*", linestyle='')
        ax2.set_ylabel("Difference between LPN max and station elevation (m)", color="blue", fontsize=14)

        #fig.savefig(f'{savedir}/{num_poste}_{xpid}.pdf', formatout='pdf',  bbox_inches='tight')
        fig.savefig(f'{savedir}/{num_poste}_{xpid}.pdf', format='pdf',  bbox_inches='tight')
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
            plt.plot(ref, marker='_', markersize=30, markeredgewidth=3, color='dark')
        #fig.savefig(f'{savedir}/assim_{xpid}_{num_poste}_{date}.pdf', formatout='pdf',  bbox_inches='tight')
        fig.savefig(f'{savedir}/assim_{xpid}_{num_poste}_{date}.pdf', format='pdf',  bbox_inches='tight')

        #sns.violinplot(data=data, y='24 hour precipitation (mm)', split=True, hue='Simulation')

    def eval_simu(self):
        pass

if __name__ == "__main__":

    evaluation = Evaluation()
    #evaluation.ensemble_attributes()
    evaluation.plot_scores()

