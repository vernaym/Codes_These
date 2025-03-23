
import os
import numpy as np
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt

import CRPS.CRPS as pscore


fontsize = 20
matplotlib.rcParams.update({'font.size': fontsize})

savedir = '/home/vernaym/These/NO_TRANSFER/figures/illustration'

obs = 3

for i in range(1, 5):

    if i == 4:
        mu1 = obs - 1
        mu2 = obs - 1
    else:
        mu1 = obs
        mu2 = obs

    if i == 1:
        nmembers = 17
    else:
        nmembers = 10000

    std1 = 1
    simu1 = np.sort(np.random.normal(loc=mu1, scale=std1, size=nmembers))
    mean1 = np.mean(simu1)
    spread1 = np.std(simu1)

    std2 = 0.2
    simu2 = np.sort(np.random.normal(loc=mu2, scale=std2, size=nmembers))
    mean2 = np.mean(simu2)
    spread2 = np.std(simu2)

    proba = np.arange(1, nmembers + 1) / nmembers

    fig, ax = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    ax[0].axvline(obs, ymin=0, color='k', label=f'Observation={obs:.1f} m', linewidth=4)
    x = np.linspace(0, 5, 100)
    ax[0].plot(x, norm.pdf(x, loc=mu1, scale=std1) / mu1, color='blue', label=f'Moyenne={mu1:.1f} m, Dispersion={std1:.1f} m')
    if i == 1:
        tmp = np.sort(np.random.normal(loc=mu1, scale=std1, size=17))
        ax[0].bar(tmp, norm.pdf(tmp, loc=mu1, scale=std1) / mu1, color='blue', width=0.01)

    ax[1].vlines(obs, ymin=0, ymax=1, color='k')

    if i == 1:
        simu1 = np.sort(np.random.normal(loc=mu1, scale=std1, size=10000))
        mean1 = np.mean(simu1)
        spread1 = np.std(simu1)
        proba = np.arange(1, 10000 + 1) / 10000

    # Simu 1
    proba_0 = - (proba[0] - simu1[0] * (proba[1] - proba[0]) / (simu1[1] - simu1[0])) / ((proba[1] - proba[0]) / (simu1[1] - simu1[0]))
    ax[1].plot(np.sort(np.append([0, proba_0, 4], simu1)), np.sort(np.append([0, 0, 1], proba)), color='blue')
    proba_obs_1 = proba[simu1 < obs][-1] + (obs - simu1[simu1 < obs][-1]) * (proba[simu1 >= obs][0] - proba[simu1 < obs][-1]) / (simu1[simu1 >= obs][0] - simu1[simu1 < obs][-1])
    x1 = np.sort(np.append(simu1[simu1 < obs], [proba_0, obs]))
    y1 = np.sort(np.append(proba[simu1 < obs], [0, proba_obs_1]))
    x2 = np.sort(np.append([1, proba_obs_1], proba[simu1 >= obs]))
    y2 = np.sort(np.append([obs, 4], simu1[simu1 >= obs]))
    #crps1 = np.sum(y1[1:] * (x1[1:] - x1[0:-1])) + np.sum(y2[1:] * (x2[1:] - x2[0:-1]))
    crps1 = pscore(simu1, obs).compute()[0]
    print(f'mean1={mean1}')
    print(f'crps1={crps1}')
    ax[1].fill_between(np.sort(np.append(simu1[simu1 < obs], [proba_0, obs])), 0, np.sort(np.append(proba[simu1 < obs], [0, proba_obs_1])), color='blue', alpha=0.5, label=f'CRPS={crps1:.2f} m')
    ax[1].fill_between(np.sort(np.append([obs, 4], simu1[simu1 >= obs])), np.sort(np.append([1, proba_obs_1], proba[simu1 >= obs])), 1, color='blue', alpha=0.5)

    # Simu 2
    if i > 2:
        ax[0].plot(x, norm.pdf(x, loc=mu2, scale=std2) / (2 * mu2), color='red', label=f'Moyenne={mu2:.1f} m, Dispersion={std2:.1f} m')
        #ax[0].bar(simu2, norm.pdf(simu2, loc=mu2, scale=std2) / mu2, color='red', width=0.01)
        proba_0 = - (proba[0] - simu2[0] * (proba[1] - proba[0]) / (simu2[1] - simu2[0])) / ((proba[1] - proba[0]) / (simu2[1] - simu2[0]))
        if len(simu2[simu2 >= obs]) > 0:
            proba_obs_2 = proba[simu2 < obs][-1] + (obs - simu2[simu2 < obs][-1]) * (proba[simu2 >= obs][0] - proba[simu2 < obs][-1]) / (simu2[simu2 >= obs][0] - simu2[simu2 < obs][-1])
        else:
            proba_obs_2 = 1
        x1 = np.sort(np.append(simu2[simu2 < obs], [proba_0, obs]))
        y1 = np.sort(np.append(proba[simu2 < obs], [0, proba_obs_2]))
        x2 = np.sort(np.append([1, proba_obs_2], proba[simu2 >= obs]))
        y2 = np.sort(np.append([obs, 4], simu2[simu2 >= obs]))
        #crps2 = np.sum(y1[1:] * (x1[1:] - x1[0:-1])) + np.sum(y2[1:] * (x2[1:] - x2[0:-1]))
        crps2 = pscore(simu2, obs).compute()[0]
        print(f'mean2={mean2}')
        print(f'crps2={crps2}')
        ax[1].plot(np.sort(np.append([0, proba_0, 4], simu2)), np.sort(np.append([0, 0, 1], proba)), color='red')
        ax[1].fill_between(np.sort(np.append(simu2[simu2 < obs], [proba_0, obs])), 0, np.sort(np.append(proba[simu2 < obs], [0, proba_obs_2])), color='red', alpha=0.5, label=f'CRPS={crps2:.2f} m')
        ax[1].fill_between(np.sort(np.append([obs, 4], simu2[simu2 >= obs])), np.sort(np.append([1, proba_obs_2], proba[simu2 >= obs])), 1, color='red', alpha=0.5)

    ax[0].set_xlim(0, 5)
    ax[0].set_ylim(0, 0.5)
    ax[0].set_ylabel('Probabilité', fontsize=fontsize)
    ax[1].set_xlim(0, 5)
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel('Hauteur de neige (m)', fontsize=fontsize)
    ax[1].set_ylabel('Fonction de répartition', fontsize=fontsize)

    ax[0].legend(loc='upper left')
    ax[1].legend(loc='upper left')

    fig.tight_layout()
    fig.savefig(os.path.join(savedir, f'illustration_CRPS_{i}.pdf'))

    plt.close(fig)
