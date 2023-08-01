import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


#obs=pd.read_csv('/home/vernaym/These/DATA/obs_quotidiennes_RR_2021080106_2022071906.data', sep=';')
obs=pd.read_csv('/home/vernaym/These/DATA/obs_horaires_auto_RR_20211101_20220430.csv', sep=';')
elevations = obs.groupby(['num_poste']).alti.mean()
elevations = np.sort(elevations)[::-1]
fig, ax = plt.subplots()
ax.plot(range(len(elevations)), elevations, linestyle='', marker='+', markersize=5)
ax.set_xlabel('ANTILOPE rain-gauges', fontsize=14)
ax.set_ylabel('Elevation (m)', fontsize=14)
fig.savefig('/home/vernaym/These/figures/elevation_pluvios_antilope.pdf', format='pdf')

