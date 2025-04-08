import xarray as xr
import pandas as pd
import numpy as np

ds = xr.open_dataset('Random_Sampling_2021080206_2022080106_daily_nivometeo_alp.nc')
simu = ds.sel({'member': 0}).rename({'rr': 'simu'}).to_dataframe().drop(columns=['member', 'lon', 'lat'])


df = pd.read_csv('/home/vernaym/These/NO_TRANSFER/DATA/obs_nivometeo_daily_RR_20210801_20220801.csv', sep=';')
obs = df.rename(columns={'Q.dat': 'time', 'Q.num_poste': 'num_poste', 'poste_nivo.lat_dg': 'lats', 'poste_nivo.lon_dg': 'lons', 'Q.rr': 'obs'}).drop(columns=['hist_reseau_poste.reseau_poste', 'poste_nivo.massif_nivo', 'poste_nivo.nom_usuel', 'poste_nivo.alti'])
obs['time'] = pd.to_datetime(obs['time'])+pd.Timedelta(hours=30)
obs = obs.set_index(['num_poste', 'time'])

data = pd.concat([obs, simu], axis=1)

data = data[(~np.isnan(data['obs'])) & (~np.isnan(data['simu']))]
data = data.reset_index()
out = data.groupby('num_poste').agg({'time': 'count', 'lats': 'min', 'lons': 'min', 'obs': 'sum', 'simu': 'sum'})
out = out[out['time'] > 50]
out['ratio'] = out['simu'] / out['obs']

out.to_csv('scores_2021110106_2022043006_nivometeo_alpes.csv', sep=';')


