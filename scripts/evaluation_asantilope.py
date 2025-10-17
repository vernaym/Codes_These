import xarray as xr
import pandas as pd
import numpy as np

df = pd.read_csv('/home/vernaym/These/NO_TRANSFER/DATA/obs_nivometeo_daily_RR_20211201_20220430.csv', sep=';')
obs = df.rename(columns={'Q.dat': 'time', 'Q.num_poste': 'num_poste', 'poste_nivo.lat_dg': 'lat',
    'poste_nivo.lon_dg': 'lon', 'Q.rr': 'obs'}).drop(
        columns=['hist_reseau_poste.reseau_poste', 'poste_nivo.massif_nivo', 'poste_nivo.nom_usuel', 'poste_nivo.alti']
)
obs['time'] = pd.to_datetime(obs['time']) + pd.Timedelta(hours=30)
stations = obs.groupby('num_poste').agg({'lat': 'min', 'lon': 'min'})
obs = obs.set_index(['num_poste', 'time'])

ds = xr.open_dataset('Random_Sampling_2021080206_2022080106_daily_GrandesRousses.nc')
simu = ds.sel(lat=xr.DataArray(stations.lat, dims='num_poste'), lon=xr.DataArray(stations.lon, dims='num_poste'),
        member=0, method='nearest').rename({'rr': 'simu'}).to_dataframe().drop(columns=['member', 'lon', 'lat'])

data = pd.concat([obs, simu], axis=1)

data = data[(~np.isnan(data['obs'])) & (~np.isnan(data['simu']))]
data = data.reset_index()

data['error'] = data['simu'] - data['obs']

data['square_error'] = data['error'] ** 2

out = data.groupby('num_poste').agg({'time': 'count', 'lat': 'min', 'lon': 'min', 'obs': 'sum', 'simu': 'sum',
    'square_error': 'mean', 'error': 'mean'})
out['square_error'] = np.sqrt(out['square_error'])
out = out.rename(columns={'square_error': 'rmse', 'error': 'mean_bias'})
out = out[out['time'] > 50]
out['ratio'] = out['simu'] / out['obs']

out.to_csv('scores_2021110106_2022043006_nivometeo_alpes.csv', sep=';')
