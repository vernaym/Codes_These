
import xarray as xr
import pandas as pd

ds = xr.open_dataset('Random_Sampling_2021080206_2022080106_daily_alp.nc')
df = pd.read_csv('/home/vernaym/These/NO_TRANSFER/DATA/obs_nivometeo_daily_RR_20211201_20220430.csv', sep=';')

out  = ds.sel(
    lat = xr.DataArray(
        df['poste_nivo.lat_dg'].unique(),
        dims   = 'num_poste',
        coords = {'num_poste': df['Q.num_poste'].unique()},
    ),
    lon = xr.DataArray(
        df['poste_nivo.lon_dg'].unique(),
        dims = 'num_poste',
        coords = {'num_poste': df['Q.num_poste'].unique()},
    ),
    method='nearest'
)

out = out.to_netcdf('Random_Sampling_2021080206_2022080106_daily_nivometeo_alp.nc')
