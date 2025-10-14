
import xarray as xr
from snowtools.utils import xarray_snowtools

listfiles = [
    'ANTILOPEJP1H_PRECIP_SOL_FRANXL1S100_2018080106_2019080106.nc',
    'ANTILOPEJP1H_PRECIP_SOL_FRANXL1S100_2018080106_2019080106.nc',
    'ANTILOPEJP1H_PRECIP_SOL_FRANXL1S100_2018080106_2019080106.nc',
]
ds = xr.open_mfdataset(listfiles, engine='snowtools', concat_dim='time', combine='nested')
ds = ds.drop(['step', 'SOL', 'valid_time'])
ds = ds.rename({'Precipitation': 'cumul'})

cumul = ds.sum('time')
cumul.to_netcdf('CUMUL_ANTILOPEJP1H_2018080106_2021080106.nc')


winter = ds.where(ds.time.dt.month.isin([12, 1, 2, 3]), drop=True)
winter_cumul = winter.sum('time')
winter_cumul = xarray_snowtools.preprocess(winter_cumul)
winter_cumul.to_netcdf('CUMUL_ANTILOPEJP1H_WINTER_2018_2021_dec-mar.nc')
