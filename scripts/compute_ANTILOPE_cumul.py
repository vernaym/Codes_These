
import xarray as xr
import numpy as np

datebegin = '2021-12-15'
dateend   = '2022-03-31'
ds=xr.open_dataset('/cnrm/cen/users/NO_SAVE/vernaym/workdir/EDELWEISS/DATA/Precipitation_2021080106_2022080106.nc')  # On sxcen
ds.rr.data = np.where(ds.rr.data==9999, np.nan, ds.rr.data)
tmp=ds.sel(time=slice(datebegin, dateend)).sum('time').rename({'rr': 'cumul'})
tmp.to_netcdf(f'CUMUL_ANTILOPE_{datebegin}_{dateend}_alp.nc')
