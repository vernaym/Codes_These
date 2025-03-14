import xarray as xr
import numpy as np
import glob
import os

import matplotlib.pyplot as plt


filenames=[os.path.join(f'mb{member:04d}', 'hourly', 'Precipitation_2021080206_2022080106.nc' ) for member in range(17)]
ds=xr.open_mfdataset(filenames, concat_dim='member', combine='nested')
ds=ds.rename({'Precipitation': 'rr'})
ds=ds.compute()

ds.rr.data = np.where(np.isfinite(ds.rr.data), ds.rr.data, np.nan)
ds=ds.where((ds.longitude>=6.010) & (ds.longitude<=6.490) & (ds.latitude>=44.990) & (ds.latitude<=45.240), drop=True)
tmp=ds.sum('time')
im=plt.imshow(tmp.std('member').rr, cmap=plt.cm.Purples, origin='lower')
plt.colorbar(im)
plt.tight_layout()
plt.savefig('Spread_hourly.pdf', format='pdf')
plt.close('all')
im=plt.imshow(tmp.mean('member').rr, cmap=plt.cm.YlGnBu, origin='lower')
plt.colorbar(im)
plt.tight_layout()
plt.savefig('Mean_hourly.pdf', format='pdf')
