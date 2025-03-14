import xarray as xr
import numpy as np

import matplotlib.pyplot as plt

ds=xr.open_dataset('Random_Sampling_2021080206_2022080106_daily_GrandesRousses.nc')
ds.rr.data = np.where(np.isfinite(ds.rr.data), ds.rr.data, np.nan)
ds=ds.where((ds.lon>=6.010) & (ds.lon<=6.490) & (ds.lat>=44.990) & (ds.lat<=45.240), drop=True)

ds = ds.sel(time=slice('2021-11-01', '2022-04-30'))

tmp=ds.sum('time')
im=plt.imshow(tmp.std('member').rr, cmap=plt.cm.Purples, origin='lower')
plt.colorbar(im)
plt.tight_layout()
plt.savefig('Spread.pdf', format='pdf')
plt.close('all')
im=plt.imshow(tmp.mean('member').rr, cmap=plt.cm.YlGnBu, origin='lower')
plt.colorbar(im)
plt.tight_layout()
plt.savefig('Mean.pdf', format='pdf')
