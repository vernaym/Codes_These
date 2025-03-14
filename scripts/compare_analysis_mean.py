
import sys

import xarray as xr
import numpy as np

import matplotlib.pyplot as plt

xpid1 = sys.argv[1]
xpid2 = sys.argv[2]
#xpid1 = 'RS42'
#xpid2 = 'RS46'

def preprocess(ds):
    ds.rr.data = np.where(np.isfinite(ds.rr.data), ds.rr.data, np.nan)
    ds=ds.where((ds.lon>=6.010) & (ds.lon<=6.490) & (ds.lat>=44.990) & (ds.lat<=45.240), drop=True)
    out=ds.sum('time').mean('member').rr
    return out

ds1=xr.open_dataset(f'{xpid1}/Random_Sampling_2021080206_2022080106_daily_GrandesRousses.nc')
m1 = preprocess(ds1)
ds2=xr.open_dataset(f'{xpid2}/Random_Sampling_2021080206_2022080106_daily_GrandesRousses.nc')
m2 = preprocess(ds2)

im=plt.imshow(m1-m2, cmap=plt.cm.RdBu, origin='lower')
plt.colorbar(im)
plt.tight_layout()
plt.savefig(f'diff_{xpid1}_{xpid2}.pdf', format='pdf')
plt.close('all')
