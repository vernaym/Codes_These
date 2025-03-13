
import os
import xarray as xr
import numpy as np

from sklearn.linear_model import LinearRegression

from snowtools.scripts.extract.vortex import vortexIO as io
import snowtools.tools.xarray_preprocess as xrp

import matplotlib.pyplot as plt

product = 'RS27'
winter = False


def vertical_gradient(precipitation, axis=None, ZS=None):
    x = ZS.data.flatten().reshape((-1, 1))
    y = precipitation.flatten()
    reg = LinearRegression().fit(x, y)
    #z = reg.predict(x)
    #r2 = np.round(reg.score(x, y), 3)
    out = xr.DataArray(data=np.array([[reg.coef_[0]]]), dims=('xx', 'yy'), coords={'xx':[1], 'yy':[1]})
    #out = reg.coef_[0]
    return out

geometry = 'GrandesRousses250m'
io.get_const('uenv:dem.2@vernaym', 'relief', geometry, filename='TARGET_RELIEF.nc',
            gvar='RELIEF_GRANDESROUSSES250M_4326')
# Get Domain's DEM in case ZS not in simulation file
fullmnt = xr.open_dataset('TARGET_RELIEF.nc')  # Target domain's Digital Elevation Model
fullmnt = xrp.preprocess(fullmnt, decode_time=False)
fullmnt = fullmnt['elevation']
#mnt = mnt.sel({'xx': slice(6.05, 6.2), 'yy': slice(45.05, 45.25)})
#mnt = fullmnt.sel({'yy': slice(45.10, 45.15), 'xx': slice(6.0, 6.15)})
mnt = fullmnt.sel({'yy': slice(45.0, 45.20), 'xx': slice(6.0, 6.2)})

if product == 'AROME':
    arome = xr.open_dataset('AROME_hours_grandesrousses_eurw1s40_2021080106_2022080106.nc')
    ds = arome.tp
    ds = ds.interp(longitude=mnt.xx, latitude=mnt.yy, method='linear')
elif product.startswith('RS'):
    rootdir = f'/cnrm/cen/users/NO_SAVE/vernaym/cache/vortex/edelweiss/grandesrousses250m/{product}@vernaym/mb0000/meteo'
    filename = 'FORCING_2021080206_2022080106.nc'
    rs = xr.open_dataset(os.path.join(rootdir, filename))
    ds = (rs['Rainf'] + rs['Snowf']) * 3600
    #ds = ds.sel({'x': slice(939625., 951125.), 'y': slice(6443609, 6466270)})
    ds = ds.sel({'x': ds.x[np.where(np.isin(mnt.xx.data, fullmnt.xx.data))], 'y': ds.y[np.where(np.isin(mnt.yy.data, fullmnt.yy.data))]})

#hourly_gradient = xr.apply_ufunc(vertical_gradient, arome.tp, mnt, input_core_dims=[["time"], []])
gp = ds.groupby('time', restore_coord_dims=True)
hourly_gradient = gp.reduce(vertical_gradient, ZS=mnt)
#hourly_gradient = gp.apply(vertical_gradient, shortcut=True, args=[mnt])

hourly_gradient = hourly_gradient.squeeze()

if winter:
    hg_winter = hourly_gradient.sel(time=slice('2021-11-01', '2022-04-30'))
    pos = hg_winter.where(hourly_gradient > 0).count()
    neg = hg_winter.where(hourly_gradient < 0).count()
    hg_winter.plot()
    savename = f'analyse_gradient_{product}_nov-apr.pdf'
else:
    pos = hourly_gradient.where(hourly_gradient > 0).count()
    neg = hourly_gradient.where(hourly_gradient < 0).count()
    hourly_gradient.plot()
    savename = f'analyse_gradient_{product}.pdf'

plt.title(f'Nb pos: {pos.data}, nb neg: {neg.data}')
plt.savefig(savename)



