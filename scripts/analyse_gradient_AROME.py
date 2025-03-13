
import xarray as xr
import numpy as np

from sklearn.linear_model import LinearRegression

from snowtools.scripts.extract.vortex import vortexIO as io
import snowtools.tools.xarray_preprocess as xrp

import matplotlib.pyplot as plt


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
mnt = xr.open_dataset('TARGET_RELIEF.nc')  # Target domain's Digital Elevation Model
mnt = xrp.preprocess(mnt, decode_time=False)
mnt = mnt['elevation']
mnt = mnt.sel({'xx': slice(6.05, 6.2), 'yy': slice(45.05, 45.25)})

arome = xr.open_dataset('AROME_hours_grandesrousses_eurw1s40_2021080106_2022080106.nc')
arome = arome.interp(longitude=mnt.xx, latitude=mnt.yy, method='linear')

#hourly_gradient = xr.apply_ufunc(vertical_gradient, arome.tp, mnt, input_core_dims=[["time"], []])
gp = arome.tp.groupby('time', restore_coord_dims=True)
hourly_gradient = gp.reduce(vertical_gradient, ZS=mnt)
#hourly_gradient = gp.apply(vertical_gradient, shortcut=True, args=[mnt])

hourly_gradient = hourly_gradient.squeeze()

winter = False

if winter:
    hg_winter = hourly_gradient.sel(time=slice('2021-11-01', '2022-04-30'))
    pos = hg_winter.where(hourly_gradient > 0).count()
    neg = hg_winter.where(hourly_gradient < 0).count()
    hg_winter.plot()
    savename = 'analyse_gradient_AROME_nov-apr.pdf'
else:
    pos = hourly_gradient.where(hourly_gradient > 0).count()
    neg = hourly_gradient.where(hourly_gradient < 0).count()
    hourly_gradient.plot()
    savename = 'analyse_gradient_AROME.pdf'

plt.title(f'Nb pos: {pos.data}, nb neg: {neg.data}')
plt.savefig(savename)



