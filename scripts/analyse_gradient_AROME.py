
import os
import xarray as xr
import numpy as np

from sklearn.linear_model import LinearRegression

from snowtools.scripts.extract.vortex import vortexIO as io
import snowtools.tools.xarray_preprocess as xrp

import matplotlib.pyplot as plt
import matplotlib.animation as animation

winter = False


def plot_time_serie(dataset):

    fig, ax = plt.subplots()

    def animate(time):
        ax.scatter(dataset.isel(time=time).data.flatten(), mnt.data.flatten())

    ani = animation.FuncAnimation(fig, animate, 24, interval=400, blit=False)
    ani.save('animation.gif')


def vertical_gradient(precipitation, axis=None, ZS=None):
    x = ZS.data.flatten().reshape((-1, 1))
    y = precipitation.flatten()
    reg = LinearRegression().fit(x, y)
    # z = reg.predict(x)
    # r2 = np.round(reg.score(x, y), 3)
    out = xr.DataArray(data=np.array([[reg.coef_[0]]]), dims=('xx', 'yy'),
            coords={'xx': [1], 'yy': [1]})
    # out = reg.coef_[0]
    return out


def plot_date(dataset, datebegin, dateend, xpid):
    dataset = dataset.sel({'time': slice(datebegin, dateend)}).sum('time')
    fig, ax = plt.subplots(figsize=(12, 6))
    dataset.plot(ax=ax, cmap=plt.cm.YlGnBu, vmin=0)
    fig.savefig(f'precipitation_{xpid}_{datebegin}_{dateend}.pdf')


geometry = 'GrandesRousses250m'
io.get_const('uenv:dem.2@vernaym', 'relief', geometry, filename='TARGET_RELIEF.nc',
            gvar='RELIEF_GRANDESROUSSES250M_4326')
# Get Domain's DEM in case ZS not in simulation file
fullmnt = xr.open_dataset('TARGET_RELIEF.nc')  # Target domain's Digital Elevation Model
fullmnt = xrp.preprocess(fullmnt, decode_time=False)
fullmnt = fullmnt['elevation']
# mnt = fullmnt.sel({'xx': slice(6.05, 6.2), 'yy': slice(45.05, 45.25)})  # Domaine intermédiaire
# mnt = fullmnt.sel({'yy': slice(45.10, 45.15), 'xx': slice(6.0, 6.15)})  # petit domaine
# mnt = fullmnt.sel({'yy': slice(45.0, 45.20), 'xx': slice(6.0, 6.2)})  # Sud Ouest
mnt = fullmnt.sel({'yy': slice(45.139, 45.141), 'xx': slice(6.036, 6.13)})  # transect

products = ['AROME', 'RS27', 'RS42']
for product in products:

    if product == 'AROME':
        arome = xr.open_dataset('AROME_hours_grandesrousses_eurw1s40_2021080106_2022080106.nc')
        ds = arome.tp
        ds = ds.interp(longitude=mnt.xx, latitude=mnt.yy, method='linear')
    elif product.startswith('RS'):
        rootdir = f'/cnrm/cen/users/NO_SAVE/vernaym/cache/vortex/edelweiss/grandesrousses250m/{product}@vernaym/' \
            'mb0000/meteo'
        filename = 'FORCING_2021080206_2022080106.nc'
        rs = xr.open_dataset(os.path.join(rootdir, filename))
        ds = (rs['Rainf'] + rs['Snowf']) * 3600
        # plot_date(ds, '2021-12-28-07', '2021-12-29-06', product)
        ds = ds.sel({'x': ds.x[np.where(np.isin(fullmnt.xx.data, mnt.xx.data))],
            'y': ds.y[np.where(np.isin(fullmnt.yy.data, mnt.yy.data))]})

    # plot_time_serie(ds.sel({'time': slice('2021-12-28-07', '2021-12-29-06')}))

    gp = ds.groupby('time', restore_coord_dims=True)
    hourly_gradient = gp.reduce(vertical_gradient, ZS=mnt)

    hourly_gradient = hourly_gradient.squeeze()

    if winter:
        hg_winter = hourly_gradient.sel(time=slice('2021-11-01', '2022-04-30'))
        pos = hg_winter.where(hourly_gradient > 0).count()
        neg = hg_winter.where(hourly_gradient < 0).count()
        hg_winter.plot(label=f'{product} (nb pos: {pos.data}, nb neg: {neg.data})')
        # savename = f'analyse_gradient_{product}_nov-apr.pdf'
    else:
        pos = hourly_gradient.where(hourly_gradient > 0).count()
        neg = hourly_gradient.where(hourly_gradient < 0).count()
        hourly_gradient.plot(label=f'{product} (nb pos: {pos.data}, nb neg: {neg.data})')
        # savename = f'analyse_gradient_{product}.pdf'

    # plt.title(f'Nb pos: {pos.data}, nb neg: {neg.data}')

plt.legend()

savename = 'analyse_gradients_' + '_'.join(products)
if winter:
    plt.savefig(f'{savename}_nov-apr.pdf')
else:
    plt.savefig(f'{savename}.pdf')
