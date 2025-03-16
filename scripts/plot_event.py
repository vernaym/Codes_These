
import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle


# datadir = '/mnt/lfs/d10/mrns/users/NO_SAVE/vernaym/workdir/EDELWEISS/DATA'
datadir = '/home/vernaym/These/NO_TRANSFER/DATA'
savedir = '/home/vernaym/These/NO_TRANSFER/figures/analyse_situation_LPN'

lonmin = 6.010
lonmax = 6.490
latmin = 44.990
latmax  = 45.240

# 38191400;L Alpe d Huez (SATA);1860;45.097667;6.072833;51
# 38191408;L Alpe d Huez 2350;2340;45.105000;6.097500;51


def timelapse(da, savename):

    fig, ax = plt.subplots()

    da.isel(time=0).plot(cmap=plt.cm.terrain, vmin=0, vmax=4000)

    def animate(time):
        # da.isel(time=time).plot(cmap=plt.cm.YlGnBu, vmin=0, vmax=5, add_colorbar=False)
        da.isel(time=time).plot(cmap=plt.cm.terrain, vmin=0, vmax=4000, add_colorbar=False)
        plt.title(f"Time: {da.isel(time=time).time.data}")

    ani = animation.FuncAnimation(fig, animate, len(da.time.data), interval=400, blit=False, repeat=False)

    ani.save(os.path.join(savedir, savename), fps=30)


def interactive(da, savename, cmap=plt.cm.gist_earth, vmin=None, vmax=None):
    """
    https://stackoverflow.com/questions/29407665/how-to-add-a-time-control-panel-to-a-funcanimation-from-matplotlib
    """
    from matplotlib.widgets import Slider, Button

    da = da.where((da.longitude >= lonmin) & (da.longitude <= lonmax) & (da.latitude >= latmin) &
            (da.latitude <= latmax), drop=True)
    huez = da.sel({'latitude': 45.1, 'longitude': 6.1}, method='nearest').squeeze()

    if vmin is None:
        vmin = da.min().data
    if vmax is None:
        vmax = da.max().data

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})

    axtime = plt.axes([0.15, 0.01, 0.5, 0.05])
    resetax = plt.axes([0.7, 0.01, 0.1, 0.05])
    stime = Slider(axtime, 'Time', 0, len(da.time.data) - 1, valinit=0, valstep=1)

    # da.isel(time=0).plot(ax=ax, cmap=plt.cm.terrain, vmin=0, vmax=4000)
    da.isel(time=0).plot(ax=ax[0], cmap=cmap, vmin=vmin, vmax=vmax)
    huez.plot(ax=ax[1], color='k')
    vline = ax[1].axvline(huez.isel(time=0).time.data)

    def update(val):
        time = stime.val
        da.isel(time=time).plot(ax=ax[0], cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
        ax[0].set_title(f"Time: {da.isel(time=time).time.data}")

        vline.set_data(vline.set_data([huez.isel(time=time).time.data, huez.isel(time=time).time.data], [0, 1]))
        # huez.plot(ax=ax[1], color='k')
        # ax[1].axvspan(huez.isel(time=0).time.data, huez.isel(time=time).time.data, alpha=0.2, color='grey')
        # ax[1].axvline(huez.isel(time=time).time.data)
        fig.canvas.draw()

    # Routines to reset and update sliding bar
    def reset(event):
        stime.reset()

    stime.on_changed(update)
    button = Button(resetax, 'Reset')
    button.on_clicked(reset)
    pickle.dump(fig, open(os.path.join(savedir, savename), 'wb'))
    plt.show()
    plt.close()


# antilope = xr.open_dataarray(os.path.join(datadir, 'ANTILOPEH_2021080106_2022080106_alp.nc'))
# timelapse(antilope.sel(time=slice('2021-12-28 00', '2021-12-30 00')),
#    savename='ANTILOPE_alp_2021-12-28_2021-12-30.gif')

isowbt = xr.open_dataarray(os.path.join(datadir, 'ISO_WETBT.arome-3dvarfr_2021080106_2022080106.nc'))
iso1wbt = isowbt.sel({'ISO_TPW': 27415}).drop_vars('time').rename({'valid_time': 'time'})
# timelapse(iso1wbt.sel(time=slice('2021-12-28 00', '2021-12-30 00')), savename='LPN_2021-12-28_2021-12-30.gif')
savename = 'LPN_2021-12-28_2021-12-30.pickle'
interactive(iso1wbt.sel(time=slice('2021-12-28 00', '2021-12-30 00')), savename=savename)

# ax = pickle.load(open(os.path.join(savedir, savename), 'rb'))
# plt.show()
