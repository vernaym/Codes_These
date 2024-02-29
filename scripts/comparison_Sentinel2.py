import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

import vortexIO


datebegin = '2021080106'  # TODO : passer en argument
#datebegin = '2021080207'  # TODO : passer en argument
dateend = '2022080106'  # TODO : passer en argument
#xpid = 'RS25_no_pappus@vernaym'  # TODO : passer en argument
xpid = 'safran@vernaym'  # TODO : passer en argument
members = 17  # TODO : passer en argument
members = None  # TODO : passer en argument
geometry = 'GrandesRousses250m'


mntdir = '/home/vernaym/These/DATA'
workdir = '/home/vernaym/workdir/EDELWEISS/diag'

altitude_bands = np.arange(1900, 3600, 300)  # Define altitude bands (1900-3600m with 300m intervals)


def maskgf(arr, method='nearest'):
    """
      Masks an input array (arr) using a reference mask dataset.

      Args:
          arr    : The input array to be masked.
          method : The interpolation method to use when resampling the glacier mask
                   to the same resolution as the input array. Valid options are
                   'nearest', 'linear', 'cubic', etc. (default: 'nearest').

      Returns:
          A new array with the same shape as the input array, where values are masked
          out based on the glacier mask. Masked values are set to NaN.
    """

  # Load the glacier mask dataset
    #masque=xr.open_dataset('/home/vernaym/These/DATA/mask/masque_foret_glacier.nc').Band1.interp_like(arr, method=method)
    masque = xr.open_dataset('/home/vernaym/These/DATA/mask/masque_glacier2017_foret_ville_riviere.nc')['Band1']

  # Interpolate the glacier mask to the same resolution as the input array
    masque = masque.interp_like(arr, method=method)

  # Mask the input array based on the glacier mask
    return arr.where(masque == 0)

def lcscd(data, threshold=.2):
    """
    Compute the following diagnostic variables from PRO DSN_T_ISBA (snow depth) variable :
    * LCSCD  : Longest Concurent Snow Cover Duration period
    * LCSMOD : Snow Melt Out Date of the Longest Concurent snow cover period
    * LCSOD  : Snow Cover Onset date of the Longest Concurent snow cover period
    """

    data = xr.where(data > threshold, True, False)
    cumulative = data.cumsum(dim='time')-data.cumsum(dim='time').where(data.values == 0).ffill(dim='time').fillna(0)
    scd = (cumulative.max(dim = 'time')).rename('scd_concurent')
    mod = (cumulative.argmax(dim = 'time') +1).rename('mod')
    sod = (mod - scd).rename('sod')
    sd = data.where(data == True, np.nan).count(dim = 'time').rename('sd')
    return xr.merge([scd,mod,sod,sd])

def decode_time(pro):
    """
    Manually decode time variable since other variables can not be decoded automatically
    """
    ds = xr.Dataset({"time": pro.time})
    ds = xr.decode_cf(ds)
    pro['time'] = ds.time
    return pro

def compare(obs, var='LCSMOD'):
    plt.imshow(np.flipud(simu[var].data))
    plt.colorbar()
    plt.show()
    plt.imshow(np.flipud(obs.Band1.data))
    plt.colorbar()
    plt.show()
    diff=simu.LCSMOD.data-obs.Band1.data
    plt.imshow(np.flipud(diff))
    plt.colorbar()
    plt.show()

def read_mnt():
    latmax = 45.240
    latmin = 44.990
    lonmin = 6.010
    lonmax = 6.490
    mnt = xr.open_dataset(os.path.join(mntdir, "MNTLouisGRoussecorrected.nc"))
    return mnt

def plot_error_fields(obs, var):
    if members is None:
        simu = xr.open_dataset(f'DIAG.nc', decode_times=False)
    else:
        simu = xr.open_mfdataset(f'mb*/DIAG.nc', combine='nested', concat_dim='member', decode_times=False)
    #simu = simu.rename({'xx':'x', 'yy':'y'})
    # TODO : résoudre le problème de décallage des coordonnées en amont
    simu['x'] = obs['x']
    simu['y'] = obs['y']

    if members is not None:
        simu['member'] = range(members)
        tmp = simu.mean(dim='member')
    else:
        tmp = simu
    tmp = tmp.compute()
    diff = tmp[var]-obs['Band1']
    plt.imshow(np.flipud(diff.data), cmap='RdBu')
    plt.colorbar()
    plt.savefig(f'diff_{var}.pdf', format='pdf')

    plt.close('all')

def filter_simu(subdir, mnt):
    diagname = os.path.join(subdir, 'DIAG.nc')
    simu = xr.open_dataset(diagname, decode_times=False)
    #simu = simu.rename({'xx':'x', 'yy':'y'})
    # TODO : résoudre le problème de décallage des coordonnées en amont
    simu['x'] = mnt['x']
    simu['y'] = mnt['y']
    filtered_simu = per_alt(simu[var], altitude_bands, mnt)
    df = filtered_simu.to_dataframe(name=var).dropna().reset_index()
    return df

def plot_ange(obs, var, mask=True):
    mnt = read_mnt()

    filtered_obs = per_alt(obs.Band1, altitude_bands, mnt)
    obs_df = filtered_obs.to_dataframe(name=var).dropna().reset_index()

    simu_df= None
    if members is None:
        subdir = ''
        simu_df = filter_simu(subdir, mnt)
    else:
        for member in range(members):
            subdir = f'mb{member:03d}'
            if simu_df is not None:
                simu_df = pd.concat([simu_df, filter_simu(subdir, mnt)])
            else:
                simu_df = df

    dataplot = pd.concat([simu_df, obs_df], keys=['simu', 'obs']).drop(columns=['x','y'])  # unecessarilly large DataFrame (duplicate index) ?
    #dataplot = pd.concat([obs_df, simu_df], axis=1)  # TODO : drop duplicated x/y/middle_slices_ZS columns
    #dataplot = pd.concat([obs_df, simu_df['simu']], axis=1)  # WARNING : this does not ensure that x/y/middle_slices_ZS columns match !
    dataplot.columns = dataplot.columns.str.replace('middle_slices_ZS', 'Elevation Bands (m)')
    #dataplot.columns = dataplot.columns.str.replace('middle_slices_ZS', 'Z')


    sns.set(rc={"figure.figsize":(12, 15)})
    sns.set_theme(style="whitegrid",font_scale=1.7)
    g=sns.violinplot(
            dataplot.reset_index().rename(columns={'level_0': 'forcing'}),  # data
            y = 'Elevation Bands (m)',  # y-axis
            x = var,  # X-axis
            inner = 'box',  # ?
            #inner = 'stick',  # To plot each individual data of the violinplot
            hue = 'forcing',  # Legend 'title'
            scale = 'width',  # ?
            bw = 'scott',  # ?
            cut = 0,  # ?
            orient = 'h',  # Horizontal violinplots
            palette = ("#6ACC64", "silver"),  # color palette (1 per DF column)
            #facet_kws = {'legend_out': True},  # Does not work on sxcen
        )
    plt.ylim(reversed(plt.ylim()))
    plt.xlim([0, 375])

    if mask:
        plt.savefig(f'{var}_mask.pdf', format='pdf')
    else:
        plt.savefig(f'{var}_nomask.pdf', format='pdf')

    plt.close('all')


def per_alt(data, ls_alt, mnt): # ls_alt = np.arange(0,4200,300) (for example)
    """
      Groups data into slices based on altitude ranges.

      Args:
          data: The input dataset containing the data to be grouped.
          ls_alt: A list of altitude values defining the boundaries of each slice.
                  Values should be in ascending order.
          elevation_label: The name of the variable in the dataset containing
                  elevation values (default: 'ZS').

      Returns:
          A new dataset with the same variables as the input data, but with an
          additional dimension 'middle_slices_ZS' corresponding to the mean altitude
          slices. Each element along this dimension represents data within a
          specific altitude range.
    """
    data_per_alt = []
    for i in range(0,len(ls_alt)-1):
        data_per_alt.append(data.where((mnt['ZS'] >= ls_alt[i]) & (mnt['ZS'] < ls_alt[i+1])))
    data_per_alt = xr.concat(data_per_alt, dim='middle_slices_ZS')
    data_per_alt['middle_slices_ZS'] = ls_alt[1:] - (ls_alt[1] - ls_alt[0])/2

    return data_per_alt

def diag(subdir):
    proname = os.path.join(subdir, 'PRO.nc')
    pro = xr.open_dataset(proname, decode_times=False)
    pro = decode_time(pro)
    diag = lcscd(pro.DSN_T_ISBA.resample(time='1D').mean())
    diag = diag.rename({'xx':'x', 'yy':'y'})

    mask = True
    if mask:
        # mask glacier/forest covered pixels
        diag = maskgf(diag)
        block = 'mask'
    else:
        block = 'nomask'

    # Write DIAG file and remove PRO
    diag.to_netcdf(os.path.join(subdir, 'DIAG.nc'))
    os.remove(os.path.join(subdir, 'PRO.nc'))

    return block


if __name__ == '__main__':

    os.chdir(workdir)

    # Retrieve PRO files with Vortex
    vortexIO.get_pro(datebegin, dateend, xpid, geometry, members=members)

    if members is None:
        subdir = ''
        block =diag(subdir)
    else:
        for member in range(17):
            print(f'Member {member}')
            subdir = f'mb{member:03d}'
            block = diag(subdir)

    # Compare simulated with Sentinel2 data
    # TODO : put/get Sentinel2 data from hendrix
    for var in ['scd_concurent', 'mod']:
        # open Sentinel2 data
        if var == 'mod':
            obs = xr.open_dataset('/home/vernaym/These/DATA/Sentinel2/20210901_L3B-SNOW_SMD_R2.nc')
        elif var == 'scd_concurent':
            obs = xr.open_dataset('/home/vernaym/These/DATA/Sentinel2/20210901_L3B-SNOW_SCD_R2.nc')

        #compare(obs, var=var)
        plot_ange(obs, var)  # Violinplots by elevation range
        plot_error_fields(obs, var)  # Field difference

    # Archive DIAG files with Vortex
    vortexIO.put_diag(datebegin, dateend, xpid, geometry, members=members, block=block)

    # Clean data
    if members is None:
        os.remove('DIAG.nc')
    else:
        for member in range(members):
            os.remove(f'mb{member:03d}/DIAG.nc')
