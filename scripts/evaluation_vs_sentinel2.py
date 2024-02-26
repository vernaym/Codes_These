import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mntdir = '/home/vernaym/These/DATA'
prodir = '/home/vernaym/workdir/EDELWEISS/pro'

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

def plot_error_fields(obs, var='LCSMOD'):
    simu = xr.open_mfdataset(f'mb*/DIAG.nc', combine='nested', concat_dim='member', decode_times=False)
    simu['member'] = range(1,17)
    simu = simu.rename({'xx':'x', 'yy':'y'})
    simu['x'] = obs['x']
    simu['y'] = obs['y']
    tmp = simu.mean(dim='member')
    tmp = tmp.compute()
    diff = tmp[var]-obs['Band1']
    plt.imshow(np.flipud(diff.data), cmap='RdBu')
    plt.colorbar()
    plt.show()

def plot_ange(obs, var='LCSMOD'):
    altitude_bands = np.arange(1900, 3600, 300)  # Define altitude bands (1900-3600m with 300m intervals)
    mnt = read_mnt()

    filtered_obs = per_alt(obs.Band1, altitude_bands, mnt)
    obs_df = filtered_obs.to_dataframe(name=var).dropna().reset_index()

    simu_df= None
    for member in range(1, 17):
        simu = xr.open_dataset(f'mb{member:03d}/DIAG.nc', decode_times=False)
        simu = simu.rename({'xx':'x', 'yy':'y'})
        # TODO : résoudre le problème de décallage des coordonnées
        simu['x'] = mnt['x']
        simu['y'] = mnt['y']
        filtered_simu = per_alt(simu[var], altitude_bands, mnt)
        df = filtered_simu.to_dataframe(name=var).dropna().reset_index()
        if simu_df is not None:
            simu_df = pd.concat([simu_df, df])
        else:
            simu_df = df

    dataplot = pd.concat([simu_df, obs_df], keys=['simu', 'obs']).drop(columns=['x','y'])  # unecessarilly large DataFrame (duplicate index) ?
    #dataplot = pd.concat([obs_df, simu_df], axis=1)  # TODO : drop duplicated x/y/middle_slices_ZS columns
    #dataplot = pd.concat([obs_df, simu_df['simu']], axis=1)  # WARNING : this does not ensure that x/y/middle_slices_ZS columns match !
    dataplot.columns = dataplot.columns.str.replace('middle_slices_ZS', 'Elevation Bands (m)')
    #dataplot.columns = dataplot.columns.str.replace('middle_slices_ZS', 'Z')


    sns.set(rc={"figure.figsize":(12, 15)})
    sns.set_theme(style="whitegrid",font_scale=1.7)
    g=sns.violinplot(dataplot.reset_index().rename(columns={'level_0': 'forcing'}) ,y='Elevation Bands (m)',x=var,inner='box',hue='forcing',scale='width',bw='scott',\
               cut=0,orient='h',palette=("#6ACC64", "silver"),facet_kws={'legend_out': True})
    plt.ylim(reversed(plt.ylim()))

    plt.savefig(f'{var}.pdf', format='pdf')


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


if __name__ == '__main__':

    #var = 'LCSMOD'
    var = 'LCSCD'

    os.chdir(prodir)

    if var == 'LCSMOD':
        obs = xr.open_dataset('/home/vernaym/These/DATA/Sentinel2/20210901_L3B-SNOW_SMD_R2.nc')
    elif var == 'LCSCD':
        obs = xr.open_dataset('/home/vernaym/These/DATA/Sentinel2/20210901_L3B-SNOW_SCD_R2.nc')
    #compare(obs, var=var)
    #plot_ange(obs, var=var)
    plot_error_fields(obs, var=var)



