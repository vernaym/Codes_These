import os
import numpy as np
import pandas as pd
import xarray as xr
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

import vortexIO


members_map = dict(
        safran        = None,
        safran_pappus = None,
        RS25          = 16,
    )
geometry = 'GrandesRousses250m'


mntdir = '/home/vernaym/These/DATA'
workdir = '/home/vernaym/workdir/EDELWEISS/diag'

altitude_bands = np.arange(1900, 3600, 300)  # Define altitude bands (1900-3600m with 300m intervals)

def parse_command_line():
    description = "Computation of Sentinel2-like diagnostics (snow melt-out date, snow cover duration) associated \
                   to a SURFEX simulation"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-b', '--datebegin', type=str, help="First date covered by the simulation file, format YYYYMMDDHH.")
    parser.add_argument('-e', '--dateend', type=str, help="Last date covered by the simulation file, format YYYYMMDDHH.")
    parser.add_argument('-x', '--xpids', nargs='+', type=str, help="XPID(s) of the simulation(s) format XP_NAME@username")
    args = parser.parse_args()
    return args

def read_mnt():
    latmax = 45.240
    latmin = 44.990
    lonmin = 6.010
    lonmax = 6.490
    mnt = xr.open_dataset(os.path.join(mntdir, "MNTLouisGRoussecorrected.nc"))
    return mnt

def plot_error_fields(xpids, obs, var):
    for xpid in xpids:
        shortid = xpid.split('@')[0]
        members = members_map[shortid]
        if members is None:
            simu = xr.open_dataset(f'DIAG_{shortid}.nc', decode_times=False)
        else:
            simu = xr.open_mfdataset(f'mb*/DIAG_{shortid}.nc', combine='nested', concat_dim='member', decode_times=False)
        #simu = simu.rename({'xx':'x', 'yy':'y'})
        # TODO : résoudre le problème de décallage des coordonnées en amont (dans OPTIONS.nam)
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
        plt.savefig(f'diff_{var}_{shortid}.pdf', format='pdf')

        plt.close('all')

def plot_ange(xpids, obs, var, mask=True):
    mnt = read_mnt()

    filtered_obs = per_alt(obs.Band1, altitude_bands, mnt)
    dataplot = filtered_obs.to_dataframe(name='obs').dropna().reset_index().drop(columns=['x','y'])

    for xpid in xpids:
        shortid = xpid.split('@')[0]
        members = members_map[shortid]
        if members is None:
            subdir = ''
            df = filter_simu(shortid, subdir, mnt)
        else:
            df = None
            for member in range(1, members+1):
                subdir = f'mb{member:03d}'
                dfm = filter_simu(shortid, subdir, mnt)
                if df is not None:
                    df = pd.concat([df, dfm], ignore_index=True)
                else:
                    df = dfm

        dataplot = pd.concat([dataplot, df])

#        dataplot = pd.concat([simu_df, obs_df], keys=['simu', 'obs']).drop(columns=['x','y'])  # unecessarilly large DataFrame (duplicate index) ?
    dataplot.columns = dataplot.columns.str.replace('middle_slices_ZS', 'Elevation Bands (m)')
    dataplot = dataplot.melt('Elevation Bands (m)', var_name='experiment', value_name=var)

    sns.set(rc={"figure.figsize":(12, 15)})
    sns.set_theme(style="whitegrid",font_scale=1.7)
    g = sns.violinplot(
            #dataplot.reset_index().rename(columns={'level_0': 'forcing'}),  # data
            dataplot,  # data
            y = 'Elevation Bands (m)',  # y-axis
            x = var,  # X-axis
            inner = 'box',  # ?
            #inner = 'stick',  # To plot each individual data of the violinplot
            hue = 'experiment',  # Legend 'title'
            scale = 'width',  # ?
            bw = 'scott',  # ?
            cut = 0,  # ?
            orient = 'h',  # Horizontal violinplots
            #palette = ("#6ACC64", "silver"),  # color palette (1 per DF column)
            #facet_kws = {'legend_out': True},  # Does not work on sxcen
        )
    plt.ylim(reversed(plt.ylim()))
    plt.xlim([0, 375])

    plt.savefig(f'{var}.pdf', format='pdf')

    plt.close('all')

def filter_simu(xpid, subdir, mnt):
    diagname = os.path.join(subdir, f'DIAG_{xpid}.nc')
    simu = xr.open_dataset(diagname, decode_times=False)
    #simu = simu.rename({'xx':'x', 'yy':'y'})
    # TODO : résoudre le problème de décallage des coordonnées en amont
    simu['x'] = mnt['x']
    simu['y'] = mnt['y']
    filtered_simu = per_alt(simu[var], altitude_bands, mnt)
    df = filtered_simu.to_dataframe(name=xpid).dropna().reset_index().drop(columns=['x','y'])
    return df

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

    os.chdir(workdir)

    args = parse_command_line()
    datebegin = args.datebegin
    dateend   = args.dateend
    xpids     = args.xpids

    # 1. Get all input data
    for xpid in xpids:
        shortid = xpid.split('@')[0]
        # VERRUE
        if shortid.startswith('safran'):
            deb = '2021080106'
        else:
            deb = datebegin  # 2021080207
        members = members_map[shortid]
        # Get DIAG files with Vortex
        vortexIO.get_diag(deb, dateend, xpid, geometry, members=members, filename=f'DIAG_{shortid}.nc')

    # 2. Compare simulated data with Sentinel2 data
    # TODO : put/get Sentinel2 data from hendrix
    for var in ['scd_concurent', 'mod']:
        # open Sentinel2 data
        if var == 'mod':
            obs = xr.open_dataset('/home/vernaym/These/DATA/Sentinel2/20210901_L3B-SNOW_SMD_R2.nc')
        elif var == 'scd_concurent':
            obs = xr.open_dataset('/home/vernaym/These/DATA/Sentinel2/20210901_L3B-SNOW_SCD_R2.nc')

        #compare(obs, var=var)
        plot_ange(xpids, obs, var)  # Violinplots by elevation range
        plot_error_fields(xpids, obs, var)  # Field difference

    # 3. Clean data
    for xpid in xpids:
        shortid = xpid.split('@')[0]
        members = members_map[shortid]
        if members is None:
            os.remove(f'DIAG_{shortid}.nc')
        else:
            for member in range(1, members+1):
                os.remove(f'mb{member:03d}/DIAG_{shortid}.nc')
