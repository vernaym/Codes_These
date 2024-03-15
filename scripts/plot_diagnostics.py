import os
import numpy as np
import pandas as pd
import xarray as xr
import argparse
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from snowtools.scripts.extract.vortex import vortexIO

members_map = dict(
        safran         = None,
        safran_pappus  = None,
        RawData        = None,
        RawData_pappus = None,
        RS25           = 16,
        RS27           = 1,  # post-processed ANTILOPE only
        RS27_pappus    = 1, # post-processed ANTILOPE only
        EnKF36         = 16,
        EnKF36_pappus  = 16,
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
        fig, ax = plt.subplots(figsize=(16,10))
        shortid = xpid.split('@')[0]
        members = members_map[shortid]
        if members is None:
            simu = xr.open_dataset(f'DIAG_{shortid}.nc', decode_times=False)
        else:
            # Verrue : trouver une solution standard plus propre
            if members == 1:
                simu = xr.open_dataset(f'mb000/DIAG_{shortid}.nc', decode_times=False)  # "Deterministic" member=0 by default (WARNING : different from the current 's2m oper' convention)
            else:
                simu = xr.open_mfdataset([f'mb{member:03d}/DIAG_{shortid}.nc' for member in range(members)], combine='nested', concat_dim='member', decode_times=False)
        if 'x' in simu.keys():
            simu = simu.rename({'x':'xx', 'y':'yy'})
        # TODO : résoudre le problème de décallage des coordonnées en amont (dans OPTIONS.nam)
        simu['xx'] = obs['xx']
        simu['yy'] = obs['yy']

        if members is not None and members > 1:
            simu['member'] = range(members)
            tmp = simu.mean(dim='member')
        else:
            tmp = simu
        tmp = tmp.compute()
        diff = tmp[var]-obs['Band1']
        cmap = matplotlib.cm.RdBu
        cmap.set_bad('grey', 1.)
        im = ax.imshow(np.flipud(diff.data), cmap=cmap, vmin=-100, vmax=100)
        fig.colorbar(im)
        fig.savefig(f'diff_{var}_{shortid}.pdf', format='pdf')

        plt.close('all')

def plot_ange(xpids, obs, var, mask=True):
    mnt = read_mnt()

    filtered_obs = per_alt(obs.Band1, altitude_bands, mnt)
    dataplot = filtered_obs.to_dataframe(name='obs').dropna().reset_index().drop(columns=['xx','yy'])

    for xpid in xpids:
        print(xpid)
        shortid = xpid.split('@')[0]
        members = members_map[shortid]
        if members is None:
            subdir = ''
            df = filter_simu(shortid, subdir, mnt)
        else:
            df = None
            for member in range(members):
                print(member)
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
            #palette=("#6ACC64","#6ACC64","#EE854A","#EE854A","#4878D0","#4878D0","silver",),
            palette=("silver", "#D65F5F", "#D65F5F", "#4878D0", "#4878D0", "#6ACC64", "#6ACC64", "#EE854A", "#EE854A"),
            #palette = ("#6ACC64", "silver"),  # color palette (1 per DF column)
            #facet_kws = {'legend_out': True},  # Does not work on sxcen
        )
    plt.ylim(reversed(plt.ylim()))
    plt.xlim([0, 375])

    # Set Ange's hatches
    import matplotlib as mpl
    d=0
    for i, violin in enumerate(g.findobj(mpl.collections.PolyCollection)):
        #print(i)
        if i == 8 or i==17 or i==26 or i==35 or i==44 or i==53:
            d=d+1
            continue
        if (i-d) % 2 :
            continue
        else:
            violin.set_hatch(r'\\\\')

    # Set Ange's color
    import matplotlib.patches as mpatches
    colors = ["silver", "#D65F5F", "#D65F5F", "#4878D0", "#4878D0", "#6ACC64", "#6ACC64", "#EE854A", "#EE854A"]
    circ0 = mpatches.Patch(facecolor=colors[0],label='Sentinel 2 A obs')
    circ1 = mpatches.Patch( facecolor=colors[1],hatch=r'\\\\',label='Safran')
    circ2= mpatches.Patch( facecolor=colors[2],label='Safran Pappus')
    circ3 = mpatches.Patch(facecolor=colors[3],hatch=r'\\\\',label='Raw ANTILOPE')
    circ4 = mpatches.Patch( facecolor=colors[4],label='Raw ANTILOPE Pappus')
    circ5= mpatches.Patch( facecolor=colors[5],hatch=r'\\\\',label='AS-ANTILOPE')
    circ6 = mpatches.Patch(facecolor=colors[6],label='AS-ANTILOPE Pappus')
    circ7 = mpatches.Patch(facecolor=colors[7],label='Ensemble Analysis')
    #plt.legend(handles = [circ0,circ1,circ2,circ3,circ4,circ5,circ6,circ7])
    plt.legend(handles = [circ0,circ1,circ2,circ3,circ4,circ5,circ6])

    plt.savefig(f'{var}.pdf', format='pdf')

    plt.close('all')

def filter_simu(xpid, subdir, mnt):
    diagname = os.path.join(subdir, f'DIAG_{xpid}.nc')
    simu = xr.open_dataset(diagname, decode_times=False)
    # TODO : gérer le problème de coordonnées pour éviter les "rename" très lents !
    if 'x' in simu.keys():
        simu = simu.rename({'x':'xx', 'y':'yy'})
    # TODO : résoudre le problème de décallage des coordonnées en amont
    simu['xx'] = mnt['xx']
    simu['yy'] = mnt['yy']
    filtered_simu = per_alt(simu[var], altitude_bands, mnt)
    df = filtered_simu.to_dataframe(name=xpid).dropna().reset_index().drop(columns=['xx','yy'])
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
        # TODO : gérer ça plus proprement
        if '@' not in xpid:
            user = os.environ["USER"]
            xpid = f'{xpid}@{user}'
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
    # TODO : concaténer les 2 variables dans 1 seul fichier
    # TODO : put/get Sentinel2 data from hendrix (already implemented in vortexIO)
    for var in ['scd_concurent', 'mod']:
        # open Sentinel2 data
        if var == 'mod':
            #obs = xr.open_dataset('/home/vernaym/These/DATA/Sentinel2/20210901_L3B-SNOW_SMD_R2.nc')
            obs = xr.open_dataset('/home/vernaym/These/DATA/Sentinel2/SMOD_20210901.nc')
        elif var == 'scd_concurent':
            obs = xr.open_dataset('/home/vernaym/These/DATA/Sentinel2/SCD_20210901.nc')

        # compare(obs, var=var)
        plot_ange(xpids, obs, var)  # Violinplots by elevation range
        plot_error_fields(xpids, obs, var)  # Field difference

    # 3. Clean data
    for xpid in xpids:
        shortid = xpid.split('@')[0]
        members = members_map[shortid]
        if members is None:
            os.remove(f'DIAG_{shortid}.nc')
        else:
            for member in range(members):
                os.remove(f'mb{member:03d}/DIAG_{shortid}.nc')
