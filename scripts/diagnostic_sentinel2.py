import os
import numpy as np
import pandas as pd
import xarray as xr
import argparse

import vortexIO


#datebegin = '2021080106'  # TODO : passer en argument
#datebegin = '2021080207'  # TODO : passer en argument
#dateend = '2022080106'  # TODO : passer en argument
#xpid = 'RS25_no_pappus@vernaym'  # TODO : passer en argument
#members = 17  # TODO : passer en argument
#members = None  # TODO : passer en argument
geometry = 'GrandesRousses250m'


mntdir = '/home/vernaym/These/DATA'
workdir = '/home/vernaym/workdir/EDELWEISS/diag'



def parse_command_line():
    description = "Computation of Sentinel2-like diagnostics (snow melt-out date, snow cover duration) associated \
                   to a SURFEX simulation"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-b', '--datebegin', type=str, help="First date covered by the simulation file, format YYYYMMDDHH.")
    parser.add_argument('-e', '--dateend', type=str, help="Last date covered by the simulation file, format YYYYMMDDHH.")
    parser.add_argument('-x', '--xpid', type=str, help="XPID of the simulation format XP_NAME@username")
    parser.add_argument('-m', '--members', type=int, default=None, help="Number of members associated to the experiment")
    args = parser.parse_args()
    return args


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

def diag(subdir):
    proname = os.path.join(subdir, 'PRO.nc')
    pro = xr.open_dataset(proname, decode_times=False)
    pro = decode_time(pro)
    diag = lcscd(pro.DSN_T_ISBA.resample(time='1D').mean())
    diag = diag.rename({'xx':'x', 'yy':'y'})


    # mask glacier/forest covered pixels
    diag = maskgf(diag)

    # Write DIAG file and remove PRO
    diag.to_netcdf(os.path.join(subdir, 'DIAG.nc'))
    os.remove(os.path.join(subdir, 'PRO.nc'))


if __name__ == '__main__':

    args = parse_command_line()
    datebegin = args.datebegin
    dateend   = args.dateend
    xpid      = args.xpid
    members   = args.members

    os.chdir(workdir)

    # Retrieve PRO files with Vortex
    vortexIO.get_pro(datebegin, dateend, xpid, geometry, members=members)

    if members is None:
        subdir = ''
        diag(subdir)
    else:
        for member in range(17):
            print(f'Member {member}')
            subdir = f'mb{member:03d}'
            diag(subdir)

    # Archive DIAG files with Vortex
    vortexIO.put_diag(datebegin, dateend, xpid, geometry, members=members)

    # Clean data
    if members is None:
        os.remove('DIAG.nc')
    else:
        for member in range(members):
            os.remove(f'mb{member:03d}/DIAG.nc')
