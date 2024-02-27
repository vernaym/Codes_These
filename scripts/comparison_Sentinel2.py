import os
import numpy as np
import xarray as xr

import vortexIO


datebegin = '2021080207'  # TODO : passer en argument
dateend = '2022080106'  # TODO : passer en argument
xpid = 'XP00@vernaym'  # TODO : passer en argument
geometry = 'GrandesRousses250m'

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


if __name__ == '__main__':
    os.chdir('/mnt/lfs/d10/mrns/users/NO_SAVE/vernaym/workdir/eval_with_Sentinel2')
    vortexIO.get_pro(datebegin, dateend, xpid, geometry, members=16)
    for member in range(1, 17):
        print(f'Member {member}')
        pro = xr.open_dataset(f'mb{member:03d}/PRO.nc', decode_times=False)
        pro = decode_time(pro)
        diag = lcscd(pro.DSN_T_ISBA)
        diag.rename({'xx':'x', 'yy':'y'})

        #pro = update_pro(pro, temporal_aggreg='1D')
        #diag = pro[['LCSCD', 'LCSMOD', 'LCSOD']]

        mask = True
        if mask:
            # mask glacier/forest covered pixels
            diag = maskgf(diag)
            block = 'mask'
        else:
            block = 'nomask'
        diag.to_netcdf(f'mb{member:03d}/DIAG.nc')
        os.remove(f'mb{member:03d}/PRO.nc')

    vortexIO.put_diag(datebegin, dateend, xpid, geometry, members=16, block=block)

    for member in range(1, 17):
        os.remove(f'mb{member:03d}/DIAG.nc')
