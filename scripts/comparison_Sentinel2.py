import os
import numpy as np
import xarray as xr

import vortexIO


datebegin = '2021080207'  # TODO : passer en argument
dateend = '2022080106'  # TODO : passer en argument
xpid = 'XP00@vernaym'  # TODO : passer en argument
geometry = 'GrandesRousses250m'

def maskgf(pro, method='nearest'):
    """
      Masks an input array (pro) using a reference mask dataset.

      Args:
          pro: The input array to be masked.
          method: The interpolation method to use when resampling the glacier mask
                  to the same resolution as the input array. Valid options are
                  'nearest', 'linear', 'cubic', etc. (default: 'nearest').

      Returns:
          A new array with the same shape as the input array, where values are masked
          out based on the glacier mask. Masked values are set to NaN.
    """

  # Load the glacier mask dataset
    #masque=xr.open_dataset('/home/vernaym/These/DATA/mask/masque_foret_glacier.nc').Band1.interp_like(pro,method=method)
    masque = xr.open_dataset('/home/vernaym/These/DATA/mask/masque_glacier2017_foret_ville_riviere.nc')['Band1']

  # Interpolate the glacier mask to the same resolution as the input array
    masque = masque.interp_like(pro, method=method)

  # Mask the input array based on the glacier mask
    return pro.where(masque == 0)

def check_pro(pro, temporal_aggreg, out_aggreg):
    """
    Check PRO file:
    * period
    * temporal resolution
    * variables (SCA)
    """
    if pro.time[0].values > np.datetime64(str(pro.time.dt.year[0].values)+'-09-01'):
        print('ERROR : Start date after 1 sept',pro.time_coverage_start)
    else:
        pro = pro.sel(time=slice(str(pro.time.dt.year[0].values)+'-09-01',pro.time_coverage_end))
        print('Time coverage :'+str(pro.time[0].values)+ ' => '+ str(pro.time[-1].values))
    #temporal_aggreg='1H' or '1D'
    if temporal_aggreg != '1H' and temporal_aggreg != '1D':
        raise("erreur resampling 1H or 1D ")
    if out_aggreg != '1H' and out_aggreg != '1D':
        raise("erreur out aggreg 1H or 1D ")
    pro = checkSCA(pro)

    return pro

def checkSCA(pro):
    """
    Check if Snow Cover Area already in pro variables
    If not, compute it.
    """
    if 'SCA' not in list(pro.data_vars):
        pro = SCA(pro)
    return pro

def LCSCD_core(t):
    """
    Look for Longest Continuous Snow Cover Duration in the simulation
    """
    n = t.shape[0]
    if np.all(t == t[0]) :
        if t[0] == 0:  # point toujours sans neige selon tempaggreg
            return (0,0)
        if t[0] == 1:  # point toujours avec neige selon tempaggreg
            return (0, -1)  #(pro.time[-1]-pro.time[0])/ np.timedelta64(1, 'D')
    else :
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(t[:-1], t[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]
        run_values = t[loc_run_start]
        run_lengths = np.diff(np.append(run_starts, n))

        argmax=np.argmax(np.multiply(np.trunc(run_values), run_lengths))
        # on multiplie run_lenght avec np.trunc(run_values) pour supprimer les longues periode sans neige 
        # on prends argmax=l'argument des arrays run_XXX correspondant a la plus longue des periodes avec neige
        if run_starts[argmax]+run_lengths[argmax] > n-1:
            return (run_starts[argmax], run_starts[argmax] + run_lengths[argmax] - 1) # le -1 pour eviter un crash si neige jusqua fin de simu on termine en non inclue
        else :
            return (run_starts[argmax], run_starts[argmax] + run_lengths[argmax])

def Update_pro(pro, temporal_aggreg='1H', out_aggreg = '1D'):
    """
    Add the following variables in the PRO file:
    * LCSCD  : Longest Concurent Snow Cover Duration period
    * LCSMOD : Snow Melt Out Date of the Longest Concurent snow cover period
    * LCSOD  : Snow Cover Onset date of the Longest Concurent snow cover period
    """

    description_map = dict(
            LCSCD  = "Longest Concurent Snow Cover Duration in days, time aggregated in 1 hour steps",
            LCSMOD = "Melt out date of the Longest Concurent Snow Cover Duration in days, time aggregated in 1 day steps",
            LCSOD  = "Snow onset date of the Longest Concurent Snow Cover Duration in days, time aggregated in 1 hour steps",
        )

    pro = check_pro(pro, temporal_aggreg, out_aggreg)

    # Identify snow-covered pixels
    sc = np.ceil(pro.SCA.resample(time=temporal_aggreg).mean())# /!\ on moyenne (obligé) et on tronque au sup avec ceil
    # Identify Longest Concurent Snow Cover Duration period
    lcscd = np.apply_along_axis(LCSCD_core, sc.get_axis_num('time'), sc)

    newvar = dict()
    if out_aggreg == '1H':
        newvar['LCSCD']  = (sc.time.to_numpy()[lcscd[1,:,:]]-sc.time.to_numpy()[lcscd[0,:,:]]).astype('timedelta64[h]') / np.timedelta64(1, 'h')
        newvar['LCSMOD'] = (sc.time.to_numpy()[lcscd[1,:,:]])
        newvar['LCSOD']  = (sc.time.to_numpy()[lcscd[0,:,:]])
        units = "hours"
    elif out_aggreg == '1D':
        newvar['LCSCD']  = (sc.time.to_numpy()[lcscd[1,:,:]]-sc.time.to_numpy()[lcscd[0,:,:]]) / np.timedelta64(1, 'D')
        newvar['LCSMOD'] = (sc.time.to_numpy()[lcscd[1,:,:]]-sc.time.to_numpy()[0]) / np.timedelta64(1, 'D')
        newvar['LCSOD']  = (sc.time.to_numpy()[lcscd[0,:,:]]-sc.time.to_numpy()[0]) / np.timedelta64(1, 'D')
        units = "days"

    for key, value in newvar.items():
        # TODO : Mettre les données au format ("y", "x") et gérer la projection + ajouter un le MNT
        pro[key] = xr.DataArray(
                data  = value,
                dims  = ["y", "x"],
                attrs = dict(
                    description = description_map[key],
                    units = units,
                ),
            )

    return pro

def SCA(pro, h_lim=.2):
    """
    h_lim = snow depth min below which the pixel is considered snow-free
    returns :
    * 0 for snow-free pixels
    * 1 for snow-covered pixels
    """
    pro['SCA']=(xr.where(pro.DSN_T_ISBA<=h_lim,False,True))
    return pro

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
        pro = Update_pro(pro, temporal_aggreg='1D')
        diag = pro[['LCSCD', 'LCSMOD', 'LCSOD']]

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
