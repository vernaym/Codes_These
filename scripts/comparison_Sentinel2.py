import os
import numpy as np
import xarray as xr
import vortex
import cen
from vortex import toolbox
from bronx.stdtypes.date import Date, Period

import footprints

toolbox.active_now = True

t = vortex.ticket()

datebegin = Date('2021080207')  # TODO : passer en argument
dateend = Date('2022080106')  # TODO : passer en argument
xpid = 'XP00@vernaym'  # TODO : passer en argument
geometry = 'GrandesRousses250m'
namespace = 'vortex.multi.fr'
proname = f'PRO_{datebegin.ymdh}_{dateend.ymdh}.nc'

def get_pro():
    tbpro = toolbox.input(
        local          = f'mb[member]/{proname}',
        experiment     = xpid,
        geometry       = geometry,
        datebegin      = datebegin,
        dateend        = dateend,
        date           = dateend,
        nativefmt      = 'netcdf',
        kind           = 'SnowpackSimulation',
        vapp           = 'edelweiss',
        vconf          = '[geometry:tag]',
        model          = 'surfex',
        namespace      = namespace,
        namebuild      = 'flat@cen',
        block          = 'pro',
        member         = footprints.util.rangex(1, 16),
        #member         = 1,
        fatal          = True,
    ),
    print(t.prompt, 'tbpro =', tbpro)
    print()

def save_diag():
    tbpro = toolbox.output(
        local          = f'mb[member]/DIAG.nc',
        experiment     = xpid,
        geometry       = geometry,
        begindate      = datebegin,
        enddate        = dateend,
        scope          = 'SesonalSnowCoverDiagnostic',
        date           = dateend,
        nativefmt      = 'netcdf',
        kind           = 'diagnostics',
        vapp           = 'edelweiss',
        vconf          = '[geometry:tag]',
        model          = 'surfex',
        namespace      = namespace,
        namebuild      = 'flat@cen',
        block          = 'pro',
        member         = footprints.util.rangex(1, 16),
        #member         = 1,
        fatal          = True,
    ),
    print(t.prompt, 'tbpro =', tbpro)
    print()

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

#def LCSCD(pro, temporal_aggreg='1H', out_aggreg = '1D', maskforest=True):
def Update_pro(pro, temporal_aggreg='1H', out_aggreg = '1D', maskforest=True):
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

    if out_aggreg=='1H':
        maskforest=False
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
                dims  = ["yy", "xx"],
                attrs = dict(
                    description = description_map[key],
                    units = units,
                ),
            )
        if maskforest==True :
            pro[key] = foretmask(pro[key])

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

def foretmask(pro):
    filename = 'A DEFINIR'  # TODO
    fm = xr.open_dataset(filename).mask
    try :
        return xr.where(fm,np.nan,pro)
    except :
        return xr.where(fm.reindex(y=list(reversed(fm.y))),np.nan,pro)

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
    get_pro()
    for member in range(1, 17):
        print(f'Member {member}')
        pro = xr.open_dataset(f'mb{member:03d}/{proname}', decode_times=False)
        pro = decode_time(pro)
        pro = Update_pro(pro, temporal_aggreg='1D', maskforest=False)
        pro = pro[['LCSCD', 'LCSMOD', 'LCSOD']]
        pro.to_netcdf(f'mb{member:03d}/DIAG.nc')
        os.remove(f'mb{member:03d}/{proname}')
    save_diag()
