import numpy as np
import xarray as xr
import vortex
import cen
from vortex import toolbox
from bronx.stdtypes.date import Date, Period

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
        #member         = footprints.util.rangex(1, self.conf.nmembers),
        member         = 1,
        fatal          = True,
    ),
    print(t.prompt, 'tbpro =', tbpro)
    print()


def LCSCD_core (t):
    #t=np.ceil(t)
    n=t.shape[0]
    #print(t.shape)
    if np.all(t==t[0]) :
        #print('non')# point constant aucour du temps a la resolution temporelle cible
        if t[0]==0: # point toujours sans neige selon tempaggreg
            return (0,0)
            #print(idx,idy)
        if t[0]==1: # point toujours avec neige selon tempaggreg
            return (0,-1)#(pro.time[-1]-pro.time[0])/ np.timedelta64(1, 'D')

    else :
        #run_values, run_starts, run_lengths = find_runs(t)# on arrondi au supperieur l'array grace a np.ceil
         # ensure array
        # find_runs function 
        #t = np.asanyarray(t)
            # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(t[:-1], t[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]
            # find run values
        run_values = t[loc_run_start]
            # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        #print(run_values, run_starts, run_lengths)
        #np.trunc(run_values) pour blinder le SCAgroupé.mean().ceil() car on veut des valeurs binaires
        argmax=np.argmax(np.multiply(np.trunc(run_values),run_lengths)) 
        # on multiplie run_lenght avec np.trunc(run_values) pour supprimer les longues periode sans neige 
        # on prends argmax=l'argument des arrays run_XXX correspondant a la plus longue des periodes avec neige
        #print(run_starts,run_values,run_lengths,run_values[argmax])
        #print(t.shape)
        if run_starts[argmax]+run_lengths[argmax]>n-1:
            return (run_starts[argmax],run_starts[argmax]+run_lengths[argmax]-1) # le -1 pour eviter un crash si neige jusqua fin de simu on termine en non inclue
        else :
            return (run_starts[argmax],run_starts[argmax]+run_lengths[argmax]) 

def LCSCD(pro, temporal_aggreg='1H', out_aggreg = '1D', maskforest=True):
    """Durée d'eneigement de la plus longue periode continue"""
    if pro.time[0].values>np.datetime64(str(pro.time.dt.year[0].values)+'-09-01'):
        print('ERROR : Start date after 1 sept',pro.time_coverage_start)
    else:
        pro=pro.sel(time=slice(np.str(pro.time.dt.year[0].values)+'-09-01',pro.time_coverage_end))
        print('Time coverage :'+str(pro.time[0].values)+ ' => '+ str(pro.time[-1].values))
    #nombre d'heure de la plus longue periode d'enneigement continue depuis le premier septembre
    if 'SCA' in list(pro.data_vars): # chekc if sca exisit
        None
    else:
        pro=SCA(pro)

    #temporal_aggreg='1H' or '1D'
    if temporal_aggreg != '1H' and temporal_aggreg != '1D':
        print("erreur resampling 1H or 1D ")
    if out_aggreg != '1H' and out_aggreg != '1D':
        print("erreur out aggreg 1H or 1D ")
        #return False
    t=np.ceil(pro.SCA.resample(time=temporal_aggreg).mean())# /!\ on moyenne (obligé) et on tronque au sup avec ceil
    #protection for LCSCD_core find_runs fuction
    #if t.ndim != 1:
    #    raise ValueError('only 1D array supported')
    #    n = t.shape[0]
    # handle empty array
    #if n == 0:
    #    raise ValueError('only 1D array supported')
    # end of protection
    # apply LCSCD_CORE
    out=np.apply_along_axis(LCSCD_core, t.get_axis_num('time'), t)
    #out=xr.apply_ufunc(LCSCD_core,t,input_core_dims=[['time']])

    if out_aggreg == '1H':
        out=(t.time.to_numpy()[out[1,:,:]]-t.time.to_numpy()[out[0,:,:]]).astype('timedelta64[h]') / np.timedelta64(1, 'h')
    if out_aggreg == '1D':
        out=(t.time.to_numpy()[out[1,:,:]]-t.time.to_numpy()[out[0,:,:]]) / np.timedelta64(1, 'D')
    if out_aggreg == '1H':
        try : pro['LCSCD']=xr.DataArray(data=out,dims=["x", "y"],attrs=dict(description="Longest Concurent Snow Cover Duration in days, time aggregated in 1 hour steps"),) 
        except : pro['LCSCD']=xr.DataArray(data=out,dims=["y", "x"],attrs=dict(description="Longest Concurent Snow Cover Duration in days, time aggregated in 1 hour steps"),) 
        if maskforest==True :
            pro['LCSCD']=foretmask(pro.LCSCD)
        pro.LCSCD.assign_attrs(units="hours")
    if out_aggreg == '1D':
        try : pro['LCSCD']=xr.DataArray(data=out.astype('float'),dims=["x", "y"],attrs=dict(description="Longest Concurent Snow Cover Duration in days, time aggregated in 1 day steps"),) 
        except : pro['LCSCD']=xr.DataArray(data=out.astype('float'),dims=["y", "x"],attrs=dict(description="Longest Concurent Snow Cover Duration in days, time aggregated in 1 day steps"),)
        if maskforest==True :
            pro['LCSCD']=foretmask(pro.LCSCD)
        pro.LCSCD.assign_attrs(units="days")
    return pro

def LCSMOD(pro, temporal_aggreg='1H', out_aggreg='1D', maskforest=True):
    """date de fonte de la plus longue periode d'enneigement continue en jour depuis le 1er septembre"""

        #temporal_aggreg='1H' or '1D'
    if temporal_aggreg != '1H' and temporal_aggreg != '1D':
        print("erreur resampling 1H or 1D ")
        #temporal_aggreg='1H' or '1D'
    if out_aggreg != '1H' and out_aggreg != '1D':
        print("erreur out aggreg 1H or 1D ")
    if out_aggreg=='1H':
        maskforest=False
    if pro.time[0].values>np.datetime64(str(pro.time.dt.year[0].values)+'-09-01'):
        print('ERROR : Start date before 1 sept',pro.time_coverage_start)
    else:
        pro=pro.sel(time=slice(np.str(pro.time.dt.year[0].values)+'-09-01',pro.time_coverage_end))
        print('Time coverage :'+str(pro.time[0].values)+ ' => '+ str(pro.time[-1].values))
    #nombre d'heure de la plus longue periode d'enneigement continue depuis le premier septembre
    if 'SCA' in list(pro.data_vars): # chekc if sca exisit
        None
    else:
        pro=SCA(pro)

        #return False
    t=np.ceil(pro.SCA.resample(time=temporal_aggreg).mean())# /!\ on moyenne (obligé) et on tronque au sup avec ceil
    #protection for LCSCD_core find_runs fuction
    #if t.ndim != 1:
    #    raise ValueError('only 1D array supported')
    #    n = t.shape[0]
    # handle empty array
    #if n == 0:
    #    raise ValueError('only 1D array supported')
    # end of protection
    # apply LCSCD_CORE
    out=np.apply_along_axis(LCSCD_core,t.get_axis_num('time'),t)
    #out=xr.apply_ufunc(LCSCD_core,t,input_core_dims=[['time']])

    if out_aggreg == '1H':
        out=(t.time.to_numpy()[out[1,:,:]])
        #print(out)
    if out_aggreg == '1D':
        #print(t.time.to_numpy()[0])
        #print(t.time.to_numpy()[out[1,:,:]])
        out=(t.time.to_numpy()[out[1,:,:]]-t.time.to_numpy()[0])/ np.timedelta64(1, 'D')
        #print(out)    
    if out_aggreg == '1H':
        try : pro['LCSMOD']=xr.DataArray(data=out,dims=["x", "y"],attrs=dict(description="Melt out date of the Longest Concurent Snow Cover Duration in days, time aggregated in 1 hour steps"),) 
        except : pro['LCSMOD']=xr.DataArray(data=out,dims=["y", "x"],attrs=dict(description="Melt out date of the Longest Concurent Snow Cover Duration in days, time aggregated in 1 hour steps"),) 
        if maskforest==True :
            pro['LCSMOD']=foretmask(pro.LCSMOD)
        pro.LCSMOD.assign_attrs(units="days")
    if out_aggreg == '1D':
        try : pro['LCSMOD']=xr.DataArray(data=out.astype('float'),dims=["x", "y"],attrs=dict(description="Melt out date of the Longest Concurent Snow Cover Duration in days, time aggregated in 1 day steps"),) 
        except : pro['LCSMOD']=xr.DataArray(data=out.astype('float'),dims=["y", "x"],attrs=dict(description="Melt out date of the Longest Concurent Snow Cover Duration in days, time aggregated in 1 day steps"),)
        if maskforest==True :
            pro['LCSMOD']=foretmask(pro.LCSMOD)
        pro.LCSMOD.assign_attrs(units="days")
    return pro

def LCSOD(pro, temporal_aggreg='1H', out_aggreg='1D', maskforest=True):
    """date de début d'enneigement de la plus longue periode d'enneigement continue en jour depuis le 1er septembre"""

    # temporal_aggreg='1H' or '1D'
    if temporal_aggreg != '1H' and temporal_aggreg != '1D':
        print("erreur resampling 1H or 1D ")
    # out_aggreg='1H' or '1D'
    if out_aggreg != '1H' and out_aggreg != '1D':
        print("erreur out aggreg 1H or 1D ")
    if out_aggreg=='1H':
        maskforest=False
    if pro.time[0].values>np.datetime64(str(pro.time.dt.year[0].values)+'-09-01'):
        print('ERROR : Start date after 1 sept', pro.time_coverage_start)
    else:
        pro=pro.sel(time=slice(str(pro.time.dt.year[0].values)+'-09-01',pro.time_coverage_end))
        print('Time coverage :'+str(pro.time[0].values)+ ' => '+ str(pro.time[-1].values))
    #nombre d'heure de la plus longue periode d'enneigement continue depuis le premier septembre
    if 'SCA' in list(pro.data_vars): # chekc if sca exisit
        None
    else:
        pro=SCA(pro)

    t = np.ceil(pro.SCA.resample(time=temporal_aggreg).mean())# /!\ on moyenne (obligé) et on tronque au sup avec ceil
    #protection for LCSCD_core find_runs fuction
    #if t.ndim != 1:
    #    raise ValueError('only 1D array supported')
    #    n = t.shape[0]
    # handle empty array
    #if n == 0:
    #    raise ValueError('only 1D array supported')
    # end of protection
    # apply LCSCD_CORE
    out = np.apply_along_axis(LCSCD_core,t.get_axis_num('time'),t)
    #out=xr.apply_ufunc(LCSCD_core,t,input_core_dims=[['time']])

    if out_aggreg == '1H':
        out=(t.time.to_numpy()[out[0,:,:]])
        #print(out)
    if out_aggreg == '1D':
        #print(t.time.to_numpy()[0])
        #print(t.time.to_numpy()[out[1,:,:]])
        out=(t.time.to_numpy()[out[0,:,:]]-t.time.to_numpy()[0])/ np.timedelta64(1, 'D')
        #print(out)    
    if out_aggreg == '1H':
        try : pro['LCSOD']=xr.DataArray(data=out,dims=["x", "y"],attrs=dict(description="Snow onset date of the Longest Concurent Snow Cover Duration in days, time aggregated in 1 hour steps"),) 
        except : pro['LCSOD']=xr.DataArray(data=out,dims=["y", "x"],attrs=dict(description="Snow onset date of the Longest Concurent Snow Cover Duration in days, time aggregated in 1 hour steps"),) 
        if maskforest==True :
            pro['LCSOD'] = foretmask(pro.LCSOD)
        pro.LCSOD.assign_attrs(units="days")
    if out_aggreg == '1D':
        try : pro['LCSOD']=xr.DataArray(data=out.astype('float'),dims=["x", "y"],attrs=dict(description="Snow onset date of the Longest Concurent Snow Cover Duration in days, time aggregated in 1 day steps"),) 
        except : pro['LCSOD']=xr.DataArray(data=out.astype('float'),dims=["y", "x"],attrs=dict(description="Snow onset date of the Longest Concurent Snow Cover Duration in days, time aggregated in 1 day steps"),)
        if maskforest==True :
            pro['LCSOD']=foretmask(pro.LCSOD)
        pro.LCSOD.assign_attrs(units="days")
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
    get_pro()
    for member in range(1, 17):
        pro = xr.open_dataset(f'mb{member:03d}/{proname}', decode_times=False)
        pro = decode_time(pro)
        pro = LCSOD(pro, temporal_aggreg='1D', maskforest=False)
        pro = LCSMOD(pro, temporal_aggreg='1D', maskforest=False)
        pro = LCSCD(pro, temporal_aggreg='1D', maskforest=False)

        import pdb
        pdb.set_trace()
