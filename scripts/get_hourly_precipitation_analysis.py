import vortex
from cen.data import flow
import footprints
from vortex import toolbox
from bronx.stdtypes.date import Date, Period

toolbox.active_now = True

start = Date(2021, 12, 27, 7)
stop = Date(2021, 12, 30, 6)

date = start
while date <= stop:
    print(date)

    datebegin = date.replace(hour=6)
    dateend = date + Period(days=1)

    tbin = toolbox.input(
        role           = 'Precipitation analysis',
        kind           = 'Precipitation',
        vapp           = 'edelweiss',
        vconf          = '[geometry:area]',
        source_app     = 'antilope',
        source_conf    = 'RandomSampling',
        cutoff         = 'assimilation',
        filename       = 'precipitation_[datebegin:ymd6h]_[dateend:ymd6h]_mb[member].nc',
        experiment     = 'XP25@vernaym',
        geometry       = 'GrandesRousses1km',
        nativefmt      = 'netcdf',
        model          = 'edelweiss',
        date           = dateend.ymd6h,
        datebegin      = datebegin.ymd6h,
        dateend        = dateend.ymd6h,
        namespace      = 'vortex.multi.fr',
        member         = footprints.util.rangex(1,16,1),
        block          = 'analysis',
        intent         = 'inout',
    ),

    date = date + Period(days=1)






