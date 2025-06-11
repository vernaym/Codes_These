#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import xarray as xr

from vortex import toolbox
import cen

from bronx.stdtypes.date import Date

from These.radar.Preprocessing_ANTILOPE import AntilopePreprocessing

toolbox.active_now = True


def usage():
    print("USAGE make_map.py datebegin dateend")
    print("format des dates : YYYYMMDD (précipitations de YYYYMMD-1 6h à YYYYMMDD6H")
    sys.exit(1)


try:
    datebegin = Date(sys.argv[1])
    dateend = Date(sys.argv[2])

except Exception as e:
    usage()
    raise e


def get_antilope(domain, filename='ANTILOPE.nc', obs_auto=None):

    toolbox.input(
        kind           = 'Precipitation',
        vapp           = 'antilope',
        vconf          = '[geometry:tag]',
        date           = '[dateend]',
        geometry       = domain,
        datebegin      = datebegin.ymdh,
        dateend        = dateend.ymdh,
        experiment     = 'oper@vernaym',
        local          = filename,
        block          = 'Hourly',
        namebuild      = 'flat@cen',
        namespace      = 'vortex.multi.fr',
    )


domain = 'alp'

filename = f'ANTILOPE_{domain}_{datebegin.ymdh}_{dateend.ymdh}.nc'
antilope = get_antilope(domain, filename=filename)

pp = AntilopePreprocessing(domain, filename, datebegin=datebegin, dateend=dateend)
antilope = pp.run()
out = antilope.analysis.rename('Precipitation')
outname = f'ASANTILOPE_{domain}_{datebegin.ymdh}_{dateend.ymdh}.nc'
out.to_netcdf(outname)

toolbox.output(
    kind           = 'Precipitation',
    vapp           = 'antilope',
    vconf          = domain,
    date           = '[dateend]',
    geometry       = 'FRANXL1S100',
    datebegin      = datebegin.ymdh,
    dateend        = dateend.ymdh,
    experiment     = 'RS51@vernaym',
    local          = outname,
    block          = 'Hourly',
    namebuild      = 'flat@cen',
    namespace      = 'vortex.multi.fr',
)
