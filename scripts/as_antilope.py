#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from vortex import toolbox
import cen

from bronx.stdtypes.date import Date

from These.radar.Preprocessing_ANTILOPE import AntilopePreprocessing

toolbox.active_now = True


def parse_command_line():

    description = "ANTILOPE-based ensemble precipitation analysis"

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-b', '--datebegin', type=str, required=True,
                        help="First date covered by the simulation file, format YYYYMMDDHH.")

    parser.add_argument('-e', '--dateend', type=str, required=True,
                        help="Last date covered by the simulation file, format YYYYMMDDHH.")

    parser.add_argument('-d', '--domain', type=str, default='alp',
                        help="Simulation domain")

    parser.add_argument('-m', '--members', type=int, default=1,
                        help="Number of ensemble members (default = deterministic")

    args = parser.parse_args()

    args.datebegin = Date(args.datebegin)
    args.dateend = Date(args.dateend)

    return args


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


if __name__ == '__main__':
    args = parse_command_line()
    datebegin = args.datebegin
    dateend   = args.dateend
    domain    = args.domain

    filename = f'ANTILOPE_{domain}_{datebegin.ymdh}_{dateend.ymdh}.nc'
    antilope = get_antilope(domain, filename=filename)

    pp = AntilopePreprocessing(domain, filename, datebegin=datebegin, dateend=dateend)
    antilope = pp.run()

    if args.members == 1:

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

    elif args.members > 1:

        antilope.analysis.expand_dims('member')

        draw_gauss = AntilopePreprocessing.random_draw(distribution='normal', members=args.members)
        draw_gauss2 = AntilopePreprocessing.random_draw(distribution='normal', members=args.members)
