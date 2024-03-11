"""
Script to simulate the exposure times and mock reduced 4MOST spectra
for a given input target catalog and a set of spectral rules and rulesets.
"""

__author__ = 'JK Krogager'

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from argparse import ArgumentParser
import numpy as np
import warnings
import os
import sys
import datetime

from qmostetc import SEDTemplate, QMostObservatory, Ruleset, Rule, Filter, L1DXU


def load_rulesets(qmost, ruleset_fname, rules_fname):
    rules = Rule.read(qmost, rules_fname)
    ruleset_list = Ruleset.read(ruleset_fname)
    rulesets = {}
    for rs in ruleset_list:
        rs.set_rules(rules)
        rulesets[rs.name] = rs
    return rulesets


def update_header(hdu_list, target):
    specuid = np.random.randint(10000000)
    hdu_list[0].header['OBID'] = 101
    hdu_list[0].header['OBID1'] = 101
    hdu_list[0].header['ESO TEL AIRM END'] = target['AIRMASS']
    hdu_list[0].header['ESO TEL AIRM START'] = target['AIRMASS']
    hdu_list[0].header['ESO TEL AMBI FWHM END'] = target['SEEING']
    hdu_list[0].header['ESO TEL AMBI FWHM START'] = target['SEEING']
    hdu_list[0].header['ESO TEL AMBI MOON'] = target['MOON']
    hdu_list[0].header['PROG_ID'] = 'QMOST-ETC'
    hdu_list[0].header['MJD-OBS'] = Time.now().mjd
    hdu_list[0].header['MJD-END'] = Time.now().mjd
    hdu_list[0].header['OBJ_UID'] = hash(target['NAME'])
    hdu_list[0].header['OBJ_NME'] = target['NAME']
    hdu_list[1].header['OBJ_UID'] = hash(target['NAME'])
    hdu_list[1].header['OBJ_NME'] = target['NAME']
    hdu_list[0].header['TRG_UID'] = hash(target['NAME'])
    hdu_list[0].header['TRG_NME'] = target['NAME']
    hdu_list[1].header['TRG_UID'] = hash(target['NAME'])
    hdu_list[1].header['TRG_NME'] = target['NAME']
    hdu_list[1].header['TRG_MAG'] = target['MAG']
    hdu_list[0].header['SPECUID'] = specuid
    hdu_list[1].header['SPECUID'] = specuid
    hdu_list[1].header['TRG_Z'] = target['REDSHIFT_ESTIMATE']
    hdu_list[1].header['TRG_TMP'] = os.path.basename(target['TEMPLATE'])
    return hdu_list


def process_catalog(catalog, *, ruleset_fname, rules_fname,
                    output_dir='l1_data', template_path='',
                    airmass=1.2, seeing=0.8, moon='dark',
                    CR_rate=1.67e-7, l1_type='joined', N_targets=None):

    catalog['MOON'] = moon
    catalog['SEEING'] = seeing
    catalog['AIRMASS'] = airmass

    # Object to simulate the 4MOST observatory, including atmosphere,
    # telescope, spectrograph, CCD.
    spectrograph = 'lrs' if catalog['RESOLUTION'][0] == 1 else 'hrs'
    qmost = QMostObservatory(spectrograph)
    alt = np.arccos(1. / airmass) * 180 / np.pi * u.deg
    rulesets = load_rulesets(qmost, ruleset_fname, rules_fname)
    obs = qmost(alt, seeing*u.arcsec, moon)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    warnings.simplefilter('ignore', u.UnitsWarning)
    warnings.simplefilter('ignore', fits.card.VerifyWarning)
    print("Applying 4MOST ETC to the catalog:")
    exptime_log = []
    if N_targets:
        idx = np.random.choice(np.arange(len(catalog)), N_targets, replace=False)
        catalog = catalog[idx]

    for num, row in enumerate(catalog, 1):
        ruleset_name = row['RULESET']
        target_name = row['NAME']
        ruleset = rulesets[ruleset_name]
        etc = ruleset.etc(alt, seeing*u.arcsec, moon)
        template_fname = os.path.join(template_path, row['TEMPLATE'])
        SED = SEDTemplate(template_fname)

        # Add the target spectrum from the template with a magnitude
        mag_type_str = row['MAG_TYPE']
        survey, band, ab_vega = mag_type_str.split('_')
        mag_type = [filt_id for filt_id in Filter.list()
                    if survey in filt_id and '.'+band in filt_id][0]
        mag_unit = u.ABmag
        if ab_vega != 'AB':
            print("Warning not AB magnitude in catalog... may be incorrect")
        mag = row['MAG'] * mag_unit
        SED = SED.redshift(row['REDSHIFT_ESTIMATE'])
        etc.set_target(SED(mag, mag_type), 'point')
        obs.set_target(SED(mag, mag_type), 'point')

        # Get and print the exposure time
        texp = etc.get_exptime()
        exptime_log.append({'NAME': target_name, 'MAG': row['MAG'],
                            'TEXP': texp, 'REDSHIFT': row['REDSHIFT_ESTIMATE']})

        res = obs.expose(texp)
        if np.isnan(res['target']).any():
            res['target'][np.isnan(res['target'])] = 0.
        if np.isnan(res['sky']).any():
            res['sky'][np.isnan(res['sky'])] = 0.

        # Add cosmic rays:
        N_pix = len(res)
        N_cosmic = np.random.poisson(CR_rate * texp.value * N_pix * 0.8)
        idx = np.random.choice(np.arange(N_pix), N_cosmic, replace=False)
        CR_boost = 10**np.random.normal(3.5, 0.1, N_cosmic) * u.electron
        res['target'][idx] += CR_boost
        res['noise'][idx] = np.sqrt(res['noise'][idx]**2 + CR_boost * u.electron)

        dxu = L1DXU(qmost, res, texp)

        # Write individual L1 files
        if l1_type[0].upper() == 'A':
            for arm_name in qmost.keys():
                INST = 'L' if spectrograph == 'lrs' else 'H'
                INST += arm_name.upper()[0]
                INST += '1'
                output_arm = os.path.join(output_dir, f'{target_name}_{INST}.fits')
                try:
                    hdu_list = dxu.per_arm(arm_name)
                    hdu_list = update_header(hdu_list, row)
                    hdu_list.writeto(output_arm, overwrite=True)
                except ValueError as e:
                    print(f"Failed to save the spectrum: {row['TEMPLATE']}")
                    print(f"for arm: {arm_name}")

        if spectrograph.lower() == 'lrs':
            # Create JOINED L1 SPECTRUM:
            output = os.path.join(output_dir, f"{target_name}_LJ1.fits")
            try:
                hdu_list = dxu.joined()
                hdu_list = update_header(hdu_list, row)
                hdu_list.writeto(output, overwrite=True)
            except ValueError as e:
                print(f"Failed to save the joined spectrum: {row['TEMPLATE']}")

        sys.stdout.write(f"\r{num}/{len(catalog)}")
        sys.stdout.flush()
    exptimes = Table(exptime_log)
    exptimes.meta['comments'] = ['Exposure times in seconds']
    log_fname = os.path.join(output_dir, 'exposure_times.csv')
    exptimes.write(log_fname,
                   formats={'TEXP': '%.1f', 'MAG': '%.2f', 'REDSHIFT': '%.4f'},
                   overwrite=True, comment='# ')
    print("")



def main():
    parser = ArgumentParser(description="Generate simulated spectra from 4MOST Target Catalog")
    parser.add_argument("input", type=str,
                        help="input target FITS catalog")
    parser.add_argument('--airmass', type=float, default=1.2)
    parser.add_argument('--moon', type=str, default='dark', choices=['dark', 'gray', 'bright'])
    parser.add_argument('--seeing', type=float, default=0.8)
    parser.add_argument('-n', '--number', type=int, default=None)
    parser.add_argument('--rules', type=str, help='Rules definition (FITS or CSV)')
    parser.add_argument('--ruleset', type=str, help='Ruleset definition (FITS or CSV)')
    parser.add_argument('--temp-dir', type=str, default='./', help='Directory of spectral templates')
    parser.add_argument("-o", "--output", type=str, default='l1_data',
                        help="output directory [default=./l1_data]")
    parser.add_argument('--arm', type=str, default='J', choices=['J', 'joined', 'ALL', 'a'])


    args = parser.parse_args()

    t1 = datetime.datetime.now()
    catalog = Table.read(args.input)

    process_catalog(catalog,
                    ruleset_fname=args.ruleset,
                    rules_fname=args.rules,
                    output_dir=args.output,
                    template_path=args.temp_dir,
                    airmass=args.airmass,
                    seeing=args.seeing,
                    moon=args.moon,
                    l1_type=args.arm,
                    N_targets=args.number)
    t2 = datetime.datetime.now()
    dt = t2 - t1
    if args.number:
        N_targets = args.number
    else:
        N_targets = len(catalog)
    print(f"Finished simulation of {N_targets} targets in {dt.total_seconds():.1f} seconds")


if __name__ == '__main__':
    main()

