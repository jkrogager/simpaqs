"""
Apply the 4MOST ETC to the noiseless quasar spectra with absorption lines
from `simulate_quasars.py`. The templates are randomly assigned an observed
magnitude between 18 and 20.5 (by default) in the r-band (by default).
The airmass and seeing are assigned randomly. The moon phase is assumed
to be dark by default and we use a default 1200 sec exposure time.
"""

__author__ = 'Jens-Kristian Krogager'
__email__ = 'jens-kristian.krogager@univ-lyon1.fr'

import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from argparse import ArgumentParser
import os
import warnings
import sys
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

from qmostetc import QMostObservatory, SEDTemplate, L1DXU


class Target:
    # TEMPLATE,REDSHIFT,MAG,MAG_TYPE,EXPTIME,AIRMASS,MOON,SEEING,SPECTRO
    def __init__(self, row):
        self.template = row['TEMPLATE']
        self.redshift = row['REDSHIFT']
        self.mag = row['MAG']
        self.mag_type = row['MAG_TYPE']
        self.exptime = row['EXPTIME']
        self.airmass = row['AIRMASS']
        self.moon = row['MOON']
        self.seeing = row['SEEING']
        self.spectro = row['SPECTRO']

        self.ra = np.random.uniform(0., 360.)
        self.dec = np.random.uniform(-5., -80.)
        # assign TRG_UID:
        self.uid = np.random.randint(10000000)

        # assign CNAME:
        model_id = row['ID']
        # self.name = f"{model_id}_z{self.redshift:.1f}_mag{self.mag:.1f}"
        self.name = f"{model_id}"


def run_ETC_target(qmost, target, output_dir, template_path='output/quasar_models', CR_rate=1.67e-7):
    """
    Run the ETC for a given target. The function generates the separate spectra per arm
    and the joined spectrum. The simulations add random cosmic rays.

    target : `:class:Target`
        Target information

    output_dir : str
        Directory to put the spectra

    template_path : str
        Directory to the spectral templates

    CR_rate : float
        Cosmic ray rate per second per pixel
    """
    # CR_rate = 0 # If we assume sigma-clipping is done to remove cosmic rays

    # The observation with certain conditions: airmass (zenith angle), seeing,
    # moon brightness
    airmass = np.arccos(1./target.airmass)*180/np.pi * u.deg
    obs = qmost(airmass, target.seeing*u.arcsec, target.moon)

    # The target spectrum template.
    template_filename = os.path.join(template_path, target.template)
    spectrum = SEDTemplate(template_filename)

    # Add the target spectrum from the template with a magnitude
    obs.set_target(spectrum(target.mag*u.ABmag, target.mag_type), 'point')
    
    # Retrieve the result table
    texp = target.exptime*u.second
    res = obs.expose(texp)
    if np.isnan(res['target']).any():
        res['target'][np.isnan(res['target'])] = 0.
    if np.isnan(res['sky']).any():
        res['sky'][np.isnan(res['sky'])] = 0.

    # Add cosmic rays:
    # N_pix = len(res)
    # N_cosmic = np.random.poisson(CR_rate * target.exptime * N_pix * 0.8)
    # idx = np.random.choice(np.arange(N_pix), N_cosmic, replace=False)
    # CR_boost = 10**np.random.normal(2.0, 0.5, N_cosmic) * u.electron
    # res['target'][idx] += CR_boost
    # res['noise'][idx] = np.sqrt(res['noise'][idx]**2 + CR_boost * u.electron)

    dxu = L1DXU(obs.observatory, res, texp)

    # Write individual L1 files
    # for arm_name in obs.keys():
    #     INST = 'L' if target.spectro.lower() == 'lrs' else 'H'
    #     INST += arm_name.upper()[0]
    #     INST += '1'
    #     output_arm = os.path.join(output_dir, f'{target.name}_{INST}.fits')
    #     try:
    #         hdu_list = dxu.per_arm(arm_name)
    #         hdu_list = update_header(hdu_list, target)
    #         hdu_list.writeto(output_arm, overwrite=True)
    #     except ValueError as e:
    #         print(f"Failed to save the spectrum: {target.template}")
    #         print(f"for arm: {arm_name}")

    if target.spectro.lower() == 'lrs':
        # Create JOINED L1 SPECTRUM:
        output = os.path.join(output_dir, f'{target.name}_LJ1.fits')
        try:
            hdu_list = dxu.joined()
            hdu_list = update_header(hdu_list, target)
            hdu_list.writeto(output, overwrite=True)
        except ValueError as e:
            print(f"Failed to save the joined spectrum: {target.template}")


def update_header(hdu_list, target):
    specuid = np.random.randint(10000000)
    hdu_list[0].header['OBID'] = 101
    hdu_list[0].header['OBID1'] = 101
    hdu_list[0].header['PROG_ID'] = 'SIMPAQS'
    hdu_list[0].header['MJD-OBS'] = Time.now().mjd
    hdu_list[0].header['MJD-END'] = Time.now().mjd
    hdu_list[0].header['TRG_UID'] = target.uid
    hdu_list[0].header['TRG_NME'] = target.name
    hdu_list[1].header['TRG_UID'] = target.uid
    hdu_list[1].header['TRG_NME'] = target.name
    hdu_list[0].header['OBJ_UID'] = target.uid
    hdu_list[0].header['OBJ_NME'] = target.name
    hdu_list[1].header['OBJ_UID'] = target.uid
    hdu_list[1].header['OBJ_NME'] = target.name
    hdu_list[0].header['SPECUID'] = specuid
    hdu_list[1].header['SPECUID'] = specuid

    hdu_list[1].header['TRG_MAG'] = target.mag
    hdu_list[1].header['TRG_Z'] = target.redshift
    hdu_list[1].header['TRG_TMP'] = os.path.basename(target.template)

    hdu_list[0].header['ESO TEL AIRM END'] = target.airmass
    hdu_list[0].header['ESO TEL AIRM START'] = target.airmass
    hdu_list[0].header['ESO TEL AMBI FWHM END'] = target.seeing
    hdu_list[0].header['ESO TEL AMBI FWHM START'] = target.seeing
    hdu_list[0].header['ESO TEL AMBI MOON'] = target.moon
    return hdu_list


def process_catalog(catalog, band='DECam.r', mag_min=18., mag_max=20.5, template_path='',
                    exptime=1200, moon='dark', spectro='LRS', output='l1_data'):
    catalog['TEMPLATE'] = ['%s.fits' % f for f in catalog['ID']]
    catalog['EXPTIME'] = exptime
    catalog['MOON'] = moon
    catalog['SEEING'] = np.random.uniform(0.8, 1.0, len(catalog))
    catalog['AIRMASS'] = np.random.uniform(1.1, 1.3, len(catalog))
    catalog['SPECTRO'] = spectro
    catalog['MAG_TYPE'] = band
    catalog['MAG'] = np.random.uniform(mag_min, mag_max, len(catalog))

    if not os.path.exists(output):
        os.makedirs(output)
    
    warnings.simplefilter('ignore', u.UnitsWarning)
    warnings.simplefilter('ignore', fits.card.VerifyWarning)
    print("Applying 4MOST ETC to the catalog:")
    # Object to simulate the 4MOST observatory, including atmosphere,
    # telescope, spectrograph, CCD.
    qmost = QMostObservatory(spectro.lower())
    for num, row in enumerate(tqdm(catalog), 1):
        target = Target(row)
        run_ETC_target(qmost, target, output, template_path=template_path)
    catalog.write(os.path.join(output, 'observations.csv'), overwrite=True)


if __name__ == '__main__':
    parser = ArgumentParser(description="Generate simulated spectra from input catalog")
    parser.add_argument("input", type=str,
                        help="input catalog file, mandatory columns: TEMPLATE/ID, REDSHIFT, MAG, MAG_TYPE")
    parser.add_argument('--airmass', type=float, default=1.1)
    parser.add_argument('--exptime', type=float, default=1200)
    parser.add_argument('--moon', type=str, default='dark')
    parser.add_argument('--seeing', type=float, default=0.8)
    parser.add_argument('--spectro', type=str, default='LRS', choices=['LRS', 'HRS'])
    parser.add_argument('--path', type=str, default='output/quasar_models')
    parser.add_argument("-o", "--output", type=str, default='output/l1_data/',
                        help="output directory [default=./output/l1_data]")

    args = parser.parse_args()

    catalog = Table.read(args.input)
    process_catalog(catalog, exptime=args.exptime, moon=args.moon,
                    spectro=args.spectro, output=args.output,
                    template_path=args.path)

