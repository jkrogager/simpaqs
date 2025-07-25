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
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm


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


## --airmass = 'high'  # AIRMASS = 1.05 / 1.2 / 1.45
airmass_conversion = {
        'high': 1.45,
        'mid': 1.2,
        'low': 1.05,
        }

## --moon_phase = 'dark' # DARK: FLI=0.2,  GREY: FLI=0.5
sky_conversion = {
        'dark': 0.2,
        'grey': 0.5,
        }

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'data/')

# Gaia G-band filter curve:
# G_filter = np.loadtxt(data_path + 'Gaia_G.tab')
G_filter = Table.read(data_path + 'SDSS.R.fits')


# Read noise per pixel:
RON = {'blue': 2.286,
       'green': 2.265,
       'red': 2.206,
       }
all_arms = ['blue', 'green', 'red']

# Final Wavelength Grid for joint spectrum:
N_pix_in_spectrum = 23201
wl_joint = np.linspace(3700., 9500, N_pix_in_spectrum)

# Cosmic Ray rate per second per pixel:
CR_rate = 1.67e-7


def get_transmission(airmass='mid'):
    # Include various airmasses (mid, low, high)...
    sky_trans = fits.getdata(data_path + 'sky_transmission.fits')
    trans_joint = np.interp(wl_joint, sky_trans['WAVE'], sky_trans['TRANS'], right=1., left=1.)
    return trans_joint

def get_sky_model(airmass='mid', moon_phase='dark'):
    sky_model_per_arm = dict()
    for arm in all_arms:
        # Sky model for given airmass and moon phase:
        model_data = fits.getdata(data_path + f'sky_model_{moon_phase}.fits', arm)
        sky_model_per_arm[arm] = {'WAVE': model_data['WAVE'], 'FLUX': model_data[airmass]}
    return sky_model_per_arm


def get_efficiency(airmass='mid'):
    """
    Returns a Dict with three keys: blue, green, red.
    Each entry is a FITS Table with the given total system efficiency per arm:
        WAVE  Q_EFF
    """
    # Instrument, atmosphere, telescope + fibre efficiencies:
    system_eff = dict()
    for arm in all_arms:
        tab_throughput = fits.getdata(data_path + 'efficiency_4most.fits', arm)
        system_eff[arm] = tab_throughput[airmass]
    return system_eff


def synthetic_G_band(wavelength, flux):
    T = np.interp(wavelength, G_filter['wavelength'].value, G_filter['trans'].value,
                  left=0, right=0)
    synphot = np.sum(flux*T)/np.sum(T)
    return synphot


def normalize_flux(wl, flux, mag):
    """
    Normalize the flux to a given magnitude in the r-band.
    The magnitude is assumed to be in AB system.
    """
    f0 = synthetic_G_band(wl, flux)
    wl_band = 6424.9269
    f_band = 1./(wl_band)**2 * 10**(-(mag + 2.406) / 2.5)
    norm_factor = f_band / f0

    return flux * norm_factor


def apply_noise(temp_wl, temp_flux, sky_model, throughput, transmission, t_exp=2400):
    """
    temp_wl : array of wavelengths in the input template (units: Angstrom)
    temp_flux : array of fluxes in the input template (units: erg/s/cm2/A)
    sky_model : FITS-Table with three extensions {BLUE, GREEN, RED}
                The table contains the following columns: WAVE, FLUX
    throughput : FITS-Table of effective throughput including instrument, fibre and telescope
                 with three extensions {BLUE, GREEN, RED}
    transmission : array of atmospheric transmission model (telluric absorption).
                   Other absorption features can be included as well, such as metal absorption lines.
    t_exp : Exposure time in seconds (default=1200)
    filename : Filename of the noisy spectrum. If not given, the spectrum is not saved.
    """
    # Collecting area of 4.1m telescope:
    # A_tel = np.pi*((4.1/2)**2 - (1.2/2)**2)*1.e4   # cm^2
    A_tel = 120715.7  # cm^2

    # Number of spatial pixels contributing to one spectral pixel
    # i.e., the projected fibre-width on the detector
    N_pix = 6

    # Source flux in units of photons/s/cm2/A
    hc = 1.9865e-08
    photon_flux = temp_flux * temp_wl / hc

    # Calculate each arm separately and stitch at the end:
    var_all = list()
    flux_all = list()
    qual_all = list()
    for arm_num, arm in enumerate(all_arms):
        # Look up the skymodel for the given arm:
        sky = sky_model[arm]  # in units of electrons
        wl = sky['WAVE']

        # Instrument, atmosphere, telescope + fibre efficiencies:
        Q_eff = throughput[arm]

        # Interpolate source flux to sky grid:
        source = np.interp(wl, temp_wl, photon_flux)
        pixel_size = np.diff(wl)
        pixel_size = np.append(pixel_size, pixel_size[-1])

        # Spectral Quantum Factor (empirical from ESO ETC):
        SQF = 1./np.sqrt(2)

        source = source * A_tel * pixel_size * t_exp * Q_eff * SQF
        sky = sky['FLUX'] * t_exp

        # Insert cosmic rays:
        l_CR = CR_rate * t_exp * len(wl)
        N_CR = np.random.poisson(l_CR)
        CR_index = np.random.choice(np.arange(len(wl)), N_CR, replace=False)
        source[CR_index] += 10**np.random.normal(2.2, 0.2, N_CR)
        qual = np.zeros_like(wl)
        qual[CR_index] = 1

        # snr = source / np.sqrt(source + sky + N_pix*RON[arm]**2)
        noise = np.sqrt(source + sky + N_pix*RON[arm]**2)
        sensitivity = A_tel * pixel_size * t_exp * Q_eff * SQF * wl / hc
        err_arm = np.interp(wl_joint, wl, noise/sensitivity, left=np.nan, right=np.nan)
        flux_arm = np.interp(wl_joint, wl, source/sensitivity, left=np.nan, right=np.nan)
        qual_arm = np.interp(wl_joint, wl, qual, left=0, right=0)
        qual_arm = 1*(qual_arm > 0)
        var_all.append(err_arm**2)
        flux_all.append(flux_arm)
        qual_all.append(qual_arm)
    flux_all = np.array(flux_all)
    var_all = np.array(var_all)
    flux_joint = np.nansum(flux_all/var_all, axis=0) / np.nansum(1./var_all, axis=0)
    flux_joint = flux_joint * transmission
    err_joint = np.sqrt(1./np.nansum(1./var_all, axis=0))
    qual_joint = 1*(np.sum(qual_all, axis=0) > 0)
    noise = np.random.normal(0., 1., N_pix_in_spectrum)
    flux_joint = flux_joint + noise*err_joint

    return {'WAVE': wl_joint,
            'FLUX': flux_joint,
            'ERR_FLUX': err_joint,
            'QUAL': qual_joint}


def save_mock_spectrum(spec_dict, target, filename):
    hdu = fits.HDUList()

    hdr = fits.Header()
    hdr['EXPTIME']  = (target.exptime, 'Total integration time per data element (s)')
    hdr['ORIGIN']   = ('ESO-PARANAL', 'Observatory or facility')
    hdr['TELESCOP'] = ('ESO-VISTA', 'ESO telescope designation')
    hdr['INSTRUME'] = ('QMOST   ', 'Instrument name')
    hdr['OBJ_RA'] = (target.ra, 'Right Ascension of target (deg)')
    hdr['OBJ_DEC'] = (target.dec, 'Declination of target (deg)')
    hdr['QZC_Z'] = (target.redshift, 'Target Redshift')
    hdr['OBJ_ZERR'] = ((1 + target.mag)*0.01, 'Redshift uncertainty')
    hdr['OBJ_ZWRN'] = (0, 'Redshift warning flag')
    hdr['OBJ_ZTPL'] = ('QSO', 'Template used for the redshift determination')
    hdr['OBJ_CLSS'] = ('QSO', 'Object classification')
    hdr['OBJ_UID'] = (target.uid, 'Unique identifier of target')
    hdr['OBJ_NME'] = (target.name, 'Name of target')
    hdr['4L1_SNR'] = (np.nanmedian(spec_dict['FLUX'] / spec_dict['ERR_FLUX']),
                      'Median SNR of the spectrum')
    hdr['TEMPLATE'] = (target.template, 'Template used for the simulation')
    hdr['SPECUID'] = (np.random.randint(10000000), 'Unique identifier of the spectrum')
    hdr['MJD-OBS'] = Time.now().mjd

    prim = fits.PrimaryHDU(header=hdr)
    hdu.append(prim)

    # `pre` is the length of the array at 0.25Å sampling
    pre = f'{len(spec_dict["WAVE"])}'
    col_wl = fits.Column(name='WAVE', array=np.array([spec_dict['WAVE']]),
                         format=pre+'E', unit='Angstrom', disp='F9.4')
    col_flux = fits.Column(name='FLUX', array=np.array([spec_dict['FLUX']]),
                           format=pre+'E', unit='erg/(s*cm^2*Angstrom)', disp='E13.5E2')
    col_err = fits.Column(name='ERR_FLUX', array=np.array([spec_dict['ERR_FLUX']]),
                          format=pre+'E', unit='erg/(s*cm^2*Angstrom)', disp='E13.5E2')
    col_qual = fits.Column(name='QUAL', array=np.array([spec_dict['QUAL']]),
                           format=pre+'J', null=-1, disp='I1')
    tab = fits.BinTableHDU.from_columns([col_wl, col_flux, col_err, col_qual])
    tab.name = 'PHASE3SPECTRUM'
    hdu.append(tab)
    hdu.writeto(filename, overwrite=True, output_verify='silentfix')


def run_ETC_target(target, output_dir, template_path='output/quasar_models'):
    """
    Run the ETC for a given target. The function generates the separate spectra per arm
    and the joined spectrum. The simulations add random cosmic rays.

    target : `:class:Target`
        Target information

    output_dir : str
        Directory to put the spectra

    template_path : str
        Directory to the spectral templates
    """

    # The target spectrum template.
    template_filename = os.path.join(template_path, target.template)
    spectrum = Table.read(template_filename, format='fits')

    temp_wl = spectrum['LAMBDA']
    temp_flux = spectrum['FLUX_DENSITY']
    temp_flux = normalize_flux(temp_wl, temp_flux, target.mag)

    # Create Mock Observation
    transmission = get_transmission(airmass=target.airmass)
    sky_model = get_sky_model(target.airmass, target.moon)
    throughput = get_efficiency(target.airmass)
    output_filename = os.path.join(output_dir, f'{target.name}_L1.fits')
    spec_dict = apply_noise(temp_wl, temp_flux, sky_model, throughput, transmission,
                            t_exp=target.exptime)
    # save to FITS file
    save_mock_spectrum(spec_dict, target, filename=output_filename)


def process_catalog(catalog, band='DECam.r', mag_min=18., mag_max=20.5, template_path='',
                    exptime=2400, moon='dark', airmass='mid', output='l1_data'):
    catalog['TEMPLATE'] = ['%s.fits' % f for f in catalog['ID']]
    catalog['EXPTIME'] = exptime
    catalog['MOON'] = moon
    catalog['SEEING'] = 0.8
    catalog['AIRMASS'] = airmass
    catalog['SPECTRO'] = 'LRS'
    catalog['MAG_TYPE'] = band
    catalog['MAG'] = np.random.uniform(mag_min, mag_max, len(catalog))

    if not os.path.exists(output):
        os.makedirs(output)

    warnings.simplefilter('ignore', u.UnitsWarning)
    warnings.simplefilter('ignore', fits.card.VerifyWarning)
    print("Applying 4MOST ETC to the catalog:")
    for num, row in enumerate(tqdm(catalog), 1):
        target = Target(row)
        run_ETC_target(target, output, template_path=template_path)
    catalog.write(os.path.join(output, 'observations.csv'), overwrite=True)


if __name__ == '__main__':
    parser = ArgumentParser(description="Generate simulated spectra from input catalog")
    parser.add_argument("input", type=str,
                        help="input catalog file, mandatory columns: TEMPLATE/ID, REDSHIFT, MAG, MAG_TYPE")
    parser.add_argument('--airmass', type=float, default='mid')
    parser.add_argument('--exptime', type=float, default=2400)
    parser.add_argument('--moon', type=str, default='dark')
    parser.add_argument('--path', type=str, default='output/quasar_models')
    parser.add_argument("-o", "--output", type=str, default='output/l1_data/',
                        help="output directory [default=./output/l1_data]")

    args = parser.parse_args()

    catalog = Table.read(args.input)
    process_catalog(catalog, exptime=args.exptime, moon=args.moon,
                    spectro=args.spectro, output=args.output,
                    template_path=args.path)
