"""
This script generates a sample of randomized, noiseless, normalized quasar absorption spectra
at the resolution of 4MOST LRS.

The absorption systems are drawn from a power-law Lya-alpha column density distribution function
normalized to the redshift evolution of the absorber incidence with a random redshift assigned.

For any absorber with log(NHI) > 18 we add metal absorption with a random metallicity, random
number of absorption components, a random velocity width and a random depletion strength.
We use the following distributions for the absorber parameters:
    For log(NHI) > 20.:
        [M/H] = Gaussian(-1.5, 0.5)
        N_comps = Random Int(2, 10)
        dV_90 = Uniform(100, 500)
        delta = Uniform(0., 2.0)
        H2 absorption is included in 10% of systems

    For 18 < log(NHI) <= 20.:
        [M/H] = Gaussian(-1.8, 0.3)
        N_comps = Random Int(1, 3)
        dV_90 = 50
        delta = 0.1

For each component of the absorption line, a random relative velocity is drawn within the full
velocity width given by dV_90. Each component then has a random broadening parameter assigned,
for low-ionization states, this is taken from randomly between `b_min` and `b_max` (5 and 15 km/s)
by default. For high ionization lines, CIV and SiIV, we use a randomized scaling factor which
is on average a factor of 10 higher than the low ions. Moreover, the high-ions are distributed
over a larger dV_90 range (by a factor of 1 to 3).

Dust depletion is taken into account using the depletion sequences by Konstantopoulou et al. 2022,
following the methodology by De Cia et al. 2016. We do not model alpha-element enhancement.

The templates are convolved by the LSF of the low-resolution 4MOST spectrograph and resampled
onto the final joined spectral format and saved as a template FITS table with two columns
LAMBDA and FLUX_DENSITY

"""

__author__ = 'Jens-Kristian Krogager'
__email__ = 'jens-kristian.krogager@univ-lyon1.fr'

from astropy.io import fits
from astropy.table import Table
import datetime

import numpy as np

import VoigtFit
from VoigtFit.funcs.voigt import Voigt, convolve
from VoigtFit import show_transitions
from VoigtFit.utils.Asplund import solar
from VoigtFit.utils import depletion
from VoigtFit.container.regions import load_lsf
from tqdm import tqdm

import glob
import sys
import os

import lya


depletion_sequence = depletion.coeffs


def add_metals(z_sys, logNHI, Z, delta, dV_90, N_comps, wl, logN_weight=100, b_min=5., b_max=15., f_lim=1.e-2):
    """
    Create a synthetic metal profile for singly ionized species (OI in case of oxygen).

    Parameters
    ----------
    z_sys : float
        The systemic absorption redshift

    logNHI : float
        The neutral hydrogen column density, log(NHI / cm^-2)

    Z : float
        The total metallicity (dust + gas phase)

    delta : float
        The depletion as parametrized by [Zn/Fe] following De Cia et al. 2016

    dV_90 : float
        The Velocity width of the profile. Random components are drawn symmetrically
        around the `z_sys` with the extremas located at -dV_90/2 and +dV_90/2.

    N_comps : int
        Number of components per absorption line

    wl : np.array
        The wavelength array on which to evaluate the absorption profile

    logN_weight : float [default = 100]
        The weighting of individual components. The total column density is distributed
        randomly among components based on the weight drawn from the interval [1, logN_weight].
        The column density scales are then normalized to unity and multiplied by the
        total column for each species.

    b_min : float [default = 5]
        A random b-parameter in units of km/s is drawn from the interval [b_min, b_max].

    b_max : float [default = 15]
        A random b-parameter in units of km/s is drawn from the interval [b_min, b_max].

    f_lim : float [default = 1.e-2]
        The minimum limit of oscillator strength to be considered. Only lines stronger
        than this limit will be included in the calculations

    Returns
    -------
    transmission : np.array
        The calculated transmission spectrum evaluated on the input `wl` grid

    parameters : dict
        Dictionary of velocity structure parameters and metal column densities
        for each metal species.

    log : list
        List of messages containing the parameters of the random realization

    linelist : list
        List of all lines included: their line-tag (ex: FeII_2600) and their observed wavelength
    """
    tau = np.zeros_like(wl)
    log = []
    linelist = []
    parameters = {}

    # Make velocity structure:
    N_scale = np.random.uniform(1, logN_weight, N_comps)
    N_scale /= N_scale.sum()
    b = np.random.uniform(b_min, b_max, N_comps)
    if N_comps == 1:
        z = np.array([z_sys])
        v_offset = np.array([0])
    else:
        dv = np.random.uniform(-1., 1., N_comps)
        v_offset = dv * dV_90/(dv.max() - dv.min())
        z = z_sys + v_offset/299792*(z_sys+1)
    b_str = ', '.join(["%.2f" % b_i for b_i in b])
    z_str = ', '.join(["%.6f" % z_i for z_i in z])
    v_str = ', '.join(["%.1f" % v_i for v_i in v_offset])
    log.append("b (km/s): " + b_str)
    log.append("v_rel (km/s): " + v_str)
    log.append("z: " + z_str)
    log.append(" --- Metal Columns II --- ")
    for X, (A2, B2) in depletion_sequence.items():
        if X not in solar:
            continue
        X_sun, _ = solar[X]
        logN_X = Z + logNHI + (X_sun - 12) + A2 + B2*delta
        N = 10**logN_X * N_scale
        if X == 'O':
            ion = f'{X}I'
        else:
            ion = f'{X}II'
        transitions = show_transitions(ion, lower=1000.)
        logN_str = ', '.join(["%.2f" % np.log10(N_i) for N_i in N])
        log.append(f"{ion}: " + logN_str)
        parameters[ion] = np.log10(N)
        for trans in transitions:
            for z_i, b_i, N_i in zip(z, b, N):
                if np.log10(N_i) < 10.:
                    continue
                tau += Voigt(wl, trans['l0'], trans['f'], N_i, b_i*1.e5, trans['gam'], z=z_i)
                linelist.append([trans['trans'], trans['l0']*(z_i+1)])

    # Add high-ions (CIV and SiIV):
    N_scale_IV = np.random.uniform(1, logN_weight, N_comps)
    N_scale_IV /= N_scale_IV.sum()
    b_IV = b.copy() * 10**(np.random.normal(1., 0.05))
    if N_comps == 1:
        z_IV = np.array([z_sys])
        v_offset_IV = np.random.normal(0., 75., size=(1,))
    else:
        dv = np.random.uniform(-1., 1., N_comps)
        v_stretch = np.random.uniform(1., 3., N_comps)
        v_offset_IV = dv * v_stretch * dV_90/(dv.max() - dv.min())
        z_IV = z_sys + v_offset_IV/299792*(z_sys+1)
    b_str = ', '.join(["%.2f" % b_i for b_i in b_IV])
    z_str = ', '.join(["%.6f" % z_i for z_i in z_IV])
    v_str = ', '.join(["%.1f" % v_i for v_i in v_offset_IV])
    log.append("b (km/s): " + b_str)
    log.append("v_rel (km/s): " + v_str)
    log.append("z: " + z_str)
    log.append(" --- Metal Columns IV --- ")
    for X in ['C', 'Si']:
        if X == 'C':
            logN_X = 1.2*Z + 15.8
        elif X == 'Si':
            logN_X = 1.2*Z + 15.3
        N_IV = 10**logN_X * N_scale_IV
        ion = f'{X}IV'
        transitions = show_transitions(ion, lower=1000., flim=f_lim)
        logN_str = ', '.join(["%.2f" % np.log10(N_i) for N_i in N_IV])
        log.append(f"{ion}: " + logN_str)
        parameters[ion] = np.log10(N)
        for trans in transitions:
            for z_i, b_i, N_i in zip(z_IV, b_IV, N_IV):
                if np.log10(N_i) < 10.:
                    continue
                tau += Voigt(wl, trans['l0'], trans['f'], N_i, b_i*1.e5, trans['gam'], z=z_i)
                linelist.append([trans['trans'], trans['l0']*(z_i+1)])

    transmission = np.exp(-tau)

    parameters['b'] = b
    parameters['z'] = z
    parameters['vel'] = v_offset
    parameters['b_IV'] = b_IV
    parameters['z_IV'] = z_IV
    parameters['vel_IV'] = v_offset_IV

    return transmission, parameters, log, linelist


def add_H2(z, wl, logN):
    H2_TEMPLATES = glob.glob('molecules/H2_template*.fits')
    temp_fname = np.random.choice(H2_TEMPLATES)
    H2 = Table.read(temp_fname)
    T = H2.meta['TEMP']
    logN_ref = H2.meta['LOG_NH2']
    tau = np.interp(wl, H2['WAVE']*(1+z), H2['TAU'], left=0., right=0.)
    tau *= 10**(logN-logN_ref)
    transmission = np.exp(-tau)
    return transmission, T


def add_CI(z, wl, logN, T=None):
    if T:
        CI_templates = glob.glob(f'molecules/CI_template_T{T:.0f}_*.fits')
        if len(CI_templates) == 0:
            CI_templates = glob.glob('molecules/CI_template*.fits')
            print(f"No template found matching the given temperature {T:.0f}. Try `T=None`")
    else:
        CI_templates = glob.glob('molecules/CI_template*.fits')
    temp_fname = np.random.choice(CI_templates)
    CI = Table.read(temp_fname)
    T = CI.meta['TEMP']
    n = CI.meta['DENSITY']
    logN_ref = CI.meta['LOG_NCI']
    tau = np.interp(wl, CI['WAVE']*(1+z), CI['TAU'], left=0., right=0.)
    tau *= 10**(logN-logN_ref)
    transmission = np.exp(-tau)
    return transmission, T, n


def make_absorber(z_qso, filenum=1, output_dir='output/abs_templates'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    wl = np.arange(3600, 9600, 0.1)
    wl_qmost = np.arange(2990, 11000, 0.25)
    kernel = load_lsf('resolution/4MOST_LR_kernel.txt', wl)

    # Draw random samle of absorbers:
    # The calculation is split into subsets in redshift space
    # due to the limitations of the redshift distribution approximation
    # used in `lya.py`
    z_edges = np.linspace(0, z_qso, 5)
    absorbers = []
    P_list = []
    for z1, z2 in zip(z_edges[:-1], z_edges[1:]):
        p_i, abs_i = lya.lya_transmission_noconv(z1, z2, wl)
        P_list.append(p_i)
        absorbers += abs_i
    P_lya = np.prod(P_list, axis=0)

    metal_profiles = []
    H2_profiles = []
    H2_profiles.append(np.ones_like(wl))
    meta_data = []
    logNH2 = 0
    logNCI = 0
    T_01 = 0
    n_H = 0
    for z, logNHI in absorbers:
        if logNHI > 20.:
            Z = np.random.normal(-1.5, 0.5)
            N_comps = np.random.randint(3, 10)
            dV = np.random.uniform(200, 500)
            delta = 0.73*Z + 1.26 + np.random.normal(0., 0.2)
            H2_random_number = np.random.uniform(0, 100)
            if z > 2.4 and H2_random_number < 20:
                logNH2 = np.random.uniform(19, 21)
                logNCI = np.random.uniform(13, 15)
                this_H2_profile, T_01 = add_H2(z, wl, logNH2)
                this_CI_profile, T_CI, n_H = add_CI(z, wl, logNCI, T=T_01)
                H2_profiles.append(this_H2_profile * this_CI_profile)
        else:
            Z = np.random.normal(-1.8, 0.3)
            N_comps = np.random.randint(1, 3)
            dV = 50.
            delta = 0.1
        this_profile, pars, log, _ = add_metals(z, logNHI, Z, delta, dV, N_comps, wl)
        metal_profiles.append(this_profile)
        meta_data.append({'pars': pars,
                          'log': log,
                          'z_sys': z,
                          'logNHI': logNHI,
                          'Z_tot': Z,
                          'dV90': dV,
                          'delta': delta,
                          'logNH2': logNH2,
                          'logNCI': logNCI,
                          'T_01': T_01,
                          'n_H': n_H})
    P_metals = np.prod(metal_profiles, axis=0)
    P_H2 = np.prod(H2_profiles, axis=0)

    transmission = P_lya * P_metals * P_H2
    profile_conv = convolve(transmission, kernel)
    profile_obs = np.interp(wl_qmost, wl, profile_conv)

    # Format output FITS table:
    hdu = fits.HDUList()
    hdr = fits.Header()
    hdr['AUTHOR'] = __author__
    hdr['COMMENT'] = 'Absorption Line Template'
    hdr['VERSION'] = 1.0
    hdr['Z_QSO'] = z_qso
    prim = fits.PrimaryHDU(header=hdr)

    col_wl = fits.Column(name='LAMBDA', unit='Angstrom', format='1D',
                         array=wl_qmost)
    col_flux = fits.Column(name='FLUX_DENSITY', unit='erg/(s cm**2 Angstrom)', format='1E',
                           array=profile_obs)
    tab = fits.BinTableHDU.from_columns([col_wl, col_flux], header=hdr)
    tab.name = 'TEMPLATE'
    hdu.append(tab)

    for num, item in enumerate(meta_data):
        h = fits.Header()
        h['REDSHIFT'] = item['z_sys']
        h['LOG_NHI'] = item['logNHI']
        h['LOG_NH2'] = item['logNH2']
        h['LOG_NCI'] = item['logNCI']
        h['TEMP'] = item['T_01']
        h['DENSITY'] = item['n_H']
        h['Z_TOT'] = item['Z_tot']
        h['DV_90'] = item['dV90']
        h['DELTA'] = item['delta']
        h['N_COMP'] = len(item['pars']['z'])
        for line in item['log']:
            h['COMMENT'] = line
        pars_tab = fits.BinTableHDU(Table(item['pars']), header=h)
        if item['logNHI'] > 20.3 and item['z_sys'] > 2.05:
            pars_tab.name = 'DLA'
        else:
            pars_tab.name = 'ABS'
        hdu.append(pars_tab)

    filename = f'{output_dir}/PAQS_absorber_{filenum:06}.fits'
    hdu.writeto(filename, overwrite=True)
    template_info = {
            'FILE': filename,
            'Z_QSO': z_qso,
            'N_ABS': len(meta_data),
            'N_DLA': len([item for item in meta_data if item['logNHI'] > 20.3 and item['z_sys'] > 2.05]),
            }
    abs_info = []
    # Keep track of meta data:
    for item in meta_data:
        abs_info.append({'FILE': filename,
                         'Z_ABS': item['z_sys'],
                         'LOG_NHI': item['logNHI'],
                         'LOG_NH2': item['logNH2'],
                         'LOG_NCI': item['logNCI'],
                         'TEMP': item['T_01'],
                         'DENSITY': item['n_H'],
                         'Z_TOT': item['Z_tot'],
                         'DELTA': item['delta'],
                         'DV_90': item['dV90'],
                         'Z_QSO': z_qso,
                         })
    return template_info, abs_info


def make_absorber_templates(N_total, z_min=1.0, z_max=4.0, output_dir='output/abs_templates', verbose=True):
    info_list = []
    abs_info_list = []
    quasar_redshifts = np.random.uniform(z_min, z_max, N_total)
    start = datetime.datetime.now()
    if verbose:
        print("Making absorber templates:")
        #sys.stdout.write("\r %i / %i" % (0, N_total))
    for num, z_qso in enumerate(tqdm(quasar_redshifts), 1):
        temp_info, abs_info = make_absorber(z_qso, filenum=num, output_dir=output_dir)
        info_list.append(temp_info)
        abs_info_list += abs_info
        #if verbose:
            #sys.stdout.write("\r %i / %i" % (num, N_total))
            #sys.stdout.flush()
    filelog = Table(info_list)
    filelog.write("output/list_templates.csv", format='csv', overwrite=True)
    abslog = Table(abs_info_list)
    abslog.write("output/list_absorbers.csv", format='csv', overwrite=True)
    all_dlas = (abslog['LOG_NHI'] > 20.3) & (abslog['Z_ABS'] > 2.05)
    DLAlog = abslog[all_dlas]
    DLAlog.write("output/list_dlas.csv", format='csv', overwrite=True)
    if verbose:
        print("")
        print("Wrote summary files:")
        print("Template list: output/list_templates.csv")
        print("Absorber list: output/list_absorbers.csv")
        print("DLA list: output/list_dlas.csv")
        print(f"Finished in {datetime.datetime.now() - start}")
    return filelog, abslog, DLAlog


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser('Make absorption templates')
    parser.add_argument("number", type=int,
                        help="Number of random absorption sightlines to generate")
    parser.add_argument("--z_min", type=float, default=1.,
                        help="Minimum redshift to simulate  [default=1]")
    parser.add_argument("--z_max", type=float, default=4.,
                        help="maximum redshift to simulate  [default=4]")
    parser.add_argument("-o", "--output", type=str, default='output/abs_templates',
                        help="Output directory [default=output/abs_templates]")

    args = parser.parse_args()
    templates, absorbers, DLAs = make_absorber_templates(args.number, args.z_min, args.z_max, args.output)


if __name__ == '__main__':
    main()

