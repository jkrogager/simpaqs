"""
"""

__author__ = 'Jens-Kristian Krogager'
__email__ = 'jens-kristian.krogager@univ-lyon1.fr'

from astropy.io import fits
from astropy.table import Table
import datetime
from tqdm import tqdm
import yaml
import numpy as np

import VoigtFit
from VoigtFit.funcs.voigt import Voigt, convolve
from VoigtFit import show_transitions
from VoigtFit.utils.Asplund import solar
from VoigtFit.utils import depletion
from VoigtFit.container.regions import load_lsf

import glob
import sys
import os

import lya


depletion_sequence = depletion.coeffs


def add_metals(z_sys, logNHI, Z, delta, dV_90, N_comps, wl, logN_weight=30, b_min=5., b_max=15.):
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

    logN_weight : float [default = 30]
        The weighting of individual components. The total column density is distributed
        randomly among components based on the weight drawn from the interval [1, logN_weight].
        The column density scales are then normalized to unity and multiplied by the
        total column for each species.

    b_min : float [default = 5]
        A random b-parameter in units of km/s is drawn from the interval [b_min, b_max].

    b_max : float [default = 15]
        A random b-parameter in units of km/s is drawn from the interval [b_min, b_max].

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
        comp_max = np.argmax(N_scale)
        dv = dv - dv[comp_max]
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
        transitions = show_transitions(ion, lower=912.)
        logN_str = ', '.join(["%.2f" % np.log10(N_i) for N_i in N])
        log.append(f"{ion}: " + logN_str)
        parameters[ion] = np.log10(N)
        for trans in transitions:
            for z_i, b_i, N_i in zip(z, b, N):
                tau += Voigt(wl, trans['l0'], trans['f'], N_i, b_i*1.e5, trans['gam'], z=z_i)
                linelist.append([trans['trans'], trans['l0']*(z_i+1)])

    # Add high-ions (CIV and SiIV):
    N_scale_IV = np.random.uniform(1, logN_weight, N_comps)
    N_scale_IV /= N_scale_IV.sum()
    b_IV = np.random.uniform(25, 45, N_comps)
    if N_comps == 1:
        z_IV = np.array([z_sys])
        v_offset_IV = np.random.normal(0., 25., size=(1,))
    else:
        dv = np.random.uniform(-1., 1., N_comps)
        dv = dv - dv[np.argmax(N_scale_IV)]
        v_stretch = np.random.uniform(1., 1.5, N_comps)
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
        transitions = show_transitions(ion, lower=1000.)
        logN_str = ', '.join(["%.2f" % np.log10(N_i) for N_i in N_IV])
        log.append(f"{ion}: " + logN_str)
        parameters[ion] = np.log10(N_IV)
        for trans in transitions:
            for z_i, b_i, N_i in zip(z_IV, b_IV, N_IV):
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


def make_absorber_from_pars(pars, output_dir='output/user_templates'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    wl = np.arange(2900, 9600, 0.25)
    wl_qmost = np.arange(3000, 9500, 0.25)
    kernel = load_lsf('resolution/4MOST_LRS_kernel.txt', wl)

    # Draw random samle of absorbers:
    # The calculation is split into subsets in redshift space
    # due to the limitations of the redshift distribution approximation
    # used in `lya.py`
    z_qso = pars['z_qso']
    z_edges = np.linspace(3000/1216.-1, z_qso, 5)
    P_list = []
    for z1, z2 in zip(z_edges[:-1], z_edges[1:]):
        HI_pars = [(sys['z_abs'], 15., 10**sys['logNHI'])
                   for sys in pars['absorbers']
                   if z1 < sys['z_abs'] < z2]
        if len(HI_pars) == 0:
            HI_pars = None
        p_i, abs_i = lya.lya_transmission_noconv(z1, z2, wl, absorbers=HI_pars, NHI_limit=1e18)
        P_list.append(p_i)
    P_lya = np.prod(P_list, axis=0)

    profiles = [np.ones_like(wl)]
    for system in pars['absorbers']:
        z_abs = system['z_abs']
        logNHI = system['logNHI']
        logNCI = system['logNCI']
        logNH2 = system['logNH2']
        Z = system['logZ']
        delta = system['delta']
        dV = system['dv90']
        N_comps = system['N']
        P_H2, T_01 = add_H2(z_abs, wl, logNH2)
        P_CI, T_CI, n_H = add_CI(z_abs, wl, logNCI, T=T_01)
        P_metals, _, _, _ = add_metals(z_abs, logNHI, Z, delta, dV, N_comps, wl)
        profiles.append(P_lya * P_metals * P_H2 * P_CI)
    transmission = np.prod(profiles, axis=0)

    profile_conv = convolve(transmission, kernel)
    profile_obs = np.interp(wl_qmost, wl, profile_conv)

    # Format output FITS table:
    hdu = fits.HDUList()
    hdr = fits.Header()
    hdr['AUTHOR'] = __author__
    hdr['COMMENT'] = 'Absorption Line Template'
    hdr['VERSION'] = 1.0
    prim = fits.PrimaryHDU(header=hdr)

    col_wl = fits.Column(name='LAMBDA', unit='Angstrom', format='1D',
                         array=wl_qmost)
    col_flux = fits.Column(name='FLUX_DENSITY', unit='erg/(s cm**2 Angstrom)', format='1E',
                           array=profile_obs)
    tab = fits.BinTableHDU.from_columns([col_wl, col_flux], header=hdr)
    tab.name = 'TEMPLATE'
    hdu.append(tab)

    filename = f"{output_dir}/{pars['name']}.fits"
    hdu.writeto(filename, overwrite=True)
    return filename



def main():
    from argparse import ArgumentParser
    parser = ArgumentParser('Make absorption templates')
    parser.add_argument("pars", type=str, nargs='+',
                        help="Parameter file with absorption systems to generate")
    parser.add_argument("-o", "--output", type=str, default='output/user_templates',
                        help="Output directory [default=output/user_templates]")

    args = parser.parse_args()

    for fname in args.pars:
        with open(fname) as pars_file:
            pars = yaml.full_load(pars_file)

        print(f"Making template: {pars['name']} at z_qso = {pars['z_qso']} with {len(pars['absorbers'])} systems")
        filename = make_absorber_from_pars(pars, output_dir=args.output)


if __name__ == '__main__':
    main()

