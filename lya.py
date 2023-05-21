"""
Generate Random Ly-alpha Forest Spectra

Includes column densities from 12 < logN < 22.
The approach is only approximate and only simulates
the first 10 Lyman-
"""

__author__ = 'Jens-Kristian Krogager'
__email__ = 'jens-kristian.krogager@univ-lyon1.fr'

import numpy as np
import powerlaw
from scipy.signal import fftconvolve, gaussian

HI_data = [(1215.6696, 4.16e-01, 6.265e+08, 1.008),
           (1025.7201, 7.91e-02, 1.897e+08, 1.008),
           (972.5345, 2.90e-02, 8.127e+07, 1.008),
           (949.7407, 1.39e-02, 4.204e+07, 1.008),
           (937.8011, 0.0078, 24500000., 1.008),
           (930.7459, 0.00481, 15510000., 1.008),
           (926.2233, 0.00318, 10430000., 1.008),
           (923.148, 0.00222, 7344000., 1.008),
           ]


def abs_length(zmin, zmax):
    z = np.linspace(zmin, zmax, 500)
    dX = (1+z)**2 / np.sqrt(0.3*(1+z)**3 + 0.7)
    return np.trapz(dX, z)


def get_absorbers(zmin, zmax, verbose=False):
    """
    Draw a random sample of absorbers from the NHI distribution function, f(NHI),
    parametrized as a single power-law following Kim et al. (2013), A&A 552, A77

    zmin : float
        Minimum redshift over which to sample f(NHI)

    zmax : float
        Maximum redshift over which to sample f(NHI)

    Returns
    -------
    abs_data : np.array(N, 3)
        Array containing three columns:
        [absorber redshift, b-parameter, column density]
        for N randomly drawn absorbers
    """
    z0 = 0.5*(zmin + zmax)
    # Approximate the distribution function as a single power-law:
    # Kim et al. A&A 552, A77 (2013)
    logN = np.linspace(13., 21.5, 1000)
    logf = -1.43*logN + 7.34

    # The total number expected is then:
    X = abs_length(zmin, zmax)
    dndz_scale = 10**(1.61*np.log10((z0+1)/(2.5+1)))
    N_avg = int(np.trapz(10**logf, 10**logN) * X * dndz_scale)
    N = np.random.poisson(N_avg)
    if verbose:
        print(f"Generating {N} random Lya forest lines")

    # The single power-law overshoots the number of high-N systems:
    # random_NHI = powerlaw.Power_Law(xmin=1.e13, parameters=[1.40]).generate_random(N)
    # Use broken power-law:
    N2 = int(N*0.00032)
    N1 = N - N2
    random_NHI1 = powerlaw.Power_Law(xmin=1.e13, xmax=2.e21, parameters=[1.40]).generate_random(N1)
    good = random_NHI1 < 2.e21
    if np.sum(good) != N1:
        N2 += N1-np.sum(good)
    random_NHI2 = powerlaw.Power_Law(xmin=2.e21, xmax=1.e22, parameters=[3.48]).generate_random(N2)
    random_NHI = np.concatenate([random_NHI1[good], random_NHI2])
    # Should technically be evenly distributed in X(z)
    # but if I use small intervals in z, this is negligible
    random_z = np.random.uniform(zmin, zmax, N)
    # Use log-normal distribution of b~30:
    random_b = 10**np.random.normal(1.5, 0.05, N)

    return np.column_stack([random_z, random_b, random_NHI])


def lya_transmission_noconv(zmin, zmax, wl):
    LLS = list()

    abs_data = get_absorbers(zmin, zmax)
    lmin = np.min(wl)
    lmax = np.max(wl)

    tau = np.zeros_like(wl)
    for z, b, NHI in abs_data:
        l_LL = 911.7641 * (z+1)
        if NHI > 1.e18:
            LLS.append((z, np.log10(NHI)))

        for l0, f, gam, _ in HI_data:
            tau += Voigt(wl, l0, f, NHI, b*1.e5, gam, z=z)
        if (NHI > 2.e17) & (l_LL > lmin):
            # add Lyman-Limit:
            tau[wl <= l_LL] += NHI/10**17.2 * (wl[wl <= l_LL]/l_LL)**-3
    tau[tau > 10.] = 10.
    profile = np.exp(-tau)
    return profile, LLS


def lya_transmission(zmin, zmax, wl_obs=None, R=5000., fwhm=1.0, loglam=True):
    """
    Calculate the Lya-forest transmission from the random
    realization of (z, b, NHI) generated by `get_absorbers`.

    zmin, zmax : float
        Lower and upper redshift range to simulate

    wl : np.array  [optional]
        Optional wavelength grid to use for the calculation of optical depth

    R : float
        Spectral resolving power  (R = dl/lambda), only used if the spectrum
        is sampled logarithmically, i.e., constant resolution.

    fwhm : float
        Spectral FWHM in Angstrom, only used if the spectrum is sampled linearly,
        i.e., wavelength dependent resolution (fixed fwhm in Å)

    loglam : boolean
        Sample the spectrum in logarithmic bins?

    Returns
    -------
    wl_obs : np.array
        simulated wavelength range

    profile_obs : np.array
        simulated Ly-alpha forest profile

    DLAs : list(float)
        List of column densities and redshifts of DLAs (logNHI > 20.3)
    """
    # HI Ly-alpha parameters:
    l0, f, gam = 1215.6696, 0.139, 6.265e+08
    DLAs = list()

    abs_data = get_absorbers(zmin, zmax)

    l_cen = ((zmin + zmax)/2. + 1) * l0
    lmin, lmax = l0*(zmin+1), l0*(zmax+1)
    if loglam:
        dl = l_cen / R / 3.
        N_pix = int((lmax - lmin) / dl)
        if wl_obs is None:
            wl = np.logspace(np.log10(lmin-50*dl), np.log10(lmax+50*dl), N_pix+100)
            wl_obs = np.linspace(lmin, lmax, N_pix)
        else:
            lmin = np.min(wl_obs)
            lmax = np.max(wl_obs)
            N_pix = len(wl_obs)
            dl = np.median(np.diff(wl_obs))
            wl = np.logspace(np.log10(lmin-50*dl), np.log10(lmax+50*dl), N_pix+100)
        pxs = np.diff(wl)[0] / wl[0] * 299792.458
        kernel = (299792.458/R) / pxs / 2.35482
        LSF = gaussian(10*int(kernel) + 1, kernel)
        LSF = LSF/LSF.sum()
    else:
        dl = fwhm / 5.
        N_pix = int((lmax - lmin) / dl)
        if wl_obs is None:
            wl = np.linspace(lmin-50*dl, lmax+50*dl, N_pix+100)
            wl_obs = np.linspace(lmin, lmax, N_pix)
        else:
            lmin = np.min(wl_obs)
            lmax = np.max(wl_obs)
            dl = np.median(np.diff(wl_obs))
            N_pix = len(wl_obs)
            wl = np.linspace(lmin-50*dl, lmax+50*dl, N_pix+100)
        LSF = gaussian(5*int(fwhm) + 1, fwhm/2.35482)
        LSF = LSF/LSF.sum()

    tau = np.zeros_like(wl)
    for z, b, NHI in abs_data:
        l_LL = 911.7641 * (z+1)
        if NHI > 1.e20:
            DLAs.append((z, np.log10(NHI)))

        for l0, f, gam, _ in HI_data:
            tau += Voigt(wl, l0, f, NHI, b*1.e5, gam, z=z)
        if (NHI > 2.e17) & (l_LL > lmin):
            # add Lyman-Limit:
            tau[wl <= l_LL] += NHI/10**17.2 * (wl[wl <= l_LL]/l_LL)**-3
    tau[tau > 10.] = 10.
    profile = np.exp(-tau)

    profile_broad = fftconvolve(profile, LSF, 'same')
    profile_obs = np.interp(wl_obs, wl, profile_broad)

    return wl_obs, profile_obs, DLAs


def H(a, x):
    """Voigt Profile Approximation from T. Tepper-Garcia (2006, 2007)."""
    P = x**2
    H0 = np.exp(-x**2)
    Q = 1.5/x**2
    return H0 - a/np.sqrt(np.pi)/P * (H0*H0*(4.*P*P + 7.*P + 4. + Q) - Q - 1)


def Voigt(wl, l0, f, N, b, gam, z=0):
    """
    Calculate the optical depth Voigt profile.

    wl : array_like, shape (N)
        Wavelength grid in Angstroms at which to evaluate the optical depth.
    l0 : float
        Rest frame transition wavelength in Angstroms.
    f : float
        Oscillator strength.
    N : float
        Column density in units of cm^-2.
    b : float
        Velocity width of the Voigt profile in cm/s.
    gam : float
        Radiation damping constant, or Einstein constant (A_ul)
    z : float
        The redshift of the observed wavelength grid `l`.
    """
    # ==== PARAMETERS ==================

    c = 2.99792e10        # cm/s
    m_e = 9.1094e-28      # g
    e = 4.8032e-10        # cgs units

    C_a = np.sqrt(np.pi)*e**2*f*l0*1.e-8/m_e/c/b
    a = l0*1.e-8*gam/(4.*np.pi*b)

    wl = wl/(z+1.)
    x = (c / b) * (1. - l0/wl)

    tau = np.float64(C_a) * N * H(a, x)
    return tau

