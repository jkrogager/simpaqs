import numpy as np
import matplotlib.pyplot as plt
import os

from astropy.io import fits
from astropy.table import Table
from astropy import units as u

from VoigtFit.utils import molecules
from VoigtFit.utils.molecules import population_of_level
from VoigtFit.container.lines import lineList
from VoigtFit.funcs.voigt import Voigt
from VoigtFit import show_transitions


CI_MODEL_FILE = 'popratio_CI_3Dgrid.dat'

def get_CI_excitation(T=100, density=100):
    """
    Calculate the fine-structure excitation ratios of CI J=1/J=0
    and J=2/J=0 assuming a given temperature in K and density in cm^-3.

    Parameters
    ----------
    T : float
        The gas temperature in Kelvin used to set the relative strength
        of fine-structure levels. Must be between 10 and 1000 K.

    density : float
        The gas density in particles per cubic centimeter (cm^-3).
        Must be between 1 and 1000 cm^-3.

    Returns
    -------
    R1 : float
        The ratio of J=1 to ground-level excitation

    R2 : float
        The ratio of J=2 to ground-level excitation

    Raises
    ------
    ValueError
        If the given value of temperature or density is outside the popratio
        grid of models.
    """
    # Load the pre-calculated CI grid of models
    # The numbers are written in the first line of the output:
    with open(CI_MODEL_FILE) as model_file:
        first_line = model_file.readline()

    first_line = first_line[1:]
    dims = first_line.strip().split()
    n1, n2, n3 = [int(num) for num in dims]

    x1, x2, x3, R1, R2 = np.loadtxt(CI_MODEL_FILE, unpack=True)
    model_T = x1[::n2*n3]
    model_n = x2[:n2*n3:n3]
    urad = x3[:n3]

    R1 = R1.reshape((n1, n2, n3))
    R2 = R2.reshape((n1, n2, n3))

    if not model_T.min() < T < model_T.max():
        msg = f"Invalid input: {T}"
        msg += f"Input temperature must be between {model_T.min()} and {model_T.max()} K"
        raise ValueError(msg)

    if not model_n.min() < density < model_n.max():
        msg = f"Invalid input: {density}"
        msg += f"Input density must be between {model_n.min()} and {model_n.max()} cm^-3"
        raise ValueError(msg)

    idx_T = np.argmin(np.abs(model_T - T))
    idx_n = np.argmin(np.abs(model_n - density))
    idx_u = 0
    return R1[idx_T, idx_n, idx_u], R2[idx_T, idx_n, idx_u]


def make_CI_template(logN=13, b_turb=2, T=100, density=100):
    """
    Make a CI template with fine-structure levels populated based
    on temperature (in K) and density (in cm^-3).
    
    Parameters
    ----------
    logN : float
        The total CI column density assumed for the optical depth calculation.

    b_turb : float
        The turbulent broadening in km/s. Added in quadrature to the thermal
        component based on the input temperature `T`.

    T : float
        The gas temperature in Kelvin used to set the relative strength
        of fine-structure levels. Must be between 10 and 1000 K.

    density : float
        The gas density in particles per cubic centimeter (cm^-3).
        Must be between 1 and 1000 cm^-3.

    Returns
    -------
    tau : np.ndarray
        Array of the optical depth of H2 transitions

    Raises
    ------
    ValueError
        If the given value of temperature or density is outside the popratio
        grid of models.
    """
    R1, R2 = get_CI_excitation(T, density)
    N_rel = np.array([1, R1, R2])
    N_rel = N_rel / np.sum(N_rel)

    b = np.sqrt(b_turb**2 + 0.0166287*T/2.016)

    wl = np.arange(940, 1660, 0.05)
    tau = np.zeros_like(wl)
    for num, ion in enumerate(['CI', 'CIa', 'CIb']):
        transitions = show_transitions(ion)
        for trans in transitions:
            N_j = N_rel[num] * 10**logN
            tau += Voigt(wl, trans['l0'], trans['f'], N_j, b*1.e5, trans['gam'], z=0)

    return wl, tau


def make_H2_template(T=100, logN=20, b_turb=2):
    """
    Make a generic optical depth template of H2 Lyman-Werner transitions
    assuming a fixed excitation temperature (T) and total column density
    of 10^20 cm^-2. The broadening parameter is calculated from the temperature
    combined in quadrature with an assumed turbulent broadening of 2 km/s.
    The optical depth is calculated over 912 to 1200 Angstrom in rest-frame.

    Parameters
    ----------
    T : float
        H2 excitation temperature used to set the relative strength
        of rotational levels.

    logN : float
        The total H2 column density assumed for the optical depth calculation.

    b_turb : float
        The turbulent broadening in km/s

    Returns
    -------
    tau : np.ndarray
        Array of the optical depth of H2 transitions
    """
    # Calculate J-level populations
    N_rel = np.array([population_of_level('H2', T, j) for j in range(8)])
    N_rel /= np.sum(N_rel)

    b = np.sqrt(b_turb**2 + 0.0166287*T/2.016)

    wl = np.arange(912, 1200, 0.1)
    tau = np.zeros_like(wl)
    for band, lines in molecules.H2.items():
        transitions = sum(lines, [])
        for trans in transitions:
            idx = lineList['trans'].tolist().index(trans)
            props = lineList[idx]
            J = int(props['ion'][-1])
            N_j = N_rel[J] * 10**logN
            tau += Voigt(wl, props['l0'], props['f'], N_j, b*1.e5, props['gam'], z=0)

    return wl, tau


def generate_set_of_templates(Tmin=50, Tmax=300, nmin=50, nmax=500, N=6):
    if not os.path.exists('molecules'):
        os.mkdir('molecules')

    T_range = np.linspace(Tmin, Tmax, N)
    logN_ref = 20
    for T in T_range:
        wl, tau = make_H2_template(T=T, logN=logN_ref)
        tab = Table({'WAVE': wl, 'TAU': tau})
        tab.meta['TEMP'] = (T, "Excitation temperature of H2")
        tab.meta['LOG_NH2'] = logN_ref
        tab.write(f'molecules/H2_template_T{T:.0f}.fits', overwrite=True)

    n_range = np.linspace(nmin, nmax, N)
    logN_ref = 14
    for T, n in zip(T_range, n_range):
        wl, tau = make_CI_template(T=T, density=n, logN=logN_ref)
        tab = Table({'WAVE': wl, 'TAU': tau})
        tab.meta['TEMP'] = (T, "Gas temperature in Kelvin")
        tab.meta['DENSITY'] = (n, "Gas density in cm^-3")
        tab.meta['LOG_NCI'] = logN_ref
        tab.write(f'molecules/CI_template_T{T:.0f}_n{n:.0f}.fits', overwrite=True)


if __name__ == '__main__':
    generate_set_of_templates()

