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


def make_H2_template(T=100, b_turb=2):
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
            N_j = N_rel[J] * 1e20
            if np.log10(N_j) < 10.:
                continue
            tau += Voigt(wl, props['l0'], props['f'], N_j, b*1.e5, props['gam'], z=0)

    return wl, tau


def generate_set_of_templates(Tmin=50, Tmax=300, N=6):
    if not os.path.exists('molecules'):
        os.mkdir('molecules')

    T_range = np.linspace(Tmin, Tmax, N)
    for T in T_range:
        wl, tau = make_H2_template(T=T)
        tab = Table({'WAVE': wl, 'TAU': tau})
        tab.meta['T_01'] = T
        tab.write(f'molecules/H2_template_T{T:.0f}.fits', overwrite=True)


if __name__ == '__main__':
    generate_set_of_templates()

