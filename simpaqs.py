
import sys
import numpy as np
from astropy import table

from simulate_absorbers import make_absorber_templates
from simulate_quasars import add_quasar_continuum
from simulate_spectra import process_catalog

__author__ = 'Jens-Kristian Krogager'
__email__ = 'jens-kristian.krogager@univ-lyon1.fr'


# -- Input parameters
simID = 'test_DLA2'
Z_MIN = 2.2
Z_MAX = 4.0
EXPTIME = 1200  # seconds
MOON = 'dark'
MAG_MIN = 17
MAG_MAX = 20
OUTPUT_DIR = f'output/l1_data_{simID}'
ABS_MODELS_DIR = f'output/abs_{simID}'
QSO_MODELS_DIR = f'output/quasars_{simID}'
BAL = True
FULL_FOREST = False
FORCE_DLA = True
NHI_MAX = 1e20
##########################

# Set NHI_MAX to 1e19 or so, to reject absorption systems with column density higher
# than this limit. This is useful to generate a sample without any DLAs.

# To generate a sample of spectra where *all* quasars have a DLA, use: FORCE_DLA = True
# Note that `FORCE_DLA` overrides the use of NHI_MAX

# Setting FULL_FOREST = True will generate lyman-alpha absorbers and metal absorption
# for systems below the detectable limit of the blue cutoff (lambda < 3700 Å).

N_TOTAL = int(sys.argv[1])
# np.random.seed(20230521)

abs_template_list, abslog, DLAlog = make_absorber_templates(N_TOTAL,
                                                            z_min=Z_MIN,
                                                            z_max=Z_MAX,
                                                            verbose=True,
                                                            force_DLA=FORCE_DLA,
                                                            NHI_max=NHI_MAX,
                                                            full_forest=FULL_FOREST,
                                                            output_dir=ABS_MODELS_DIR)

# abs_template_list = table.Table.read("test/abs/list_templates.csv") # For midway inspection

model_input = add_quasar_continuum(abs_template_list, BAL=BAL, output_dir=QSO_MODELS_DIR,
                                   # dust_mode='flat',
                                   )

# model_input = table.Table.read('output/quasar_models/model_input.csv') # For midway inspection

process_catalog(model_input, mag_min=MAG_MIN, mag_max=MAG_MAX, template_path=QSO_MODELS_DIR,
                exptime=EXPTIME, moon=MOON, output=OUTPUT_DIR)

# Save parameters:

variables = {'SimID': simID,
             'Z_MIN': Z_MIN, 'Z_MAX': Z_MAX,
             'BAL': BAL, 'FULL_FOREST': FULL_FOREST, 
             'FORCE_DLA': FORCE_DLA, 
             'NHI_MAX': NHI_MAX,
             'EXPTIME': EXPTIME, 'MOON': MOON,
             'MAG_MIN': MAG_MIN, 'MAG_MAX': MAG_MAX,
             'OUTPUT_DIR': OUTPUT_DIR,
             'ABS_MODELS_DIR': ABS_MODELS_DIR,
             'QSO_MODELS_DIR': QSO_MODELS_DIR,
}

with open(f"{OUTPUT_DIR}/config_{simID}.txt", "w") as cfg:
    for parname, par in variables.items():
        cfg.write(f"{parname}: {par}\n")

