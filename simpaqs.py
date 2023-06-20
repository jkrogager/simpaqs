
import sys

from simulate_absorbers import make_absorber_templates
from simulate_quasars import add_quasar_continuum
from simulate_spectra import process_catalog
import numpy as np
from astropy import table

__author__ = 'Jens-Kristian Krogager'
__email__ = 'jens-kristian.krogager@univ-lyon1.fr'


# -- Input parameters
Z_MIN = 2
Z_MAX = 4
EXPTIME = 3600  # seconds
MOON = 'dark'
MAG_MIN = 18
MAG_MAX = 20.5
OUTPUT_DIR = 'output/l1_data'
##########################

N_TOTAL = int(sys.argv[1])
np.random.seed(20230521)

abs_template_list, abslog, DLAlog = make_absorber_templates(N_TOTAL, z_min=Z_MIN, z_max=Z_MAX, verbose=True)

#abs_template_list = table.Table.read("output/list_templates.csv") # For midway inspection

model_input = add_quasar_continuum(abs_template_list)

#model_input = table.Table.read('output/quasar_models/model_input.csv') # For midway inspection

process_catalog(model_input, mag_min=MAG_MIN, mag_max=MAG_MAX,
                exptime=EXPTIME, moon=MOON, output=OUTPUT_DIR)

