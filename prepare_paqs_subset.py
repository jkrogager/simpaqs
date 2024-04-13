"""
Take the PAQS target catalog and prepare a small subset of targets
for spectral simulations using SIMPAQS with realistic exposure times
based on target magnitudes and spectral success criteria.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import os
import glob


catalog_path = '/Users/krogager/Projects/4GPAQS/targets/paqs_catalog/output/12APR2024'
catalog_fname = glob.glob(os.path.join(catalog_path, 'PAQS_catalog*.fits.gz'))[0]

models = Table.read('output/quasar_models/model_input.csv')
temp_fname = [f'{name}.fits' for name in models['ID']]

catalog = Table.read(catalog_fname)

N = len(models)
idx = np.random.choice(np.arange(len(catalog)), N, replace=False)
subset = catalog[idx]
subset['TEMPLATE_REDSHIFT'] = models['REDSHIFT']
subset['TEMPLATE'] = temp_fname

subset.write('PAQS_simulated_catalog.fits', overwrite=True)


