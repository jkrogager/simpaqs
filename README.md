# PAQS Spectral Simulator


Code to generate a sample of random quasars with realistic absorption systems:
Lyman-alpha forest, Lyman-limit systems and DLAs, including their metal lines.
The metal lines only include singly ionized lines plus CIV and SiIV for now.


## Installation

Clone the code:
    
    git clone 
    cd simpaqs
    python -m pip install -r requirements.txt


Install `simqso`:

    git clone git@github.com:imcgreer/simqso.git
    cd simqso
    python setup.py install


Install 4MOST ETC:

    QMOST_PYPI=https://gitlab.4most.eu/api/v4/projects/212/packages/pypi/simple
    python -m pip install --extra-index-url $QMOST_PYPI qmostetc


## Run the full simulator

   python simpaqs.py  **N**

where `N` is the number of spectra to generate.
 

## Output
The code generates: 
 - absorption transmission profiles (`abs_templates/`)
 - noiseless quasar continuum models with absorption (`quasar_models/`)
 - simulated L1 4MOST spectra with noise and cosmics (individual arms and joined; `l1_data/`)

The code also saves a list of the absorption templates generated,
the absorption systems and their properties, as well as a list of DLAs.
The quasar continuum parameters are also saved in the `quasar_models` directory,
and lastly, the observational parameters are saved in the catalog file 'observations.csv'
in the `l1_data` directory.


## Methodology

See the individual documentation in the three simulator modules:
`simulate_absorbers.py`, `simulate_quasars.py` and `simulate_spectra.py`.

