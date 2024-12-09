# PAQS Spectral Simulator


Code to generate a sample of random quasars with realistic absorption systems:
Lyman-alpha forest, Lyman-limit systems and DLAs, including their metal lines
and molecular lines in 10% of the DLAs.
The metal lines only include singly ionized lines plus CIV and SiIV for now.

The code also allows a simulation and exposure time estimation of a 4MOST target catalog.
For this see the section about *ETC simulator* below.


## Installation

Clone the code:
    
    git clone 
    cd simpaqs
    python -m pip install -r requirements.txt


Install 4MOST ETC:

    QMOST_PYPI=https://gitlab.4most.eu/api/v4/projects/212/packages/pypi/simple
    python -m pip install --extra-index-url $QMOST_PYPI qmostetc


Install `simqso` (not needed if only running `simulate_catalog.py`):

    git clone git@github.com:imcgreer/simqso.git
    cd simqso
    python setup.py install


## Run the ETC and L1 simulation

The script `simulate_catalog.py` will take a 4MOST target catalog with an associated
set of spectral templates as well as rules and rulesets to generate a list of exposure
times per target as well as mock L1 spectra (joined LRS spectra by default). This is run as follows:

    python simulate_catalog.py  catalog_name.fits  --temp-dir templates/ --rules rules.csv --ruleset ruleset.csv --output l1_data

The output will be placed in the folder `output/l1_data` by default. The default conditions are as follows:

 - seeing : 0.8 arcsec
 - airmass : 1.2
 - moon phase : dark

These conditions can be changed using the command line options: `--seeing`, `--airmass`, `--moon`.
By default the script generates simulated joined L1 spectra with realistic noise and cosmic ray hits, if the targets are LRS targets. If you also want individual arms for each target, use the option `--arm ALL`. This will also generate spectra in case of HRS targets that do not have a joined counterpart.

The catalog, templates, rules and rulesets must follow the 4FS file formats!

### Output

The output directory is given in the command line `--output` and will contain the simulated L1 spectra (see the L1 pipeline documentationand DXU for format definitions). This folder will also contain a file `exposure_times.csv` which lists the target name, magnitude, estimated exposure time and redshift.



## Run the full simulator

A full run of the simulator can be done using the wrapper script:

    python simpaqs.py  N


where `N` is the number of spectra to generate.
Alternatively, the individual steps can be customized further using the dedicated scripts. 


### Output
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
[simulate absorbers](simulate_absorbers.py), [simulate quasars](simulate_quasars.py),
and [simulate spectra](simulate_spectra.py).


## Create a mock target catalog and simulate it

Using the `prepare_paqs_subset.py` script, you can pick a random subset of the full target catalog. Note that the file path of the catalog is hard-coded in the script. The script assigns a template filename to each entry in the catalog.
Next you can run the catalog through the simulator which takes our rules and ruleset definitions into account in order to calculate the exposure times for each target. Below I use the BAL models created using `simulate_quasars.py 500 --bal`.

    python3 simulate_catalog.py PAQS_simulated_BAL_catalog.fits
        --rules ../../targets/paqs_catalog/output/12APR2024/PAQS_20240412_rules.csv
        --ruleset ../../targets/paqs_catalog/output/12APR2024/PAQS_20240412_ruleset.csv
        --temp-dir output/BAL_models
        -o output/simpaqs_l1_v2 --arm J --prog PAQS

The `--prog` option add the value 'PAQS' to the FITS header of the output.

