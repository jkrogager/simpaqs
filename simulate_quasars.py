"""
Simulate quasar continuum and emission lines using `simqso`

The script takes a list of absorption templates from `simulate_absorbers.py`
and generates a random quasar model at the given quasar redshift.
Each absorption template is generated using two different random quasar continuum models.

The power-law slopes of the continuum model are varied using a gaussian distribution
of the slopes:
    GaussianSampler(-1.7, 0.1) and GaussianSampler(-0.3, 0.2)
with a break at 1215 Å.

The luminosities are randomly generated by sampling a random black hole mass
and a random Eddington ratio from log-normal distributions:
    M_BH : GaussianSampler(8.67, 0.5)
    R_Edd = GaussianSampler(-0.83, 0.4)

The rest-frame luminosity density at 1450Å is then calculated by assuming
a bolometric correction of 5 (Richards et al. 2006).

Fe lines are included using a re-scaled template of the Vestergaard & Wilkes (2001)
template. Lastly, broad lines are included following the Baldwin effect.

The quasar continuum model is multiplied by the absorption transmission profile
to generate a noiseless quasar model with foreground absorption following the same
spectral template format as 4MOST with two columns: LAMBDA and FLUX_DENSITY

The script generates a list of template input parameters in the output directory
with the following parameters:
    ID: quasar model id
    z: quasar redshift
    absMag: absolute magnitude (at 1450Å)
    smcDustEBV: E(B-V) in mag following the SMC extinction curve
    LOG_MBH: log of black hole mass in units of M_sun
    LOG_REDD: log of Eddington ratio 
    ABS_TEMP: absorption template file (for reference)

"""

__author__ = 'Jens-Kristian Krogager'
__email__ = 'jens-kristian.krogager@univ-lyon1.fr'

from astropy.cosmology import Planck13
from astropy import table
import numpy as np
from scipy import stats
import os
import glob
from scipy.interpolate import interp1d
import pickle

from simqso.sqgrids import *
from simqso import sqbase
from simqso.sqrun import buildSpectraBulk, buildQsoSpectrum
from tqdm import tqdm


here = os.path.abspath(os.path.dirname(__file__))

def load_bal_templates():
    """
    Load all the rest-frame BAL templates into a dictionary.
    """
    models = {}
    temp_fnames = {'felobal': os.path.join(here, 'BALs/felobal_ot_pro1_spec.pkl'),
                   'hibal_1': os.path.join(here, 'BALs/hibal_1_pro1_spec.pkl'),
                   'hibal_2': os.path.join(here, 'BALs/hibal_2_pro1_spec.pkl'),
                   }
    for key, fname in temp_fnames.items():
        with open(fname, 'rb') as pkl_file:
            wl, trans = pickle.load(pkl_file)
            models[key] = (wl, trans)
    return models


def add_quasar_continuum(templates, dust_mode='exponential', BAL=False, output_dir='output/quasar_models'):
    """
    Add simulated quasar continuum to the list of absorption templates
    generated by `simulate_absorbers.py`. 

    templates : astropy.table.Table
        Table or similar containing the fields: FILE, Z_QSO
    """
    # Get wavelength grid from the first template:
    temp = Table.read(templates['FILE'][0], hdu=1)
    wave = temp['LAMBDA']

    if BAL:
        bal_models = load_bal_templates()

    # just make up a few magnitude and redshift points
    nqso = len(templates)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    z_all = templates['Z_QSO']
    # Sampling MBH and LEdd from Rakshit, Stalin & Kotilainen (2020)
    # https://iopscience.iop.org/article/10.3847/1538-4365/ab99c5
    logM_BH = np.random.normal(8.67, 0.5, nqso)
    logR_Edd = np.random.normal(-0.83, 0.4, nqso)
    logL_bol = np.log10(3.2e4) + logM_BH + logR_Edd
    BC = 5.  # Richards et al. (2006)
    M_all = 4.8 - 2.5*logL_bol + 2.5*np.log10(BC)

    M = AbsMagVar(FixedSampler(M_all), restWave=1450)
    z = RedshiftVar(FixedSampler(z_all))
    # Include dust sampled from E(B-V)
    if dust_mode.lower() == 'exponential':
        # based roughly on Krawczyk et al. 2015
        Ebv_all = stats.expon.rvs(loc=0., scale=0.05, size=nqso)
    else:
        Ebv_all = stats.uniform.rvs(0., 0.5, size=nqso)
    dust = SMCDustVar(FixedSampler(Ebv_all))

    qsos = QsoSimPoints([M, z, dust], cosmo=Planck13, units='luminosity')

    # use the canonical values for power law continuum slopes in FUV/NUV, with breakpoint at 1215A
    contVar = BrokenPowerLawContinuumVar([GaussianSampler(-1.7, 0.1),
                                          GaussianSampler(-0.3, 0.2)],
                                         [1215.])

    # generate lines using the Baldwin Effect emission line model from BOSS DR9
    emLineVar = generateBEffEmissionLines(qsos.absMag)

    # the default iron template from Vestergaard & Wilkes 2001 was modified to fit BOSS spectra
    fescales = [(0, 1540, 0.5),
                (1540, 1680, 2.0),
                (1680, 1868, 1.6),
                (1868, 2140, 1.0),
                (2140, 3500, 1.0)]
    feVar = FeTemplateVar(VW01FeTemplateGrid(qsos.z, wave, scales=fescales))

    # Now add the features to the QSO grid
    qsos.addVars([contVar, emLineVar, feVar])

    # ready to generate spectra
    meta, spectra = buildSpectraBulk(wave, qsos, saveSpectra=True)

    print("Creating quasar models:")

    # Save the templates:
    all_ids = []
    all_bal_types = []
    dnum = len(glob.glob(f'{output_dir}/PAQS_quasar_*.fits')) + 1
    for num, spec in enumerate(tqdm(spectra)):
        model_id = f'PAQS_quasar_{num+dnum:06}'
        filename = f'{output_dir}/{model_id}.fits'
        all_ids.append(model_id)
        absorber = Table.read(templates['FILE'][num], hdu=1)
        
        # Filter any value greater than 1 in the abs_templates 
        if (absorber['FLUX_DENSITY']>1).any():
            i_min = np.where(absorber['FLUX_DENSITY']>1)[0][0]
            i_max = np.where(absorber['FLUX_DENSITY']>1)[0][-1]
            wav_min = absorber['LAMBDA'][i_min]
            wav_max = absorber['LAMBDA'][i_max]
            f = interp1d([absorber['LAMBDA'][i_min-1], absorber['LAMBDA'][i_max+1]], 
                      [absorber['FLUX_DENSITY'][i_min-1], absorber['FLUX_DENSITY'][i_max+1]])
            absorber['FLUX_DENSITY'][absorber['FLUX_DENSITY']>1] = f(absorber['LAMBDA'][absorber['FLUX_DENSITY']>1])
            #print(f'\ninterpolation launched for {model_id} from wavelength {wav_min:.2f} to {wav_max:.2f}
            # for appearance of values greater than 1 in the abs_template')
        
        spec = spec * absorber['FLUX_DENSITY']
        bal_type = np.random.choice(['hibal_1', 'hibal_2', 'felobal', 'none'], p=[0.2, 0.2, 0.2, 0.4])
        all_bal_types.append(bal_type)
        if BAL and bal_type != 'none':
            # Incude a random BAL template
            bal_wl, models = bal_models[bal_type]
            model_num = np.random.randint(0, len(models))
            if len(models[model_num]) == 2:
                _, trans = models[model_num]
            else:
                trans = models[model_num]
            bal_trans = np.interp(wave, bal_wl*(z_all[num] + 1), trans)
            spec = spec * bal_trans

        hdu = fits.HDUList()
        hdr = fits.Header()
        hdr['AUTHOR'] = 'JK Krogager'
        hdr['COMMENT'] = 'Synthetic quasar + absorber model based on simqso'
        hdr['REDSHIFT'] = z_all[num]
        hdr['MAG'] = (M_all[num], "Abs. magnitude 1450")
        hdr['EBV'] = (Ebv_all[num], "Mag")
        hdr['LOG_MBH'] = (logM_BH[num], "log(M_BH / Msun)")
        hdr['LOG_REDD'] = (logR_Edd[num], "log(R_Edd)")
        hdr['ID'] = model_id
        hdr['ABS_ID'] = templates['FILE'][num]
        hdr['BAL_TYPE'] = bal_type
        prim = fits.PrimaryHDU(header=hdr)
        hdu.append(prim)
        col_wl = fits.Column(name='LAMBDA', array=wave, format='1D', unit='Angstrom')
        col_flux = fits.Column(name='FLUX_DENSITY', array=spec, format='1E', unit='erg/(s cm**2 Angstrom)')
        tab = fits.BinTableHDU.from_columns([col_wl, col_flux])
        tab.name = 'TEMPLATE'
        hdu.append(tab)
        hdu.writeto(filename, overwrite=True, output_verify='silentfix')

    qsos.data['ID'] = all_ids
    qsos.data['LOG_MBH'] = logM_BH
    qsos.data['LOG_REDD'] = logR_Edd
    qsos.data['ABS_TEMP'] = templates['FILE']
    qsos.data['BAL_TYPE'] = all_bal_types
    if os.path.exists(f'{output_dir}/model_parameters.fits'):
        tab = Table.read(f'{output_dir}/model_parameters.fits')
        tab = table.vstack([tab, qsos.data])
    else:
        tab = qsos.data
    tab.write(f'{output_dir}/model_parameters.fits', overwrite=True)
    # Write input parameter table:
    subset = qsos.data['ID', 'z', 'absMag', 'smcDustEBV', 'LOG_MBH', 'LOG_REDD', 'BAL_TYPE', 'ABS_TEMP']
    subset.rename_column('z', 'REDSHIFT')
    if os.path.exists(f'{output_dir}/model_input.csv'):
        old_subset = Table.read(f'{output_dir}/model_input.csv')
        subset = table.vstack([old_subset, subset])
    subset.write(f'{output_dir}/model_input.csv', overwrite=True)
    return subset



def main():
    from argparse import ArgumentParser
    parser = ArgumentParser('Add quasar continuum to absorber templates')
    parser.add_argument("templates", type=str,
                        help="List of absorber templates (output/list_templates.csv)")
    parser.add_argument("-n", "--number", type=int, default=3,
                        help="Number of random quasar realizations per absorber [default=3]")
    parser.add_argument('--dust', type=str, default='exponential')
    parser.add_argument('--bal', action='store_true')
    parser.add_argument("--dir", type=str, default='output/quasar_models',
                        help="Output directory [default=output/quasar_models]")

    args = parser.parse_args()
    
    print("Adding quasar continuum models to absorber templates")
    for _ in range(args.number):
        templates = Table.read(args.templates)
        add_quasar_continuum(templates, dust_mode=args.dust, BAL=args.bal, output_dir=args.dir)


if __name__ == '__main__':
    main()

