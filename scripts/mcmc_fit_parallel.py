from astropy.table import Table, join
from astropy.cosmology import Planck18 as cosmo
from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from scipy.optimize import minimize
from scipy.special import erfc
import pandas as pd
import numpy as np
import matplotlib.colors as colors
import sncosmo
import lcdata
import yaml
import parsnip
import multiprocessing as mp

# Ensure the correct multiprocessing method is set
mp.set_start_method('fork', force=True)

# Function to process each light curve (transient)
def process_light_curve(lc, ps_model_path, idx, ds, output_dir):
    try:
        # Load the model within each process
        ps_model = parsnip.load_model(ps_model_path)
        lc['fluxerr'] = np.abs(lc['fluxerr'])

        # Convert byte strings to regular strings
        lc['band'] = [b.decode('utf-8') if isinstance(b, bytes) else b for b in lc['band']]
        lc['zpsys'] = [b.decode('utf-8') if isinstance(b, bytes) else b for b in lc['zpsys']]

        lc_pp = ps_model.preprocess(lc)
        t0 = lc_pp.meta['parsnip_reference_time']
        z = lc.meta['redshift']
        lc_id = lc.meta['object_id']

        # Set up models
        source = parsnip.ParsnipSncosmoSource(ps_model_path)
        model = sncosmo.Model(source=source)
        model_salt = sncosmo.Model(source='salt2')

        model.set(z=z, color=0.0, s1=0.1, s2=0.1, s3=0.1)
        model.set(t0=t0)
        model_salt.set(z=z)
        model_salt.set(t0=t0)

        # Parsnip fitting
        result, fitted_model = sncosmo.mcmc_lc(
            lc, model,
            ['t0', 'amplitude', 'color', 's1', 's2', 's3'],
            bounds={'s1': (-5, 5), 's2': (-5, 5), 's3': (-5, 5), 'color': (-3, 3)},
            nwalkers=20
        )

        # Check for large errors in s parameters
        df = pd.DataFrame(result.samples, columns=['t0', 'amplitude', 'color', 's1', 's2', 's3'])
        if (result.errors.get('s1', 0) >= 2 or 
            result.errors.get('s2', 0) >= 2 or 
            result.errors.get('s3', 0) >= 2):
            return {"skipped": True, "lc_id": lc_id}

        # Save the plot
        sncosmo.plot_lc(lc, model=fitted_model, errors=result.errors)
        plt.suptitle('Using Parsnip model', ha='right')
        plt.savefig(f'{output_dir}/parsnip_fits/SN{idx}_{lc_id}_{ds}.pdf')
        plt.clf()

        # Save Parsnip parameters
        parsnip_params = {
            't0': result.parameters[0],
            't0_error': result.errors.get('t0', None),
            'color': result.parameters[2],
            'color_err': result.errors.get('color', None),
            's1': result.parameters[3],
            's1_error': result.errors.get('s1', None),
            's2': result.parameters[4],
            's2_error': result.errors.get('s2', None),
            's3': result.parameters[5],
            's3_error': result.errors.get('s3', None),
            'z': z
        }

        # Return success and data
        return {"skipped": False, "lc_id": lc_id, "parsnip_params": parsnip_params, "samples": result.samples}
    
    except Exception as e:
        return {"skipped": True, "lc_id": lc_id, "error": str(e)}

# Main function to fit and save parameters
def fit_and_save_parameters(dataset, ps_model_path, output_dir='./fits/'):
    parsnip_params = {'t0': [], 't0_error': [], 's1': [], 's1_error': [], 's2': [], 's2_error': [],
                      's3': [], 's3_error': [], 'color': [], 'color_err': [], 'z': [], 'zpsys': 'ab'}
    
    skipped_ids = []
    samples = {'object_id': [], 'mcmc_samples': []}

    dataset = lcdata.read_hdf5(f'./{dataset}.h5')

    # Multiprocessing pool
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(process_light_curve, [(lc, ps_model_path, idx, dataset, output_dir) for idx, lc in enumerate(dataset.light_curves)])

    # Collect results
    for result in results:
        if result["skipped"]:
            skipped_ids.append(result["lc_id"])
        else:
            parsnip_params['t0'].append(result["parsnip_params"]['t0'])
            parsnip_params['t0_error'].append(result["parsnip_params"]['t0_error'])
            parsnip_params['color'].append(result["parsnip_params"]['color'])
            parsnip_params['color_err'].append(result["parsnip_params"]['color_err'])
            parsnip_params['s1'].append(result["parsnip_params"]['s1'])
            parsnip_params['s1_error'].append(result["parsnip_params"]['s1_error'])
            parsnip_params['s2'].append(result["parsnip_params"]['s2'])
            parsnip_params['s2_error'].append(result["parsnip_params"]['s2_error'])
            parsnip_params['s3'].append(result["parsnip_params"]['s3'])
            parsnip_params['s3_error'].append(result["parsnip_params"]['s3_error'])
            parsnip_params['z'].append(result["parsnip_params"]['z'])

            samples['object_id'].append(result["lc_id"])
            samples['mcmc_samples'].append(result["samples"])

    return parsnip_params, skipped_ids, samples

# Run the main function
datasets = ['sdss_data', 'sdss_blue', 'ps_data', 'ps_blue', 'csp_data', 'csp_blue', 'swift_data', 'swift_blue']
ps_model_path = './model_7e-5.pt'
for dataset in datasets:
    parsnip_params, skipped_ids, samples = fit_and_save_parameters(dataset, ps_model_path)

    # Save results to YAML files
    with open(f'parsnip_params_blue_{dataset}.yml', 'w') as outfile:
        yaml.dump(parsnip_params, outfile, default_flow_style=False)
    
    with open(f'skipped_ids_blue_{dataset}.yml', 'w') as outfile:
        yaml.dump(skipped_ids, outfile, default_flow_style=False)
    
    with open(f'mcmc_samples_blue_{dataset}.yml', 'w') as outfile:
        yaml.dump(samples, outfile, default_flow_style=False)

