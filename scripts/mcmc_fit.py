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
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Run Parsnip model fitting on light curves.")
parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset chunk (HDF5 file).")
parser.add_argument('--start', type=int, required=True, help="Start index for light curves.")
parser.add_argument('--end', type=int, required=True, help="End index for light curves.")

args = parser.parse_args()

#Load the model

def fit_and_save_parameters(dataset, start, end):
    # Initialize parameter storage
    parsnip_params = {
        't0': [], 't0_error': [],
        's1': [], 's1_error': [],
        's2': [], 's2_error': [],
        's3': [], 's3_error': [],
        'color': [], 'color_err': [],
        'z': [], 'zpsys': 'ab'
    }
    
    salt_params = {
        'x1': [], 'x1_error': [],
        'color': [], 'color_err': [],
        'z': [], 'zpsys': 'ab'
    }

    skipped_ids = []
    
    samples = {'object_id': [], 'mean': [], 'std': [], 'covariance': []}

    # Load the Parsnip model and dataset
    ps_model_path='./model_7e-5.pt'
    print(f"Loading Parsnip model with name {ps_model_path}...")
    ps_model = parsnip.load_model(ps_model_path)
    dataset_name = dataset
    print(f"Loading dataset {dataset_name}...")
    dataset = lcdata.read_hdf5(f'./data/{dataset}.h5')
    # Process only the light curves within the range [start, end)
    for idx in range(start, min(end, len(dataset.light_curves))):
            print(f"Processing light curve {idx}...")
            lc = dataset.light_curves[idx]
            lc['fluxerr'] = np.abs(lc['fluxerr'])

            # Convert byte strings to regular strings
            lc['band'] = [b.decode('utf-8') if isinstance(b, bytes) else b for b in lc['band']]
            lc['zpsys'] = [b.decode('utf-8') if isinstance(b, bytes) else b for b in lc['zpsys']]

            lc_pp = ps_model.preprocess(dataset[idx])
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

            try:
                # Parsnip fitting and plotting
                result, fitted_model = sncosmo.mcmc_lc(
                    lc, model,
                    ['t0', 'amplitude', 'color', 's1', 's2', 's3'], # parameters to vary
                     bounds={'s1': (-5, 5), 's2': (-5, 5), 's3': (-5, 5), 'color': (-3,3)},
                     nwalkers=20  
                )

                # Check for large errors in s parameters
                df = pd.DataFrame(result.samples, columns=['t0', 'amplitude', 'color', 's1', 's2', 's3'])
                print(f"SN {lc_id} blue filter mcmc fitting parameters are: \n")
                print(df.mean())
                print(f"SN {lc_id} blue filter mcmc fitting parameter uncertainties are: \n")
                print(df.std())
                print(f"SN {lc_id} blue filter mcmc fitting parameter covariance matrix is: \n")
                print(df.cov())
                
                df['amplitude'] = -2.5 * np.log(df['amplitude'])
                
                  # Calculate mean, std, and covariance
                mean = df.mean().to_dict()
                std = df.std().to_dict()
                covariance = df.cov().values
    
                samples['object_id'].append(lc_id)
                samples['mean'].append(mean)
                samples['std'].append(std)
                samples['covariance'].append(covariance)

                if (result.errors.get('s1', 0) >= 2 or 
                    result.errors.get('s2', 0) >= 2 or 
                    result.errors.get('s3', 0) >= 2):
                    skipped_ids.append(lc_id)
                    print(f"SN {idx} for {dataset_name} with object id {lc_id} contains large s errors")
                    continue

		        
                sncosmo.plot_lc(lc, model=fitted_model, errors=result.errors)
                plt.suptitle('Using Parsnip model', ha='right')
                plt.savefig(f'./fits/parsnip_fits/SN{idx}_{lc_id}_{dataset_name}.pdf')
                plt.clf()

                # Save Parsnip parameters
                parsnip_params['t0'].append(result.parameters[0])
                parsnip_params['t0_error'].append(result.errors.get('t0', None))
                parsnip_params['color'].append(result.parameters[2])
                parsnip_params['color_err'].append(result.errors.get('color', None))
                parsnip_params['s1'].append(result.parameters[3])

                parsnip_params['s1_error'].append(result.errors.get('s1', None))
                parsnip_params['s2'].append(result.parameters[4])
                parsnip_params['s2_error'].append(result.errors.get('s2', None))
                parsnip_params['s3'].append(result.parameters[5])
                parsnip_params['s3_error'].append(result.errors.get('s3', None))
                parsnip_params['z'].append(z)

             

                #print(f'Finished plotting SN {idx} for {dataset_name} using SALT2 model')

            except Exception as e:
                print(f"Error in fitting or plotting: {e}")
    #Saving to yml files.
    print(f"Saving results...")
    with open(f'./parsnip_params/parsnip_params_{dataset_name}_{start}-{end}.yml', 'w') as outfile:
        yaml.dump(parsnip_params, outfile, default_flow_style=False)
        
    with open(f'./mcmc_samples/mcmc_samples_{dataset_name}_{start}-{end}.yml', 'w') as outfile:
        yaml.dump(samples, outfile, default_flow_style=False)

    return parsnip_params, salt_params, skipped_ids, samples

ps_model_path = './model_7e-5.pt'
parsnip_params, salt_params, skipped_ids, samples = fit_and_save_parameters(args.dataset, args.start, args.end)

