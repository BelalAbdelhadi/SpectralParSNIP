import matplotlib.pyplot as plt
import numpy as np
import sncosmo
import lcdata
import yaml
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Run SALT2 model fitting on light curves.")
parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset chunk (HDF5 file).")
parser.add_argument('--start', type=int, required=True, help="Start index for light curves.")
parser.add_argument('--end', type=int, required=True, help="End index for light curves.")

args = parser.parse_args()

# Function to fit SALT2 model and save parameters
def fit_and_save_salt_parameters(dataset, start, end):
    # Initialize storage for SALT2 parameters
    salt_params = {
        't0': [], 't0_error': [],
        'x1': [], 'x1_error': [],
        'color': [], 'color_error': [],
        'amplitude': [], 'amplitude_error': [],
        'z': [], 'zpsys': 'ab'
    }

    samples = {'object_id': [], 'mean': [], 'std': [], 'covariance': []}

    skipped_ids = []
    
    # Load the dataset
    dataset_name = dataset
    print(f"Loading dataset {dataset_name}...")
    dataset = lcdata.read_hdf5(f'./data/{dataset}.h5')

    # Process only the light curves within the range [start, end)
    for idx in range(start, min(end, len(dataset.light_curves))):
        print(f"Processing light curve {idx}...")
        lc = dataset.light_curves[idx]
        lc['fluxerr'] = np.abs(lc['fluxerr'])  # Ensure fluxerr is positive

        # Convert byte strings to regular strings
        lc['band'] = [b.decode('utf-8') if isinstance(b, bytes) else b for b in lc['band']]
        lc['zpsys'] = [b.decode('utf-8') if isinstance(b, bytes) else b for b in lc['zpsys']]

        z = lc.meta['redshift']
        lc_id = lc.meta['object_id']
        t0 = lc.meta.get('t0', None)

        # Set up SALT2 model
        model_salt = sncosmo.Model(source='salt2')

        model_salt.set(z=z)
        if t0 is not None:
            model_salt.set(t0=t0)
        else:
            model_salt.set(t0=0.0)  # You can change the default if needed

        try:
            # Fit SALT2 model to light curve
            result, fitted_model = sncosmo.mcmc_lc(
                lc, model_salt,
                ['t0', 'x1', 'color', 'amplitude'],  # parameters to vary
                bounds={'x1': (-5, 5), 'color': (-3, 3)},  # bounds for parameters
                nwalkers=20
            )

            # Check for large errors in x1 or color
            df = pd.DataFrame(result.samples, columns=['t0', 'x1', 'color', 'amplitude'])
            print(f"SN {lc_id} SALT2 MCMC fitting parameters are: \n{df.mean()}")
            print(f"SN {lc_id} SALT2 MCMC fitting parameter uncertainties are: \n{df.std()}")
            print(f"SN {lc_id} SALT2 MCMC fitting parameter covariance matrix is: \n{df.cov()}")

            # Calculate mean, std, and covariance
            mean = df.mean().to_dict()
            std = df.std().to_dict()
            covariance = df.cov().values

            samples['object_id'].append(lc_id)
            samples['mean'].append(mean)
            samples['std'].append(std)
            samples['covariance'].append(covariance)

            if result.errors.get('x1', 0) >= 2 or result.errors.get('color', 0) >= 2:
                skipped_ids.append(lc_id)
                print(f"SN {idx} for {dataset_name} with object id {lc_id} contains large x1 or color errors")
                continue

            # Save SALT2 parameters
            salt_params['t0'].append(result.parameters[0])
            salt_params['t0_error'].append(result.errors.get('t0', None))
            salt_params['x1'].append(result.parameters[1])
            salt_params['x1_error'].append(result.errors.get('x1', None))
            salt_params['color'].append(result.parameters[2])
            salt_params['color_error'].append(result.errors.get('color', None))
            salt_params['amplitude'].append(result.parameters[3])
            salt_params['amplitude_error'].append(result.errors.get('amplitude', None))
            salt_params['z'].append(z)

            # Plot light curve using SALT2 model
            sncosmo.plot_lc(lc, model=fitted_model, errors=result.errors)
            plt.suptitle('Using SALT2 model', ha='right')
            plt.savefig(f'./fits/salt_fits/SN{idx}_{lc_id}_{dataset_name}.pdf')
            plt.clf()

        except Exception as e:
            print(f"Error in fitting or plotting: {e}")

    # Saving results
    print(f"Saving results...")
    with open(f'./salt_params/salt_params_{dataset_name}_{start}-{end}.yml', 'w') as outfile:
        yaml.dump(salt_params, outfile, default_flow_style=False)

    with open(f'./salt_samples/salt_samples_{dataset_name}_{start}-{end}.yml', 'w') as outfile:
        yaml.dump(samples, outfile, default_flow_style=False)

    return salt_params, skipped_ids, samples

# Example usage:
salt_params, skipped_ids, samples = fit_and_save_salt_parameters(args.dataset, args.start, args.end)
