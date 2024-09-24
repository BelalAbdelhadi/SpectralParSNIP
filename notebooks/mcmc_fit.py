
from astropy.table import Table, join
from astropy.cosmology import Planck18 as cosmo
from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from scipy.optimize import minimize
from scipy.special import erfc

import numpy as np
import matplotlib.colors as colors
import sncosmo
import lcdata

import parsnip

#Load the model
model = parsnip.load_model('../parsnip-mod/models/model_7e-5.pt')
#Load the dataset
ps_dataset_raw = lcdata.read_hdf5('../lc_data/sdss_data.h5')
ps_dataset = model.preprocess(ps_dataset_raw)
ps_train, ps_test = parsnip.split_train_test(ps_dataset)
ps_predictions = model.predict_dataset(ps_dataset)

def fit_and_save_parameters(datasets, ps_model_path, output_dir='./fits/'):
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

    # Load the Parsnip model
    ps_model = parsnip.load_model(ps_model_path)
    
    # Loop through datasets
    for ds in datasets:
        table_rows = []
        dataset = lcdata.read_hdf5(f'./{ds}.h5')
        for idx, lc in enumerate(dataset.light_curves):
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
                    ['t0', 'amplitude', 'color', 's1', 's2', 's3'], 
                     bounds={'s1': (-5, 5), 's2': (-5, 5), 's3': (-5, 5), 'color': (-3,3)},
                     nwalkers=20  # parameters to vary
                )

                # Check for large errors in s parameters
                if (result.errors.get('s1', 0) >= 2 or 
                    result.errors.get('s2', 0) >= 2 or 
                    result.errors.get('s3', 0) >= 2):
                    skipped_ids.append(lc_id)
                    print(f"SN {idx} for {ds} with object id {lc_id} contains large s errors")
                    continue

                table_rows.append(sncosmo.flatten_result(result))
                
                sncosmo.plot_lc(lc, model=fitted_model, errors=result.errors)
                plt.suptitle('Using Parsnip model', ha='right')
                plt.savefig(f'{output_dir}/parsnip_fits/SN{idx}_{lc_id}.pdf')
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

                #print(f'Finished plotting SN {idx} for {ds} using Parsnip model')

                # Salt2 fitting and plotting
                result_salt, fitted_model_salt = sncosmo.mcmc_lc(
                    lc, model_salt,
                    ['t0', 'x0', 'x1', 'c'],
                    nwalkers=20  # parameters to vary
                )
                sncosmo.plot_lc(lc, model=fitted_model_salt, errors=result_salt.errors)
                plt.suptitle('Using SALT2 model', ha='right')
                plt.savefig(f'{output_dir}/salt_fits/SN{idx}_{lc_id}.pdf')
                plt.clf()

                # Save SALT2 parameters
                salt_params['x1'].append(result_salt.parameters[2])
                salt_params['x1_error'].append(result_salt.errors.get('x1', None))
                salt_params['color'].append(result_salt.parameters[3])
                salt_params['color_err'].append(result_salt.errors.get('c', None))
                salt_params['z'].append(z)

                #print(f'Finished plotting SN {idx} for {ds} using SALT2 model')

            except Exception as e:
                print(f"Error in fitting or plotting: {e}")
    results = Table(table_rows)
    return parsnip_params, salt_params, skipped_ids, results

# Example usage
datasets = ['sdss_data']
ps_model_path = '../parsnip-mod/models/model_7e-5.pt'
parsnip_params, salt_params, skipped_ids, results = fit_and_save_parameters(datasets, ps_model_path)