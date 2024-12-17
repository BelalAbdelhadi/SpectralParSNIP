from astropy.table import Table, join
from astropy.cosmology import Planck18 as cosmo
from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from scipy.optimize import minimize
from scipy.special import erfc

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
from sklearn.metrics import roc_curve, auc
import warnings

import sncosmo
import lcdata
import parsnip

warnings.filterwarnings('ignore')

# Load and register bands
bands = []
waves = np.load('snemo_waves.npy')
for i in range(0, len(waves), 12):   
    wavelengths = waves[i:i+12]
    transmission = np.ones_like(wavelengths)
    band = sncosmo.Bandpass(wavelengths, transmission, name=f'band_{int(i/12)}')
    sncosmo.registry.register(band, force=True)
    bands.append(band)

# Models to iterate over
model_paths = ['./models/model_7e-5.pt', './models/model_real_finale.pt']
datasets = ['sdss_data', 'ps_data', 'csp_data', 'swift_data', 'snemo_train_2']

# Loop over each model and dataset
for model_path in model_paths:
    model_name = os.path.basename(model_path).split('.')[0]
    model = parsnip.load_model(model_path)
    
    for dataset_name in datasets:
        dataset = lcdata.read_hdf5(f'./{dataset_name}.h5')
        
        # Loop over light curves 6 through 10
        for i in range(6, 11):
            lc = dataset.light_curves[i]
            lc_meta = dataset.meta[i]
            obj_id = lc_meta['object_id']
            
            # Plot light curve and spectrum
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            axs[0].set_title(f'Light Curve Plot for {obj_id} with model {model_name}')
            parsnip.plot_light_curve(lc, model=model, ax=axs[0])
            axs[0].set_xlim(left=-75)
            
            axs[1].set_title(f'Spectrum for {obj_id} with model {model_name}')
            t_max = np.argmax(lc['flux'])
            parsnip.plot_spectrum(lc, model, lc['time'][t_max], ax=axs[1])
            
            # Save plot
            plt.savefig(f'./figures/SN{obj_id} from {dataset_name} with {model_name}.pdf')
            plt.close(fig)
            
            print(f"Plotted Light curve and Spectra for {obj_id} in {dataset_name} using {model_name}")
