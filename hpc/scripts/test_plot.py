import numpy as np 
import sncosmo
import parsnip 
import matplotlib.pyplot as plt
import warnings
import lcdata
from astropy.table import Table, vstack

warnings.filterwarnings("ignore")

bands = []
waves = np.load('snemo_waves.npy')
for i in range(0,len(waves),12):   
    wavelengths = waves[i:i+12]
    transmission = np.ones_like(wavelengths)
    band = sncosmo.Bandpass(wavelengths, transmission, name = f'band_{int(i/12)}')
    sncosmo.registry.register(band, force=True)
    bands.append(band)

model = parsnip.load_model('model_7e-5.pt') 
datasets = ['snemo_test_2', 'csp_data', 'ps_data', 'swift_data', 'sdss_data', 'snemo_train_2']
for ds in datasets:
    dataset_stack = lcdata.read_hdf5(f'{ds}.h5')

    for idx, sn in enumerate(dataset_stack.light_curves):
        try:
            prep_ds = parsnip.preprocess_light_curve(sn, model.settings)
            fig, axs = plt.subplots(1,3, figsize=(21,7))
            axs[0].scatter(prep_ds['time'], prep_ds['flux'], c ='r', alpha=0.3)
            axs[0].scatter(sn['time'],sn['flux'],c='b', alpha=0.3)
            axs[0].set_title(f'Prep vs raw for SN {idx+1}')
            axs[0].legend(['Prep data', 'Raw data'])
            axs[1].set_title(f'Plot for SN{idx+1} light curve')
            parsnip.plot_light_curve(sn, model=model,ax = axs[1])

            print(f'plotted light curve for SN{idx+1}_{ds}')

            t_max = np.argmax(sn['flux'])

            axs[2].set_title(f'Spectrum for SN{idx+1}')
            parsnip.plot_spectrum(sn, model, sn['time'][t_max], ax=axs[2])

            plt.savefig(f'./plots/lcs/{ds}/SN_{idx+1}.jpg')
        except ValueError as e:
            print(f"Skipping SN{idx+1} due to ValueError: {e}")
