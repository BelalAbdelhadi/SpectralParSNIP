import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
import lcdata
import sncosmo
import parsnip
import warnings

warnings.filterwarnings("ignore")
bands = []
waves = np.load('snemo_waves.npy')
for i in range(0,len(waves),12):   
    wavelengths = waves[i:i+12]
    transmission = np.ones_like(wavelengths)
    band = sncosmo.Bandpass(wavelengths, transmission, name = f'band_{int(i/12)}')
    sncosmo.registry.register(band, force=True)
    bands.append(band)


dataset_real = parsnip.load_dataset('./snemo_test.h5')
dataset_stack = parsnip.load_dataset('./stacked_data_2.h5')
model_real = parsnip.load_model('./model_real.pt')
model_spectrum = parsnip.load_model('./model_stacked_2.pt')
#Plot light curves using parsnip
for i in range(len(dataset.light_curves)):
    lc = dataset_real.light_curves[i]
    lc_spec = dataset_stack.light_curves[i]

    time_max = np.argmax(lc['flux'])
    zero_phase = np.argmin(np.abs(lc_spec['time']))

    fig, ax=plt.subplots(2,2, figsize=(28,12))
    plt.suptitle(f'Spectra for SN no.{i+1}')
    ax[0][0].set_title(f'Synthetic model on synthetic dataset')
    ax[0][1].set_title(f'Spectrum model on real dataset')
    ax[1][0].set_title(f'Real model on synthetic dataset')
    ax[1][1].set_title(f'Real model on real dataset')

    parsnip.plot_spectrum(lc_spec, model_spectrum, time=lc_spec['time'][zero_phase], c='r', ax=ax[0][0]);
    parsnip.plot_spectrum(lc, model_spectrum, time=lc['time'][time_max], c='r', ax=ax[0][1]);
    parsnip.plot_spectrum(lc_spec, model_real, time=lc_spec['time'][zero_phase], c='b', ax=ax[1][0]);
    parsnip.plot_spectrum(lc, model_real, time=lc['time'][time_max], c='b', ax=ax[1][1]);
    plt.savefig(f'./plots/real_model/SN no. {i+1}.png')
    print(f'Plot for SN  no.{i+1} is done')
