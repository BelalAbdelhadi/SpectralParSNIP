import numpy as np
import sncosmo
import parsnip


model = parsnip.load_model('./model_1.pt')
waves = model.model_wave

bands = []
for i in range(0,300,10):
    wavelengths = waves[i:i+10]
    transmission = np.ones_like(wavelengths)
    band = sncosmo.Bandpass(wavelengths, transmission, name = f'band_{int(i/10)}')
    sncosmo.registry.register(band, force=True)
    bands.append(band)