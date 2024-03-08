import parsnip
import sncosmo
import lcdata 
import numpy as np
import matplotlib.pyplot as plt
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
    
    
source = parsnip.ParsnipSncosmoSource('model_7e-5.pt')
model = sncosmo.Model(source=source)
model_salt = sncosmo.Model(source='salt2') 
datasets = ['csp_data', 'ps_data', 'swift_data', 'sdss_data']

ps_model = parsnip.load_model('model_reg_phase_int5.pt')
for ds in datasets:
    dataset = lcdata.read_hdf5(f'{ds}.h5')
    for idx, lc in enumerate(dataset.light_curves):
        lc_pp = ps_model.preprocess(dataset[idx])
        t0 = lc_pp.meta['parsnip_reference_time']
        # Convert byte strings to regular strings
        lc['band'] = [b.decode('utf-8') if isinstance(b, bytes) else b for b in lc['band']]
        lc['zpsys'] = [b.decode('utf-8') if isinstance(b, bytes) else b for b in lc['zpsys']]

        z = lc.meta['redshift']
        model.set(z=z)
        model.set(t0=t0)
        
        model_salt.set(z=z)
        model_salt.set(t0=t0)

        # Ensure your model and model_salt are defined correctly here
        # For example: model = sncosmo.Model( ... )
        
        try:
            # Parsnip fitting and plotting
            result, fitted_model = sncosmo.fit_lc(
                lc, model,
                ['t0', 'amplitude', 'color', 's1', 's2', 's3'],  # parameters to vary
                
            )
            sncosmo.plot_lc(lc, model=fitted_model, errors=result.errors)
            plt.suptitle('Using Parsnip model',ha='right')
            plt.savefig(f'fits/{ds}/SN{idx}_parsnip.jpg')
            plt.clf()
            print(f'Finished plotting SN {idx} for {ds} using parsnip model')
            
            
            # Salt2 fitting and plotting
            result_salt, fitted_model_salt = sncosmo.fit_lc(
                lc, model_salt,
                ['t0', 'x0', 'x1', 'c'],  # parameters to vary
            )
            sncosmo.plot_lc(lc, model=fitted_model_salt, errors=result_salt.errors)
            plt.suptitle('Using SALT2 model', ha='right')
            plt.savefig(f'fits/{ds}/SN{idx}_salt.jpg')
            plt.clf()
            print(f'Finished plotting SN {idx} for {ds} using salt model')


        except Exception as e:
            print(f"Error in fitting or plotting: {e}")
