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

import warnings
warnings.filterwarnings('ignore')

models = os.listdir('./models')
reg_models = []
for model in models: 
    if '-' in model:
        reg_models.append(model)

dataset = lcdata.read_hdf5('./csp_data.h5')
lc= dataset.light_curves[20]
lc_meta= dataset.meta[20]
obj_id = lc_meta['object_id']
for model in reg_models:
    parsnip_model = parsnip.load_model(f'./models/{model}')
    plt.figure(figsize=(20,20))
    parsnip.plot_light_curve(lc, model=parsnip_model)
    plt.title(f'{obj_id} light curve using {model}')
    print(f'Plotted Light curve for {model}')
    plt.savefig(f'./figures/{obj_id}_{model}.pdf')