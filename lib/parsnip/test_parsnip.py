import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.time import Time
from astropy.io import ascii
import astropy.units as u
import lcdata
import sncosmo
import parsnip
import warnings
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

model = parsnip.load_model('./model_1.pt')
waves = model.model_wave