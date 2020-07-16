#%% Load Python modules
# # Importa Bibliotecas Python

# bibliotecas para as redes neurais
import tensorflow as tf
tf.keras.backend.clear_session()
from keras.layers.merge import Concatenate
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# bibliotecas matemáticas
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler

# bibliotecas para os modelos de séries temporais
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.vector_ar.var_model import VAR
from pmdarima.arima.utils import ndiffs
import pmdarima as pm

# bilbiotecas de utilidades do sistema
import sys
import os
import pickle
from datetime import datetime
import time
import logging
from IPython import get_ipython

# As novas versõess do Pandas e Matplotlib trazem diversas mensagens de aviso ao desenvolvedor. Vamos desativar isso.
# bibliotecas para visualização dos dados
import warnings
import matplotlib.cbook
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# %% Parâmetros para formatação dos gráficos
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['text.color'] = 'k'
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 25
matplotlib.style.use('fivethirtyeight')

from IPython import *

get_ipython().run_line_magic('matplotlib', 'inline')

from tqdm import tqdm

# %% Definições para o log do Tensorflow/Tensorboard
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# %% Definições das pastas para armazenar arquivos produzidos pela simulação
MODELS_FLD = os.path.join('..', 'models')
FIGS_FLD = os.path.join('..', 'figs')
LOGS_FLD = os.path.join('..', 'logs')
PKL_FLD = os.path.join('..', 'pkl')
os.environ['NUMEXPR_MAX_THREADS'] = '9'