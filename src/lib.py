#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:37:23 2020

@author: Carlos Eduardo Veras Neves
"""

#%% Importa Bibliotecas

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


# As novas versões do Pandas e Matplotlib trazem diversas mensagens de aviso ao desenvolvedor. Vamos desativar isso.
# bibliotecas para visualização dos dados
import warnings
import matplotlib.cbook
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


#%% Parâmetros para formatação dos gráficos
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['text.color'] = 'k'
from matplotlib.pylab import rcParams 
rcParams['figure.figsize'] = 20,25
matplotlib.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

from tqdm import tqdm


#%% Definições para o log do Tensorflow/Tensorboard
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#%% Definições das pastas para armazenar arquivos produzidos pela simulação
MODELS_FLD = os.path.join('..','models')
FIGS_FLD = os.path.join('..','figs')
LOGS_FLD = os.path.join('..','logs')
PKL_FLD = os.path.join('..','pkl')
os.environ['NUMEXPR_MAX_THREADS'] = '9'
#%% Função para criação dos diretórios
def makedirs(fld):
    """
    

    Parameters
    ----------
    fld : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if not os.path.exists(fld):
        os.makedirs(fld)


#%% Carrega os dados
def load_data():
    """
    

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    """
    
    # Carrega os dados
    github_repo = 'https://raw.githubusercontent.com/carlosvneves/doutorado/master/'
    desembolsos = pd.read_csv(github_repo + 'desembolsos.csv')
    pib = pd.read_csv(github_repo +'pib.csv')
    fbcf = pd.read_csv(github_repo +'fbcf.csv') 
    
    fbcf.index = pd.to_datetime(fbcf['date'])
    fbcf.drop(['date'],inplace=True, axis = 1)
    
       
    pib.index = pd.to_datetime(pib['date'])
    pib.drop(['date'],inplace=True, axis = 1)

    
    desembolsos.index = pd.to_datetime(desembolsos['date'])
    desembolsos.drop(['date'], inplace=True, axis = 1)
       
    data = desembolsos.groupby(pd.PeriodIndex(desembolsos.index, freq='Q')).mean()
    data = data.loc['1996Q1':]
    data.index = data.index.to_timestamp(freq='Q')
    
    for col in data.columns:
        data[col] = data[col]/pib['pib'].values * 100
       
       
    # Corte da série de acordo com a análise de tendência
    start = '2002Q1'
    
    data['Investimento'] = fbcf['fbcf'].values/pib['pib'].values *100
    data = data.loc[start:]
    
    print(data.describe())


    # Visualiza os dados originais
    plt.figure()
    plt.plot(data.index,data[['Investimento']],
             label = 'Investimentos como % do PIB')
    plt.xlabel('Ano')
    plt.ylabel('FBCF (%PIB)')
    plt.legend()
    plt.title('Formação Bruta de Capital Fixo como % do PIB')
    plt.savefig('{}/investimentos'.format(FIGS_FLD))
    plt.show()
    
    plt.figure()
    plt.plot(data.index,data[['Agropecuária']],
             label = 'Aropecuária como % do PIB')
    plt.plot(data.index,data[['Indústria']],
             label = 'Induśtria como % do PIB')
    plt.plot(data.index,data[['Infraestrutura']],
             label = 'Infraestrutura como % do PIB')
    plt.plot(data.index,data[['Comércio e serviços']],
             label = 'Comércio e Serviços como % do PIB')
    plt.plot(data.index,data[['Total']],
             label = 'Total como % do PIB')
    plt.title('Desembolsos do BNDES como % do PIB')
    plt.xlabel('Ano')
    plt.ylabel('FBCF (%PIB)')
    plt.legend()
    plt.savefig('{}/desembolsos'.format(FIGS_FLD))
    plt.show()
      
    
    
    data[['Agropecuária','Indústria','Infraestrutura','Comércio e serviços', 'Total']]
    
    
    # Ajuste Sazonal (X-13 ARIMA) das séries para otimizar a modelagem
    X13_PATH = os.path.join('..','x13')
    
    data_sa = pd.DataFrame(data)
    data_sa.rename(columns=lambda x: x[0:3], inplace=True)
    
    for col in data_sa:
        sa = sm.tsa.x13_arima_analysis(data_sa[col],x12path=X13_PATH)
        data_sa[col] = sa.seasadj.values
    
    data_sa.tail()
    
    
    # Visualiza dados com ajuste sazonal 
    
    plt.figure()
    plt.plot(data_sa.index,data_sa[['Inv']],
             label = 'Investimentos como % do PIB')
    plt.xlabel('Ano')
    plt.ylabel('FBCF (%PIB)')
    plt.legend()
    plt.title('Formação Bruta de Capital Fixo como % do PIB - ajuste sazonal')
    plt.savefig('{}/investimentos-sa'.format(FIGS_FLD))
    plt.show()
    
    plt.figure()
    plt.plot(data_sa.index,data_sa[['Agr']],
             label = 'Aropecuária como % do PIB')
    plt.plot(data_sa.index,data_sa[['Ind']],
             label = 'Induśtria como % do PIB')
    plt.plot(data_sa.index,data_sa[['Inf']],
             label = 'Infraestrutura como % do PIB')
    plt.plot(data_sa.index,data_sa[['Com']],
             label = 'Comércio e Serviços como % do PIB')
    plt.plot(data_sa.index,data_sa[['Tot']],
             label = 'Total como % do PIB')
    plt.title('Desembolsos do BNDES como % do PIB - ajuste sazonal')
    plt.xlabel('Ano')
    plt.ylabel('FBCF (%PIB)')
    plt.legend()
    plt.savefig('{}/desembolsos-sa'.format(FIGS_FLD))
    plt.show() 
    #  Unsample dos dados de Trim para Mensal
    upsampled = data_sa.resample('M')
    interpolated = upsampled.interpolate(method='linear')
    interpolated.tail(24)
    
    data = interpolated
    
    print('##'*25)
    print('# Matriz de Correlação #')        
    print(data.corr())
    print('##'*25)
    
    return data

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

### autocorrelation prediction
def autocor_pred(real, pred, lag=1):
    return pearsonr(real[:-lag], pred[lag:])[0]

### COMPUTE METRICS ON TEST DATA ###
def compute_mae(data, true, pred, pred_simple):
    diz_error_lstm = {}
    diz_error_var_lstm = {}
    
    
    for i,col in enumerate(data.columns):
        
        error = mean_absolute_error(true[:,i], pred_simple[:,i])
        diz_error_lstm[col] = error
        
        error = mean_absolute_error(true[:,i], pred[:,i])
        diz_error_var_lstm[col] = error
        
    plt.figure(figsize=(14,5))
    plt.bar(np.arange(len(diz_error_lstm))-0.15, diz_error_lstm.values(), alpha=0.5, width=0.3, label='lstm')
    plt.bar(np.arange(len(diz_error_var_lstm))+0.15, diz_error_var_lstm.values(), alpha=0.5, width=0.3, label='var_lstm')
    plt.xticks(range(len(diz_error_lstm)), diz_error_lstm.keys())
    plt.ylabel('MAE'); plt.legend()
    np.set_printoptions(False)
    plt.show()
        
    return  diz_error_lstm, diz_error_var_lstm

## COMPUTE METRICS ON TEST DATA ###
def compute_mse(data, true, pred, pred_simple):
    diz_error_lstm = {}
    diz_error_var_lstm = {}
    
    
    for i,col in enumerate(data.columns):
        
        error = mean_squared_error(true[:,i], pred_simple[:,i])
        diz_error_lstm[col] = error
        
        error = mean_squared_error(true[:,i], pred[:,i])
        diz_error_var_lstm[col] = error
               
    plt.figure(figsize=(14,5))
    plt.bar(np.arange(len(diz_error_lstm))-0.15, diz_error_lstm.values(), alpha=0.5, width=0.3, label='lstm')
    plt.bar(np.arange(len(diz_error_var_lstm))+0.15, diz_error_var_lstm.values(), alpha=0.5, width=0.3, label='var_lstm')
    plt.xticks(range(len(diz_error_lstm)), diz_error_lstm.keys())
    plt.ylabel('MSE'); plt.legend()
    np.set_printoptions(False)
    plt.show()
    
    return  diz_error_lstm, diz_error_var_lstm

def compute_autocor(data, true, pred, pred_simple):
    diz_ac_lstm = {}
    diz_ac_var_lstm = {}
    
    
    for i,col in enumerate(data.columns):
        
        ac = autocor_pred(true[:,i], pred_simple[:,i])
        diz_ac_lstm[col] = ac
        
        ac = autocor_pred(true[:,i], pred[:,i])
        diz_ac_var_lstm[col] = ac
    
        
    plt.figure(figsize=(14,5))
    plt.bar(np.arange(len(diz_ac_lstm))-0.15, diz_ac_lstm.values(), alpha=0.5, width=0.3, label='lstm')
    plt.bar(np.arange(len(diz_ac_var_lstm))+0.15, diz_ac_var_lstm.values(), alpha=0.5, width=0.3, label='var_lstm')
    plt.xticks(range(len(diz_ac_lstm)), diz_ac_lstm.keys())
    plt.ylabel('correlation lag1'); plt.legend()
    np.set_printoptions(False)   
    plt.show()
    
    return diz_ac_lstm, diz_ac_var_lstm


    

    
    