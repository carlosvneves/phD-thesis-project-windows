#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 15:39:14 2020

@author: Carlos Eduardo Veras Neves - PhD candidate
University of Brasilia - UnB
Department of Economics - Applied Economics

Thesis Supervisor: Geovana Lorena Bertussi, PhD.
Title:Professor of Department of Economics - UnB

Classe para construção das diversas arquiteturas de redes neurais.
"""
from lib import *

class Models:

    MODEL_ARCH = ''

    #%% Constrói o modelo LSTM
    def lstm(self):
        """
        Função que constrói o modelo baseado em neurônios LSTM.
        Usa a arquitetura CuDNN para execução mais rápida.

       Returns
       -------
       model : keras.model,
           Modelo com arquitetura baseada em neurônios LSTM.

        """


        global MODEL_ARCH

        # nome da arquitetura modelo
        MODEL_ARCH = 'LSTM'

        #número de neurônios nas camadas ocultas
        hidden_nodes = int(self.neurons*2/3)
        dropout = 0.2

        # modelo de rede de acordo com a configuração
        model = keras.Sequential()

        # CUDNN LSTM implementation
        model.add(keras.layers.LSTM(units = self.neurons, activation = 'tanh',
                             recurrent_activation = 'sigmoid',
                   recurrent_dropout = 0,unroll = False,
                   use_bias = True,
                   input_shape=(self.x_shape, self.y_shape)))
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(hidden_nodes, activation = 'relu'))
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(1))

        return model

    #%% Constrói o modelo LSTM - stacked
    def lstm_stacked(self):
        """
        Função que constrói o modelo baseado em neurônios LSTM empilhados.

        Returns
        -------
        model : keras.model,
            Modelo com arquitetura baseada em neurônios LSTM empilhados.


        """
        global MODEL_ARCH

        # nome da arquitetura modelo
        MODEL_ARCH ='LSTM-S'


        #número de neurônios nas camadas ocultas
        hidden_nodes = int(self.neurons*2/3)
        dropout = 0.2
        # modelo de rede de acordo com a configuração
        model = keras.Sequential()


        # Stacked LSTM model
        model.add(keras.layers.LSTM(units = self.neurons, activation = 'relu',recurrent_activation = 'sigmoid',
                          recurrent_dropout = 0,unroll = False, use_bias = True, return_sequences = True,
                         input_shape=(self.x_shape, self.y_shape)))
        # adicionar camada LSTM para usar o disposito de recorrência
        model.add(keras.layers.LSTM(units = self.neurons, activation = 'relu'))
        model.add(keras.layers.Dense(hidden_nodes, activation = 'tanh'))
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(1))

        return model

    #%% Constrói o modelo LSTM - bidirecional
    def lstm_bidirectional(self):
        """
        Função que constrói o modelo baseado em neurônios LSTM-Bidirecional

        Returns
        -------
        model : keras.model,
            Modelo com arquitetura baseada em neurônios LSTM-Bidirecional.


        """
        global MODEL_ARCH

        # nome da arquitetura modelo
        MODEL_ARCH = 'LSTM-B'

        #número de neurônios nas camadas ocultas
        hidden_nodes = int(self.neurons*2/3)
        dropout = 0.2
        # modelo de rede de acordo com a configuração
        model = keras.Sequential()

        # CUDNN LSTM implementation
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(self.neurons, activation = 'relu'), 
                                      input_shape=(self.x_shape, self.y_shape)))
        model.add(keras.layers.Dropout(dropout)) # camada para evitar overfitting (20% dos pesos são zerados)
        model.add(keras.layers.Dense(hidden_nodes, activation = 'relu'))
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(1))

        return model


    #%% Constrói o modelo LSTM - bidirecional
    def gru(self):
        """
        Função que constrói o modelo baseado em neurônios GRU.

        Returns
        -------
        model : keras.model,
            Modelo com arquitetura baseada em neurônios GRU.


        """
        global MODEL_ARCH
        # nome da arquitetura modelo
        MODEL_ARCH = 'GRU'

        #número de neurônios nas camadas ocultas
        hidden_nodes = int(self.neurons*2/3)
        dropout = 0.2
        # modelo de rede de acordo com a configuração
        model = keras.Sequential()

        # CUDNN LSTM implementation
        model.add(keras.layers.Conv1D(filters=self.neurons, kernel_size=8, strides=4, padding="valid",
             input_shape=(self.x_shape, self.y_shape)))
        model.add(keras.layers.GRU(self.neurons, return_sequences=True))
        model.add(keras.layers.Dropout(dropout)) # camada para evitar overfitting (20% dos pesos são zerados)
        model.add(keras.layers.GRU(hidden_nodes))
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(1))

        return model

    def cnn_lstm(self):
        """
        Função que constrói o modelo baseado em neurônios CNN e LSTM.


        Returns
        -------
        model : keras.model
            Modelo com arquitetura baseada em neurônios CNN e LSTM empilhados.

        """
        
        global MODEL_ARCH
        # nome da arquitetura modelo
        MODEL_ARCH = 'CNN-LSTM'

        #número de neurônios nas camadas ocultas
        hidden_nodes = int(self.neurons*2/3)
        dropout = 0.2
        # modelo de rede de acordo com a configuração
        model = keras.Sequential()
        
        
        # Usando a implementação CuDNNLSTM - mais rápida
        model.add(keras.layers.LSTM(self.neurons,activation = 'tanh',recurrent_activation = 'sigmoid',
                       return_sequences=True,recurrent_dropout = 0,unroll = False, use_bias = True, 
                       input_shape=(self.x_shape, self.y_shape)))
        model.add(keras.layers.Dropout(dropout)) # camada para evitar overfitting 
        model.add(keras.layers.Conv1D(filters=self.neurons, kernel_size=8, activation='tanh', 
                         input_shape=(self.x_shape, self.y_shape)))
        model.add(keras.layers.MaxPooling1D(pool_size=1))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(dropout)) # camada para evitar overfitting 
        model.add(keras.layers.Dense(hidden_nodes, activation='relu'))
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(1))
        
        #print(model.summary())
        
        return model

    

    def var_lstm(self, x_shape, seq_length = 36):
        """
        

        Parameters
        ----------
        x_shape : TYPE
            DESCRIPTION.
        seq_length : TYPE, optional
            DESCRIPTION. The default is 36.

        Returns
        -------
        model : TYPE
            DESCRIPTION.

        """
    
        global MODEL_ARCH

        # nome da arquitetura modelo
        MODEL_ARCH = 'VAR-NN'

        #número de neurônios nas camadas ocultas
        hidden_nodes = int(self.neurons*2/3)
        dropout = 0.2
    
    
    
        opt = keras.optimizers.RMSprop(lr=1e-4)
        
        inp = keras.layers.Input(shape=(seq_length, x_shape))
        
        x = keras.layers.LSTM(units = self.neurons, activation = 'tanh',
                             recurrent_activation = 'sigmoid',
                   recurrent_dropout = 0,unroll = False,
                   use_bias = True)(inp)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.Dense(hidden_nodes, activation='relu')(x)
        x = keras.layers.Dropout(dropout)(x)
        
        out = keras.layers.Dense(5)(x)
        
        model = keras.models.Model(inp, out)
        model.compile(optimizer=opt, loss='mse')
    
        return model

    def set_x_shape(self,x_shape):
        """


        Parameters
        ----------
        x_shape : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.x_shape = x_shape

    def set_y_shape(self,y_shape):
        """


        Parameters
        ----------
        y_shape : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.y_shape = y_shape

    def set_neurons(self,neurons):
        """


        Parameters
        ----------
        neurons : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.neurons = neurons

    def get_model_name(self):
        """


        Returns
        -------
        None.

        """
        return self.model_name


    #%% Class constructor
    def __init__(self):

            self.neurons = 0
            self.x_shape = 0
            self.y_shape = 0

