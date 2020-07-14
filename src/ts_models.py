#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 22:27:37 2020

@author: nnlinux
"""
from lib import *



class TS_Models:

    ### PLOTTING UTILITY FUNCTIONS ###
    
    def plot_series(self,train_date, train, test_date, test, name):
        """
        

        Parameters
        ----------
        train_date : TYPE
            DESCRIPTION.
        train : TYPE
            DESCRIPTION.
        test_date : TYPE
            DESCRIPTION.
        test : TYPE
            DESCRIPTION.
        name : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        plt.figure(figsize=(16,4))
        
        
        plt.plot(train_date, train[name], label='train')
        plt.plot(test_date, test[name], label='test')
        plt.ylabel(name); plt.legend()
        plt.show()
 
        
    def plot_autocor(self,name, df):
        """
        

        Parameters
        ----------
        name : TYPE
            DESCRIPTION.
        df : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        plt.figure(figsize=(16,4))
        
        # pd.plotting.autocorrelation_plot(df[name])
        # plt.title(name)
        # plt.show()
        
        timeLags = np.arange(1,36)
        plt.plot([df[name].autocorr(dt) for dt in timeLags])
        plt.title(name); plt.ylabel('autocorr'); plt.xlabel('time lags')
        plt.show()
    
    def ARIMA(self, data):
        # fit stepwise auto-ARIMA
        stepwise_fit = pm.auto_arima(data, start_p=1, start_q=1,
                             max_p=5, max_q=5, m=12,
                             start_P=0, seasonal=True,
                             d=1, D=1, trace=True,
                             error_action='ignore',  # don't want to know if an order does not work
                             suppress_warnings=True,  # don't want convergence warnings
                             stepwise=True)  # set to stepwise
        
        print(' ### Modelo ARIMA ajustado ### \n')
        print(stepwise_fit.summary())
        print(' ############################# \n')
        
        return stepwise_fit
    
    def VAR(self,data_diff, best_order):
        """
        

        Parameters
        ----------
        data_diff : TYPE
            DESCRIPTION.
        best_order : TYPE
            DESCRIPTION.

        Returns
        -------
        var : TYPE
            DESCRIPTION.
        var_result : TYPE
            DESCRIPTION.

        """
        
        ### FIT FINAL VAR WITH LAG CORRESPONTING TO THE BEST AIC ###

        var = VAR(endog=data_diff.values)
        var_result = var.fit(maxlags=best_order)
        
        print(' ### Modelo VAR ajustado ### \n')
        print(var_result.summary())
        print(' ############################# \n')    
    
        return var, var_result
    
    def adf_test(self,series,title=''):
        """
        Pass in a time series and an optional title, returns an ADF report
        """
        print(f'Augmented Dickey-Fuller Test: {title}')
        result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
        
        labels = ['ADF test statistic','p-value','# lags used','# observations']
        out = pd.Series(result[0:4],index=labels)
    
        for key,val in result[4].items():
            out[f'critical value ({key})']=val
            
        print(out.to_string())          # .to_string() removes the line "dtype: float64"
        
        if result[1] <= 0.10:
            print("Strong evidence against the null hypothesis")
            print("Reject the null hypothesis")
            print("Data has no unit root and is stationary")
        else:
            print("Weak evidence against the null hypothesis")
            print("Fail to reject the null hypothesis")
            print("Data has a unit root and is non-stationary")
            
    def VAR_bestorder(self,data_diff, max_order = 5):
        """
        

        Parameters
        ----------
        data_diff : TYPE
            DESCRIPTION.
        max_order : TYPE, optional
            DESCRIPTION. The default is 10.

        Returns
        -------
        best_order : TYPE
            DESCRIPTION.
        best_aic : TYPE
            DESCRIPTION.

        """
        
        ### FIND BEST VAR ORDER ###

        AIC = {}
        best_aic, best_order = np.inf, 0
        
        for i in range(1,max_order):
            model = VAR(endog=data_diff.values)
            model_result = model.fit(maxlags=i)
            AIC[i] = model_result.aic
            
            if AIC[i] < best_aic:
                best_aic = AIC[i]
                best_order = i
                
        print('BEST ORDER', best_order, 'BEST AIC:', best_aic)
        
        
        ### PLOT AICs ### 

        plt.figure(figsize=(14,5))
        plt.plot(range(len(AIC)), list(AIC.values()))
        plt.plot([best_order-1], [best_aic], marker='o', markersize=8, color="red")
        plt.xticks(range(len(AIC)), range(1,50))
        plt.xlabel('lags'); plt.ylabel('AIC')
        np.set_printoptions(False)
        plt.show()
        
        return best_order, best_aic
    
    
    ### UTILITY FUNCTION FOR RETRIVE VAR PREDICTIONS ###
    def retrive_ARIMA_prediction(self,arima_result, period, steps):
        """
        

        Parameters
        ----------
        var_result : TYPE
            DESCRIPTION.
        period : TYPE
            DESCRIPTION.
        prior : TYPE
            DESCRIPTION.
        prior_init : TYPE
            DESCRIPTION.
        steps : TYPE
            DESCRIPTION.

        Returns
        -------
        final_pred : TYPE
            DESCRIPTION.

        """
        
        pred = arima_result.predict(steps)
        #init = prior_init.loc[0].tail(period).values
        
        if steps > period:
            id_period = list(range(period))*(steps//period)
            id_period = id_period + list(range(steps-len(id_period)))
        else:
            id_period = list(range(steps))
        
        #final_pred = np.zeros((steps, prior.shape[1]))
        final_pred = np.zeros((steps))
        
        for j, (i,p) in enumerate(zip(id_period, pred)):
            
            #final_pred[j] = init[i]+p
            final_pred[j] = p
            #init[i] = init[i]+p    
            
        return final_pred
    
    
    
    ### UTILITY FUNCTION FOR RETRIVE VAR PREDICTIONS ###
    def retrive_VAR_prediction(self,var_result, period, prior, prior_init, steps):
        """
        

        Parameters
        ----------
        var_result : TYPE
            DESCRIPTION.
        period : TYPE
            DESCRIPTION.
        prior : TYPE
            DESCRIPTION.
        prior_init : TYPE
            DESCRIPTION.
        steps : TYPE
            DESCRIPTION.

        Returns
        -------
        final_pred : TYPE
            DESCRIPTION.

        """
        
        pred = var_result.forecast(np.asarray(prior), steps=steps)
        init = prior_init.tail(period).values
        
        if steps > period:
            id_period = list(range(period))*(steps//period)
            id_period = id_period + list(range(steps-len(id_period)))
        else:
            id_period = list(range(steps))
        
        final_pred = np.zeros((steps, prior.shape[1]))
        for j, (i,p) in enumerate(zip(id_period, pred)):
            final_pred[j] = init[i]+p
            init[i] = init[i]+p    
            
        return final_pred
    
        

    