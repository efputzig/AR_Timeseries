#!/usr/bin/env python
""" Functions for analyzing time series
__main__: Generates plots using the functions for data analysis
          -reads in data from INFILE or generates it from make_AR_TS
          -generates an ACF and plots with error bounds
          -generates PACF and plots with error bounds
          -calculates MS deviation of model from fit and plots for a range of AR model order
Functions:
-AutoCorrFn: measures correlation function
-AR_coefficients: generates specified number of AR coefficients from a correlation function
-PartAutoCorrFn: measures PACF using AR_coefficients 
-AR_MeanSqError: calculate and return mean-squared error in AR model
"""
__author__="Elias Putzig"
__version__="1.0.0"

# Global Variable(s)
READDATAFROMFILE=False
GENERATEDMODELORDER=5
SEED=43323
INFILE='insampledata.csv'
if READDATAFROMFILE:
    TAG="Test"
else:
    TAG="Generated"
ACFPLOTFILE='{}Sample_ACF.png'.format(TAG)
PACFPLOTFILE='{}Sample_PACF.png'.format(TAG)
ERRORPLOTFILE='{}FitError.png'.format(TAG)

# Modules
import numpy as np
import matplotlib.pyplot as plt
import make_AR_TS

def AutoCorrFn(timeseries, maxlags, SubtractMean=False):
    """Measure autocorrelation of 1-d array (time series) for a lag of up to maxlags
    # Note: This return is normalized by the variance 
    """
    ar_L=len(timeseries)
    assert ar_L>maxlags, "Array length {} too short for autocorr-length {}".format(ar_L,maxlags)
    ar_m=timeseries.mean()
    ar_v=timeseries.var()
    cf=np.zeros(shape=(maxlags,))
    cf[0]=ar_v
    if not SubtractMean:
        for i in range(1,maxlags):
            cf[i]=sum(timeseries[i:]*timeseries[:-i])/(ar_L-i)
    else:
        for i in range(1,maxlags):
            cf[i]=sum((timeseries[i:]-ar_m)*(timeseries[:-i]-ar_m))/(ar_L-i)
    return cf/ar_v

def AR_coefficients(Ncoeff, acf):
    """Generates Ncoeff coefficinents of an autoregressive model with
    autocorrelation function acf. Does this by inverting matrix from
    Yule-Walker equations (A).
    Returns sigma2=(noise variance)/(time-series variance), AR coefficients
    """
    assert len(acf)>Ncoeff, "Cannot generate {} coeff. with acf of length {}".format(Ncoeff, len(acf))
    # Construct matrix from Yule-Walker Eqs.
    A=np.zeros((Ncoeff, Ncoeff))
    for i in range(Ncoeff):
        for j in range(Ncoeff):
            A[i][j]=acf[abs(i-j)]
    # Invert the matrix
    Ainv=np.linalg.inv(A)
    # Calculate the coefficients of AR model
    coeff=np.dot(Ainv,acf[1:Ncoeff+1])
    # Calculate the variance
    sigma2=acf[0]-np.dot(coeff,acf[1:Ncoeff+1])
    return sigma2, coeff

def PartAutoCorrFn(timeseries, maxlags):
    """Measure partial autocorrelation function (PACF) for up to maxlags lags
    using AR_coefficients PACF(i)=K_{i,i}
    where K_{i,j} is the i'th coefficient of a model with j total coefficients.
    Note: Levinson–Durbin recursion would be faster.
    """
    assert len(timeseries)>4*maxlags, "{} is too many lags for PACF of timeseries length {}.".format(maxlags,len(timeseries))
    ts_L=len(timeseries)
    ts_m=timeseries.mean()
    ACF=AutoCorrFn(timeseries, maxlags+1)
    PACF=np.zeros(maxlags)
    PACF[0]=1.0
    for k in range(1,maxlags):
        _, coeff = AR_coefficients(k,ACF)
        PACF[k]= coeff[-1]
    return PACF

def AR_MeanSqError(FitCoefficients, timeseries):
    """Calculate and return mean-squared error in AR model
    with coefficients FitCoefficients by comparing with timeseries.
    Returns: mean-squared error, array of errors
    """
    L_fit=len(FitCoefficients)
    L_ts=len(timeseries)
    assert L_ts>4*L_fit, "Cannot measure error to fit with {} coefficients with timeseries of length {}.".format(L_fit,L_ts)
    error=np.zeros(L_ts-L_fit)
    model_prediction=np.zeros(L_ts-L_fit)
    for i in range(L_fit,L_ts):
        model_prediction[i-L_fit]=np.dot(FitCoefficients[::-1],timeseries[i-L_fit:i])
        error[i-L_fit]=timeseries[i]-model_prediction[i-L_fit]
    MSE=np.sum(error*error)/len(error)
    return MSE,error,model_prediction
        
                                                                                                     
    
if __name__ == '__main__':
    """ Generates plots of ACF, PACF, and fitting error. Sections as follows:
    -Read insampledata, or generate data using make_AR_TS,
    -Plot ACF with error bounds,
    -Plot PACF with error bounds,
    -Estimate error predictions for k AR_coefficients, and plot vs k
    """
    if READDATAFROMFILE:
        tsa=np.genfromtxt(INFILE) # tsa = time series array
    else:
        coeff, tsa = make_AR_TS.make_AR_TS(GENERATEDMODELORDER,seed=SEED)
        print("Coefficients: ",coeff)


    ##### Plot ACF #####
    maxlags_acf=200
    acf=AutoCorrFn(tsa,maxlags_acf)
    xints=np.arange(1,maxlags_acf)
    #The Plot (starts from 1 in case acf[0]=1>>acf[i] for i>0)
    plt.plot(xints,acf[1:], 'o', label='Sample ACF(k)')
    plt.xlabel('lag (k)')
    plt.ylabel('ACF')
    if READDATAFROMFILE:
        acfplot_title='Normalized ACF for infile'
    else:
        acfplot_title='Normalized ACF for Model of Order {}'.format(GENERATEDMODELORDER)
    plt.title(acfplot_title)
    # ACF error bounds: using length of 2/(len(time_series)-lags)
    xvals=np.linspace(1,maxlags_acf-1,500)
    yvals=2/np.sqrt(len(tsa)-xvals)
    plt.fill_between(xvals,yvals,-yvals, facecolor='red',alpha=0.15, label='95% Confidence bounds (for null hyp.)')
    plt.legend(loc='upper right')
    plt.savefig(ACFPLOTFILE)
    plt.clf()

    ##### Plot PACF #####
    maxlags_pacf=55
    pacf=PartAutoCorrFn(tsa,maxlags_pacf)
    xints=np.arange(1,maxlags_pacf)
    #The Plot (starts from 1 in case pacf[0]=1>>acf[i] for i>0)
    plt.plot(xints,pacf[1:], 'o', label='Sample PACF(k)')
    plt.xlabel('lag (k)')
    plt.ylabel('PACF')
    if READDATAFROMFILE:
        pacfplot_title='PACF for infile'
    else:
        pacfplot_title='PACF for Model of Order {}'.format(GENERATEDMODELORDER)
    plt.title(pacfplot_title)
    # PACF error bounds: using length of 2/(len(time_series)-lags)
    xvals=np.linspace(1,maxlags_pacf-1,500)
    yvals=2/np.sqrt(len(tsa)-xvals)
    plt.fill_between(xvals,yvals,-yvals, facecolor='red',alpha=0.15, label='95% Confidence bounds (for null hyp.)')
    plt.legend(loc='upper right')
    plt.savefig(PACFPLOTFILE)
    plt.clf()

    ##### Estimate Error Predictions AR_coefficients #####
    FittingSet=tsa[:-len(tsa)//10]
    TestingSet=tsa[-len(tsa)//10:]
    sigma2_fs=FittingSet.var()
    # For fits with two different choices of numbers of coefficients
    Ncoeff_1=2
    Ncoeff_2=50
    ACF_fs=AutoCorrFn(FittingSet,max(Ncoeff_1,Ncoeff_2)+1)
    s2_1, K_1 = AR_coefficients(Ncoeff_1, ACF_fs)
    s2_2, K_2 = AR_coefficients(Ncoeff_2, ACF_fs)
    MSE_1,_,_= AR_MeanSqError(K_1, np.append(FittingSet[-Ncoeff_1:],TestingSet))
    MSE_2,_,_= AR_MeanSqError(K_2, np.append(FittingSet[-Ncoeff_2:],TestingSet))
    print("MSE from fits of length {}:{}, and {},{}".format(Ncoeff_1,MSE_1,Ncoeff_2,MSE_2))
    # Plot for AR fit error and noise amplitude prediction vs number of coefficients
    MaxCoeff=20
    ACF_fs=AutoCorrFn(FittingSet,MaxCoeff+2)
    MSError=np.zeros(MaxCoeff+1)
    PredictedNoiseSigma2=np.zeros(MaxCoeff+1)
    MSError[0]=TestingSet.var()
    PredictedNoiseSigma2[0]=sigma2_fs
    for i in range(1,MaxCoeff+1):
        s2_i,K_i=AR_coefficients(i, ACF_fs)
        MSError_i,_,_=AR_MeanSqError(K_i, np.append(FittingSet[-i:],TestingSet))
        MSError[i]=MSError_i
        PredictedNoiseSigma2[i]=s2_i*sigma2_fs
        if not READDATAFROMFILE:
            if i==GENERATEDMODELORDER:
                print("Measured Coefficients: ",K_i)
    # Make the plot
    plt.plot(MSError, label='Mean-Squared Error')
    plt.plot(PredictedNoiseSigma2, label='Predicted Noise Variance')
    plt.xlabel('Order of AR Fit')
    plt.ylabel('Error')
    if READDATAFROMFILE:
        errorplot_title='MSE and Noise Amplitude'
    else:
        errorplot_title='MSE and Noise Amplitude for Model of Order {}'.format(GENERATEDMODELORDER)
    plt.title(errorplot_title)
    plt.legend(loc='center right')
    plt.savefig(ERRORPLOTFILE)
    plt.clf()
    
    
    
