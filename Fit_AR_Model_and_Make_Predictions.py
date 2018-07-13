#!/usr/bin/env python
""" Script fits AR(35) model to `insampledata.csv', 
then uses the model to make predictions for the next 
element of time series in `outsampledata.csv
"""
__author__="Elias Putzig"
__version__="0.0.1"

import numpy as np
import TimeSeriesStats as TSS

MODELORDER=35

##### Fit the model to the time series data #####
#tsa -> time series array
tsa=np.genfromtxt("insampledata.csv")
acf=TSS.AutoCorrFn(tsa,MODELORDER+2)
# Get noise variance and coefficients for an AR(MODELORDER) fit
sigma2_t,K_coeff = TSS.AR_coefficients(MODELORDER,acf)
sigma2_t *= tsa.var()
print( "Noise amplitude predicted by the AR({}) model: {}".format(MODELORDER,sigma2_t) )

##### Make predictions for outsampledata #####
# Read data from file, and arange so that we can iterate quickly (.T.copy())
osa=np.loadtxt(open("outsampledata.csv", "rb"), delimiter=",", skiprows=1).T.copy()
predictions=np.zeros(osa.shape[0])
for i in range(osa.shape[0]):
    # Reverse the coefficients, and dot with the end of the osa[i]
    # prediction[i]=Expectation{osa[i,100]}=K_coeff[0]*osa[i,99]+K_coeff[1]*osa[i,98]+...
    predictions[i]=np.dot(K_coeff[::-1],osa[i,-MODELORDER:])
np.savetxt("predicteddata.csv",predictions)

                          
