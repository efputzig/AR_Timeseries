#!/usr/bin/env python
""" Create an autoregressive time series.
set_AR_coeff(order): sets coefficients of a wide-sense stationary AR model
make_AR_TS(order,length,sigma,...):Returns autoregressive coefficients and a timeseries

"""
__author__="Elias Putzig"
__version__="1.0"


import numpy as np

def set_AR_coeff(order):
    """ Sets coefficients such that they are wide-sense stationary.
    Condition: roots of polynomial x^p-Sum_[i=0->p]{coeff[i]*x^(p-i-1)}
      lie inside the unit circle (wikipedia: autoregressive_model)
    """
    coeff=np.random.randn(order+1)
    coeff[0]=-1.0
    roots=np.roots(coeff)
    # Modify roots which are >1 using a mask (np.where(condition,1,0))
    roots=roots + np.where(np.absolute(roots)>=1,1,0)* (np.random.rand()/(1.01*np.absolute(roots))-roots)
    coeff=np.real( np.poly(roots) )#*np.random.rand()/1.01 # Convert back
    coeff=-coeff[1:]
    return coeff


def make_AR_TS(order=4, sigma=1.0, length=10000, seed=11):
    """ Returns an autoregressive coefficients and a timeseries
    sigma: the stdev of the white noise in the AR model
    order: the order of the AR model; i.e. the number of coefficients
    length: the number of elements in the series 
    coeff: the coefficients of the AR model
    """
    assert length>2*order, "Choose a length that is greater than the 2*order."
    np.random.seed(seed)
    coeff=set_AR_coeff(order)
    ## Generate the data from the model ##
    tsa=sigma*np.random.randn(2*length) ## starting with 2xlength    
    for n in range(len(tsa)):
        for m in range(len(coeff)):
            if n>m: 
                tsa[n]=tsa[n]+coeff[m]*tsa[n-m-1]
    return coeff, tsa[length:]
    
    
    
if __name__ == '__main__':
    pass    #do nothing - code deleted
