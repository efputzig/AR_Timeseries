#!/Usr/bin/env python
""" Make plots of a time series to check its behavior
Output: A file (currently Check_Stationary) with a plots
Plots: mean with sem, variance, and 

"""
__author__="Elias Putzig"
__version__="0.0.1"

# Global Variables
FILENAME="Check_Stationary.png"

# Modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import TimeSeriesStats as TSS                     


def check_stationary(arr, M, AutocorrL=40, filename="Check_Stationary.png"):
    # Break 1-d array into M pieces, and plot ave, var, autocorr up to lag of AutocorrL
    assert M>1 and isinstance(M,int), "M must be a integer, greater than 1. (M={})".format(M)
    assert isinstance(arr,np.ndarray), "arr must be a numpy array."
    L=len(arr)//M # length of the pieces
    # Calculate average, variance, and error
    if (len(arr)%M==0):
        ave=np.mean(arr.reshape(M,L), axis=1)
        var=np.var(arr.reshape(M,L), axis=1)        
    else:
        # Leave off the end of arr to make even pieces
        ave=np.mean(arr[:-(len(arr)%M)].reshape(M,L), axis=1)
        var=np.var(arr[:-(len(arr)%M)].reshape(M,L), axis=1)
    error=np.sqrt(var/L)
    # Autocorrelations of 2-3 pieces
    ac1=TSS.AutoCorrFn(arr[0:L],AutocorrL)
    ac2=TSS.AutoCorrFn(arr[(M-1)*L:M*L],AutocorrL)
    if (M>2): ac3=TSS.AutoCorrFn(arr[(M//2)*L:(M//2+1)*L],AutocorrL)
    # Make the plots on a grid: ave, var \ autocorr
    plt.clf()
    fig=plt.figure()
    gs=gridspec.GridSpec(2,2)
    ax1=plt.subplot(gs[0,0])
    ax2=plt.subplot(gs[0,1])
    ax3=plt.subplot(gs[1,:])
    # Average
    ax1.errorbar(np.arange(M), ave, yerr=error, fmt='o')
    ax1.set_title('Average and SEM of {} Segments'.format(M))
    ax1.set_ylabel('Seg. Average')
    ax1.set_xlabel('Segment')
    # Variance
    ax2.plot(np.arange(M), var, 'bs')
    ax2.set_title('Variance of Segments')
    ax2.set_ylabel('Seg. Variance')
    ax2.set_xlabel('Segment')
    # Autocorrelation
    ac_lag=np.arange(1,AutocorrL)
    ax3.plot(ac_lag,ac1[1:],label='Seg. 1')
    ax3.plot(ac_lag,ac2[1:],label='Seg. {}'.format(M))
    if (M>2): ax3.plot(ac_lag,ac3[1:],label='Seg. {}'.format(M//2+1))
    ax3.set_title("Autocorrelation Fns of {} Segments".format(min(3,M)))
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels,loc='upper right')
    ax3.set_xlabel('lag')
    ax3.set_ylabel('ACF')
    # Make sure titles don't overlap
    gs.tight_layout(fig)
    plt.savefig(FILENAME)
    plt.close(fig)
    
    
if __name__ == '__main__':
    # Read insampledata and run 'check_stationary' 
    tsa=np.genfromtxt('insampledata.csv')
    check_stationary(tsa, 5)
    
    

    
    
