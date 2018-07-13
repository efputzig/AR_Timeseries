import datetime
import pandas_datareader as pdr
import pandas as pd
# Temporary fix to broken yahoo API
# from https://github.com/ranaroussi/fix-yahoo-finance
import fix_yahoo_finance as yf                   
yf.pdr_override()  

aapl=yf.download('AAPL',
                  start=datetime.datetime(2006,10,1),
                  end=datetime.datetime(2012,1,1))

# Write data to a csv
aapl.to_csv('aapl_ohlc.csv')

# Read it back in
df = pd.read_csv('aapl_ohlc.csv', header=0, index_col='Date', parse_dates=True)
