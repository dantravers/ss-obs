# Script to calculate statistics from the hourly results outputs
# Basic script calls the existing statistics calculations, but framework can be easily extended

import os
import sys

import numpy as np
import pandas as pd

import power as pw
from stats_funcs import generate_error_stats
from model_run import assign_month_hour

def main():
    """
    Arguments
    ---------
    arg1 : name of folder where the results files are
    arg2 : filepath of output file to write calculated statistics to.
    arg3 : name of existing statistics file containing metadata (optional).  Assume it is in same directory as results.
    """ 

    # input ss_ids.list and output files
    if len(sys.argv) > 2:
        filedir = sys.argv[1]
        outfile = sys.argv[2]
    else:
        raise(Exception)
    if len(sys.argv) > 3:
        statsfile = sys.argv[3]
    else:
        statsfile = ''

    pwr = pw.Power(2)
    resstats = pd.DataFrame([])
    for res_file in os.listdir(filedir):
        if res_file[-7:] == 'res.csv':
            print(os.path.join(filedir, res_file))
            temp = pd.read_csv(os.path.join(filedir, res_file), index_col=0, parse_dates=['run', 'datetime'], dayfirst=True)
            tempstats = temp.groupby(['run', 'ss_id']).apply(calcstats, pwr)
            tempstats.index = tempstats.index.droplevel(1)
            resstats = resstats.append(tempstats)
    resstats.reset_index(inplace=True)
    if len(statsfile) > 0:
        totstats = pd.read_csv(os.path.join(filedir, statsfile), index_col=0, parse_dates=['run'], dayfirst=True)
        resstats = pd.merge(totstats[['model','lags', 'w_id', 'ss_id', 'days_ahead', 'run']], \
            resstats, 
            how='right', 
            on=['run', 'ss_id'],
            #left_index=True, right_index=True, 
            suffixes=('', '_calc')
        )
    resstats.to_csv(outfile)

def calcstats(g, pwr):
    cap = g.loc[:, 'outturn'].max()
    df = generate_outturn_avg_max(assign_month_hour(g.set_index('datetime')), cap, pwr.epex, pwr.sbsp)
    df['ss_id'] = g.loc[:, 'ss_id'].max() # this is taking the ss_id and adding to results.  Should be constant across g.
    return(df)

def generate_outturn_avg_max(df, cap, epex=None, sbsp=None):
    """ Example calculates max and avg of outturn. 

    Parameters
    ----------
    df : DataFrame
        A dataframe containing a column 'forecast' and a column 'outturn'.
        Usually indexed by datetime, but not obligatory.
    """

    out_avg = df.outturn.mean()
    out_max = df.outturn.max()
    
    return(pd.DataFrame( {
                        'outturn_avg' : np.around(out_avg, decimals=3),
                        'outturn_max' : np.around(out_max, decimals=3)
                        }, 
                        columns = ['outturn_avg', 'outturn_max'],
                        index=[0]) )

def generate_new_stat_template(df, cap, epex=None, sbsp=None):
    """ New simple statistic calculation.  
    Intention is that this function is overwritten / upgraded as I develop new statistics.

    Parameters
    ----------
    df : DataFrame
        A dataframe containing a column 'forecast' and a column 'outturn'.
        Usually indexed by datetime, but not obligatory.
    cap : Float
        Capacity used in normalization. 
    epex : DataFrame
        Dataframe indexed by datetime containing the epex prices under column label "price".  
        This is the day ahead epex hourly auction settlement prices.  This is used to calculate the price power is hedged at 
        if we assume day ahead hedging (very simplified).
    sbsp : DataFrame
        DataFrame indexed by datetime containing the balancing market system buy-sell prices under columns "ssp", "sbp".
        We use just "ssp" in calculations to calculate cash-out prices for being in imbalance.
    """

    if (epex is not None) & (sbsp is not None):
        initial_len = len(df)
        df = pd.merge(df, epex, how='inner', left_index=True, right_index=True)
        if len(df) < initial_len/2:
            print('Prices present for less than half the forecasted points.  Still calcualting statistics.') # ** should be myprint, verbosity=2
        df['hedge'] = df.forecast * df.price / 1000
        val_kWh = df['hedge'].sum() / df['outturn'].sum() #sign -ve for losses in generation, +ve for losses in load.
    else:
        print('Required prices NOT passed to statistics calculation, so not calculating price-based statistics.')# ** myprint, verbosity=3
        val_kWh = np.nan
    # all other statistics
    return(pd.DataFrame( {
                        'val_kWh' : np.around(val_kWh, decimals=3)
                        }, 
                        columns = ['val_kWh'],
                        index=[0]) )

if __name__ == "__main__":
	main()