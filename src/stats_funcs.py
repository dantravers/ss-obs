## Statistics functions for Model_run.py

import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
from location_data_utils import apply_weekday

#import matplotlib.pyplot as plt
#import seaborn as sns

def generate_error_stats(result, cap, epex=None, sbsp=None, splits=True):
    """ Function to generate the error statistics based on a dataframe of actual outturn and predicted outturn readings

    Parameters
    ----------
    result : DataFrame
        Expects a DataFrame with columns for 'forecast' and 'outturn'.  It is likely that this is indexed
        by datetime on axis 0, but not required.  If splits=True, it is required to have columns for 'month' and 'hour' also.
    cap : Float
        Capacity of the variable being used.  This is used in normalizing the results to be percentages.
    epex : DataFrame
        Dataframe indexed by datetime containing the epex prices under column label "price".  
        This is the day ahead epex hourly auction settlement prices.  This is used to calculate the price power is hedged at 
        if we assume day ahead hedging.
    sbsp : DataFrame
        DataFrame indexed by datetime containing the balancing market system buy-sell prices under columns "ssp", "sbp".
        We use just "ssp" in calculations to calculate cash-out prices for being in imbalance.
    splits : Boolean
        If False, then statistics are calculated across all periods passed in, and only these total statistics are generated.
        If True, the statistics will be generated for each split of month-hour independently as well as across all periods
        together.  These month-hour statistics can be useful for heatmap analysis of the forecast performance.    
    """

    stat_temp = stats(result, cap, epex, sbsp)
    stat_temp['month'] = np.nan
    stat_temp['hour'] = np.nan
    if splits: 
        no_zeros_result = result.groupby(['month', 'hour']).filter(lambda x: x['outturn'].sum()>0)
        stat_temp = stat_temp.append(no_zeros_result.groupby(['month', 'hour']).\
                                 apply(stats, (cap, epex, sbsp)).reset_index(), ignore_index=True)
    return(stat_temp[['month', 'hour', 'cap_used', 'count', 'MBE', 'MAE', 'RMSE', 'cash_pct', 'lg_over', 'lg_under', 'wMBE', 'wMAE', 'wRMSE', 'hrly_val', 'Rsqd']])

def stats(df, cap, epex=None, sbsp=None):
    """ Function to return statistics suite of errors of forecast-actual.
    All statistics are normalized by capacity.

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
    
    df['diff'] = df['forecast'] - df['outturn']
    df['abs_diff'] = np.abs(df['diff'])
    df['s_diff'] = df['diff'] * df['diff']
    # captures large errors: errors in 95% & 5% of the distribution of outturns.
    lg_over = df[df['diff'] > 1.65 * df.outturn.std()].shape[0] / df.shape[0] * 100 
    lg_under = df[df['diff'] < -1.65 * df.outturn.std()].shape[0] / df.shape[0] * 100 
    count=  df.count()['forecast'], 
    MBE = df.mean()['diff'] / cap * 100, 
    MAE = df.mean()['abs_diff'] / cap * 100, 
    RMSE = (df.mean()['s_diff'] ** 0.5) / cap * 100, 
    wMBE = np.average(df['diff'], weights=df['outturn']) / cap * 100, 
    wMAE = np.average(df['abs_diff'], weights=df['outturn']) / cap * 100,
    wRMSE = np.average(df['s_diff'], weights=df['outturn']) ** 0.5 / cap * 100,
    Rsqd = (1 - (df.std()['diff'] / df.std()['outturn'])**2)
    # statistic on the price impact of forecast errors - this does inner merge on df, so you lose some points -->
    # don't do before other stats!
    if (epex is not None) & (sbsp is not None):
        df = add_financial_cols(df, epex, sbsp)
        cash_loss_pct = df.sum()['cash_out'] / df['hedge'].sum() * 100   #!! THIS IS THE WRONG STATISTIC - REPLACE WITH MEAN OF fcst_loss!!
        hrly_val = (df.sum()['hedge'] + df.sum()['cash_out']) / len(df)
    else:
        print('Required prices passed to x-validation, so not calculating price-based statistics.')# ** myprint, verbosity=3
        cash_loss_pct = np.nan
        hrly_val = np.nan
    # all other statistics
    return(pd.DataFrame( {
                       'cap_used' : cap, 
                       'count': count, 
                       'MBE': MBE, 
                       'MAE': MAE, 
                       'RMSE': RMSE, 
                       'cash_pct' : np.around(cash_loss_pct, decimals=2), 
                       'lg_over' : np.around(lg_over, decimals=2),
                       'lg_under' : np.around(lg_under, decimals=2),
                        'wMBE': wMBE, 
                        'wMAE': wMAE,
                        'wRMSE': wRMSE,
                        'hrly_val' : np.around(hrly_val, decimals=3), 
                        'Rsqd': Rsqd
                        }, 
                        columns = ['cap_used', 'count', 'MBE', 'MAE', 'RMSE', 'cash_pct', 'lg_over', 'lg_under', 'wMBE', 'wMAE', 'wRMSE', 'hrly_val', 'Rsqd'],
                        index=[0]) )

def add_financial_cols(df, epex, sbsp):
    initial_len = len(df)
    df = pd.merge(df, epex, how='inner', left_index=True, right_index=True)
    df = pd.merge(df, sbsp, how='inner', left_index=True, right_index=True)
    if len(df) < initial_len/2:
        print('Prices present for less than half the forecasted points.  Still calcualting statistics.')
    # ** should be myprint, verbosity=2df['hedge'] = df.forecast * df.price / 1000
    df['diff'] = df.forecast - df.outturn
    df['price_diff'] = df.price - df.ssp
    df['perfect'] = df.outturn * df.price / 1000
    df['hedge'] = df.forecast * df.price / 1000
    df['cash_out'] = (df.outturn - df.forecast) * df.ssp / 1000
    df['fcst_loss'] = df['diff'] * df.price_diff / 1000
    # df['value'] = df['hedge']  + df['cash_out'] - replaced old formula with equivalent below
    df['value'] = df.perfect  + df.fcst_loss
    return(df)

def heatmap_summary_stats(run_stats):
    pass
    """    try: 
        total = run_stats[run_stats['month'].isnull() & run_stats['hour'].isnull()]
        print(total[['count', 'MBE', 'MAE', 'RMSE', 'wMBE', 'wMAE', 'wRMSE']])
    except KeyError:
        print('No total statistics in results')
    try:
        month_hour = run_stats[run_stats['month'].notna() & run_stats['hour'].notna()]
        f = plt.figure(figsize = (14,5))
        f.add_subplot(1, 2, 1, title='MBE')
        sns.heatmap(month_hour.pivot(index='month', columns='hour')['MBE'], cmap='BrBG').invert_yaxis()
        f.add_subplot(1, 2, 2, title='MAE')
        sns.heatmap(month_hour.pivot(index='month', columns='hour')['MAE'], cmap='Blues').invert_yaxis()
    except KeyError:
        print('No grouping statistics in results')"""

def scatter_results(results):
    pass
    """# Scatter plot of a dataframe containing columns of outturn and differences
    if 'diff' not in results.columns.values: 
        results['diff'] = results['forecast'] - results['outturn']
    grid = sns.jointplot(x='outturn', y='diff', data=results, kind='reg', ratio=10)
    grid.fig.set_figwidth(16)
    grid.fig.set_figheight(4)"""

def set_month_hour(df):
    """ Function to assign month and hour columns to dataframe based on df containing datetime index
    """
    
    df = df.assign(hour=df.index.hour)
    df = df.assign(month=df.index.shift(-1, freq='h').month)
    return(df)

def value_factor(df, power):
    # calculate value_factor and cv_value_factor values on raw results
    # df is assumed to have columns for datetime, outturn, forecast, price, ssp
    
    value_fac = gen_wtd_avg = time_wtd = fwd_cv_wtd = fwd_cv_fac = np.nan
    df.set_index('datetime', inplace=True)
    if (len(df) >= 30):
        df = df.assign(date=pd.to_datetime(df.index.date))
        # get daily average prices for each date in prce history and merge with hourly detail data
        daily_avg_prices = power.epex.groupby(power.epex.index.floor('d')).mean()
        df = pd.merge(df, daily_avg_prices['price'], how='inner', left_on='date', right_index=True, suffixes=('', '_avg'))
        # calculate the time weighted prices for each site (denominator in value factor)   
        #  where we take into account only the days on which the site generates 
        time_wtd = df.groupby(df.index.floor('d'))['price_avg'].mean().mean()
        # generation weighted prices (numerator in value factor formula)
        gen_wtd_avg = np.average(df.price, weights=df.outturn)
        # create value factor df: 
        value_fac = gen_wtd_avg / time_wtd * 100
        
        #Forward curve weighted price aggregation
        prices = apply_fwd_curve_cols(power.epex)
        df = apply_fwd_curve_cols(df)
        grouped_df = pd.merge(df.groupby(['weekday', 'month', 'hour'])['outturn'].mean().reset_index(),\
                              prices.groupby(['weekday', 'month', 'hour'])['price'].mean().reset_index(), \
                              how='left', on=['weekday', 'month', 'hour'])
        fwd_cv_wtd = np.average(grouped_df['price'], weights=grouped_df['outturn'])
        fwd_cv_fac = fwd_cv_wtd / time_wtd * 100
    else:
        print('Prices present for less than 30.  Not calculating statistics.') # ** should be myprint, verbosity=2
    return( pd.Series([value_fac, time_wtd, gen_wtd_avg, fwd_cv_wtd, fwd_cv_fac], 
                      index=['value_factor', 'time_wtd_prc', 'gen_wtd_prc', 'fwd_cv_wtd_prc', 'fwd_cv_fac'])
          )

def apply_fwd_curve_cols(df):
    df = apply_weekday(df, 'grouped_we', False)
    df = df.assign(month = df.index.month)
    df = df.assign(hour = df.index.hour)
    return(df)