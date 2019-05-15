## Statistics functions for Model_run.py

import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_error_stats(result, cap, splits=True):
    """ Function to generate the error statistics based on a dataframe of actual outturn and predicted outturn readings

    Parameters
    ----------
    result : DataFrame
        Expects a DataFrame with columns for 'forecast' and 'outturn'.  It is likely that this is indexed
        by datetime on axis 0, but not required.  If splits=True, it is required to have columns for 'month' and 'hour' also.
    cap : Float
        Capacity of the variable being used.  This is used in normalizing the results to be percentages.
    splits : Boolean
        If False, then statistics are calculated across all periods passed in, and only these total statistics are generated.
        If True, the statistics will be generated for each split of month-hour independently as well as across all periods
        together.  These month-hour statistics can be useful for heatmap analysis of the forecast performance.    
    """

    stat_temp = stats(result, cap)
    stat_temp['month'] = np.nan
    stat_temp['hour'] = np.nan
    if splits: 
        no_zeros_result = result.groupby(['month', 'hour']).filter(lambda x: x['outturn'].sum()>0)
        stat_temp = stat_temp.append(no_zeros_result.groupby(['month', 'hour']).\
                                 apply(stats, cap=cap).reset_index(), ignore_index=True)
    return(stat_temp[['month', 'hour', 'count', 'MBE', 'MAE', 'RMSE', 'wMBE', 'wMAE', 'wRMSE', 'Rsqd']])

def stats(df, cap):
    """ Function to return statistics suite of errors of forecast-actual.
    All statistics are normalized by capacity.

    Parameters
    ----------
    df : DataFrame
        A dataframe containing a column 'forecast' and a column 'outturn'.
        Usually indexed by datetime, but not obligatory.
    cap : Float
        Capacity used in normalization. 
    """
    
    df['diff'] = df['forecast'] - df['outturn']
    df['abs_diff'] = np.abs(df['diff'])
    df['s_diff'] = df['diff'] * df['diff']
    return(pd.DataFrame( {
                       'count': df.count()['forecast'], 
                       'MBE': df.mean()['diff'] / cap * 100, 
                       'MAE': df.mean()['abs_diff'] / cap * 100, 
                       'RMSE': (df.mean()['s_diff'] ** 0.5) / cap * 100, 
                        'wMBE': np.average(df['diff'], weights=df['outturn']) / cap * 100, 
                        'wMAE': np.average(df['abs_diff'], weights=df['outturn']) / cap * 100,
                        'wRMSE': np.average(df['s_diff'], weights=df['outturn']) ** 0.5 / cap * 100,
                        'Rsqd': (1 - (df.std()['diff'] / df.std()['outturn'])**2)
                      }, 
                    columns = ['count', 'MBE', 'MAE', 'RMSE', 'wMBE', 'wMAE', 'wRMSE', 'Rsqd'],
                    index=[0]) )

def heatmap_summary_stats(run_stats):
    try: 
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
        print('No grouping statistics in results')

def scatter_results(results):
    # Scatter plot of a dataframe containing columns of outturn and differences
    if 'diff' not in results.columns.values: 
        results['diff'] = results['forecast'] - results['outturn']
    grid = sns.jointplot(x='outturn', y='diff', data=results, kind='reg', ratio=10)
    grid.fig.set_figwidth(16)
    grid.fig.set_figheight(4)

def bar_compare(data1, data2, hue_selection, title1, title2):
    #Bar chart comparing two sets of statistics from multiple runs, passed in long form
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,5), sharey=True)
    sns.barplot(x='statistic', y='value', hue=hue_selection, data=data1, ax=ax1)
    sns.barplot(x='statistic', y='value', hue=hue_selection, data=data2, ax=ax2)
    ax1.set_title(title1)
    ax2.set_title(title2)