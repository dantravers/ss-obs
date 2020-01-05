# Script to calculate statistics from the hourly results outputs
# Basic script calls the existing statistics calculations, but framework can be easily extended

import os
import sys

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
            tempstats = temp.groupby('run').apply(calcstats, pwr)
            tempstats.index = tempstats.index.droplevel(1)
            resstats = resstats.append(tempstats)
    if len(statsfile) > 0:
        totstats = pd.read_csv(os.path.join(filedir, statsfile), index_col=0, parse_dates=['run'], dayfirst=True)
        resstats = pd.merge(totstats[['model','lags', 'w_id', 'ss_id', 'grouped', 'days_ahead', 'run']].set_index('run', drop=True), \
            resstats, 
            how='right', 
            left_index=True, right_index=True, 
            suffixes=('', '_calc'))
    resstats.to_csv(outfile)

def calcstats(g, pwr):
    cap = g.loc[:, 'outturn'].max()
    df = generate_error_stats(assign_month_hour(g.set_index('datetime')), cap, pwr.epex, pwr.sbsp, splits=False)
    df['ss_id'] = g.loc[:, 'ss_id'].max()
    return(df)

if __name__ == "__main__":
	main()