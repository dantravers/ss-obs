# Script to collect statistics from model stats file and concatenate
# Could be alternatively done via bash command

import os
import sys

import pandas as pd

def main():
    """
    Arguments
    ---------
    arg1 : name of folder where the stats files are
    arg2 : filepath of output file to write statistics to.
    """ 

    # input ss_ids.list and output files
    if len(sys.argv) > 2:
        filedir = sys.argv[1]
        statsfile = sys.argv[2]
    else:
        raise(Exception)
 
    totstats = pd.DataFrame([])
    for res_file in os.listdir(filedir):
        if (res_file[-4:] == '.csv') & (res_file[-7:-4] != 'res'):
            print(os.path.join(filedir, res_file))
            temp = pd.read_csv(os.path.join(filedir, res_file), index_col=0, parse_dates=['run'], dayfirst=True)
            totstats = totstats.append(temp, sort=True)
    totstats.to_csv(os.path.join(filedir, statsfile))

if __name__ == "__main__":
	main()