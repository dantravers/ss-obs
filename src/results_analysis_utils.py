import os
import numpy as np
import pandas as pd
import power as pw
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from stats_funcs import add_financial_cols

WINHOME = '/mnt/c/users/dantravers'

class FcstDetails:
    def __init__(self, ss_ids, STATS="/GoogleDrive/Projects/Poster Map/fcst_runs/", DATA="/Documents/dbs/fcst_runs/2020_07/"):
        self.STATS_DIR = WINHOME + STATS
        self.DATA_DIR = WINHOME + DATA
        self.catalogue = pd.read_csv(os.path.join(self.STATS_DIR, "2020_07_1/catalogue_from_detailed_results.csv"))
        self.data = get_ss_id_data_list(ss_ids, self.DATA_DIR, self.catalogue)
        self.res = pd.read_csv(os.path.join(self.STATS_DIR, '2020_07/results_new_w_fcst_loss.csv'), dayfirst=True, index_col=0)
        self.res['value_kwp'] = (self.res.perfect + self.res.fcst_loss) / self.res.kWp
        self.res['fcst_loss_kwp'] = self.res.fcst_loss / self.res.kWp

    def get(self, ss_id):
        return(self.data.loc[:, ss_id])

    def get_rich(self, ss_id):
        df = self.get(ss_id).copy()
        df['month'] = df.index.month
        df['year'] = df.index.year
        return(df)
    
    def get_meta(self, ss_ids):
        return(self.res[self.res.ss_id.isin(ss_ids)][['ss_id', 'count', 'no_days', 'hrly_val',
                      'lat', 'lon', 'orient', 'tilt', 'kWp', 'yield', 'MAEa', 'MAE2', 'perfect', 
                                                'fcst_loss', 'fcst_loss_pct', 'value_kwp', 'fcst_loss_kwp']])

    def timeseries_graph(self, ss_id):
        return(plot_timeseries(self.get(ss_id), ss_id))

    def quantile_graphs(self, ss_id, q=50):
        ss_data = self.get(ss_id).copy()
        ss_data['q_price'] = pd.qcut(ss_data.price_diff, q)
        ss_data['q_error'] = pd.qcut(ss_data['diff'], q)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
        ax1 = make_quantile_axes(ax1, ss_data.groupby('q_error')[['fcst_loss', 'diff']].mean()\
                                 , q, ss_data.perfect.mean()/100)
        ax1.set_title('Site {} - Forecast losses in {} quantiles of forecast volume error'.format(ss_id, q))
        ax2 = make_quantile_axes(ax2, ss_data.groupby('q_price')[['fcst_loss', 'price_diff']].mean()\
                                 , q, ss_data.perfect.mean()/100)
        ax2.set_title('Site {} - Forecast losses in {} quantiles of price deltas'.format(ss_id, q))
        fig.show()

    def list_ids(self):
        return(self.data.columns.unique(0).values.tolist())

def get_ss_id_data_list(ss_ids, DATA=WINHOME+"/Documents/dbs/fcst_runs/2020_07/", catalogue=pd.DataFrame([])):
    """ Function to return the df containing the forecast and actuals from the raw csv restults
    ss_ids is a list containing the ids requested."""
    power = pw.Power()
    result = pd.DataFrame([])
    for ss_id in ss_ids:
        file_id = catalogue[catalogue.ss_id==ss_id].iloc[0, 2]
        resfile = "{}_results_res1.csv".format(file_id)
        df = pd.read_csv(os.path.join(DATA, resfile), parse_dates=['run'], dayfirst=True, index_col=0)
        df = df.set_index('datetime')
        df = df[df.ss_id==ss_id]
        df = add_financial_cols(df, power.epex, power.sbsp)
        df = df.drop(['volume', 'sbp'], axis=1)
        result = result.append(df)
    result.set_index('ss_id', append=True, inplace=True)
    result = result.unstack(1).swaplevel(0, 1, 1)
    return(result)

def plot_timeseries(ss_data, ss_id):
    # function to return a figure object with a timeseries plot in plot.ly
    price_diff = go.Scatter(x=ss_data.index, y = ss_data['price_diff'].values, name='Price difference')
    vol_diff = go.Scatter(x=ss_data.index, y = ss_data['diff'].values, name='Fcst - actual')
    loss = go.Scatter(x=ss_data.index, y=ss_data['fcst_loss'], name='Loss')
    layout = go.Layout(title="Site {}".format(ss_id), 
                       xaxis=dict(title='Date'), 
                       yaxis={'title' : 'kWh', 'color':'red'}, 
                      yaxis2={'title' : 'kWh2', 'color' : 'blue', 'overlaying' : 'y', 'side' : 'right'})
    fig = go.Figure(data = [price_diff, vol_diff, loss], layout=layout)
    return(fig)

def make_quantile_axes(a, df, q, scale=1):
    # pass in df with row for each decile / quantile bucket.  
    # First col is fcst_loss figures, second is mean of the quantile buckets
    # q is the number of quantile buckets 
    # scale allows the y-axis to be scaled to be expressed in % of "perfect" forecast 
    #   to do this, set scale=perfectforecast/100
    axis_dict = {'diff':'Forecast error - bucket mean', 'price_diff':'Price delta - bucket mean'}
    names = ['{:.0f}'.format(x) for x in df.iloc[:, 1].tolist()]
    top = (df['fcst_loss']/(q*scale)).tolist()
    bottom = np.roll(np.cumsum((df['fcst_loss']/(q*scale)).values), 1)
    bottom[0] = 0
    color = ['Cyan' if x>0 else 'Blue' for x in top]
    a.bar( names, top, width=1, bottom=bottom, color=color)
    a.set_xlabel(axis_dict[df.columns[1]])
    a.xaxis.set_major_locator(plt.MaxNLocator(min(q, 18)))
    return(a)


