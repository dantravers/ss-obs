## Class ModelDefinition
# Dan Travers
# 25/2/19

import datetime
import pytz
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

class ModelDefinition:
    """ Class for storing the parameters of the ML model and cross-validation.

    Parameters
    ----------
    ml_model : str
        The string representing the ml model to be used.  E.g. linear_r, g_boost, etc.
        See the function model_predict for the methods supported.
    grouped_by : list
        A list of the parameters (features) to be grouped by when creating the model runs.
        Each group in modelled, trained and predicted independently of all other groups.
    no_folds : int
        Number of folds in k-fold cross-validation. If is equal to zero, then a leave-one-out
        validation is performed.
    cross_val_grp : str
        Parameter to pass into the K-fold validation grouping. Acceptable values: 
            dayofyear, week, month.
        If string is empty, then simple k-fold validation is used.
        This is designed to ensure we don't have information leakage from test set: if we train on hours adjacent in
        hour and day-of-year, we could leak quite badly.  Suggest to use week, or day-of-year.  
    max_runs : int
        Limits the maximum number of runs performed in the cross-validation.  This is to allow 
        performance to be capped at an acceptable level.  The default of zero means there is no 
        max enforced.
    shuffle : Boolean
        Indicates if data is shuffled in cross-validation.
    text : Str
        String to display to describe model parameters.
    kwargs : kwargs
        Model parameters to be passed through to each ML model when it is called.  These are based 
        off of teh scikit learn (or other) model parameters.

    Attributes
    ----------
    model : obj
        The scikit learn estimator with parameters set.
    """

    def __init__(self, ml_model='linear_r', grouped_by=[], no_folds=0, cross_val_grp='week', \
                max_runs=0, shuffle=True, text='', **kwargs):
        self.ml_model = ml_model
        self.grouped_by = grouped_by
        self.no_folds = no_folds
        self.cross_val_grp = cross_val_grp
        self.max_runs = max_runs
        self.shuffle = shuffle
        self.text = text 
        self.kwargs = kwargs

        self.set_model()

    def get_parameters(self):
        temp = pd.DataFrame({
            'ml_model' : self.ml_model, 
            'grouped_by' : str(self.grouped_by), 
            'no_folds' : self.no_folds, 
            'cross_val_grp' : self.cross_val_grp,
            'max_runs' : self.max_runs, 
            'shuffle' : self.shuffle, 
            'text' : self.text,
            'model_params' : str(self.kwargs)}, 
            index=[0])
        return temp

    def set_model(self):
        """ Function to set the appropriate ML model on model definition class.

        Parameters
        ----------
        None
        """
        if self.ml_model == 'linear_r': 
            model = LinearRegression()
        elif self.ml_model == 'linear_poly':
            model = Pipeline([('poly', PolynomialFeatures(**self.kwargs)), ('linear', LinearRegression(fit_intercept=False))])
        elif self.ml_model == 'random_f': 
            model = RandomForestRegressor(**self.kwargs)
        elif self.ml_model == 'tree': 
            model = DecisionTreeRegressor(**self.kwargs)
        elif self.ml_model == 'g_boost':
            model = GradientBoostingRegressor(**self.kwargs)
        elif self.ml_model == 'xg_boost':
            model=xgb.XGBRegressor(**self.kwargs)
        elif self.ml_model == 'average': 
            # this is a model created as a benchmark for load modelling - it takes just the average of the values across each grouping (E.g. month-hour-dayofweek)
            # Doesn't work now that I've refactored code.  
            # Need to special case this.
            print("Error: Average method no longer supported by refactoring Dec 19.")
        else:
            print('ERROR: Unsupported Machine Learning Model')
        self.model = model

    def cross_val_grp_labels(self, df):
        if self.cross_val_grp == 'dayofyear':
            grouping = pd.DataFrame(index=df.index).assign(grp=df.index.dayofyear)
        elif self.cross_val_grp == 'week':
            grouping = pd.DataFrame(index=df.index).assign(grp=pd.Int64Index(df.index.isocalendar().week))
        elif self.cross_val_grp == 'month':
            grouping = pd.DataFrame(index=df.index).assign(grp=df.index.month)
        elif self.cross_val_grp == 'year':
            grouping = pd.DataFrame(index=df.index).assign(grp=df.index.year)
        else:
            print("Grouping of cross-validation isn't supported in model_definition.")
        return(grouping)