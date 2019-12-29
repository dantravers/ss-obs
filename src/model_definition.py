## Class ModelDefinition
# Dan Travers
# 25/2/19

import datetime
import pytz
import pandas as pd

class ModelDefinition:
    """ Class for storing the parameters of the ML model and cross-validation.

    Parameters
    ----------
    ml_model : str
        The string representing the ml model to be used.  E.g. linear_r, g_boost, etc.
        See the function model_predict for the methods supported.
    grouped_by : list
        A list of the parameters (features) to be grouped by when creating the model runs.
        Each group in trained and run independently of all other groups.
    no_folds : int
        Number of folds in k-fold cross-validation. If is equal to zero, then a leave-one-out
        validation is performed.
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
    """

    def __init__(self, ml_model='linear_r', grouped_by=[], no_folds=0, \
                max_runs=0, shuffle=True, text='', **kwargs):
        self.ml_model = ml_model
        self.grouped_by = grouped_by
        self.no_folds = no_folds
        self.max_runs = max_runs
        self.shuffle = shuffle
        self.text = text 
        self.kwargs = kwargs

    def get_parameters(self):
        temp = pd.DataFrame({
            'ml_model' : self.ml_model, 
            'grouped_by' : str(self.grouped_by), 
            'no_folds' : self.no_folds, 
            'max_runs' : self.max_runs, 
            'shuffle' : self.shuffle, 
            'text' : self.text,
            'model_params' : str(self.kwargs)}, 
            index=[0])
        return temp