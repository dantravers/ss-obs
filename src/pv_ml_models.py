## Different ML Cross-Validation and processing functions
# functions will run model and cross-validate (down to leave-one-out) on our dataset

# Dan Travers
# 24/9/18

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, KFold, GroupKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from location_data_utils import apply_weekday
#import matplotlib.pyplot as plt

def coef_lr(X, y):
    # generate the coef of the linear regression model trained on the entire dataset
    coef = pd.DataFrame(X.columns.tolist(), columns=['feature'])
    lr = LinearRegression().fit(X,y)
    coef['lr_coef'] = lr.coef_
    coef['std'] = coef.apply(lambda x: np.std(X[x['feature']]), axis=1)
    coef['importance'] = abs(coef['lr_coef'] * coef['std'])
    coef['importance'] = coef['importance'] / coef['importance'].sum() * 100
    coeft = coef.transpose()
    coeft.columns = coeft.iloc[0, :]
    return(coeft.loc['importance', :])

def cross_validate(X, y, model_def):
    """ Function to cross-validate on a set of data.

    The cross-validation is performed by splitting on k-fold cross-validation (leave-
    one-out validation if number of k-folds==0).  The results from each k-fold are knitted
    together as the result returned.

    Parameters
    ----------
    X : DataFrame
    The set of features.

    y : DataFrame
    Multiple column dataframe containing the target, actual and month, hour fields.

    model_def : obj:ModelDefinition
    An instance of the ModelDefinition class.  This is a lightweight class containing the 
    ML model to be employed, parameters of the model and parameters of the cross-validation.
    """

    #prepare number of folds and any data limitation
    if model_def.no_folds == 0: 
        n_folds = X.shape[0]
    else:
        n_folds = min(X.shape[0], model_def.no_folds)
    if n_folds > 1:
        if model_def.cross_val_grp == '': 
            kf = KFold(n_splits=n_folds, shuffle=model_def.shuffle, random_state=0)
            arr = cross_val_predict(model_def.model, X, y, cv=kf)        
            raw_predict = pd.DataFrame({'forecast': arr,
                                    'outturn': y},
                                    index=y.index)
        else:
            if len(np.unique(model_def.cross_val_grp_labels(X).values)) >= model_def.no_folds:
                kf = GroupKFold(n_folds)
                arr = cross_val_predict(model_def.model, X.values, y.values, groups=model_def.cross_val_grp_labels(X).values, cv=kf)
                raw_predict = pd.DataFrame({'forecast': arr,
                                        'outturn': y},
                                        index=y.index)
            else:
                print('Skipping group as not enough group-kf splits present.')
                raw_predict = pd.DataFrame([])
        return(raw_predict)
    else:
        print('One grouping has only one element, so skipping')
        return(pd.DataFrame([]))

def cross_validate_grouped(X, y, model_def):
    """ Function to cross-validate, running each grouping in turn.

    The groupings could be month-hour (or by regime for example), so that the fitting & cross-validate is performed on each 
    month-hour combination independently.  I.e. the model is cross-validated on each group
    independently and the total cross-validated prediction is knitted together across all
    groups and returned as the result.

    Parameters
    ----------
    X : DataFrame
    The set of features.

    y : DataFrame
    Single column dataframe containing the target.

    model_def : obj:ModelDefinition
    An instance of the ModelDefinition class.  This is a lightweight class containing the 
    ML model to be employed, parameters of the model and parameters of the cross-validation.
    """

    if model_def.grouped_by:
        if 'hour' in model_def.grouped_by:
            X = X.assign(hour=X.index.hour)
        if 'month' in model_def.grouped_by:
            X = X.assign(month=X.index.month)
        if 'weekday_grouped' in model_def.grouped_by:
            X = apply_weekday(X, 'grouped', False)
            X.rename(columns={'weekday' : 'weekday_grouped'}, inplace=True)
        if 'weekday_individual' in model_def.grouped_by:
            X = apply_weekday(X, 'individual', False)
            X.rename(columns={'weekday' : 'weekday_individual'}, inplace=True)
        if 'weekday_holiday_individual' in model_def.grouped_by:
            X = apply_weekday(X, 'week_holiday_individual', False)
            X.rename(columns={'weekday' : 'weekday_holiday_individual'}, inplace=True)
        if 'holiday_individual' in model_def.grouped_by:
            X = apply_weekday(X, 'holiday_individual', False)
            X.rename(columns={'weekday' : 'holiday_individual'}, inplace=True)
        grouped = X.groupby(model_def.grouped_by)
        group_predict = pd.DataFrame([])
        for _, groupg in grouped:
            group_predict = group_predict.append(cross_validate(groupg, y.loc[groupg.index], model_def))
    else: 
        group_predict = cross_validate(X, y, model_def)
    return(group_predict)

"""def graph_feature_importance(x_train, model):
    n_features = x_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x_train.columns.values)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.show()"""