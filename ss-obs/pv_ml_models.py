## Different ML Cross-Validation and processing functions
# functions will run model and cross-validate (down to leave-one-out) on our dataset

# Dan Travers
# 24/9/18

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

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

def coef_lr_grouped(X, y, groups = []):
## function which generates coefficients by groups of data in a list E.g. month, hour, season, regime
## this function now allows ungrouped runs by passing an empty group, so meaning it can handle all linreg requests.
    group_coef = pd.DataFrame([])
    if groups: 
        grouped = X.groupby(groups)
        for table, groupg in grouped:
            yg = y.loc[groupg.index]
            temp_coef = coef_lr(groupg, yg)
            temp_coef['month'] = groupg.month.max()
            temp_coef['hour'] = groupg.hour.max()
            group_coef = group_coef.append(temp_coef)
    else:
        group_coef = group_coef.append(coef_lr(X, y))
    return(group_coef)   

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
    Single column dataframe containing the target.

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
        max_rows = X.shape[0] if model_def.max_runs == 0 else n_folds * model_def.max_rows_in_fold
        raw_predict = pd.DataFrame([])
        kf = KFold(n_splits=n_folds, shuffle=model_def.shuffle, random_state=0)
        #loop through cross-validation folds, testing each fold and gathering data
        for train_i, test_i in kf.split(X.iloc[:max_rows,]):
            predict = model_predict(X.iloc[train_i], y.iloc[train_i], X.iloc[test_i], model_def)
            temp = pd.DataFrame({'month': X.iloc[test_i].index.month, 
                                'hour': X.iloc[test_i].index.hour,
                                'forecast': predict,
                                'outturn': y.iloc[test_i]})
            raw_predict = raw_predict.append(temp)

        ## removed R^2 scores, as for the loo validation, they are all zero, so not informative
        ## essentially they tend to 0 as the number of folds increases to no_samples
        return(raw_predict)
    else:
        print('One grouping has only one element, so skipping')
        return(pd.DataFrame([]))

def cross_validate_grouped(X, y, model_def):
    """ Function to cross-validate, running each grouping in turn.

    The groupings could be month-hour (or by regime for example), so that the cross-validate is performed on each 
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
        grouped = X.groupby(model_def.grouped_by)
        group_predict = pd.DataFrame([])
        for table, groupg in grouped:
            yg = y.loc[groupg.index]
            group_predict = group_predict.append(cross_validate(groupg, yg, model_def))
    else: 
        group_predict = cross_validate(X, y, model_def)
    return(group_predict)

def model_predict(x_train, y_train, x_test, model_def, graph=False):
    """ Function to call the appropriate ML algorithm, fit and predict on train and test data respectively.

    Parameters
    ----------
    x_train : DataFrame
        The set of features to be used in training.
    y_train : DataFrame
        Single column dataframe containing the target to train against.
    x_test : DataFrame
        Set of features to be used to test the fitted model.
    model_def : obj:ModelDefinition
        An instance of the ModelDefinition class.  This is a lightweight class containing the 
        ML model to be employed and parameters of the model (as kwargs).
    graph : Boolean 
        Boolean to indicate whether to graph the feature importances.
    """

    if model_def.ml_model == 'linear_r': 
        model = LinearRegression()
        temp = model.fit(x_train, y_train).predict(x_test)
    elif model_def.ml_model == 'random_f': 
        model = RandomForestRegressor(**model_def.kwargs)
        temp = model.fit(x_train, y_train).predict(x_test)
        if graph:
            graph_feature_importance(x_train, model)
    elif model_def.ml_model == 'tree': 
        model = DecisionTreeRegressor(**model_def.kwargs)
        temp = model.fit(x_train, y_train).predict(x_test)
    elif model_def.ml_model == 'g_boost':
        model = GradientBoostingRegressor(**model_def.kwargs)
        temp = model.fit(x_train, y_train).predict(x_test)
        if graph:
            graph_feature_importance(x_train, model)
    else:
        print('Unsupported Machine Learning Model')
        temp = pd.DataFrame([])
    return(temp)

def graph_feature_importance(x_train, model):
    n_features = x_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x_train.columns.values)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.show()