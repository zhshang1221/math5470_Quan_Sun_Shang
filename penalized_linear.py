from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

import simulate_numbers as SN

import numpy as NP
NP.random.seed(0)

def Lasso(X_train, y_train, X_valid, y_valid):
    '''
    train the best Penalized Linear model under training set and validation set
    :param X_train: training features
    :param y_train: training labels
    :param X_valid: validation features
    :param y_valid: validation labels
    :return: well trained Lasso model with well tuned paras
    '''
    lasso = linear_model.Lasso()
    param = [{'alpha': NP.linspace(0, 1, 100)}]
    grid = GridSearchCV(lasso, param_grid=param)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def get_model_complexity(Lasso):
    '''
    get feature complexity for certain Lasso model
    :param Lasso: given Lasso model
    :return: corresponding model complexity
    '''
    return Lasso.n_features_in_

def get_variance_importance(Lasso):
    '''
    get variance importance for certain Lasso model
    :param Lasso: given Lasso model
    :return: corresponding model variance importance
    '''
    x_weights = list(Lasso.coef_)
    # y_weights = Lasso.y_weights_
    return x_weights

if __name__ == '__main__':
    total_data = SN.generate_random_records()
    total_features = total_data.iloc[:, 1: ]
    total_response = total_data.iloc[:, : 1]

    X_train, X_test, y_train, y_test = train_test_split(total_features, total_response, train_size=0.8)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8)

    # get prediction values
    Lasso_model = Lasso(X_train, y_train, X_valid, y_valid)
    predict_values = Lasso_model.predict(X_test)

    # get model complexity
    model_complexity = get_model_complexity(Lasso_model)
    print('\n####\nModel Complexity is', model_complexity)

    # get model variance importance
    variance_importance = get_variance_importance(Lasso_model)
    print('\n####\nVariance Importance is', variance_importance)