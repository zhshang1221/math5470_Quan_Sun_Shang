from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import simulate_numbers as SN

import numpy as NP

def pls_regression(X_train, y_train, X_valid, y_valid):
    '''
    train the best PLS Regression model under training set and validation set
    :param X_train: training features
    :param y_train: training labels
    :param X_valid: validation features
    :param y_valid: validation labels
    :return: well trained PLS model with well tuned paras
    '''
    print('Training PLS Regression Model:\n')
    best_mse, best_pls_components = NP.inf, 1
    pls_best = PLSRegression(n_components=1).fit(X_train, y_train)
    for i in range(1, min(X_train.shape[1], 100)): # find best hyper paras
        pls_temp = PLSRegression(n_components=i).fit(X_train, y_train)

        # predict values using PLS model
        temp_predict_values = pls_temp.predict(X_valid)
        temp_mse = mean_squared_error(temp_predict_values, y_valid)

        print(f'For components {i}, MSE is {temp_mse}, while best MSE is {best_mse}')
        if temp_mse < best_mse:
            best_pls_components = i
            pls_best = PLSRegression(n_components=best_pls_components).fit(X_train, y_train)
            best_mse = temp_mse
    print(f'\nPLS Regression Model with best component as {best_pls_components} and MSE being {str(best_mse)}')
    return pls_best

def get_model_complexity(pls_model):
    '''
    get feature complexity for certain PLS model
    :param pls_model: given PLS model
    :return: corresponding model complexity
    '''
    return pls_model.n_components

def get_variance_importance(pls_model):
    '''
    get variance importance for certain PLS model
    :param pls_model: given PLS model
    :return: corresponding model variance importance
    '''
    x_weights = pls_model.x_weights_
    # y_weights = pls_model.y_weights_
    return [abs(_[0]) for _ in x_weights]

if __name__ == '__main__':
    total_data = SN.generate_random_records()
    total_features = total_data.iloc[:, 1: ]
    total_response = total_data.iloc[:, : 1]

    X_train, X_test, y_train, y_test = train_test_split(total_features, total_response, train_size=0.8)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8)

    pls_model = pls_regression(X_train, y_train, X_valid, y_valid)
    pls_model.score(X_valid, y_valid)

    # get prediction values
    predict_values = pls_model.predict(X_test)

    # get model complexity
    model_complexity = get_model_complexity(pls_model)
    print('\n####\nModel Complexity is', model_complexity)

    # get model variance importance
    variance_importance = get_variance_importance(pls_model)
    print('\n####\nVariance Importance is', variance_importance)