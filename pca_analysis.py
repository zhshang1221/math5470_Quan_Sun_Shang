from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

import simulate_numbers as SN

import numpy as NP

def pca_analysis(X_train, y_train, X_valid, y_valid):
    '''
    train the best PLS Regression model under training set and validation set
    :param X_train: training features
    :param y_train: training labels
    :param X_valid: validation features
    :param y_valid: validation labels
    :return: well trained PLS model with well tuned paras
    '''
    print('Training PCA Model:\n')
    best_mse, best_pca_components = NP.inf, 1
    pca_best = PCA(n_components=1, svd_solver='full')


    for i in range(1, min(X_train.shape[1], 100)): # find best hyper paras
        temp_pca = PCA(n_components=i, svd_solver='full')

        # predict values using PLS model
        temp_pca.fit(X_train)
        temp_features = temp_pca.transform(X_train)
        basic_linear_model = linear_model.LinearRegression()
        basic_linear_model.fit(temp_features, y_train) # to train the linear model
        temp_predict_values = basic_linear_model.predict(temp_pca.transform(X_valid))

        temp_mse = mean_squared_error(temp_predict_values, y_valid)

        print(f'For components {i}, MSE is {temp_mse}, while best MSE is {best_mse}')
        if temp_mse < best_mse:
            best_pca_components = i
            pca_best = temp_pca
            best_mse = temp_mse
    print(f'\nPCA Model with best component as {best_pca_components} and MSE being {str(best_mse)}')
    return pca_best

def get_model_complexity(pca_model):
    '''
    get feature complexity for certain PLS model
    :param pca_model: given PLS model
    :return: corresponding model complexity
    '''
    return pca_model.n_components

def get_variance_importance(pca_model):
    '''
    get variance importance for certain PLS model
    :param pca_model: given PLS model
    :return: corresponding model variance importance
    '''
    x_weights = pca_model.components_[0]
    # y_weights = pca_model.y_weights_
    return [abs(_) for _ in x_weights]

if __name__ == '__main__':
    total_data = SN.generate_random_records()
    total_features = total_data.iloc[:, 1: ]
    total_response = total_data.iloc[:, : 1]

    X_train, X_test, y_train, y_test = train_test_split(total_features, total_response, train_size=0.8)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8)

    # get prediction values
    pca_model = pca_analysis(X_train, y_train, X_valid, y_valid)
    pca_features = pca_model.transform(X_train)
    basic_linear_model = linear_model.LinearRegression()
    basic_linear_model.fit(pca_features, y_train) # to train the linear model
    predict_values = basic_linear_model.predict(pca_model.transform(X_test))

    # get model complexity
    model_complexity = get_model_complexity(pca_model)
    print('\n####\nModel Complexity is', model_complexity)

    # get model variance importance
    variance_importance = get_variance_importance(pca_model)
    print('\n####\nVariance Importance is', variance_importance)