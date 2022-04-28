from group_lasso import GroupLasso
GroupLasso.LOG_LOSSES = True

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import simulate_numbers as SN

import numpy as NP
NP.random.seed(0)

def generalized_linear(X_train, y_train, X_valid, y_valid):
    '''
    train the best Generalized Linear model under training set and validation set
    :param X_train: training features
    :param y_train: training labels
    :param X_valid: validation features
    :param y_valid: validation labels
    :return: well trained PLS model with well tuned paras
    '''
    print('Training Generalized Model:\n')
    group_lasso_best = GroupLasso(
        group_reg=5,
        l1_reg=0,
        frobenius_lipschitz=True,
        scale_reg="inverse_group_size",
        subsampling_scheme=1,
        supress_warning=True,
        n_iter=1000,
        tol=1e-3,
    )
    best_mse, best_gl_group = NP.inf, 1

    for i in range(1, min(X_train.shape[1], 100)): # find best hyper paras
        temp_group_lasso = GroupLasso(
            group_reg=i,
            l1_reg=0,
            frobenius_lipschitz=True,
            scale_reg="inverse_group_size",
            subsampling_scheme=1,
            supress_warning=True,
            n_iter=1000,
            tol=1e-3,
        )

        # predict values using PLS model
        temp_group_lasso.fit(X_train, y_train)
        temp_predict_values = temp_group_lasso.predict(X_valid)

        temp_mse = mean_squared_error(temp_predict_values, y_valid)

        print(f'For group {i}, MSE is {temp_mse}, while best MSE is {best_mse}')
        if temp_mse < best_mse:
            best_gl_group = i
            group_lasso_best = temp_group_lasso
            best_mse = temp_mse
    print(f'\nPCA Model with best component as {best_gl_group} and MSE being {str(best_mse)}')
    return group_lasso_best

def get_model_complexity(generalized_linear_model):
    '''
    get feature complexity for certain PLS model
    :param generalized_linear_model: given PLS model
    :return: corresponding model complexity
    '''
    return generalized_linear_model.sparsity_mask_.sum()

def get_variance_importance(generalized_linear_model):
    '''
    get variance importance for certain PLS model
    :param generalized_linear_model: given PLS model
    :return: corresponding model variance importance
    '''
    x_weights = generalized_linear_model.coef_
    # y_weights = generalized_linear_model.y_weights_
    return [abs(_[0]) for _ in x_weights]

if __name__ == '__main__':
    total_data = SN.generate_random_records()
    total_features = total_data.iloc[:, 1: ]
    total_response = total_data.iloc[:, : 1]

    X_train, X_test, y_train, y_test = train_test_split(total_features, total_response, train_size=0.8)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8)

    # get prediction values
    generalized_linear_model = generalized_linear(X_train, y_train, X_valid, y_valid)
    predict_values = generalized_linear_model.predict(X_test)

    # get model complexity
    model_complexity = get_model_complexity(generalized_linear_model)
    print('\n####\nModel Complexity is', model_complexity)

    # get model variance importance
    variance_importance = get_variance_importance(generalized_linear_model)
    print('\n####\nVariance Importance is', variance_importance)