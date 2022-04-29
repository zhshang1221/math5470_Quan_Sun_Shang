from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import simulate_numbers as SN

import numpy as NP
NP.random.seed(0)

# gradient boosting regression trees
def GBRTReg(X_train, y_train, X_valid, y_valid):
    '''
    train the best gradient boosting regression trees model under training set and validation set
    :param X_train: training features
    :param y_train: training labels
    :param X_valid: validation features
    :param y_valid: validation labels
    :return: well trained gradient boosting regression trees model with well tuned paras
    '''
    gbrt_best = GradientBoostingRegressor(loss='huber').fit(X_train, y_train)
    for L in range(1, 5): # max depth
        for v in range(1,5): # learning rate
            learning_rate = v *0.1
            for B in range(20,100,10): # number of estimators
                gbrt_temp = GradientBoostingRegressor(loss='huber',learning_rate=learning_rate, n_estimators=B, max_depth=L).fit(X_train, y_train)
                if gbrt_temp.score(X_valid, y_valid)> gbrt_best.score(X_valid, y_valid):
                    L_best = L
                    learning_rate_best = learning_rate
                    B_best = B
                    gbrt_best = GradientBoostingRegressor(loss='huber',learning_rate=learning_rate_best, n_estimators=B_best, max_depth=L_best).fit(X_train, y_train)
    return gbrt_best


def get_model_complexity(GBRTReg):
    '''
    get feature complexity for certain PLS model
    :param generalized_linear_model: given PLS model
    :return: corresponding model complexity
    '''
    return len(GBRTReg.feature_importances_.nonzero()[0])

def get_variance_importance(GBRTReg):
    '''
    get variance importance for certain PLS model
    :param generalized_linear_model: given PLS model
    :return: corresponding model variance importance
    '''
    x_weights = list(GBRTReg.feature_importances_)
    # y_weights = generalized_linear_model.y_weights_
    return x_weights

if __name__ == '__main__':
    total_data = SN.generate_random_records()
    total_features = total_data.iloc[:, 1: ]
    total_response = total_data.iloc[:, : 1]

    X_train, X_test, y_train, y_test = train_test_split(total_features, total_response, train_size=0.8)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8)

    # get prediction values
    GBRTReg_model = GBRTReg(X_train, y_train, X_valid, y_valid)
    predict_values = GBRTReg_model.predict(X_test)

    # get model complexity
    model_complexity = get_model_complexity(GBRTReg_model)
    print('\n####\nModel Complexity is', model_complexity)

    # get model variance importance
    variance_importance = get_variance_importance(GBRTReg_model)
    print('\n####\nVariance Importance is', variance_importance)