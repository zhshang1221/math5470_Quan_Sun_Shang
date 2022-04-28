#### Paper Replication: Empirical Asset Pricing via Machine Learning

##### Codes Part

We encapsulate various statistical algorithms by separate file, whereby each having model training part, tuning of hyperparameters, and API for model complexity & variable importance inside.

```
Direcotory   
│
│─── Data Part
│   │   simulation_number: numerical simulation for testing statistical algorithms
|   |   |   generate_random_records: generate random data for testing algorithms
|   |   |   |   generate random datasets(10*1000 by default) for testing use
|   |   |   |   :param : 
|   |   |   |   :return: Pandas.DataFrame, random datasets
|
│─── Algorithms Part
|   |   pls_regression: PLS regression method
|   |   |   pls_regression: train the best PLS Regression model under training set and validation set
|   |   |   :param X_train: training features
|   |   |   :param y_train: training labels
|   |   |   :param y_train: training labels
|   |   |   :param X_valid: validation features
|   |   |   :param y_valid: validation labels
|   |   |   :return: well trained PLS model with well tuned paras
|   |   |
|   |   |   get_model_complexity: get feature complexity for certain PLS model
|   |   |   :param pls_model: given PLS model
|   |   |   :return: corresponding model complexity
|   |   |
|   |   |   get_variance_importance: get variance importance for certain PLS model
|   |   |   :param pls_model: given PLS model
|   |   |   :return: corresponding model variance importance
|   |
|   |   pca_analysis: PCA analysis method
|   |   |   pca_analysis: train the best PCA model under training set and validation set
|   |   |   :param X_train: training features
|   |   |   :param y_train: training labels
|   |   |   :param y_train: training labels
|   |   |   :param X_valid: validation features
|   |   |   :param y_valid: validation labels
|   |   |   :return: well trained PCA model with well tuned paras
|   |   |
|   |   |   get_model_complexity: get feature complexity for certain PCA model
|   |   |   :param pls_model: given PLS model
|   |   |   :return: corresponding model complexity
|   |   |
|   |   |   get_variance_importance: get variance importance for certain PCA model
|   |   |   :param pls_model: given PLS model
|   |   |   :return: corresponding model variance importance
|   |
|   |   generalized_linear: generalized linear method with Group Lasso
|   |   |   pca_analysis: train the best GL model under training set and validation set
|   |   |   :param X_train: training features
|   |   |   :param y_train: training labels
|   |   |   :param y_train: training labels
|   |   |   :param X_valid: validation features
|   |   |   :param y_valid: validation labels
|   |   |   :return: well trained GL model with well tuned paras
|   |   |
|   |   |   get_model_complexity: get feature complexity for certain GL model
|   |   |   :param pls_model: given GL model
|   |   |   :return: corresponding model complexity
|   |   |
|   |   |   get_variance_importance: get variance importance for certain GL model
|   |   |   :param pls_model: given GL model
|   |   |   :return: corresponding model variance importance
|   |
```
<br />
