from generalized_linear import generalized_linear
from pls_regression import pls_regression
from penalized_linear import Lasso
from pca_analysis import pca_analysis
from boost_tree import GBRTReg
import pandas as PD

class factor_models(object):
    '''
    params:
    -------
        feature_df: dataframe of the features，index is the datetime，columns are the features.
        return_df: dataframe of the stock returns，index is the datetime.
        first_train_end_date:The ending date for the first training.（The starting date is the first date of feature_df and return_df）。The form is '%Y%m%d'.
        last_train_end_date:The ending date for the last training.The form is '%Y%m%d'。
        freq:the expanding frequency for every expanded training task. monthly: freq='m'. annually: freq = 'A'

    methods:
    --------

    '''

    def __init__(self, feature_df, return_df, first_train_end_date, last_train_end_date, freq='m'):
        '''
        Initialize Parameters
        '''
        self._feature_df = feature_df  # The features of the stock price in form of dataframe
        self._return_df = return_df  # The returns of the stock in form of dataframe
        self.first_train_end_date = first_train_end_date  # The ending date for the first training, expanding every 1 month or 1 year afterwards
        self.last_train_end_date = last_train_end_date  # The ending date for the last training.
        self.freq = freq  # the expanding frequency for every expanded training task.

    def predict_ret(self):
        # set up a train_end_list，the training set is expanding every month or every year.
        train_end_list = PD.date_range(self.first_train_end_date, self.last_train_end_date,
                                       freq=self.freq)  # monthly: freq='m'. annually: freq = 'A'
        pred_df = PD.DataFrame()  # Store the prediction results y(out of sample returns) coming from difference models.
        for end_date in tqdm(train_end_list):  # Divide the samples monthly or annually and then train models by the end_date list
            # end_date = train_end_list[0]
            valid_date = end_date + MonthEnd()  # The validation set is the month or year right after current end date of training set. Predict the following 1 month or 1 year return once finish training
            end_date = end_date.strftime('%Y%m%d')  # transfer to string
            valid_date = valid_date.strftime('%Y%m%d')  # transfer to string

            # divide the data set
            # training set
            train_x = self._feature_df.loc[:end_date]
            train_y = self._return_df.loc[:end_date]

            # validation set
            valid_x = self._feature_df.loc[valid_date:valid_date]
            valid_y = self._return_df.loc[valid_date:valid_date]

            # build up model and train to predict the returns
            temp_pred_df = PD.DataFrame()
            temp_pred_df['real_y'] = valid_y.iloc[:, 0]

            ## OLS
            reg = linear_model.LinearRegression()
            reg.fit(train_x, train_y)
            predict_y = reg.predict(valid_x)
            temp_pred_df['OLS_y'] = predict_y  # Store the predicted results of returns from OLS

            ## OLS-3
            reg = linear_model.LinearRegression()
            reg.fit(train_x.loc[:,["size", "book_to_market", "momentum"]], train_y)
            predict_y = reg.predict(valid_x)
            temp_pred_df['OLS_y'] = predict_y  # Store the predicted results of returns from OLS

            ## penalized linear model, lasso
            lasso = Lasso(train_x, train_y, valid_x, valid_y)
            predict_y = lasso.predict(valid_x)
            temp_pred_df['lasso_y'] = predict_y  # Store the predicted results of returns from Lasso

            ## PLS
            pls = pls_regression(train_x, train_y, valid_x, valid_y)
            predict_y = pls.predict(valid_x)
            temp_pred_df['PLS_y'] = predict_y  # Store the predicted results of returns from PLS

            ## PCR
            pca = pca_analysis(train_x, train_y, valid_x, valid_y)
            temp = pca.transform(train_x)
            lr = linear_model.LinearRegression()
            lr.fit(temp, train_y)
            predict_y = lr.predict(pca.transform(valid_x))
            temp_pred_df['PCR_y'] = predict_y  # Store the predicted results of returns from PCR

            ## Generalized linear with group lasso
            group_lasso = generalized_linear(train_x, train_y, valid_x, valid_y)
            predict_y = group_lasso.predict(valid_x)
            temp_pred_df["gl_y"] = predict_y  # Store the predicted results of returns from Group Lasso

            ## GBRT，boost tree
            gbrt = GBRTReg(train_x, train_y, valid_x, valid_y)
            predict_y = gbrt.predict(valid_x)
            temp_pred_df['gbrt_y'] = predict_y  # Store the predicted results of returns from boost tree


            ## append pred_df with temp_pred_df
            pred_df = pred_df.append(temp_pred_df)
            self._pred_df = pred_df
        return pred_df

    def cal_oos(self):
        # Calculate the out-of-sample R2 according to the formula
        try:
            pred_df = self._pred_df
        except:
            pred_df = self.predict_ret()
        denominator = (pred_df['real_y'] ** 2).sum()
        numerator = pred_df.apply(lambda x: pred_df['real_y'] - x).iloc[:, 1:]
        numerator = (numerator ** 2).sum()

        roos = 1 - numerator / denominator
        roos.index = roos.index.str.rstrip('_y')
        fig, ax = plt.subplots(figsize=(16, 12))  # Plot the R2_OOS of every models
        plt.title('Out-of-sample predicting R2', fontsize=20)
        ax.bar(x=roos.index, height=roos)
        plt.show()
        return roos

if __name__ == "__main__":
    stock_data = pd.read_csv("./processed_data_monthly.csv")
    total_df = stock_data.set_index('date')  # Set the date as index
    total_df.index = pd.to_datetime(total_df.index)
    total_df = total_df.dropna(how='any')
    features_df = total_df.iloc[:, 2:]  # features
    returns_df = total_df[['mret']]  # monthly ret
    basic_factors = Factor_models(features_df, returns_df, first_train_end_date='19570228', last_train_end_date='19840430',
                                    freq='m')
    roos = basic_factors.cal_oos()
    preds = basic_factors.predict_ret()
