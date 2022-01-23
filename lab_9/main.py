import pandas as pd
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


def model_train(x_train, x_test, y_train, y_test, model):
    """

    Parameters
    ----------
    x_train : DataFrame
    x_test : : DataFrame
    y_train : Series
    y_test : Series
    model : any classifier

    Returns
    -------
    acc : numpy.float64

    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    return acc


def model_train_2features(x_train, x_test, y_train, y_test, model):
    """

    Parameters
    ----------
    x_train : DataFrame
    x_test : : DataFrame
    y_train : Series
    y_test : Series
    model : any classifier

    Returns
    -------
    acc : numpy.float64

    """

    model.fit(x_train, y_train)
    features = model.feature_importances_  # TODO: importance DOES NOT SORTED
    features = features[-2:]
    most_important_features = x_train.iloc[:, features]
    test_important_features = x_test.iloc[:, features]
    model = DecisionTreeClassifier()
    model.fit(most_important_features, y_train)
    y_pred = model.predict(test_important_features)
    acc = accuracy_score(y_test, y_pred)
    return acc


class MyRandomForest:
    def __init__(self, tree_numb=100):
        """

        Parameters
        ----------
        tree_numb : int

        """

        self._amount_of_trees = tree_numb
        self._forest_results = []
        self._forest = []
        self.predict_result = []

    def fit(self, df, df_label):
        """

        Parameters
        ----------
        df : DataFrame
        df_label : Series
        x_train : DataFrame
        y_train : Series

        """

        for i in range(self._amount_of_trees):   # TODO: RANDOM TRAINS
            x_train, _, y_train, _ = train_test_split(df, df_label, train_size=0.1)
            model = DecisionTreeClassifier()
            model.fit(x_train, y_train)
            self._forest.append(model)

    def predict(self, x_test):
        """

        Parameters
        ----------
        x_test : DataFrame
        Returns
        -------
        self.predict_result : Series

        """
        length = x_test.shape[0]
        for i in range(length):
            self.predict_result.append(0)

        for i in range(self._amount_of_trees):
            res_i = self._forest[i].predict(x_test)
            self._forest_results.append(res_i)

        for i in range(length):
            for j in range(self._amount_of_trees):
                self.predict_result[i] += self._forest_results[j][i]

        half = self._amount_of_trees // 2
        for i in range(length):
            if self.predict_result[i] >= half:
                self.predict_result[i] = 1
            else:
                self.predict_result[i] = 0
        res = self.predict_result
        return res


def main():
    df = pd.read_csv('titanic_prepared.csv', index_col=0, delimiter=',')
    df_label = df.pop('label')
    x_train, x_test, y_train, y_test = train_test_split(df, df_label, test_size=0.1)
    df[~df[['A', 'B']].apply(lambda x: np.in1d(x, x_test).all(), axis=1)]\.reset_index(drop=True)
    model_1 = DecisionTreeClassifier()
    model_2 = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model_3 = LogisticRegression(solver='liblinear', random_state=0)
    print('Decision Tree: ', model_train(x_train, x_test, y_train, y_test, model_1))
    print('XGBClassifier: ', model_train(x_train, x_test, y_train, y_test, model_2))
    print('Logistic Regression: ', model_train(x_train, x_test, y_train, y_test, model_3))
    model_4 = DecisionTreeClassifier()
    print('2 features: ', model_train_2features(x_train, x_test, y_train, y_test, model_4))
    model_5 = MyRandomForest(100)
    model_5.fit(df, df_label)
    accu = accuracy_score(y_test, model_5.predict(x_test))
    print('My Random Forest: ', accu)


if __name__ == '__main__':
    main()
