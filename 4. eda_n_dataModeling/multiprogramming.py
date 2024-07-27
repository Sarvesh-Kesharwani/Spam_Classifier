from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

import pandas as pd
import mlflow
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')
import time

mlflow.autolog()

start_time = time.time()
with mlflow.start_run():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    logReg = LogisticRegression(solver='liblinear', penalty='l1')
    svc = SVC(kernel='sigmoid', gamma=1.0)

    mnb = MultinomialNB()
    knn = KNeighborsClassifier()

    dt = DecisionTreeClassifier(max_depth=5)

    bagging = BaggingClassifier(n_estimators=50, random_state=2)
    rf = RandomForestClassifier(n_estimators=50, random_state=2)
    et = ExtraTreesClassifier(n_estimators=50, random_state=2)

    ab = AdaBoostClassifier(n_estimators=50, random_state=2)
    xgb = XGBClassifier(n_estimators=50, random_state=2)

    classifiers = {
                    'LogReg': logReg,
                    'SVC': svc,
                    'MNB': mnb,
                    'KNN': knn,
                    'DT': dt,
                    'Bagging': bagging,
                    'RF': rf,
                    'ETC': et,
                    'AdaBoost': ab,
                    'XGBoost': xgb,
                }

    def train_classifiers(name, clf, train_x, train_y, test_x, test_y):
        print(f'Training {name}:')
        clf.fit(x_train, y_train)
        print(f'Training {name} Completed.:')
        print('-'*100)
        y_test_pred = clf.predict(test_x)
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        return accuracy, precision

    def trainer(models_list):
        accuracy_lst = []
        precision_lst = []
        for name, clf in models_list.items():
            accuracy, precision = train_classifiers(name, clf, train_x, train_y, test_x, test_y)
            accuracy_lst.append(accuracy)
            precision_lst.append(precision)
        return accuracy_lst, precision_lst

    with Pool(processes=5) as pool:
        accuracy_lst, precision_lst = pool.map(trainer, classifiers)

    performance_df = pd.DataFrame({'Algorithm':classifiers.keys(), 'Accuracy':accuracy_lst, 'Precision':precision_lst})
    performance_df
end_time = time.time()
time_taken = end_time - start_time