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
from sklearn.feature_extraction.text import CountVectorizer
import time
from sklearn.metrics import accuracy_score, precision_score

data = pd.read_csv(r"C:\Users\sarvesh.kesharwani\Pictures\DSC\Spam_Classifier\3. eda_n_featureEngineering\FEed_data.csv")
performance_df = pd.DataFrame()

def data_prep():
    # preparing data and vecotrizers
    cv = CountVectorizer()
    x = cv.fit_transform(data['clean_mail'].astype(str)).toarray()
    y = data['target']
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
    return train_x, test_x, train_y, test_y

def train_classifiers(name, clf, train_x, test_x, train_y, test_y):
    print(f'Training {name}:', '^'*100)
    clf.fit(test_x, test_y)
    print(f'Training {name} Completed.:', '$'*100)
    test_y_pred = clf.predict(test_x)
    accuracy = accuracy_score(test_y, test_y_pred)
    precision = precision_score(test_y, test_y_pred)
    return accuracy, precision

def trainer(model_tuple):
    # data prep
    train_x, test_x, train_y, test_y = data_prep()
    # mlflow tracking starts here...
    # mlflow.autolog()
    mlflow.set_experiment("test2")
    with mlflow.start_run():
        # training model
        model_name, model_obj = model_tuple
        accuracy, precision = train_classifiers(model_name, model_obj, train_x, test_x, train_y, test_y)
        mlflow.set_tag('mlflow.runName', model_name)
        # mlflow.log_param('max_deapth', 4)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
    return (accuracy, precision)

if __name__ == '__main__':
    start_time = time.time()
    # preparing algo models
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
    classifiers = [
                    ('LogReg', logReg),
                    ('SVC', svc),
                    ('MNB', mnb),
                    ('KNN', knn),
                    ('DT', dt),
                    ('Bagging', bagging),
                    ('RF', rf),
                    ('ETC', et),
                    ('AdaBoost', ab),
                    ('XGBoost', xgb),
                ]

    # mp pooling
    accuracy_lst=[]
    precision_lst=[]
    accuracy_precision_lst=[]
    with Pool(processes=8) as pool:
        accuracy_precision_lst = pool.map(trainer, classifiers)
        pool.close()
        pool.join()
    print(accuracy_precision_lst)
    accuracy_lst, precision_lst = zip(*accuracy_precision_lst)
    end_time = time.time()

    # printing performances of algos
    performance_df = pd.DataFrame({'Algorithm':[classifier[0] for classifier in classifiers], 'Accuracy':accuracy_lst, 'Precision':precision_lst})
    print(performance_df)
    time_taken = end_time - start_time
    print(f"time_taken was: {round(time_taken, 2)} seconds")