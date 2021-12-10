import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statistics as stat
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import sklearn.svm as svm
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OutputCodeClassifier, OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer

from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.optimizers import SGD

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import pickle

def preprocessing_data(df_path, model):
    if model=="RFC":
        dfc = pd.read_csv(df_path)
        dfc.dropna()
        if "x_accel" in dfc.columns:
            dfc["x_accel"] = pd.eval(dfc["x_accel"])
            if "y_accel" in dfc.columns:
                dfc["y_accel"] = pd.eval(dfc["y_accel"])
                if "z_accel" in dfc.columns:
                    dfc["z_accel"] = pd.eval(dfc["z_accel"])
                    if "impedance" in dfc.columns:
                        dfc["impedance"] = pd.eval(dfc["impedance"])
                        if "gesture" in dfc.columns:
                            return dfc
                        else:
                            return "invalid csv file"
                    else:
                        return "invalid csv file"
                else:
                    return "invalid csv file"
            else:
                return "invalid csv file"
        else:
            return "invalid csv file"
    else:
        pass


def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


def load_model(filename):
    model = pickle.load(open(filename, 'rb'))
    return model


def test_model(model, X_test, y_test):
    start = time.time()
    result = model.score(X_test, y_test)
    print(f"testing score for {model}--> {result:.3f}")
    print("------------------Testing time : %s seconds ------------------" % (time.time() - start))


def scale_dataset(dataframe):
    scaler = StandardScaler()

    X = dataframe[["timestamp", "x_axis", "y_axis", "z_axis"]]
    y = dataframe["activity"]

    scaled_X = scaler.fit_transform(X)

    return scaled_X


def basic_visualization(dataframe):
    # counting
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey='none')
    fig.suptitle("basic visualization of the dataset")

    axes[0, 0].set_title("activity repartition")
    sns.histplot(ax=axes[0, 0], data=dataframe, x="activity")

    axes[0, 1].set_title("Users' records count")
    sns.histplot(ax=axes[0, 1], data=dataframe, x="user_id")

    # acceleration repartition
    df_mean = pd.DataFrame()
    df_mean["activity"] = dataframe["activity"]
    df_mean["mean_accel"] = (dataframe["x_axis"] + dataframe["y_axis"] + dataframe["z_axis"]) / 3

    axes[0, 2].set_title("mean acceleration values by activity")
    sns.violinplot(ax=axes[0, 2], data=df_mean, x="activity", y="mean_accel")

    axes[1, 0].set_title("x acceleration values by activity")
    sns.violinplot(ax=axes[1, 0], data=dataframe, x="activity", y="x_axis")

    axes[1, 1].set_title("y acceleration values by activity")
    sns.violinplot(ax=axes[1, 1], data=dataframe, x="activity", y="y_axis")

    axes[1, 2].set_title("z acceleration values by activity")
    sns.violinplot(ax=axes[1, 2], data=dataframe, x="activity", y="z_axis")

    plt.show()


def support_vector_machine(X, y):
    X_svm = X.copy()
    y_svm = y.copy()

    X_svm_train, X_svm_test, y_svm_train, y_svm_test = train_test_split(X_svm, y_svm)

    print("fitting ovo")
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X_svm_train, y_svm_train)

    print("scores ovo")
    print(clf.score(X_svm_train, y_svm_train))
    print(clf.score(X_svm_test, y_svm_test))

    print("fitting ovr")
    clf2 = svm.SVC(decision_function_shape='ovr')
    clf2.fit(X_svm_train, y_svm_train)

    print("scores ovr")
    print(clf2.score(X_svm_train, y_svm_train))
    print(clf2.score(X_svm_test, y_svm_test))

    print("fitting lin")
    clf3 = svm.LinearSVC()
    clf3.fit(X_svm_train, y_svm_train)

    print("scores lin")
    print(clf3.score(X_svm_train, y_svm_train))
    print(clf3.score(X_svm_test, y_svm_test))


def keras_model_1(X, y):
    Xc = X.copy()
    yc = y.copy()

    # encoding the categories
    encoder = LabelEncoder()
    encoder.fit(yc)
    encoded_Y = encoder.transform(yc)

    yc_dummy = to_categorical(encoded_Y)

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc_dummy)

    model = Sequential()
    model.add(Dense(3, input_dim=4, activation='softmax', kernel_initializer='he_uniform'))

    opt = SGD(lr=0.1)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # print(model.summary())

    history = model.fit(x=Xc_train, y=yc_train, epochs=10)

    train_acc = model.evaluate(Xc_train, yc_train)
    test_acc = model.evaluate(Xc_test, yc_test)

    print(f"Train loss: {train_acc[0]:.3f} - accuracy: {train_acc[1]:.3f}")
    print(f"Test loss: {test_acc[0]:.3f} - accuracy: {test_acc[1]:.3f}")


def one_versus_rest_classifier(X, y):
    X_ovr = X.copy()
    y_ovr = y.copy()

    X_ovr_train, X_ovr_test, y_ovr_train, y_ovr_test = train_test_split(X_ovr, y_ovr)

    print("fitting")
    clf = OneVsRestClassifier(SVC())
    clf.fit(X_ovr_train, y_ovr_train)

    print("scores")
    print(clf.score(X_ovr_train, y_ovr_train))
    print(clf.score(X_ovr_test, y_ovr_test))


def keras_model_2(X, y):
    X_ks2 = X.copy()
    y_ks2 = y.copy()

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y_ks2)
    encoded_Y = encoder.transform(y_ks2)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = to_categorical(encoded_Y)

    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(8, input_dim=4, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    estimator = KerasClassifier(build_fn=baseline_model, epochs=10, verbose=1)
    kfold = KFold(n_splits=5, shuffle=True)
    results = cross_val_score(estimator, X, dummy_y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


def random_forest_classifier(df, save=False, n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2,
                             min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features="auto", max_leaf_nodes=None,
                             min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None,
                             random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0,
                             max_samples=None):
    start = time.time()
    X = df[["x_accel", "y_accel", "z_accel", "impedance"]]
    y = df["gesture"]
    Xc = X.copy()
    yc = y.copy()

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc)

    clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                 min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                 max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                 bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state,
                                 verbose=verbose, warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha,
                                 max_samples=max_samples)
    clf.fit(Xc_train, yc_train)

    print("Training scores")
    print(clf.score(Xc_train, yc_train))

    print("------------------Training time : %s seconds ------------------" % (time.time() - start))
    if save:
        save_model(clf, "AiModels/RFC_model.sav")


