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


def dataset_prep(dataframe, show=False):
    dataframe.z_axis.replace(regex=True, inplace=True, to_replace=r';', value=r'')
    dataframe['z_axis'] = dataframe.z_axis.astype(np.float64)
    dataframe.dropna(axis=0, how='any', inplace=True)

    dataframe.drop(dataframe[dataframe["activity"] == "Sitting"].index, inplace=True)
    dataframe.drop(dataframe[dataframe["activity"] == "Upstairs"].index, inplace=True)
    dataframe.drop(dataframe[dataframe["activity"] == "Downstairs"].index, inplace=True)

    if show:
        print(dataframe.info())
        print(dataframe.head(2))

    return dataframe


def get_raw_data(path):
    column_names = [
        'user_id',
        'activity',
        'timestamp',
        'x_axis',
        'y_axis',
        'z_axis'
    ]
    df_original = pd.read_csv(
        path,
        header=None,
        names=column_names
    )

    df = df_original.copy()
    return df


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


def random_forest_classifier(X, y, save=False, n_estimators=100, random_state=0):
    start = time.time()
    X_rf = X.copy()
    y_rf = y.copy()

    X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y_rf)

    clf = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators)
    clf.fit(X_rf_train, y_rf_train)

    print("Training scores")
    print(clf.score(X_rf_train, y_rf_train))

    print("------------------Training time : %s seconds ------------------" % (time.time() - start))
    if save:
        return clf


if __name__ == '__main__':

    df = dataset_prep(get_raw_data('Models/DATA/WISDM/WISDM_ar_v1_1_raw.txt'))
    # basic_visualization(df)
    df.to_csv("DATA/WISDM/WISDM.csv")
    X = df[["timestamp", "x_axis", "y_axis", "z_axis"]]
    y = df["activity"]

    X_scaled = scale_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)


    # save_model(random_forest_classifier(X_scaled, y, save=True), "AiModels/RFC_model.sav")


    # model = load_model("AiModels/RFC_model.sav")
    #n_estimators = [10, 100, 500]
    #random_state = [0, 1, 5]
    #for n in n_estimators:
    #    for r in random_state:
    #        print(f" n_estimator = {n} - random_state = {r}")
    #        test_model(random_forest_classifier(X_scaled, y, save=True), X_test, y_test)



