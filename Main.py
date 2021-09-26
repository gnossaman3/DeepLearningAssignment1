import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def get_data():
    df_income = pd.read_csv('Data/income_evaluation.csv', header=0, skipinitialspace=True)
    df_music = pd.read_csv('Data/music_train.csv', header=0, skipinitialspace=True)

    income_group_labels = {}
    for col in df_income.columns:
        income_group_labels[col] = {}
        if df_income[col].dtype == type(df_income['age']):
            i = 0
            for group in df_income[col].unique():
                income_group_labels[col][group] = i
                df_income[col] = df_income[col].replace(group, i)
                i += 1

    df_music = df_music.drop(['Artist Name', 'Track Name'], axis=1).dropna(axis=0)

    return df_income, df_music, income_group_labels

if __name__ == '__main__':
    print('Preparing Data...')
    income_train, music_train, income_group_labels = get_data()
    income_train_X = income_train.loc[:, income_train.columns != 'income']
    income_train_Y = income_train['income']
    music_train_X = music_train.loc[:, music_train.columns != 'Class']
    music_train_Y = music_train['Class']
    print('Data Prepared.')
    print('Income Group Labels: ', income_group_labels)

    # Decision Tree
    print('---------------------------------Moving to Decision Tree------------------------------')
    depth = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    print('Testing Depth on Income Data...')
    income_train_scores, income_valid_scores = validation_curve(DecisionTreeClassifier(), income_train_X,
                                                                income_train_Y, param_name="max_depth",
                                                                param_range=depth, scoring="accuracy", cv=10)
    income_train_scores_avg = np.mean(income_train_scores, axis=1)
    income_valid_scores_avg = np.mean(income_valid_scores, axis=1)

    print('Done. Testing Depth on Music Data...')
    music_train_scores, music_valid_scores = validation_curve(DecisionTreeClassifier(), music_train_X, music_train_Y,
                                                              param_name="max_depth", param_range=depth,
                                                              scoring="accuracy", cv=10)
    music_train_scores_avg = np.mean(music_train_scores, axis=1)
    music_valid_scores_avg = np.mean(music_valid_scores, axis=1)

    print('Done. Creating Graphs...')
    plt.plot(depth, income_train_scores_avg, label='Training Scores')
    plt.plot(depth, income_valid_scores_avg, label='Validation Scores')
    plt.title('Decision Tree Validation Curve For max_depth HP Tuning on Income Dataset')
    plt.xlabel('max_depth')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='DTreeValidCurveDepthIncome')
    plt.close()
    dtree_best_income_depth = 7

    plt.plot(depth, music_train_scores_avg, label='Training Scores')
    plt.plot(depth, music_valid_scores_avg, label='Validation Scores')
    plt.title('Decision Tree Validation Curve For max_depth HP Tuning on Music Dataset')
    plt.xlabel('max_depth')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='DTreeValidCurveDepthMusic')
    plt.close()
    dtree_best_music_depth = 7

    print('Done. Testing Sample Split Size On Income Data...')
    split_sizes = [2, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    income_train_scores, income_valid_scores = validation_curve(DecisionTreeClassifier(), income_train_X,
                                                                income_train_Y, param_name='min_samples_split',
                                                                param_range=split_sizes, scoring='accuracy', cv=10)
    income_train_scores_avg = np.mean(income_train_scores, axis=1)
    income_valid_scores_avg = np.mean(income_valid_scores, axis=1)

    print('Done. Testing Sample Split Size on Music Data...')
    music_train_scores, music_valid_scores = validation_curve(DecisionTreeClassifier(), music_train_X, music_train_Y,
                                                              param_name="min_samples_split", param_range=split_sizes,
                                                              scoring='accuracy', cv=10)
    music_train_scores_avg = np.mean(music_train_scores, axis=1)
    music_valid_scores_avg = np.mean(music_valid_scores, axis=1)

    print("Done. Creating Graphs...")
    plt.plot(split_sizes, income_train_scores_avg, label='Training Scores')
    plt.plot(split_sizes, income_valid_scores_avg, label='Validation Scores')
    plt.title('Decision Tree Validation Curve For min_sample_split HP Tuning on Income Dataset')
    plt.xlabel('min_sample_split')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='DTreeValidCurveSampleSplitIncome')
    plt.close()
    dtree_best_income_sample_split = 80

    plt.plot(split_sizes, music_train_scores_avg, label='Training Scores')
    plt.plot(split_sizes, music_valid_scores_avg, label='Validation Scores')
    plt.title('Decision Tree Validation Curve For min_sample_split HP Tuning on Music Dataset')
    plt.xlabel('min_sample_split')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='DTreeValidCurveSampleSplitMusic')
    plt.close()
    dtree_best_music_sample_split = 60

    print('Done. Creating Learning Curve Data for Income Data...')
    train_sizes = np.linspace(.1, 1.0, 10)

    income_train_sizes, income_train_scores, income_valid_scores = learning_curve(DecisionTreeClassifier(
        max_depth=dtree_best_income_depth, min_samples_split=dtree_best_income_sample_split), income_train_X,
        income_train_Y, cv=10, train_sizes=train_sizes)
    income_train_scores_avg = np.mean(income_train_scores, axis=1)
    income_valid_scores_avg = np.mean(income_valid_scores, axis=1)

    print('Done. Creating Learning Curve Data for Music Data...')
    music_train_sizes, music_train_scores, music_valid_scores = learning_curve(DecisionTreeClassifier(
        max_depth=dtree_best_music_depth, min_samples_split=dtree_best_music_sample_split), music_train_X,
        music_train_Y, cv=10, train_sizes=train_sizes)
    music_train_scores_avg = np.mean(music_train_scores, axis=1)
    music_valid_scores_avg = np.mean(music_valid_scores, axis=1)

    print("Done. Creating Graphs...")
    plt.plot(income_train_sizes, income_train_scores_avg, label='Training Scores')
    plt.plot(income_train_sizes, income_valid_scores_avg, label='Validation Scores')
    plt.title('Decision Tree Learning Curve for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='DTreeLearningCurveIncome')
    plt.close()

    plt.plot(music_train_sizes, music_train_scores_avg, label='Training Scores')
    plt.plot(music_train_sizes, music_valid_scores_avg, label='Validation Scores')
    plt.title('Decision Tree Learning Curve for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='DTreeLearningCurveMusic')
    plt.close()

    print('Done with Decision Trees.')
    print('------------------------------Moving To Boosting------------------------------')
    depth = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

    print('Testing Depth on Income Data...')
    income_train_scores, income_valid_scores = validation_curve(GradientBoostingClassifier(), income_train_X,
                                                                income_train_Y, param_name="max_depth",
                                                                param_range=depth, scoring="accuracy", cv=10)
    income_train_scores_avg = np.mean(income_train_scores, axis=1)
    income_valid_scores_avg = np.mean(income_valid_scores, axis=1)

    print('Done. Testing Depth on Music Data...')
    music_train_scores, music_valid_scores = validation_curve(GradientBoostingClassifier(), music_train_X,
                                                              music_train_Y, param_name="max_depth", param_range=depth,
                                                              scoring="accuracy", cv=10)
    music_train_scores_avg = np.mean(music_train_scores, axis=1)
    music_valid_scores_avg = np.mean(music_valid_scores, axis=1)

    print('Done. Creating Graphs...')
    plt.plot(depth, income_train_scores_avg, label='Training Scores')
    plt.plot(depth, income_valid_scores_avg, label='Validation Scores')
    plt.title('Boosting Validation Curve For max_depth HP Tuning on Income Dataset')
    plt.xlabel('max_depth')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='BoostingValidCurveDepthIncome')
    plt.close()
    boosting_best_income_depth = 6

    plt.plot(depth, music_train_scores_avg, label='Training Scores')
    plt.plot(depth, music_valid_scores_avg, label='Validation Scores')
    plt.title('Boosting Validation Curve For max_depth HP Tuning on Music Dataset')
    plt.xlabel('max_depth')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='BoostingValidCurveDepthMusic')
    plt.close()
    boosting_best_music_depth = 3

    print('Done. Testing n_estimators on Income Data...')
    n_estimators = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450]

    income_train_scores, income_valid_scores = validation_curve(GradientBoostingClassifier(), income_train_X,
                                                                income_train_Y, param_name="n_estimators",
                                                                param_range=n_estimators, scoring="accuracy", cv=10)
    income_train_scores_avg = np.mean(income_train_scores, axis=1)
    income_valid_scores_avg = np.mean(income_valid_scores, axis=1)

    print('Done. Testing n_estimators on Music Data...')
    music_train_scores, music_valid_scores = validation_curve(GradientBoostingClassifier(), music_train_X,
                                                              music_train_Y, param_name="n_estimators",
                                                              param_range=n_estimators, scoring="accuracy", cv=10)
    music_train_scores_avg = np.mean(music_train_scores, axis=1)
    music_valid_scores_avg = np.mean(music_valid_scores, axis=1)

    print('Done. Creating Graphs...')
    plt.plot(n_estimators, income_train_scores_avg, label='Training Scores')
    plt.plot(n_estimators, income_valid_scores_avg, label='Validation Scores')
    plt.title('Boosting Validation Curve For n_estimators HP Tuning on Income Dataset')
    plt.xlabel('n_estimators')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='BoostingValidCurveNEstimatorsIncome')
    plt.close()
    boosting_best_income_n_estimators = 100

    plt.plot(n_estimators, music_train_scores_avg, label='Training Scores')
    plt.plot(n_estimators, music_valid_scores_avg, label='Validation Scores')
    plt.title('Boosting Validation Curve For n_estimators HP Tuning on Music Dataset')
    plt.xlabel('n_estimators')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='BoostingValidCurveNEstimatorsMusic')
    plt.close()
    boosting_best_music_n_estimators = 50

    print('Done. Creating Learning Curve Data for Income Data...')
    train_sizes = np.linspace(.1, 1.0, 10)

    income_train_sizes, income_train_scores, income_valid_scores = learning_curve(GradientBoostingClassifier(
        max_depth=boosting_best_income_depth, n_estimators=boosting_best_income_n_estimators), income_train_X,
        income_train_Y, cv=10, train_sizes=train_sizes)
    income_train_scores_avg = np.mean(income_train_scores, axis=1)
    income_valid_scores_avg = np.mean(income_valid_scores, axis=1)

    print('Done. Creating Learning Curve Data for Music Data...')
    music_train_sizes, music_train_scores, music_valid_scores = learning_curve(GradientBoostingClassifier(
        max_depth=boosting_best_music_depth, n_estimators=boosting_best_music_n_estimators), music_train_X,
        music_train_Y, cv=10, train_sizes=train_sizes)
    music_train_scores_avg = np.mean(music_train_scores, axis=1)
    music_valid_scores_avg = np.mean(music_valid_scores, axis=1)

    print("Done. Creating Graphs...")
    plt.plot(income_train_sizes, income_train_scores_avg, label='Training Scores')
    plt.plot(income_train_sizes, income_valid_scores_avg, label='Validation Scores')
    plt.title('Boosting Learning Curve for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='BoostingLearningCurveIncome')
    plt.close()

    plt.plot(music_train_sizes, music_train_scores_avg, label='Training Scores')
    plt.plot(music_train_sizes, music_valid_scores_avg, label='Validation Scores')
    plt.title('Boosting Learning Curve for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='BoostingLearningCurveMusic')
    plt.close()

    print('Done with Boosting.')
    print('------------------------------Moving To NN------------------------------')

    hidden_layer_sizes = [(100,), (100, 100), (100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100, 100),
                          (100, 100, 100, 100, 100, 100), (100, 100, 100, 100, 100, 100, 100),
                          (100, 100, 100, 100, 100, 100, 100, 100), (100, 100, 100, 100, 100, 100, 100, 100, 100),
                          (100, 100, 100, 100, 100, 100, 100, 100, 100, 100)]
    n_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    print('Testing Num Layers on Income Data...')
    income_train_scores, income_valid_scores = validation_curve(MLPClassifier(), income_train_X, income_train_Y,
                                                                param_name="hidden_layer_sizes",
                                                                param_range=hidden_layer_sizes, scoring="accuracy",
                                                                cv=10)
    income_train_scores_avg = np.mean(income_train_scores, axis=1)
    income_valid_scores_avg = np.mean(income_valid_scores, axis=1)

    print('Done. Testing Num Layers on Music Data...')
    music_train_scores, music_valid_scores = validation_curve(MLPClassifier(), music_train_X, music_train_Y,
                                                              param_name="hidden_layer_sizes",
                                                              param_range=hidden_layer_sizes, scoring="accuracy", cv=10)
    music_train_scores_avg = np.mean(music_train_scores, axis=1)
    music_valid_scores_avg = np.mean(music_valid_scores, axis=1)

    print('Done. Creating Graphs...')
    plt.plot(n_layers, income_train_scores_avg, label='Training Scores')
    plt.plot(n_layers, income_valid_scores_avg, label='Validation Scores')
    plt.title('NN Validation Curve For Num Layers HP Tuning on Income Dataset')
    plt.xlabel('Num Layers')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='NNValidCurveNumLayersIncome')
    plt.close()
    nn_best_income_layer_size = (100, 100, 100, 100, 100)

    plt.plot(n_layers, music_train_scores_avg, label='Training Scores')
    plt.plot(n_layers, music_valid_scores_avg, label='Validation Scores')
    plt.title('NN Validation Curve For Num Layers HP Tuning on Music Dataset')
    plt.xlabel('Num Layers')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='NNValidCurveNumLayersMusic')
    plt.close()
    nn_best_music_layer_size = (100, 100, 100, 100, 100, 100, 100)

    print('Done. Testing Activation Functions on Income Data...')
    functions = ['identity', 'logistic', 'tanh', 'relu']

    income_train_scores, income_valid_scores = validation_curve(MLPClassifier(), income_train_X, income_train_Y,
                                                                param_name="activation", param_range=functions,
                                                                scoring="accuracy", cv=10)
    income_train_scores_avg = np.mean(income_train_scores, axis=1)
    income_valid_scores_avg = np.mean(income_valid_scores, axis=1)
    print('Done. Testing Activation Functions on Music Data...')
    music_train_scores, music_valid_scores = validation_curve(MLPClassifier(), music_train_X, music_train_Y,
                                                              param_name="activation", param_range=functions,
                                                              scoring="accuracy", cv=10)
    music_train_scores_avg = np.mean(music_train_scores, axis=1)
    music_valid_scores_avg = np.mean(music_valid_scores, axis=1)

    print('Done. Creating Graphs...')
    plt.plot(functions, income_train_scores_avg, label='Training Scores')
    plt.plot(functions, income_valid_scores_avg, label='Validation Scores')
    plt.title('NN Validation Curve For Activation Functions HP Tuning on Income Dataset')
    plt.xlabel('Activation Function')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='NNValidCurveActivationFunctionIncome')
    plt.close()
    nn_best_income_act = 'tanh'

    plt.plot(functions, music_train_scores_avg, label='Training Scores')
    plt.plot(functions, music_valid_scores_avg, label='Validation Scores')
    plt.title('NN Validation Curve For Activation Functions HP Tuning on Music Dataset')
    plt.xlabel('Activation Function')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='NNValidCurveActivationFunctionMusic')
    plt.close()
    nn_best_music_act = 'logistic'

    print('Done. Creating Learning Curve Data for Income Data...')
    train_sizes = np.linspace(.1, 1.0, 10)
    income_train_sizes, income_train_scores, income_valid_scores, income_fit_times, _ = learning_curve(MLPClassifier(
        hidden_layer_sizes=nn_best_income_layer_size, activation=nn_best_income_act), income_train_X, income_train_Y,
        cv=10, train_sizes=train_sizes, return_times=True)
    income_train_scores_avg = np.mean(income_train_scores, axis=1)
    income_valid_scores_avg = np.mean(income_valid_scores, axis=1)
    income_fit_times_avg = np.mean(income_fit_times, axis=1)

    print('Done. Creating Learning Curve Data for Music Data...')
    music_train_sizes, music_train_scores, music_valid_scores, music_fit_times, _ = learning_curve(MLPClassifier(
        hidden_layer_sizes=nn_best_music_layer_size, activation=nn_best_music_act), music_train_X, music_train_Y, cv=10,
        train_sizes=train_sizes, return_times=True)
    music_train_scores_avg = np.mean(music_train_scores, axis=1)
    music_valid_scores_avg = np.mean(music_valid_scores, axis=1)
    music_fit_times_avg = np.mean(music_fit_times, axis=1)

    print("Done. Creating Graphs...")
    plt.plot(income_train_sizes, income_train_scores_avg, label='Training Scores')
    plt.plot(income_train_sizes, income_valid_scores_avg, label='Validation Scores')
    plt.title('NN Learning Curve for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='NNLearningCurveIncome')
    plt.close()

    plt.plot(income_train_sizes, income_fit_times_avg)
    plt.title('NN Learning Curve With Training Time For Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='NNLearningCurveTimesIncome')
    plt.close()

    plt.plot(music_train_sizes, music_train_scores_avg, label='Training Scores')
    plt.plot(music_train_sizes, music_valid_scores_avg, label='Validation Scores')
    plt.title('NN Learning Curve for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='NNLearningCurveMusic')
    plt.close()

    plt.plot(music_train_sizes, music_fit_times_avg)
    plt.title('NN Learning Curve With Training Time For Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='NNLearningCurveTimesMusic')
    plt.close()

    print('Done with NN.')
    print('------------------------------Moving To SVM------------------------------')

    kernels = ["rbf", "sigmoid"]

    print('Testing Kernels on Income Data...')
    income_train_scores, income_valid_scores = validation_curve(SVC(), income_train_X, income_train_Y,
                                                                param_name="kernel", param_range=kernels,
                                                                scoring="accuracy", cv=10)
    income_train_scores_avg = np.mean(income_train_scores, axis=1)
    income_valid_scores_avg = np.mean(income_valid_scores, axis=1)

    print('Done. Testing Kernels on Music Data...')
    music_train_scores, music_valid_scores = validation_curve(SVC(), music_train_X, music_train_Y, param_name="kernel",
                                                              param_range=kernels, scoring="accuracy", cv=10)
    music_train_scores_avg = np.mean(music_train_scores, axis=1)
    music_valid_scores_avg = np.mean(music_valid_scores, axis=1)

    print('Done. Creating Graphs...')
    plt.plot(kernels, income_train_scores_avg, label='Training Scores')
    plt.plot(kernels, income_valid_scores_avg, label='Validation Scores')
    plt.title('SVM Validation Curve For Kernels HP Tuning on Income Dataset')
    plt.xlabel('Kernels')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='SVMValidCurveKernelsIncome')
    plt.close()
    svm_best_income_kernels = 'rbf'

    plt.plot(kernels, music_train_scores_avg, label='Training Scores')
    plt.plot(kernels, music_valid_scores_avg, label='Validation Scores')
    plt.title('SVM Validation Curve For Kernels HP Tuning on Music Dataset')
    plt.xlabel('Kernels')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='SVMValidCurveKernelsMusic')
    plt.close()
    svm_best_music_kernels = 'rbf'

    print('Done. Testing Degrees on Income Data...')
    degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    income_train_scores, income_valid_scores = validation_curve(SVC(), income_train_X, income_train_Y,
                                                                param_name="degree", param_range=degrees,
                                                                scoring="accuracy", cv=10)
    income_train_scores_avg = np.mean(income_train_scores, axis=1)
    income_valid_scores_avg = np.mean(income_valid_scores, axis=1)

    print('Done. Testing Degrees on Music Data...')
    music_train_scores, music_valid_scores = validation_curve(SVC(), music_train_X, music_train_Y, param_name="degree",
                                                              param_range=degrees, scoring="accuracy", cv=10)
    music_train_scores_avg = np.mean(music_train_scores, axis=1)
    music_valid_scores_avg = np.mean(music_valid_scores, axis=1)

    print('Done. Creating Graphs...')
    plt.plot(degrees, income_train_scores_avg, label='Training Scores')
    plt.plot(degrees, income_valid_scores_avg, label='Validation Scores')
    plt.title('SVM Validation Curve For Degrees HP Tuning on Income Dataset')
    plt.xlabel('Degree')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='SVMValidCurveDegreeIncome')
    plt.close()
    svm_best_income_degree = 2

    plt.plot(degrees, music_train_scores_avg, label='Training Scores')
    plt.plot(degrees, music_valid_scores_avg, label='Validation Scores')
    plt.title('SVM Validation Curve For Degrees HP Tuning on Music Dataset')
    plt.xlabel('Degree')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='SVMValidCurveDegreeMusic')
    plt.close()
    svm_best_music_degree = 2

    print('Done. Creating Learning Curve Data for Income Data...')
    train_sizes = np.linspace(.1, 1.0, 10)

    income_train_sizes, income_train_scores, income_valid_scores, income_fit_times, _ = learning_curve(SVC(
        kernel=svm_best_income_kernels, degree=svm_best_income_degree), income_train_X, income_train_Y, cv=10,
        train_sizes=train_sizes, return_times=True)
    income_train_scores_avg = np.mean(income_train_scores, axis=1)
    income_valid_scores_avg = np.mean(income_valid_scores, axis=1)
    income_fit_times_avg = np.mean(income_fit_times, axis=1)

    print('Done. Creating Learning Curve Data for Music Data...')
    music_train_sizes, music_train_scores, music_valid_scores, music_fit_times, _ = learning_curve(SVC(
        kernel=svm_best_music_kernels, degree=svm_best_music_degree), music_train_X, music_train_Y, cv=10,
        train_sizes=train_sizes, return_times=True)
    music_train_scores_avg = np.mean(music_train_scores, axis=1)
    music_valid_scores_avg = np.mean(music_valid_scores, axis=1)
    music_fit_times_avg = np.mean(music_fit_times, axis=1)

    print("Done. Creating Graphs...")
    plt.plot(income_train_sizes, income_train_scores_avg, label='Training Scores')
    plt.plot(income_train_sizes, income_valid_scores_avg, label='Validation Scores')
    plt.title('SVM Learning Curve for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.ylim(0.0, 1.1)
    plt.savefig(fname='SVMLearningCurveIncome')
    plt.close()

    plt.plot(income_train_sizes, income_fit_times_avg)
    plt.title('SVM Learning Curve With Training Time For Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='SVMLearningCurveTimesIncome')
    plt.close()

    plt.plot(music_train_sizes, music_train_scores_avg, label='Training Scores')
    plt.plot(music_train_sizes, music_valid_scores_avg, label='Validation Scores')
    plt.title('SVM Learning Curve for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='SVMLearningCurveMusic')
    plt.close()

    plt.plot(music_train_sizes, music_fit_times_avg)
    plt.title('SVM Learning Curve With Training Time For Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='SVMLearningCurveTimesMusic')
    plt.close()

    print('Done with SVM.')
    print('---------------Moving To KNN---------------')

    k = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

    print('Testing K on Income Data...')
    income_train_scores, income_valid_scores = validation_curve(KNeighborsClassifier(), income_train_X, income_train_Y,
                                                                param_name="n_neighbors", param_range=k,
                                                                scoring="accuracy", cv=10)
    income_train_scores_avg = np.mean(income_train_scores, axis=1)
    income_valid_scores_avg = np.mean(income_valid_scores, axis=1)

    print('Done. Testing K on Music Data...')
    music_train_scores, music_valid_scores = validation_curve(KNeighborsClassifier(), music_train_X, music_train_Y,
                                                              param_name="n_neighbors", param_range=k,
                                                              scoring="accuracy", cv=10)
    music_train_scores_avg = np.mean(music_train_scores, axis=1)
    music_valid_scores_avg = np.mean(music_valid_scores, axis=1)

    print('Done. Creating Graphs...')
    plt.plot(k, income_train_scores_avg, label='Training Scores')
    plt.plot(k, income_valid_scores_avg, label='Validation Scores')
    plt.title('KNN Validation Curve For K HP Tuning on Income Dataset')
    plt.xlabel('K')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='KNNValidCurveKIncome')
    plt.close()
    knn_best_income_k = 20

    plt.plot(k, music_train_scores_avg, label='Training Scores')
    plt.plot(k, music_valid_scores_avg, label='Validation Scores')
    plt.title('KNN Validation Curve For K HP Tuning on Music Dataset')
    plt.xlabel('K')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='KNNValidCurveKMusic')
    plt.close()
    knn_best_music_k = 20

    print('Done. Testing Leaf Sizes on Income Data...')
    leaf_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    income_train_scores, income_valid_scores = validation_curve(KNeighborsClassifier(), income_train_X, income_train_Y,
                                                                param_name="leaf_size", param_range=leaf_sizes,
                                                                scoring="accuracy", cv=10)
    income_train_scores_avg = np.mean(income_train_scores, axis=1)
    income_valid_scores_avg = np.mean(income_valid_scores, axis=1)

    print('Done. Testing Leaf Sizes on Music Data...')
    music_train_scores, music_valid_scores = validation_curve(KNeighborsClassifier(), music_train_X, music_train_Y,
                                                              param_name="leaf_size", param_range=leaf_sizes,
                                                              scoring="accuracy", cv=10)
    music_train_scores_avg = np.mean(music_train_scores, axis=1)
    music_valid_scores_avg = np.mean(music_valid_scores, axis=1)

    print('Done. Creating Graphs...')
    plt.plot(leaf_sizes, income_train_scores_avg, label='Training Scores')
    plt.plot(leaf_sizes, income_valid_scores_avg, label='Validation Scores')
    plt.title('KNN Validation Curve For Leaf Sizes HP Tuning on Income Dataset')
    plt.xlabel('Leaf Size')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='KNNValidCurveLeafSizeIncome')
    plt.close()
    knn_best_income_leaf_size = 60

    plt.plot(leaf_sizes, music_train_scores_avg, label='Training Scores')
    plt.plot(leaf_sizes, music_valid_scores_avg, label='Validation Scores')
    plt.title('KNN Validation Curve For Leaf Sizes HP Tuning on Music Dataset')
    plt.xlabel('Leaf Size')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')
    plt.savefig(fname='KNNValidCurveLeafSizeMusic')
    plt.close()
    knn_best_music_leaf_size = 20

    print('Done. Creating Learning Curve Data for Income Data...')
    train_sizes = np.linspace(.1, 1.0, 10)

    income_train_sizes, income_train_scores, income_valid_scores = learning_curve(KNeighborsClassifier(
       n_neighbors=knn_best_income_k, leaf_size=knn_best_income_leaf_size), income_train_X, income_train_Y, cv=10,
       train_sizes=train_sizes)
    income_train_scores_avg = np.mean(income_train_scores, axis=1)
    income_valid_scores_avg = np.mean(income_valid_scores, axis=1)

    print('Done. Creating Learning Curve Data for Music Data...')
    music_train_sizes, music_train_scores, music_valid_scores = learning_curve(KNeighborsClassifier(
        n_neighbors=knn_best_music_k, leaf_size=knn_best_music_leaf_size), music_train_X, music_train_Y, cv=10,
        train_sizes=train_sizes)
    music_train_scores_avg = np.mean(music_train_scores, axis=1)
    music_valid_scores_avg = np.mean(music_valid_scores, axis=1)

    print("Done. Creating Graphs...")
    plt.plot(income_train_sizes, income_train_scores_avg, label='Training Scores')
    plt.plot(income_train_sizes, income_valid_scores_avg, label='Validation Scores')
    plt.title('KNN Learning Curve for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='KNNLearningCurveIncome')
    plt.close()

    plt.plot(music_train_sizes, music_train_scores_avg, label='Training Scores')
    plt.plot(music_train_sizes, music_valid_scores_avg, label='Validation Scores')
    plt.title('KNN Learning Curve for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='KNNLearningCurveMusic')
    plt.close()

    print('Done with KNN.')
    print('---------------------------- Comparing all Algorithms ---------------------------------')

    print('Creating Final Learning Curve Data for DTree...')
    train_sizes = np.linspace(.1, 1.0, 10)


    dtree_income_train_sizes, dtree_income_train_scores, dtree_income_test_scores, dtree_income_fit_times, _ = \
        learning_curve(DecisionTreeClassifier(max_depth=dtree_best_income_depth,
                                              min_samples_split=dtree_best_income_sample_split), income_train_X,
                       income_train_Y, cv=10, train_sizes=train_sizes, return_times=True)

    dtree_music_train_sizes, d_tree_music_train_scores, dtree_music_test_scores, dtree_music_fit_times, _ = \
        learning_curve(DecisionTreeClassifier(max_depth=dtree_best_music_depth,
                                              min_samples_split=dtree_best_music_sample_split), music_train_X, music_train_Y,
                       cv=10, train_sizes=train_sizes, return_times=True)

    dtree_income_train_scores_avg = np.mean(dtree_income_train_scores, axis=1)
    dtree_income_valid_scores_avg = np.mean(dtree_income_test_scores, axis=1)
    dtree_income_fit_times_avg = np.mean(dtree_income_fit_times, axis=1)
    dtree_music_train_scores_avg = np.mean(dtree_income_train_scores, axis=1)
    dtree_music_valid_scores_avg = np.mean(dtree_income_test_scores, axis=1)
    dtree_music_fit_times_avg = np.mean(dtree_income_fit_times, axis=1)

    print('Done. Creating Final Learning Curve Data for Boosting...')
    boost_income_train_sizes, boost_income_train_scores, boost_income_test_scores, boost_income_fit_times, _ = \
        learning_curve(GradientBoostingClassifier(max_depth=boosting_best_income_depth,
                                                  n_estimators=boosting_best_income_n_estimators), income_train_X,
                       income_train_Y, cv=10, train_sizes=train_sizes, return_times=True)

    boost_music_train_sizes, boost_music_income_train_scores, boost_music_test_scores, boost_music_fit_times, _ = \
        learning_curve(GradientBoostingClassifier(max_depth=boosting_best_music_depth,
                                                  n_estimators=boosting_best_music_n_estimators), music_train_X,
                       music_train_Y, cv=10, train_sizes=train_sizes, return_times=True)

    boost_income_train_scores_avg = np.mean(boost_income_train_scores, axis=1)
    boost_income_valid_scores_avg = np.mean(boost_income_test_scores, axis=1)
    boost_income_fit_times_avg = np.mean(boost_income_fit_times, axis=1)
    boost_music_train_scores_avg = np.mean(boost_income_train_scores, axis=1)
    boost_music_valid_scores_avg = np.mean(boost_income_test_scores, axis=1)
    boost_music_fit_times_avg = np.mean(boost_income_fit_times, axis=1)

    print('Done. Creating Final Learning Curve Data for NN...')
    nn_income_train_sizes, nn_income_train_scores, nn_income_test_scores, nn_income_fit_times, _ = \
        learning_curve(MLPClassifier(hidden_layer_sizes=nn_best_income_layer_size, activation=nn_best_income_act),
                       income_train_X, income_train_Y, cv=10, train_sizes=train_sizes, return_times=True)

    nn_music_train_sizes, nn_music_train_scores, nn_music_test_scores, nn_music_fit_times, _ = \
        learning_curve(MLPClassifier(hidden_layer_sizes=nn_best_music_layer_size, activation=nn_best_music_act),
                       music_train_X, music_train_Y, cv=10, train_sizes=train_sizes, return_times=True)

    nn_income_train_scores_avg = np.mean(nn_income_train_scores, axis=1)
    nn_income_valid_scores_avg = np.mean(nn_income_test_scores, axis=1)
    nn_income_fit_times_avg = np.mean(nn_income_fit_times, axis=1)
    nn_music_train_scores_avg = np.mean(nn_income_train_scores, axis=1)
    nn_music_valid_scores_avg = np.mean(nn_income_test_scores, axis=1)
    nn_music_fit_times_avg = np.mean(nn_income_fit_times, axis=1)

    print('Done. Creating Final Learning Curve Data for SVM...')
    svm_income_train_sizes, svm_income_train_scores, svm_income_test_scores, svm_income_fit_times, _ = learning_curve(
        SVC(kernel=svm_best_income_kernels, degree=svm_best_income_degree), income_train_X, income_train_Y, cv=10,
        train_sizes=train_sizes, return_times=True)

    svm_music_train_sizes, svm_music_train_scores, svm_music_test_scores, svm_music_fit_times, _ = learning_curve(SVC(
        kernel=svm_best_music_kernels, degree=svm_best_music_degree), music_train_X, music_train_Y, cv=10,
        train_sizes=train_sizes, return_times=True)

    svm_income_train_scores_avg = np.mean(svm_income_train_scores, axis=1)
    svm_income_valid_scores_avg = np.mean(svm_income_test_scores, axis=1)
    svm_income_fit_times_avg = np.mean(svm_income_fit_times, axis=1)
    svm_music_train_scores_avg = np.mean(svm_income_train_scores, axis=1)
    svm_music_valid_scores_avg = np.mean(svm_income_test_scores, axis=1)
    svm_music_fit_times_avg = np.mean(svm_income_fit_times, axis=1)

    print('Done. Creating Final Learning Curve Data for KNN...')
    knn_income_train_sizes, knn_income_train_scores, knn_income_test_scores, knn_income_fit_times, _ = \
        learning_curve(KNeighborsClassifier(n_neighbors=knn_best_income_k, leaf_size=knn_best_income_leaf_size),
                       income_train_X, income_train_Y, cv=10, train_sizes=train_sizes, return_times=True)

    knn_music_train_sizes, knn_music_train_scores, knn_music_test_scores, knn_music_fit_times, _ = learning_curve(
        KNeighborsClassifier(n_neighbors=knn_best_music_k, leaf_size=knn_best_music_leaf_size), music_train_X,
        music_train_Y, cv=10, train_sizes=train_sizes, return_times=True)

    knn_income_train_scores_avg = np.mean(knn_income_train_scores, axis=1)
    knn_income_valid_scores_avg = np.mean(knn_income_test_scores, axis=1)
    knn_income_fit_times_avg = np.mean(knn_income_fit_times, axis=1)
    knn_music_train_scores_avg = np.mean(knn_income_train_scores, axis=1)
    knn_music_valid_scores_avg = np.mean(knn_income_test_scores, axis=1)
    knn_music_fit_times_avg = np.mean(knn_income_fit_times, axis=1)

    print('Done. Creating Graphs.')
    plt.plot(dtree_income_train_sizes, dtree_income_valid_scores_avg, label='Decision Tree')
    plt.plot(boost_income_train_sizes, boost_income_valid_scores_avg, label='Boosting')
    plt.plot(nn_income_train_sizes, nn_income_valid_scores_avg, label='NN')
    plt.plot(svm_income_train_sizes, svm_income_valid_scores_avg, label='SVM')
    plt.plot(knn_income_train_sizes, knn_income_valid_scores_avg, label='KNN')
    plt.title('Final Learning Curve For Tuned Algorithms for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='FinalLearningCurveIncome')
    plt.close()

    plt.plot(dtree_music_train_sizes, dtree_music_valid_scores_avg, label='Decision Tree')
    plt.plot(boost_music_train_sizes, boost_music_valid_scores_avg, label='Boosting')
    plt.plot(nn_music_train_sizes, nn_music_valid_scores_avg, label='NN')
    plt.plot(svm_music_train_sizes, svm_music_valid_scores_avg, label='SVM')
    plt.plot(knn_music_train_sizes, knn_music_valid_scores_avg, label='KNN')
    plt.title('Final Learning Curve For Tuned Algorithms for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(fname='FinalLearningCurveMusic')
    plt.close()

    plt.plot(dtree_income_train_sizes, dtree_income_fit_times_avg)
    plt.plot(boost_income_train_sizes, boost_income_fit_times_avg)
    plt.plot(nn_income_train_sizes, nn_income_fit_times_avg)
    plt.plot(svm_income_train_sizes, svm_income_fit_times_avg)
    plt.plot(knn_income_train_sizes, knn_income_fit_times_avg)
    plt.title('Final Learning Curve For Tuned Algorithms for Income Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='FinalLearningCurveTimesIncome')
    plt.close()

    plt.plot(dtree_music_train_sizes, dtree_music_fit_times_avg)
    plt.plot(boost_music_train_sizes, boost_music_fit_times_avg)
    plt.plot(nn_music_train_sizes, nn_music_fit_times_avg)
    plt.plot(svm_music_train_sizes, svm_music_fit_times_avg)
    plt.plot(knn_music_train_sizes, knn_music_fit_times_avg)
    plt.title('Final Learning Curve For Tuned Algorithms for Music Dataset')
    plt.xlabel('Training Sizes')
    plt.ylabel('Fit Time')
    plt.savefig(fname='FinalLearningCurveTimesMusic')
    plt.close()

    print('Done.')