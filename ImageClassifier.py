import os
import numpy
from numpy import array
import numpy as np
import cv2
import pickle
import pylab as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import itertools

def GetDefaultParameters():
    """
        function control all parameters in the model, save the result, tune first.

        to Change: class - edit data['fold2'] to change classes.
                    path - local path to picture folder
                    tune - toTune - if you want the model tune himself instead using the default parameter's value
                    pixle_range, bleck_cell_range, c_range - control the ranges of the parameters - not nesecery
                    data - data['data'] contain info about saving the result. you define the path and the file name.

    """
    data = {}
    data['s'] = 180
    data['fold1'] = [11, 12, 40, 32, 15, 16, 43, 18, 19, 20]
    data['fold2'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    data['path'] = "D:\\101_ObjectCategories"
    data['tune'] = {
        "toTune": True,
        "pixle_range": range(6, 20),
        "bleck_cell_range": range(3, 8),
        "c_range": numpy.logspace(-2, 2, num=4, endpoint=False)
    }
    data['split'] = 20  # pic in test

    data['data'] = {
        "Results_Path": 'D:/',
        "Results_FileName": 'ResultsOfEx1.pkl',
        "save_data": False
    }
    data['prepare'] = {  # parameters for HOG
        "spatial_cell_size": 17, # range(4, 14, 2)
        "orientation_bins": 9, # range(5, 12)
        "cells_per_block": 4 # range(1, 5)
    }

    data['train'] = 1  # C for svm

    data['summary'] = 2  # 2 pic to see with the largest error

    return data


def get_data(path_data, S, fold):
    """function that get data path and create a matrix of N*S*S and labels vector
        input : path_data: path of data folder,
                S:         size want for image to resize by
                fold:      classes input
        output: loaded_data:  X's - matrix N*N*S, Y's - labels
        """
    print("--------------------laoding pictures-----------------------")
    img_count = 0
    fold_index = 0

    loaded_data = {}
    loaded_data['data'] = {}
    loaded_data['labels'] = {}
    loaded_data['originalPic'] = {}

    fold_folders = []
    folders = os.listdir(path_data)
    for i in fold:
        fold_folders.append(folders[i-1])

    loaded_data['class_names'] = fold_folders
    for fold_name in fold_folders:
        pathFol = path_data + "\\" + fold_name
        fold_index = fold_index + 1

        loaded_data['data'][fold_name] = []
        loaded_data['labels'][fold_name] = []
        loaded_data['originalPic'][fold_name] = []
        for img_name in os.listdir(pathFol):
            img_count = img_count + 1
            pathImg = pathFol + "\\" + img_name
            RBG_image = cv2.imread(pathImg)
            Gray_image = cv2.cvtColor(RBG_image, cv2.COLOR_BGR2GRAY)
            REsize_img = cv2.resize(Gray_image, (S, S))

            loaded_data['data'][fold_name].append(REsize_img)
            loaded_data['labels'][fold_name].append(fold_index)
            loaded_data['originalPic'][fold_name].append(RBG_image)

    return loaded_data


def split_train_test(fold, X, Y, pic_in_test, rgb_pic):
    splited_data = {}
    splited_data['Train_data'] = [];
    splited_data['Train_labels'] = []
    splited_data['Val_data'] = [];
    splited_data['Val_labels'] = []
    splited_data['Test_data'] = [];
    splited_data['Test_labels'] = []
    splited_data['Test_rgb_image'] = []

    for class_name in list(X):
        pic_in_class = len(list(X[class_name]))
        TT_A = range(0, pic_in_class)
        if pic_in_class > pic_in_test * 2:
            X_train, X_test = train_test_split(TT_A, train_size=pic_in_class - pic_in_test, test_size=pic_in_test)
        else:
            X_train, X_test = train_test_split(TT_A, train_size=pic_in_test, test_size=pic_in_class - pic_in_test)

        # for second partition to train and val
        if params['tune']:
            flag = True
            if len(X_train) > pic_in_test*2:
                x_train, X_val = train_test_split(X_train, train_size=pic_in_class - pic_in_test-len(X_test), test_size=pic_in_test)
            else:
                if len(X_train) > pic_in_test:
                    x_train, X_val = train_test_split(X_train, train_size=pic_in_test, test_size=pic_in_class - pic_in_test-len(X_test))
                else:
                    x_train = X_train
                    flag = False
            if flag:
                for t in sorted(X_val):
                    splited_data['Val_data'].append(X[class_name][t])
                    splited_data['Val_labels'].append(Y[class_name][t])

        for i in sorted(x_train):
            splited_data['Train_data'].append(X[class_name][i])
            splited_data['Train_labels'].append(Y[class_name][i])

        for j in sorted(X_test):
            splited_data['Test_data'].append(X[class_name][j])
            splited_data['Test_labels'].append(Y[class_name][j])
            splited_data['Test_rgb_image'].append(rgb_pic[class_name][j])

    return splited_data


def prepare(X, param):
    """"get ready to SVM, the function calculates the HOG features
        input :     X:     Data (train / test) depend the phase in the process.
                    Param: parameters to init in the HOG algorithm as spatial_cell_size & orientation_bins
        output:     prepared_data: np array contain all hog feature of images in X
"""
    spatial_cell_size = param['spatial_cell_size']
    orientation_bins = param['orientation_bins']
    cell_block = param['cells_per_block']
    prepared_data = []
    for image in range(0, len(X)):
        feature = hog(X[image], orientations=orientation_bins,
                      pixels_per_cell=(spatial_cell_size, spatial_cell_size), cells_per_block=(cell_block, cell_block))
        prepared_data.append(feature)

    return prepared_data


def Train(X, Y, param):
    """ The function fit a linear model with givven parameters
    input: X - Data
           Y - labels
           param - C value for the linear SMV
    output:
           clf - fitted linear SVM
    """
    linear_svm = LinearSVC(loss='squared_hinge', dual=True, tol=0.0001, C=param, multi_class='ovr',
                           fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0,
                           random_state=None, max_iter=2000)

    return linear_svm.fit(X, Y)


def Test(trained_svm, X):
    """ The function predict labels of X with trained_svm, get the decision function and classes for future calculations
        input:
               trained_svm - fitted linear SVM to the train data
               X - Data
        output:
               pred_labels - predicted labels
        """
    Results = {
        "Model": [],
        "DecisionMatrix": [],
        "ModelClasses": []
    }
    Results["Model"] = trained_svm.predict(X)
    Results["DecisionMatrix"] = trained_svm.decision_function(X)
    Results["ModelClasses"] = trained_svm.classes_
    return Results


def TrainWithTuning(X, Y, c_range):
    """ The function calls when you need to tune the model's parameters .
           input:
                  trained_svm - fitted linear SVM to the train data
                  X - Data
           output:
                  classifiers - predicted labels
           """
    c_range = c_range.tolist()
    classifiers = []
    for c in c_range:
        linear_svm = LinearSVC(loss='squared_hinge', dual=True, tol=0.0001, C=c, multi_class='ovr',
                               fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0,
                               random_state=None, max_iter=2000)
        clf = linear_svm.fit(X, Y)
        classifiers.append(clf)

    return classifiers


def Evaluate(y_pred, y_actualy, param):
    summary = {}
    incorrect = [y_pred != y_actualy]
    summary['error_rate'] = 1 - accuracy_score(y_actualy, y_pred)
    # print(summary['error_rate'])
    summary['confusion_mtx'] = confusion_matrix(y_actualy, y_pred)
    summary['incorrects'] = incorrect

    return summary


def save_acc_results(data, path):
    accuracy_file = open(path, 'wb')
    pickle.dump(data, accuracy_file)


def read_acc_results(name):
    accuracy_file = open(name, 'rb')
    return pickle.load(accuracy_file)


def compute_plot_big_error_pics(DecisionMatrix, ModelClasses, ResultsPredictions, X_test, Y_test):
    '''
       This function gets the error images for each class and print those who has
       the largest error. the function loop over the classes. for each class
       all the error images (compared to the real label) stacked into a list.
       and for those the distance from the wrong label is computed. in this way
       we can print the biggests error for each class
       :input Params:
           DecisionMatrix - see Test()
           ModelClasses - a List containg all of the experiment classes
           ResultsPredictions  - list of the model predictions
           SplitData - see TrainTestSplit()

       :output Params:
           LargestErrorIndexes: the test set indexes of the largests error for the experiment
    '''
    LargestErrorIndexes = []
    for i in range(0, len(ModelClasses)):

        ImagesAndScore = {
            "Image": [],
            "Score": [],
            "LargestErrorIndexes": []
        }
        for j in range(0, len(ResultsPredictions)):
            if ModelClasses[i] == Y_test[j]:
                if ResultsPredictions[j] != Y_test[j]:
                    RealClassScore = DecisionMatrix[j][i]
                    MaximalIncorrectScore = max(DecisionMatrix[j])
                    ImagesAndScore["Score"].append(RealClassScore - MaximalIncorrectScore)
                    ImagesAndScore["Image"].append([X_test[j]])
                    ImagesAndScore["LargestErrorIndexes"].append(j)
        print("error images for class ", ModelClasses[i], ":")
        if len(ImagesAndScore["Score"]) == 0:
            print("no errors for this class")
        if len(ImagesAndScore["Score"]) == 1:
            LargestErrorIndexes.append(ImagesAndScore["LargestErrorIndexes"][0])
            plt.imshow(ImagesAndScore["Image"][0][0])
            plt.show()
        if len(ImagesAndScore["Score"]) == 2:
            LargestErrorIndexes.append(ImagesAndScore["LargestErrorIndexes"][0])
            LargestErrorIndexes.append(ImagesAndScore["LargestErrorIndexes"][1])
            plt.imshow(ImagesAndScore["Image"][0][0])
            plt.show()
            plt.imshow(ImagesAndScore["Image"][1][0])
            plt.show()
        if len(ImagesAndScore["Score"]) > 2:
            index = ImagesAndScore["Score"].index(min(ImagesAndScore["Score"]))
            LargestErrorIndexes.append(ImagesAndScore["LargestErrorIndexes"][index])
            plt.imshow(ImagesAndScore["Image"][index][0])
            plt.show()
            del ImagesAndScore["Score"][index]
            del ImagesAndScore["Image"][index]
            index = ImagesAndScore["Score"].index(min(ImagesAndScore["Score"]))
            LargestErrorIndexes.append(ImagesAndScore["LargestErrorIndexes"][index])
            plt.imshow(ImagesAndScore["Image"][index][0])
            plt.show()
    return LargestErrorIndexes


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Reds):
    '''
       This function prints and plots the confusion matrix.
       Normalization can be applied by setting `normalize=True`
       :input Params:
           cm - the Confusion Matrix
           classes - a List containg all of the experiment classes
           normalize  - boolean indicates wheater to normalize the Confusion Matrix or not
           title - the Confusion Matrix title
           cmap - the colors
       :output Params: none
    '''

    numpy.set_printoptions(precision=2)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def ReportResults(summary_report, X_test, Y_test, results, param):
    """ The function calls when you need first to tune the model's parameters.
               input:
                      trained_svm - fitted linear SVM to the train data
                      X - Data
               output:
                      pred_labels - predicted labels
               """

    myset = set(Y_test)
    mynewlist = list(myset)

    plt.figure()
    plot_confusion_matrix(summary_report['confusion_mtx'], classes=results['ModelClasses'],
                          title='Confusion matrix, without normalization')
    print("the test error percentage is :", round(summary_report["error_rate"], 5))

    largest_error_img = compute_plot_big_error_pics(results['DecisionMatrix'], results["ModelClasses"],
                                                    results["Model"], X_test, Y_test)

    ResultsStatistics = {
        "ErrorRate": Summary["error_rate"],
        "ConfusionMatrix": Summary["confusion_mtx"],
        "LargestErrorIndices": largest_error_img
    }
    if param['save_data']:
        ResultsFullPath = os.path.join(param["Results_Path"], param['Results_FileName'])
        #pickle.dump(ResultsStatistics, open(ResultsFullPath, "wb"))
        save_acc_results(data=ResultsStatistics, path=ResultsFullPath)

def plot_graph(acc_array, data_range):
    """ The function plots the Error Rate

    """
    plt.plot(acc_array, data_range)
    plt.title('Error Rate - Spatial cell size')
    plt.ylabel('Error Rate')


def tune_model(Xstrain, Xsval, Ystrain, Ysval, param):
    """ The function calls when you need first to tune the model's parameters before even train the model.
        from the input of the ranges of each hiper-parameter the function find the combination with the lowest
        error rate on the validation set.
               input:
                    Xstrain, Xsval: X's values splitted to train and validation
                    Ystrain, Ysval: Y's values splitted also
                    param: ranges of Hiper-parameters

               output:
                      pred_labels - predicted labels
               """
    a = {}
    a['orientation_bins'] = 9
    tune_result = {}
    tune_result['spatial_cell_size'] = 15
    tune_result['cells_per_block'] = 3
    tune_result['Train'] = 0
    tune_result['score'] = 1
    for pixel_size in param['pixle_range']: #range(15, 18):
        for cell_block in param['bleck_cell_range']: #range(3, 6):
            a['spatial_cell_size'] = pixel_size
            a['cells_per_block'] = cell_block
            dataRep = {}
            dataRep['test'] = prepare(Xsval, a)
            dataRep['train'] = prepare(Xstrain, a)
            tuning_classifieres = TrainWithTuning(dataRep['train'], Ystrain, param['c_range'])
            for clf, c in zip(tuning_classifieres, param['c_range']): # numpy.logspace(-1, 2, num=3, endpoint=False)):
                result = Test(clf, dataRep['test'])
                Summary = Evaluate(result['Model'], Ysval, params['summary'])
                if tune_result['score'] > Summary['error_rate']:
                    tune_result['spatial_cell_size'] = pixel_size
                    tune_result['cells_per_block'] = cell_block
                    tune_result['train'] = c
                    tune_result['score'] = Summary['error_rate']
                print("C :", c, "spatial_cell_size :", pixel_size, "cells_per_block :", cell_block)
                print("the test error percentage is :", Summary["error_rate"])

    print("----------------Finish tuning. Best score are:------------------")
    print("C :", tune_result['train'], "spatial_cell_size :", tune_result['spatial_cell_size'], "cells_per_block :",
          tune_result['cells_per_block'])
    print("the test error percentage is :", tune_result['score'])

    return tune_result

if __name__ == "__main__":

    np.random.seed(7)

    params = GetDefaultParameters()
    DandL = get_data(params['path'], params['s'], params['fold2'])  # load the pictures

    splitData = split_train_test(params['fold2'], DandL['data'], DandL['labels'], params['split'],
                                 DandL['originalPic'])  # split train{train,val}, test
    dataRep = {}
    if params['tune']['toTune']:
        tuning_params = tune_model(splitData["Train_data"], splitData["Val_data"],
                                   splitData["Train_labels"], splitData['Val_labels'],params['tune'])
        params['prepare']['spatial_cell_size'] = tuning_params['spatial_cell_size']
        params['prepare']['cells_per_block'] = tuning_params['cells_per_block']
        params['train'] = tuning_params['train']

    dataRep['test'] = prepare(splitData['Test_data'], params['prepare'])
    dataRep['train'] = prepare(splitData['Train_data'], params['prepare'])

    clf = Train(dataRep["train"], splitData["Train_labels"], params['train'])
    result = Test(clf, dataRep['test'])
    Summary = Evaluate(result['Model'], splitData['Test_labels'], params['summary'])

    # Report and save results
    ReportResults(Summary, splitData['Test_rgb_image'], splitData['Test_labels'], result, params['data'])

hh = 5
