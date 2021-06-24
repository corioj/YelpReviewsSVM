# EECS 445 - Fall 2020
# Project 1 - project1.py

import pandas as pd
import numpy as np
import itertools
import string
import nltk

# added part 5 libraries
from nltk.corpus import stopwords
from sklearn import preprocessing

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from matplotlib import pyplot as plt

from helper import *

def extract_dictionary(df):
    """
    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was found).
    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary of distinct words that maps each distinct word
        to a unique index corresponding to when it was first found while
        iterating over all words in each review in the dataframe df
    """
    word_dict = {}
    # punctuation constant
    for char in string.punctuation:
        df['reviewText'] = df['reviewText'].apply(lambda x: x.replace(char, " "))
    
    # index counter
    i = 0
    
    # apply vectorized lowercase
    df['reviewText'] = df['reviewText'].str.lower()
    
    # for each msg in content
    for msg in df['reviewText']:
        # split into separate words
        contents = msg.split()
        # iterate thru split msg
        for word in contents:
            # check if current word is already found
            if word not in word_dict:
                # if not, insert it w/ key = word, val = index
                word_dict[word] = i
                i = i + 1
    # return dict
    return word_dict

def generate_feature_matrix(df, word_dict):
    """
    Reads a dataframe and the dictionary of unique words
    to generate a matrix of {1, 0} feature vectors for each review.
    Use the word_dict to find the correct index to set to 1 for each place
    in the feature vector. The resulting feature matrix should be of
    dimension (# of reviews, # of words in dictionary).
    Input:
        df: dataframe that has the ratings and labels
        word_dict: dictionary of words mapping to indices
    Returns:
        a feature matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    # go thru each review
    for i in range(0, number_of_reviews):
        # store text in a list
        msg = (df.iloc[i]['reviewText']).split()
        # go thru words in msg and mark their respective indices w a 1
        for word in msg:
            if word in word_dict:
                # word_dict[word] gives index to place 1
                feature_matrix[i][word_dict[word]] = 1
    return feature_matrix

################################## PART 5 FUNCTIONS ############################################
################################## PART 5 FUNCTIONS ############################################
################################## PART 5 FUNCTIONS ############################################
################################## PART 5 FUNCTIONS ############################################
################################## PART 5 FUNCTIONS ############################################
def extract_dictionary_part5(df):
    """
    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was found).
    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary of distinct words that maps each distinct word
        to a unique index corresponding to when it was first found while
        iterating over all words in each review in the dataframe df
    """
    word_dict = {}
    # part 5 edit
    stop_words = set(stopwords.words('english'))
    # punctuation constant
    for char in string.punctuation:
        df['reviewText'] = df['reviewText'].apply(lambda x: x.replace(char, " "))
    
    # index counter
    i = 0
    
    # apply vectorized lowercase
    df['reviewText'] = df['reviewText'].str.lower()
    
    # for each msg in content
    for msg in df['reviewText']:
        # split into separate words
        contents = [w for w in msg.split() if w not in stop_words]
        # iterate thru split msg
        for word in contents:
            # check if current word is already found
            if word not in word_dict:
                # if not, insert it w/ key = word, val = index
                word_dict[word] = i
                i = i + 1
    # return dict
    return word_dict

################################## PART 5 FUNCTIONS ############################################
################################## PART 5 FUNCTIONS ############################################
################################## PART 5 FUNCTIONS ############################################
################################## PART 5 FUNCTIONS ############################################
################################## PART 5 FUNCTIONS ############################################
def generate_feature_matrix_part5(df, word_dict):
    """
    Reads a dataframe and the dictionary of unique words
    to generate a matrix of {1, 0} feature vectors for each review.
    Use the word_dict to find the correct index to set to 1 for each place
    in the feature vector. The resulting feature matrix should be of
    dimension (# of reviews, # of words in dictionary).
    Input:
        df: dataframe that has the ratings and labels
        word_dict: dictionary of words mapping to indices
    Returns:
        a feature matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    # create a feature data frame instead of a matrix filled w/ 0's
    feature_df = pd.DataFrame(np.zeros((number_of_reviews, number_of_words)), index=np.arange(number_of_reviews), columns = list(word_dict.keys()))
    # go thru each review
    for i in range(0, number_of_reviews):
        # store review text in a list
        msg = (df.iloc[i]['reviewText']).split()
        # go thru words in review
        for word in msg:
            # if it exists, increment amount by 1
            if word in word_dict:
                feature_df.iloc[i][word] += 1
    return feature_df

################################## PART 5 FUNCTIONS ############################################
################################## PART 5 FUNCTIONS ############################################
################################## PART 5 FUNCTIONS ############################################
################################## PART 5 FUNCTIONS ############################################
################################## PART 5 FUNCTIONS ############################################


def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted labels y_pred.
    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    # TODO: Implement this function
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.

def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    # TODO: Implement this function
    #HINT: You may find the StratifiedKFold from sklearn.model_selection
    #to be useful

    #Put the performance of the model on each fold in the scores array
    scores = []
    # get scoring method
    method = metric
    # do StratifiedKFold stuff
    skf = StratifiedKFold(n_splits = k)
    
    # fit an SVC on each training fold, then get performance across testing folds and put into scores vector
    # .iloc added for part 5 data frame support
    for trainindices, testindices in skf.split(X, y):
        clf.fit(X = X.iloc[trainindices], y = y[trainindices])
        pred = clf.predict(X.iloc[testindices])
        score = 0
        if method == 'auroc':
            # decision boundary
            test = clf.decision_function(X[testindices,])
            score = metrics.roc_auc_score(y_true = y[testindices], y_score = test)
        elif method == 'accuracy':
            score = clf.score(X = X.iloc[testindices], y = y[testindices])
        elif method == 'f1-score':
            score = metrics.f1_score(y_true = y[testindices], y_pred = pred)
        elif method == 'precision':
            score = metrics.precision_score(y_true = y[testindices], y_pred = pred)
        elif method == 'sensitivity':
            # Sensitivity = (True Positive)/(True Positive + False Negative)
            tn, fp, fn, tp = metrics.confusion_matrix(y[testindices], pred, labels = [1,-1]).ravel()
            score = tp/(tp + fn)
        elif method == 'specificity':
            # Specificity = (True Negative)/(True Negative + False Positive)
            tn, fp, fn, tp = metrics.confusion_matrix(y[testindices], pred, labels = [1,-1]).ravel()
            score = tn/(tn + fp)
        scores.append(score)
    #And return the average performance across all fold splits.
    return np.array(scores).mean()

def select_classifier(penalty='l2', c=1.0, degree=1, r=0.0, class_weight='balanced'):
    """
    Return a linear svm classifier based on the given
    penalty function and regularization parameter c.
    """
    # TODO: Optionally implement this helper function if you would like to
    # instantiate your SVM classifiers in a single function. You will need
    # to use the above parameters throughout the assignment.

def select_param_linear(X, y, k=5, metric="accuracy", C_range = [], penalty='l2'):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
    Returns:
        The parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    best_C_val = C_range[0]
    best_metric = 0
    # TODO: Implement this function
    #HINT: You should be using your cv_performance function here
    #to evaluate the performance of each SVM
    # loop over C_range, fit a clf, pass into cv_performance, get best_C_val from results
    for i in C_range:
        clf = SVC(kernel = 'linear', C = i, class_weight = "balanced")
        score = cv_performance(clf = clf, X = X, y = y, k=k, metric=metric)
        print("C value: " + str(i) + ", score: " + str(score))
        if score > best_metric:
            best_C_val = i
            best_metric = score
    # SHOULD I BE RETURNING THE PERFORMANCE HERE AS WELL?
    return best_C_val, best_metric


def plot_weight(X,y,penalty,C_range):
    """
    Takes as input the training data X and labels y and plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier.
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")
    norm0 = []

    # TODO: Implement this part of the function
    #Here, for each value of c in C_range, you should
    #append to norm0 the L0-norm of the theta vector that is learned
    #when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)
    if penalty == 'l2':
        for c in C_range:
            clf = SVC(kernel = 'linear', C = c, class_weight = "balanced").fit(X,y)
            theta = clf.coef_
            norm = np.linalg.norm(theta[0], 0)
            norm0.append(norm)
    elif penalty == 'l1':
        for c in C_range:
            clf = LinearSVC(penalty = 'l1', dual = False, C = c, class_weight='balanced').fit(X,y)
            theta = clf.coef_
            norm = np.linalg.norm(theta[0], 0)
            norm0.append(norm)

    #This code will plot your L0-norm as a function of c
    plt.plot(C_range, norm0)
    plt.xscale('log')
    plt.legend(['L0-norm'])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title('Norm-'+penalty+'_penalty.png')
    plt.savefig('Norm-'+penalty+'_penalty.png')
    plt.close()


def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """
        Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
        calculating the k-fold CV performance for each setting on X, y.
        Input:
            X: (n,d) array of feature vectors, where n is the number of examples
               and d is the number of features
            y: (n,) array of binary labels {1,-1}
            k: an int specifying the number of folds (default=5)
            metric: string specifying the performance metric (default='accuracy'
                     other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                     and 'specificity')
            param_range: a (num_param, 2)-sized array containing the
                parameter values to search over. The first column should
                represent the values for C, and the second column should
                represent the values for r. Each row of this array thus
                represents a pair of parameters to be tried together.
        Returns:
            The parameter values for a quadratic-kernel SVM that maximize
            the average 5-fold CV performance as a pair (C,r)
    """
    best_C_val,best_r_val = 0.0, 0.0
    # TODO: Implement this function
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...
    # loop over C_range, fit a clf, pass into cv_performance, get best_C_val from results
    
    best_metric = 0
    # loop over C_range
    for i in param_range[0]:
        # loop over r_range
        for j in param_range[1]:
            # fit polynomial degree 2 SVM and perform CV score
            clf = SVC(kernel = 'poly', degree = 2, C = i, coef0 = j, class_weight = "balanced")
            score = cv_performance(clf = clf, X = X, y = y, k=k, metric=metric)
            # score check
            print("C value: " + str(i) + ", r value: " + str(j) + ", score: " + str(score))
            if score > best_metric:
                best_C_val = i
                best_r_val = j
                best_metric = score
    # return our tuple
    return best_C_val,best_r_val,best_metric

def main():
    # Read binary data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_matrix AND extract_dictionary
    
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    IMB_features, IMB_labels = get_imbalanced_data(dictionary_binary)
    IMB_test_features, IMB_test_labels = get_imbalanced_test(dictionary_binary)
    
    # TODO: Questions 2, 3, 4
    
    ########### 2 C, i and ii ###########
    print("2 C")
    print("The length of the dictionary is: {} \n".format(len(dictionary_binary)))
    print("The average # of non-zero features in the training set is: {} \n".format(sum(sum(X_train))/X_train.shape[0]))
    
    ########### 3.1 ###########
    
    # 3.1 D
    # constants
    print("3.1 D")
    c_vals = [10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3]#(i.e.10−3,10−2,...,102,103).
    measures = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]
    chosendata = []
    
    # iterate thru choices
    for meas in measures:
        chosen_c, performance = select_param_linear(X = X_train, y = Y_train, k=5, metric=meas, C_range = c_vals, penalty='l2')
        group = [meas, chosen_c, performance]
        chosendata.append(group)
        
    # output for the writeup 3.1 D
    for tup in chosendata:
        print(tup)
    
    # 3.1 E
    # from our table we choose accuracy to optimize
    print("3.1 E")
    chosenC = 0.1
    performance_measures = []
    # fit classifier
    part3E = SVC(kernel = 'linear', C = chosenC, class_weight = "balanced").fit(X = X_train, y = Y_train)
    
    # get prediction vector
    pred = part3E.predict(X_test)
    # go thru each measure
    for meas in measures:
        performance = 0
        if meas == 'auroc':
            # decision boundary
            test = part3E.decision_function(X_test)
            performance = metrics.roc_auc_score(y_true = Y_test, y_score = test)
        elif meas == 'accuracy':
            performance = part3E.score(X = X_test, y = Y_test)
        elif meas == 'f1-score':
            performance = metrics.f1_score(y_true = Y_test, y_pred = pred)
        elif meas == 'precision':
            performance = metrics.precision_score(y_true = Y_test, y_pred = pred)
        elif meas == 'sensitivity':
            # Sensitivity = (True Positive)/(True Positive + False Negative)
            tn, fp, fn, tp = metrics.confusion_matrix(Y_test, pred, labels = [1,-1]).ravel()
            performance = tp/(tp + fn)
        elif meas == 'specificity':
            # Specificity = (True Negative)/(True Negative + False Positive)
            tn, fp, fn, tp = metrics.confusion_matrix(Y_test, pred, labels = [1,-1]).ravel()
            performance = tn/(tn + fp)
        group = [meas, performance]
        performance_measures.append(group)
    
    # for writeup
    for tup in performance_measures:
        print(tup)
    
    # 3.1 F and G
    plot_weight(X = X_train, y = Y_train, penalty = "l2", C_range = c_vals)
    
    # 3.1 H & I
    print("3.1 H & I")
    # fit and get coefficients
    theta = part3E.coef_
    
    # positive words
    positive_ind = np.argsort(a = theta[0])[-5:-1]
    keys, values = zip(*dictionary_binary.items())
    positive_words = []
    for i in positive_ind:
        word = [k for k,v in dictionary_binary.items() if v == i]
        positive_words.append([word,theta[0,i]])
    
    # negative words
    negative_ind = np.argsort(a = theta[0])[0:4]
    negative_words = []
    for i in negative_ind:
        word = [k for k,v in dictionary_binary.items() if v == i]
        negative_words.append([word,theta[0,i]])
    
    # output
    print("Positive words: ")
    for i in positive_words:
        print(i)
        
    print("Negative words: ")
    for i in negative_words:
        print(i)
    
    ########### 3.2 ###########
    # 3.2 A & B
    
    print("3.2 B")
    # grid search
    print("grid search")
    range_r = c_vals
    params = np.array([c_vals, range_r])
    grid_c, grid_r, performancegrid = select_param_quadratic(X = X_train, y = Y_train, k=5, metric = 'auroc', param_range = params)
    
    # random search
    print("random search")
    rand_r = [pow(10,r) for r in np.random.uniform(low = -3, high = 3, size = 5)]
    rand_c = [pow(10,c) for c in np.random.uniform(low = -3, high = 3, size = 5)]
    params = np.array([rand_c, rand_r])
    rand_c, rand_r, performancerand = select_param_quadratic(X = X_train, y = Y_train, k=5, metric = 'auroc', param_range = params)
    
    # output
    print("Grid C, Grid R, Grid performance: ")
    print(str(grid_c) + " " + str(grid_r) + " " + str(performancegrid))
    print("Random C, Random R, Random performance: ")
    print(str(rand_c) + " " + str(rand_r) + " " + str(performancerand))
    
    ########### 3.4 ###########
    # 3.4 A
    print("3.4 A")
    c_vals = [10e-3, 10e-2, 10e-1, 10e0]
    best_c = 0
    best_auroc = 0
    for c in c_vals:
        clf = LinearSVC(penalty = 'l1', dual = False, C = c, class_weight='balanced')
        score = cv_performance(clf, X = X_train, y = Y_train, k=5, metric="auroc")
        if score > best_auroc:
            best_auroc = score
            best_c = c
    
    # output
    print("L1 Penalty + Hinge Loss Parameter Selection (C, AUROC)")
    print(str(best_c) + ", " +  str(best_auroc))
    
    # 3.4 B
    print("3.4 B")
    # need to pass this a linear SVC 
    plot_weight(X = X_train, y = Y_train, penalty = 'l1', C_range = c_vals)
    
    ########### 4.1 ###########
    # 4.1 B #
    # fit SVC
    print("4.1 B")
    # SVC minimizes the regular hinge loss w/ L2 penalty, so all we need is to set class weights and C value
    unbclf = SVC(C = 0.01, kernel = "linear", class_weight = {-1: 10, 1: 1}).fit(X_train, Y_train)
    # list of performances
    B41performance = []
    # get prediction vector
    pred = unbclf.predict(X_test)
    # go thru each measure
    for meas in measures:
        performance = 0
        if meas == 'auroc':
            # decision boundary
            test = unbclf.decision_function(X_test)
            performance = metrics.roc_auc_score(y_true = Y_test, y_score = test)
        elif meas == 'accuracy':
            performance = unbclf.score(X = X_test, y = Y_test)
        elif meas == 'f1-score':
            performance = metrics.f1_score(y_true = Y_test, y_pred = pred)
        elif meas == 'precision':
            performance = metrics.precision_score(y_true = Y_test, y_pred = pred)
        elif meas == 'sensitivity':
            # Sensitivity = (True Positive)/(True Positive + False Negative)
            tn, fp, fn, tp = metrics.confusion_matrix(Y_test, pred, labels = [1,-1]).ravel()
            performance = tp/(tp + fn)
        elif meas == 'specificity':
            # Specificity = (True Negative)/(True Negative + False Positive)
            tn, fp, fn, tp = metrics.confusion_matrix(Y_test, pred, labels = [1,-1]).ravel()
            performance = tn/(tn + fp)
        group = [meas, performance]
        B41performance.append(group)
    # output
    for group in B41performance:
        print(group)
    
    # 4.2 A #
    print("4.2 A")
    # fit classifier
    clfA42 = SVC(C = 0.01, kernel = "linear", class_weight = {-1: 1, 1: 1}).fit(IMB_features, IMB_labels)
    # list of performances
    B42performance = []
    # get prediction vector
    pred = clfA42.predict(IMB_test_features)
    # go thru each measure
    for meas in measures:
        performance = 0
        if meas == 'auroc':
            # decision boundary
            test = clfA42.decision_function(IMB_test_features)
            performance = metrics.roc_auc_score(y_true = IMB_test_labels, y_score = test)
        elif meas == 'accuracy':
            performance = clfA42.score(X = IMB_test_features, y = IMB_test_labels)
        elif meas == 'f1-score':
            performance = metrics.f1_score(y_true = IMB_test_labels, y_pred = pred)
        elif meas == 'precision':
            performance = metrics.precision_score(y_true = IMB_test_labels, y_pred = pred)
        elif meas == 'sensitivity':
            # Sensitivity = (True Positive)/(True Positive + False Negative)
            tn, fp, fn, tp = metrics.confusion_matrix(IMB_test_labels, pred, labels = [1,-1]).ravel()
            performance = tp/(tp + fn)
        elif meas == 'specificity':
            # Specificity = (True Negative)/(True Negative + False Positive)
            tn, fp, fn, tp = metrics.confusion_matrix(IMB_test_labels, pred, labels = [1,-1]).ravel()
            performance = tn/(tn + fp)
        group = [meas, performance]
        B42performance.append(group)
    # output
    for group in B42performance:
        print(group)
    
    # 4.3 #
    print("4.3 B")
    # investigating the helper.py reveals positive_class_size = 800, ratio = 0.25
    # based on this, I will try various Wn starting with the inverse of this ratio
    negative_weight_list = [0.25, 2, 4, 6, 8, 10]
    # fit classifiers based on this weight and choose best cross_validation value
    scores = []
    imbscore = 0
    imbweight = 0
    for weight in negative_weight_list:
        clf = SVC(C = 0.01, kernel = "linear", class_weight = {-1: weight, 1: 1})
        score = cv_performance(clf, X = IMB_features, y = IMB_labels, k=5, metric="f1-score")
        scores.append([weight, score])
        if score > imbscore:
            imbscore = score
            imbweight = weight
    
    # output
    for group in scores:
        print(group)
    
    # then we produce our testing data performance
    clf43 = SVC(C = 0.01, kernel = "linear", class_weight = {-1: 2, 1: 1}).fit(IMB_features, IMB_labels)
    # list of performances
    B43performance = []
    # get prediction vector
    pred = clf43.predict(IMB_test_features)
    # go thru each measure
    for meas in measures:
        performance = 0
        if meas == 'auroc':
            # decision boundary
            test = clf43.decision_function(IMB_test_features)
            performance = metrics.roc_auc_score(y_true = IMB_test_labels, y_score = test)
        elif meas == 'accuracy':
            performance = clf43.score(X = IMB_test_features, y = IMB_test_labels)
        elif meas == 'f1-score':
            performance = metrics.f1_score(y_true = IMB_test_labels, y_pred = pred)
        elif meas == 'precision':
            performance = metrics.precision_score(y_true = IMB_test_labels, y_pred = pred)
        elif meas == 'sensitivity':
            # Sensitivity = (True Positive)/(True Positive + False Negative)
            tn, fp, fn, tp = metrics.confusion_matrix(IMB_test_labels, pred, labels = [1,-1]).ravel()
            performance = tp/(tp + fn)
        elif meas == 'specificity':
            # Specificity = (True Negative)/(True Negative + False Positive)
            tn, fp, fn, tp = metrics.confusion_matrix(IMB_test_labels, pred, labels = [1,-1]).ravel()
            performance = tn/(tn + fp)
        group = [meas, performance]
        B43performance.append(group)
    # output
    for group in B43performance:
        print(group)
    
    # 4.4 #
    print("4.4")
    # decision score w clfA42 (Wp = 1, Wn = 1)
    IMB_score42 = clfA42.decision_function(IMB_test_features)
    fpr42, tpr42, _ = metrics.roc_curve(IMB_test_labels, IMB_score42)
    
    # predict w custom weights (Wp = 1, Wn = 2)
    IMB_score43 = clf43.decision_function(IMB_test_features)
    fpr43, tpr43, _ = metrics.roc_curve(IMB_test_labels, IMB_score43)
    
    # plot on same graph & save image
    plt.plot(fpr42, tpr42)
    plt.plot(fpr43, tpr43)
    plt.title(label = "ROC Curves")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(['1 to 1 weight CLF', 'Custom weight CLF'])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig('Q4 Weight-Based ROC Curves.png')
    plt.close()
    
    # Read multiclass data
    # TODO: Question 5: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    print("producing feature matrices")
    multiclass_features, multiclass_labels, multiclass_dictionary = get_multiclass_training_data()
    heldout_features = get_heldout_reviews(multiclass_dictionary)

    ########### 5 ###########
    print("PART 5")
    
    ### PRINT OUTPUT FOR EACH CLASSIFIER TO TRACK OVERFITTING
    print("Fit classifier")
    clf = SVC(C = 0.015, kernel = 'linear').fit(X = multiclass_features, y = multiclass_labels)
    
    print("Make predictions on held out set")
    predVec = clf.predict(heldout_features)
    
    print("Generate Challenge Labels")
    generate_challenge_labels(predVec, "corioj")

if __name__ == '__main__':
    main()
