
import sklearn.metrics


def precision(actual, pred):
    return sklearn.metrics.precision_score(actual, pred, average='macro')


def recall(actual, pred):
    return sklearn.metrics.recall_score(actual, pred, average='macro')


def f1_score(actual, pred):
    return sklearn.metrics.f1_score(actual, pred, average='macro')


def confusion_matrix(actual, pred):
    return sklearn.metrics.confusion_matrix(actual, pred)


def roc_curve(actual, pred):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(actual, pred)
    return fpr, tpr, thresholds

def det_curve(actual, pred):
    fpr, fnr, thresholds = sklearn.metrics.det_curve(actual, pred)
    return fpr, fnr, thresholds

def accuracy(actual, pred):
    return sklearn.metrics.accuracy_score(actual, pred)
