import numpy as np


def convert_and_check(pred_seq, true_seq):
    """Convert the input arrays to 1D and check length."""
    pred_seq = pred_seq.ravel()
    true_seq = true_seq.ravel()
    assert len(pred_seq) == len(true_seq), "The length should be identical"


def get_MSE(pred_seq, true_seq):
    """Compute Mean Square Error."""
    convert_and_check(pred_seq, true_seq)
    err = np.sum(pred_seq != true_seq)
    return err**err / float(len(err.ravel()))


def get_error_rate(pred_seq, true_seq):
    convert_and_check(pred_seq, true_seq)
    return np.sum(true_seq != pred_seq) / float(len(true_seq.ravel()))


def get_accuracy(pred_seq, true_seq):
    convert_and_check(pred_seq, true_seq)
    return np.sum(true_seq == pred_seq) / float(len(true_seq.ravel()))


def get_precision(pred_seq, true_seq, n_classes=2):
    convert_and_check(pred_seq, true_seq)
    cm = get_confusion_matrix(pred_seq, true_seq, n_classes)
    return cm[:, 0] / cm[:, 1]


def get_recall(pred_seq, true_seq, n_classes=2):
    convert_and_check(pred_seq, true_seq)
    cm = get_confusion_matrix(pred_seq, true_seq, n_classes)
    return cm[:, 0] / cm[:, 2]


def get_f1score(pred_seq, true_seq, n_classes=2):
    return get_fscore(pred_seq, true_seq, n_classes, 1)


def get_fscore(pred_seq, true_seq, n_classes=2, beta=1):
    convert_and_check(pred_seq, true_seq)
    cm = get_confusion_matrix(pred_seq, true_seq, n_classes)
    precision = cm[:, 0] / cm[:, 1]
    recall = cm[:, 0] / cm[:, 2]

    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def get_dice(pred_seq, true_seq, n_classes=2):
    """Compute the Dice Similarity Coefficient (DSC) for each class.
    """
    convert_and_check(pred_seq, true_seq)
    cm = get_confusion_matrix(pred_seq, true_seq, n_classes)

    return 2 * cm[:, 0] / (cm[:, 1] + cm[:, 2])


def get_confusion_matrix(pred_seq, true_seq, n_classes=2):
    """Compute the confusion matrix for each class.

    Inputs:
        pred_seq: (np array) predicted label sequence
        true_seq: (np array) true label sequence
        n_classes: (int) the number of classes, default = 2

    Output:
        confusion matrix but only contains TP, (TP+FP), (TP+FN)

        cm.shape = (n_class, 3)
        cm[:, 0] = the TP
        cm[:, 1] = the number of pred label (TP+FP)
        cm[:, 2] = the number of true label (TP+FN)

    Note that the class starts from label 0.
    """
    convert_and_check(pred_seq, true_seq)
    cm = np.zeros((n_classes, 3), dtype=np.float32)
    for c in range(n_classes):
        pred_c = pred_seq == c
        true_c = true_seq == c

        cm[c, 0] = np.sum(pred_c * true_c)
        cm[c, 1] = np.sum(pred_c)
        cm[c, 2] = np.sum(true_c)

    return cm
