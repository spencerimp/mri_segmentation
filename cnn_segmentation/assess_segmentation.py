"""
This scripts calculates the segmentation performance.
"""
from __future__ import print_function
import os
import csv
import sys
import glob
import numpy as np
from scipy.io import loadmat
from utils.metrics import (
    get_accuracy,
    get_dice,
)


def compute_performance(pred_label_files, true_label_files, n_classes, out_file,
                        verbose=False, label_list_file=None):
    """Calculate the performance given predicted and true segementation files.

    Args:
        pred_label_files: The list of predicted segmentation (.mat files)
        true_label_files: The corresponding ground truth (.mat files)
        n_classes: number of classes (including background as 0)
        verbose: True if want to see the dice score along with its class
        label_list_file: The name of all classes, only used when verbose=True

    Calculate:
        - accuracy (including class 0)
        - mean of dice scores of non-background classes
        - std of dice scores of non-background classes

    Return:
        mean of performance among all classes (n_images, 3)
        performance for all classes (n_images, n_classes-1)

    This assumes that the dimension prediction and truth are the same.
    """
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    if verbose:
        with open(label_list_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            label_list = np.array([row[1] for row in csv_reader])

    performance_mat = np.zeros((len(pred_label_files), 3))
    detailed_dice_mat = np.zeros((len(pred_label_files), n_classes-1))
    for i, pred_label_file in enumerate(pred_label_files):
        print('Calulate performance from {}'.format(pred_label_file))
        true_label_file = true_label_files[i]
        temp = loadmat(true_label_file)
        true_labels = temp['label']
        true_labels = true_labels.astype(int)

        temp = loadmat(pred_label_file)
        pred_labels = temp['label']
        pred_labels = pred_labels.astype(int)

        acc = get_accuracy(pred_labels, true_labels)
        dices = get_dice(pred_labels, true_labels, n_classes)
        detailed_dice_mat[i, :] = dices[1:]

        mean_dice = np.mean(dices)
        std_dice = np.std(dices)
        performance_mat[i, :] = [acc, mean_dice, std_dice]
        if verbose:
            print('Accuracy:{}'.format(acc))
            print('Mean dice:{}'.format(mean_dice))
            print('Std dice:{}'.format(std_dice))
            for c in range(n_classes-1):
                print('class {}:{}, dice:{}'.format(c + 1, label_list[c],
                                                    dices[c]))
    # export the mean metric
    export_performance(performance_mat, pred_label_files, out_file)

    # export the detailed dices
    file_name, file_ext = os.path.splitext(out_file)
    out_detailed_file = os.path.join(file_name+'_all_dices'+file_ext)

    if verbose:
        export_detailed_performance(detailed_dice_mat, pred_label_files,
                                    label_list, out_detailed_file)
    return performance_mat, detailed_dice_mat


def export_detailed_performance(detailed_mat, pred_label_files, label_list, out_file):
    """
    Export a matrix of (classes, files)
    """
    aug_mat = []
    # apppend each record with file name
    for i, performance_rec in enumerate(detailed_mat.tolist()):
        rec = [pred_label_files[i]]
        rec.extend(performance_rec)
        aug_mat.append(rec)

    # Add the title (label list)
    out_mat = [label_list]
    out_mat.extend(aug_mat)

    with open(out_file, 'w') as fout:
        csv_writer = csv.writer(fout, delimiter=',')
        csv_writer.writerows(out_mat)


def export_performance(performance_mat, pred_label_files, out_file):
    """Export the records into csv files.

    Each row contains
        [file_name, the record]
    Also, two records will be added to show
        1. Average of the records
        2. Std of the records
    """
    performance_mean = np.mean(performance_mat, 0)
    performance_std = np.std(performance_mat, 0)
    performance_mat = performance_mat.tolist()
    # apppend with file name
    for i, (performance_rec, pred_label_file) in enumerate(zip(performance_mat, pred_label_files)):
        rec = [pred_label_file]
        rec.extend(performance_rec)
        performance_mat[i] = rec

    # average of the files at the end
    rec = ['average']
    rec.extend(performance_mean)
    performance_mat.append(rec)

    # std of the files at the end
    rec = ['std']
    rec.extend(performance_std)
    performance_mat.append(rec)

    with open(out_file, 'w') as fout:
        fout.write('image_name,accuracy,mean_dice,std_dice\n')
        csvWriter = csv.writer(fout, delimiter=',')
        csvWriter.writerows(performance_mat)
    print('Done! Check {}'.format(out_file))


if __name__ == '__main__':
    pred_label_dir = './experiments/keras/6patches/'
    true_label_dir = './datasets/miccai/test/label_mat/'
    n_classes = 135
    out_file = './experiments/keras/6patches/performance6patches.csv'
    verbose = False
    label_list_file = './docs/MICCAI-Challenge-2012-Label-Information_v3.csv'

    pred_label_files = glob.glob(pred_label_dir+'*.mat')
    true_label_files = []
    for pred_label_file in true_label_files:
        image_name = os.path.splitext(os.path.split(pred_label_file)[-1])[0]
        true_label_file = os.path.join(true_label_dir, image_name)+'_glm.mat'
        true_label_files.append(true_label_file)

    performance_mat, detailed_performance = compute_performance(pred_label_files,
                                                                true_label_files,
                                                                n_classes,
                                                                out_file,
                                                                verbose,
                                                                label_list_file)
