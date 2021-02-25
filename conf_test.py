
"""
conf_test is a python implementation of a method that can be used to effectively calculate the amount of identity
confounding learned by classifiers developed using a record-wise data split.

The method is described in:
E. C. Neto et al., "Detecting the impact of subject characteristics on machine learning-based diagnostic applications,
" NPJ digital medicine, vol. 2, no. 1, pp. 1-6, 2019.

The permutations operation assumes that the labels in the Y array input are grouped together within the array.
"""

import numpy as np
import random
import math


def get_labels_list(y_array):
    lab_list = []
    for i in range(len(y_array)):
        if i == 0:
            label0 = y_array[i]
            lab_list.append(label0)  # add initial unique label to list
        elif y_array[i] != y_array[i - 1]:
            label1 = y_array[i]
            lab_list.append(label1)  # add any additional unique label to list
    return lab_list


def get_subjects_labels(lab_list, y_array, num_per):
    sub_lab = []
    y2 = y_array.tolist()
    for i in range(len(lab_list)):
        lab_count = y2.count(lab_list[i])
        num_lab_sub = int(lab_count / num_per)  # determine # of subjects for current label
        lab_list_sub = [lab_list[i]] * num_lab_sub  # create list of labels corresponding to # of subjects
        sub_lab = [*sub_lab, *lab_list_sub]  # unpack both iterables in a list literal and combine
    return sub_lab


def shuffle_subjects_labels(sub_lab, y_array, num_per):
    # Create list assigning an integer for each subject
    num_sub = int(y_array.shape[0] / num_per)  # determine total # of subjects
    sub_int = list(range(1, num_sub + 1))
    # Create temp lists and shuffle subject-wise labels and integers lists
    sub_lab2 = sub_lab
    sub_int2 = sub_int
    temp_list = list(zip(sub_lab2, sub_int2))
    random.shuffle(temp_list)
    sub_lab2, sub_int2 = zip(*temp_list)
    return sub_lab2, sub_int2


def split_features_data(sub_lab2, x_array, num_per):
    # Calculate number of observations in each split and keep as variables for later use
    split_1 = int(math.ceil(num_per / 2))
    split_2 = num_per - split_1
    for q in range(len(sub_lab2)):
        # Isolate observations belonging to each of two splits for current subject (2 distinct sets for X)
        range1_1 = q * num_per
        range1_2 = range1_1 + split_1
        split1_x = x_array[range1_1:range1_2, :]
        range2_1 = range1_2
        range2_2 = range2_1 + split_2
        split2_x = x_array[range2_1:range2_2, :]
        # Combine each of 2 distinct sets for X into 2 new X arrays
        if q == 0:
            new_x1 = split1_x
            new_x2 = split2_x
        else:
            new_x1 = np.concatenate((new_x1, split1_x), axis=0)
            new_x2 = np.concatenate((new_x2, split2_x), axis=0)
    return new_x1, new_x2, split_1, split_2


def expand_subjects_labels(sub_lab2, split_1, split_2):
    new_list1 = []
    new_list2 = []
    for q in range(len(sub_lab2)):
        new_sub1 = [sub_lab2[q]] * split_1
        new_list1 = [*new_list1, *new_sub1]  # unpack both iterables in a list literal and combine
        new_sub2 = [sub_lab2[q]] * split_2
        new_list2 = [*new_list2, *new_sub2]  # unpack both iterables in a list literal and combine
    new_y1 = np.array(new_list1)
    new_y2 = np.array(new_list2)
    return new_y1, new_y2


def permutations(data_set, num_per):
    # 1) Permutations implementation PART I
    # 1.1) Extract values, X, and labels, Y, from DataFrame
    array = data_set.values
    data_col_dim = data_set.shape[1] - 1
    x_array = array[:, 0:data_col_dim]
    y_array = array[:, data_col_dim]
    # 1.2) Create subject-wise labels list
    # 1.2.1) Find out all different labels in the labels array
    labels_list = get_labels_list(y_array)
    # 1.2.2) Find out number of subject per label in the labels array, and create subject-wise labels list
    sub_labels = get_subjects_labels(labels_list, y_array, num_per)
    # 1.3) Shuffle subject-wise labels (and integers lists, in case tracking positions is desired)
    sub_labels2, sub_integers2 = shuffle_subjects_labels(sub_labels, y_array, num_per)
    # 2) Permutations implementation PART II
    # 2.1) Iterate through dataset using # of subjects, split into halves and recombine into 2 distinct sets
    new_x1, new_x2, split1, split2 = split_features_data(sub_labels2, x_array, num_per)
    # 2.2) Expand back Y into into 2 new Y arrays, using saved two numbers of observations in each X split
    new_y1, new_y2 = expand_subjects_labels(sub_labels2, split1, split2)
    # X_train = newX1, X_validation = newX2, Y_train = newY1, Y_validation = newY2
    return new_x1, new_x2, new_y1, new_y2



