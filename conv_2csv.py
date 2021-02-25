
from os.path import join as p_join
import scipy.io as sio
import numpy as np
import pandas as pd


def get_variables(data_path, group_var, d_name, cc_name, l_name):
    data_dir_1 = p_join(data_path, group_var, 'stateFull_Data.mat').replace("\\", "/")  # combine strings
    data_dir_2 = p_join(data_path, group_var, 'stateFull_Data2.mat').replace("\\", "/")
    data_1 = sio.loadmat(data_dir_1)  # load .mat file
    data_2 = sio.loadmat(data_dir_2)
    freq_full_1 = data_1[d_name]  # fetch variable inside .mat file
    freq_full_2 = data_2[cc_name]
    labels = data_1[l_name]  # same labels as _2
    return freq_full_1, freq_full_2, labels


def construct_array(labels, freq_full_1, freq_full_2):
    labels_lst = labels.tolist()  # make labels array into list
    labels_lst2 = [i[0] for i in labels_lst]  # remove first square bracket around each elements
    labels_lst3 = [i[0] for i in labels_lst2]  # remove first square bracket around each elements
    labels_arr = np.asarray(labels_lst3)  # turn array back into list
    labels_arr2 = labels_arr.reshape(labels_arr.shape[0], -1)  # add dimension to array for concatenation
    data_full = np.concatenate((freq_full_1, freq_full_2, labels_arr2), axis=1)
    return data_full


def construct_header(freq_full_1, freq_full_2):
    d_headers_1 = np.asarray(["d_freq_" + str((i+1)) for i in range(freq_full_1.shape[1])])
    d_headers_2 = d_headers_1.reshape(-1, d_headers_1.shape[0])  # add dimension to array for concatenation
    cc_headers_1 = np.asarray(["cc_freq_" + str((i+1)) for i in range(freq_full_2.shape[1])])
    cc_headers_2 = cc_headers_1.reshape(-1, cc_headers_1.shape[0])  # add dimension to array for concatenation
    df_header = np.concatenate((d_headers_2, cc_headers_2, [["class"]]), axis=1)
    df_header_lst = df_header.tolist()  # make labels array into list
    df_header_lst2 = df_header_lst[0]  # remove outer square brackets
    df_header_arr = np.asarray(df_header_lst2)  # turn array back into list
    return df_header_arr


subjects_G1 = "AMD"
subjects_G2 = "normal"
str_dataPath = r'C:\Users\cnzak\Desktop\data\avRNS\biophotonics'
# ML input data : 'drusenConverter' --> [stateFull_Data]; 'ccConverter' --> [stateFull_Data2]

# Assign AMD (group 1, G1) variables
freqFull_G1_1, freqFull_G1_2, labels_G1 = get_variables(str_dataPath, subjects_G1, 'freqFull_Array', 'freqFull_Array2',
                                                        'stateFull_Labels')
# Create dataframe values and labels array for G1
dataFull_G1 = construct_array(labels_G1, freqFull_G1_1, freqFull_G1_2)

# Assign normal (group 2, G2) variables
freqFull_G2_1, freqFull_G2_2, labels_G2 = get_variables(str_dataPath, subjects_G2, 'freqFull_Array', 'freqFull_Array2',
                                                        'stateFull_Labels')
# Create dataframe values and labels array for G2
dataFull_G2 = construct_array(labels_G2, freqFull_G2_1, freqFull_G2_2)

# Create full dataframe values and labels array
dataFull = np.concatenate((dataFull_G1, dataFull_G2), axis=0)

df_headerArr = construct_header(freqFull_G1_1, freqFull_G1_2)  # create header array
df_input = pd.DataFrame(data=dataFull, index=None, columns=df_headerArr)   # combine arrays as DataFrame

dirPath_str = p_join(str_dataPath, 'df_input.csv').replace("\\", "/")  # combine directory strings
df_input.to_csv(dirPath_str, index=False)  # export DataFrame as .csv
