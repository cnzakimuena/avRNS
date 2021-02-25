
"""
spec_gen constructs a labelled dataset of spectrogram images from spatial series obtained using MATLAB for use as input
to machine learning classification algorithms.
"""

from os.path import join as p_join
import scipy
import scipy.io as sio
from scipy import signal
from scipy.fft import fftshift
# from scipy.io import wavfile
import numpy as np
import pandas as pd
import librosa
from librosa import display
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split


# Assign group variables
def get_variables(data_path, group_var, d_name, cc_name, l_name):
    data_dir_1 = p_join(data_path, group_var, 'stateFull_Data.mat').replace("\\", "/")  # combine strings
    data_dir_2 = p_join(data_path, group_var, 'stateFull_Data2.mat').replace("\\", "/")
    data_1 = sio.loadmat(data_dir_1)  # load .mat file
    data_2 = sio.loadmat(data_dir_2)
    rec_full_1 = data_1[d_name]  # fetch variable inside .mat file
    rec_full_2 = data_2[cc_name]
    labels = data_1[l_name]  # data_1 contains same labels as data_2
    return rec_full_1, rec_full_2, labels


def spatial_series_plot(rec, sp_rate, t_var):
    length_of_space = len(rec) / sp_rate
    # print(length_of_space, " mm")
    d = np.arange(0.0, rec.shape[0])/sampling_rate
    fig, ax = plt.subplots()
    ax.plot(d, rec, 'b-')
    ax.set(xlabel='Distance [$mm$]', ylabel='Magnitude', title=t_var)
    # ax.grid()
    # fig.savefig("test.png")
    plt.xlim(0, length_of_space)
    return plt.show()


def spectrogram_plot(freq, space, s_im, rec, sp_rate):
    length_of_space = len(rec) / sp_rate
    # print(length_of_space, " mm")
    plt.figure()
    # c = plt.pcolormesh(space, freq, 10 * np.log10(s_im), cmap='viridis', shading='flat')
    c = plt.pcolormesh(space, freq, 10 * np.log10(s_im), cmap='Greens', shading='gouraud')
    cbar = plt.colorbar(c)
    cbar.set_label('Power/Frequency [$dB/mm^{-1}$]')
    # z is Power/Frequency (dB/Hz)
    plt.ylabel('Frequency [$mm^{-1}$]')
    plt.xlabel('Distance [$mm$]')
    plt.xlim(0, length_of_space)
    return plt.show()


def find_spec_bounds(rec_full, sp_rate):
    for i in range(rec_full.shape[0]):
        rec = rec_full[i, :]  # single spatial series
        # # Uncomment to visualize single spatial series
        # spatial_series_plot(rec_1, sampling_rate, 'RPEb-BM Thickness')
        # (1) Generate spectrogram from spatial series
        # f, s, sxx = signal.spectrogram(rec, sp_rate, window='flattop', nperseg=40, noverlap=35, mode='psd')
        f, s, sxx = signal.spectrogram(rec, sp_rate, window='flattop', nperseg=40, noverlap=35)
        # Setting array zeros to min non-zero values to avoid log10(0) error
        sxx[sxx == 0] = np.min(sxx[np.nonzero(sxx)])
        spec = 10 * np.log10(sxx)  # power spectral density
        # # Uncomment to visualize spectrogram
        # spectrogram_plot(f, s, sxx, rec, sp_rate)
        # (2) Obtain normalization maximum and minimum values
        curr_max = spec.max()
        curr_min = spec.min()
        if i == 0:
            set_max = curr_max
            set_min = curr_min
        if curr_max > set_max:
            set_max = curr_max
        if curr_min < set_min:
            set_min = curr_min
    return set_max, set_min


def get_spec_im(rec, sp_rate, rec_max, rec_min):
    # f, s, sxx = signal.spectrogram(rec, sp_rate, window='flattop', nperseg=40, noverlap=35, mode='psd')
    f, s, sxx = signal.spectrogram(rec, sp_rate, window='flattop', nperseg=40, noverlap=35)
    # Setting array zeros to min non-zero values to avoid log10(0) error
    sxx[sxx == 0] = np.min(sxx[np.nonzero(sxx)])
    # # Uncomment to visualize spectrogram
    # spectrogram_plot(f, s, sxx, rec, sp_rate)
    spec = 10 * np.log10(sxx)  # power spectral density
    # Normalize spectrogram images to 0-255 range (based on inter- G1 and G2 maximum)
    spec = (spec - rec_min)/(rec_max-rec_min)  # signed integers normalization to 0-1 range
    spec *= 255.0/spec.max()  # normalization to 0-255 range
    # Resize images to 64x64
    res = cv2.resize(spec, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    # print('Data Type: %s' % spec1.dtype)
    # print('Min: %.3f, Max: %.3f' % (spec1.min(), spec1.max()))
    # # Uncomment to visualize normalized drusen spectrogram image
    # plt.figure()
    # plt.imshow(res, cmap='Greens', vmin=0, vmax=255)
    # plt.show()
    res = res[..., np.newaxis]  # add channel axis for concatenation
    return res


def list_to_array(labels):
    labels_lst = labels.tolist()  # make labels array into list
    labels_lst2 = [i[0] for i in labels_lst]  # remove first square bracket around each elements
    labels_lst3 = [i[0] for i in labels_lst2]  # remove first square bracket around each elements
    labels_arr = np.asarray(labels_lst3)  # turn list back into array
    labels_arr2 = labels_arr.reshape(labels_arr.shape[0], -1)  # add dimension to array for concatenation
    return labels_arr2


def get_x_array(rec_full_1, rec_full_2, sp_rate, set_max1, set_min1, set_max2, set_min2):
    x_array = np.zeros((rec_full_1.shape[0], 64, 64, 2))
    for q in range(rec_full_1.shape[0]):
        rec1 = rec_full_1[q, :]  # single drusen spatial series
        rec2 = rec_full_2[q, :]  # single cc spatial series
        # (1) Generate current spectrogram images from drusen and cc spatial series
        res_im1 = get_spec_im(rec1, sp_rate, set_max1, set_min1)
        res_im2 = get_spec_im(rec2, sp_rate, set_max2, set_min2)
        # (2) Concatenate drusen and cc spectrogram into 2-channels array of images
        res_im = np.concatenate((res_im1, res_im2), axis=2)
        # (3) Append current 2-channels array of images to x_array
        x_array[q, :, :, :] = res_im
    x_array = x_array.astype(np.uint8)  # round array elements to nearest integer
    return x_array


def get_y_array(labels_full):
    y_array = np.zeros((1, labels_full.shape[0]))
    counter = 0
    lab_list = []
    for i in range(labels_full.shape[0]):
        if i == 0:  # check if first iteration
            label0 = labels_full[i, :]
            y_array[:, i] = counter
            lab_list.append(counter)
        elif labels_full[i, :] == labels_full[i-1, :]:  # check if current label matches previous label
            y_array[:, i] = counter
        else:
            counter = counter + 1
            y_array[:, i] = counter
            lab_list.append(counter)
    cl_array = np.asarray(lab_list)
    y_array = y_array.astype(np.uint8)  # round array elements to nearest integer
    return y_array, cl_array


def split_dataset(x_array, y_array):
    y_list = y_array.tolist()
    y_list2 = y_list[0]
    x_train_orig, x_test_orig, y_train_orig, y_test_orig = train_test_split(x_array, y_list2, test_size=0.20)
    y_arr_train = np.asarray(y_train_orig)
    y_train_orig = y_arr_train[np.newaxis, ...]
    y_arr_test = np.asarray(y_test_orig)
    y_test_orig = y_arr_test[np.newaxis, ...]
    return x_train_orig, y_train_orig, x_test_orig, y_test_orig


def load_split_spec_dataset(subjects_g1, subjects_g2, str_data_path, spl_rate):
    # 1) Extract out spatial series for drusen (recFull_GX_1) abd cc (recFull_GX_2) from reading MATLAB file
    # 1.1) Assign AMD (group 1, G1) variables
    rec_full_g1_1, rec_full_g1_2, labels_g1 = get_variables(str_data_path, subjects_g1, 'recFull_Array',
                                                            'recFull_Array2', 'stateFull_Labels')
    # 1.2) Assign normal (group 2, G2) variables
    rec_full_g2_1, rec_full_g2_2, labels_g2 = get_variables(str_data_path, subjects_g2, 'recFull_Array',
                                                            'recFull_Array2', 'stateFull_Labels')
    # 2) Loops to generate ResNet_model X images input (number of images, row dim, col dim, channels depth)
    # 2.1) Obtain drusen and cc spectrogram dataset boundaries for normalization
    # combine G1 and G2 drusen series arrays
    rec_full_1 = np.concatenate((rec_full_g1_1, rec_full_g2_1))
    # find drusen dataset normalization max and min values
    rec_max1, rec_min1 = find_spec_bounds(rec_full_1, spl_rate)
    # combine G1 and G2 cc series arrays
    rec_full_2 = np.concatenate((rec_full_g1_2, rec_full_g2_2))
    # find cc dataset normalization max and min values
    rec_max2, rec_min2 = find_spec_bounds(rec_full_2, spl_rate)
    # 2.2) Loop to generate dataset of concatenated 64x64 drusen and cc spectrogram images
    x_array = get_x_array(rec_full_1, rec_full_2, spl_rate, rec_max1, rec_min1, rec_max2, rec_min2)
    # 3) Loop to generate ResNet_model Y labels and classes inputs
    # 3.1) Turn label lists into arrays and concatenate
    lab_arr_g1 = list_to_array(labels_g1)
    lab_arr_g2 = list_to_array(labels_g2)
    lab_full = np.concatenate((lab_arr_g1, lab_arr_g2), axis=0)
    # 3.2) Turn labels array into numerical array and generate classes variable
    y_array, cl_array = get_y_array(lab_full)
    # 4) Select dataset split to recreate ResNet_model input
    x_train_orig, y_train_orig, x_test_orig, y_test_orig = split_dataset(x_array, y_array)
    return x_train_orig, y_train_orig, x_test_orig, y_test_orig, cl_array


def load_spec_dataset(subjects_g1, subjects_g2, str_data_path, spl_rate):
    # 1) Extract out spatial series for drusen (recFull_GX_1) abd cc (recFull_GX_2) from reading MATLAB file
    # 1.1) Assign AMD (group 1, G1) variables
    rec_full_g1_1, rec_full_g1_2, labels_g1 = get_variables(str_data_path, subjects_g1, 'recFull_Array',
                                                            'recFull_Array2', 'stateFull_Labels')
    # 1.2) Assign normal (group 2, G2) variables
    rec_full_g2_1, rec_full_g2_2, labels_g2 = get_variables(str_data_path, subjects_g2, 'recFull_Array',
                                                            'recFull_Array2', 'stateFull_Labels')
    # 2) Loops to generate ResNet_model X images input (number of images, row dim, col dim, channels depth)
    # 2.1) Obtain drusen and cc spectrogram dataset boundaries for normalization
    # combine G1 and G2 drusen series arrays
    rec_full_1 = np.concatenate((rec_full_g1_1, rec_full_g2_1))
    # find drusen dataset normalization max and min values
    rec_max1, rec_min1 = find_spec_bounds(rec_full_1, spl_rate)
    # combine G1 and G2 cc series arrays
    rec_full_2 = np.concatenate((rec_full_g1_2, rec_full_g2_2))
    # find cc dataset normalization max and min values
    rec_max2, rec_min2 = find_spec_bounds(rec_full_2, spl_rate)
    # 2.2) Loop to generate dataset of concatenated 64x64 drusen and cc spectrogram images
    x_array = get_x_array(rec_full_1, rec_full_2, spl_rate, rec_max1, rec_min1, rec_max2, rec_min2)
    # 3) Loop to generate ResNet_model Y labels and classes inputs
    # 3.1) Turn label lists into arrays and concatenate
    lab_arr_g1 = list_to_array(labels_g1)
    lab_arr_g2 = list_to_array(labels_g2)
    lab_full = np.concatenate((lab_arr_g1, lab_arr_g2), axis=0)
    # 3.2) Turn labels array into numerical array and generate classes variable
    y_array, cl_array = get_y_array(lab_full)
    # # 4) Select dataset split to recreate ResNet_model input
    # x_train_orig, y_train_orig, x_test_orig, y_test_orig = split_dataset(x_array, y_array)
    return x_array, y_array, cl_array


def fft_plot(space_series, spl_rate):
    n = len(space_series)
    period = 1 / spl_rate
    yf = scipy.fft.fft(space_series)
    y = 2.0 / n * np.abs(yf[:n // 2])
    x = np.linspace(0.0, 1.0 / (2.0 * period), int(n / 2))
    fig, ax = plt.subplots()
    ax.plot(x, 10 * np.log10(y))
    plt.grid()
    plt.xlabel('Frequency [$mm^{-1}$]')
    plt.ylabel('Power/Frequency [$dB/mm^{-1}$]')
    return plt.show()


subjects_G1 = "AMD"
subjects_G2 = "normal"
str_dataPath = r'C:/Users/cnzak/Desktop/data/avRNS/biophotonics'
# ML input data : 'drusenConverter' --> [stateFull_Data]; 'ccConverter' --> [stateFull_Data2]
sampling_rate = 200  # sampling frequency, 600/3 = 200 px/mm
# X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_split_spec_dataset(subjects_G1, subjects_G2,
#                                                                                   str_dataPath, sampling_rate)
X_orig, Y_orig, classes = load_spec_dataset(subjects_G1, subjects_G2, str_dataPath, sampling_rate)



