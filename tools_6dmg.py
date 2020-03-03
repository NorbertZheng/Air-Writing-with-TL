import os
import csv
import numpy as np

def preprocess(path):
    n_files = 0

    for dir_path, dir_names, file_names in os.walk(path):
        for filename in file_names:
            n_files += 1

    data = np.zeros((n_files, 3, 800) ,dtype = np.float32)
    label = np.zeros((n_files), dtype = np.int64)
    index = 0

    for dir_path, dir_names, file_names in os.walk(path):
        for filename in file_names:
            array = np.loadtxt(os.path.join(dir_path, filename), delimiter=",", dtype = np.float32)
            for i in range(array.shape[1]):
                data[index][0][i] = array[1][i]
                data[index][1][i] = array[2][i]
                data[index][2][i] = array[3][i]
            # label[index] = filename[4]
            label[index] = ord(filename[6]) - ord("A")
            index += 1
    return (data, label)

preprocess("") #路径信息
