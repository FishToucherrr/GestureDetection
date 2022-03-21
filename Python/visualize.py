import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy import stats
import tensorflow as tf
import math
import random

# matplotlib inline
# plt.style.use('ggplot')



normalization_coef = 9
batch_size = 10
kernel_size = 30
depth = 20
num_hidden = 100
num_channels = 3

learning_rate = 0.0001
training_epochs = 10

filter_value = 20



def convolve1d(signal, length):
    ir = np.ones(length)/length
    #return np.convolve(y, ir, mode='same')
    
    output = np.zeros_like(signal)

    for i in range(len(signal)):
        for j in range(len(ir)):
            if i - j < 0: continue
            output[i] += signal[i - j] * ir[j]
            
    return output

def filterRecord(record, filter_value):
    x = convolve1d(record[:,0], filter_value)
    y = convolve1d(record[:,1], filter_value)
    z = convolve1d(record[:,2], filter_value)
    return np.dstack([x,y,z])[0]


def readFileData(file):
    column_names = ['timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file, header = None, names = column_names)
    
    x = data["x-axis"]
    y = data["y-axis"]
    z = data["z-axis"]
    
    return np.dstack([x,y,z])[0]

def readData(directory):
    records = []
    labels = np.empty((0))
    
    allFiles = glob.glob(directory + "/*.log")
    for file in allFiles:
        fileName = os.path.basename(file)
        (name, ext) = os.path.splitext(fileName)
        parts = name.split("_")
        if (len(parts) == 2):
            if (parts[0] == "Blue"):
                label = parts[0]
                fileData = readFileData(file)
                
                
                records.append(fileData)
                labels = np.append(labels, label)
    # print(records[0].shape)
    # exit()
    return (records, labels)

def getRecordsMaxLength(records):
    maxLen = 0
    for record in records:
        if (len(record) > maxLen):
            maxLen = len(record)
        
    return maxLen

def extendRecordsLen(records, length):
    ret = np.empty((0, length, 3))
    # print(ret.shape)

    for index in range(len(records)):
        record = records[index]
        # print(record.shape)

        if (len(record) < length):
            record = np.pad(record, ((0, length - len(record)), (0,0)), mode='constant', constant_values=0)
            
        if filter_value != 0: 
            record = filterRecord(record, filter_value)
            # print("fucking here")
            # print(record.shape)
        ret = np.append(ret, [record], axis = 0)
    #     print("\n\n")
    #     print(ret.shape)
    #     print("\n\n")

    # print(ret.shape)
    # exit()
    return ret



def normalizeRecords(records):
    return records / normalization_coef

def plotRecord(record, label):
    plt.plot(record)
    plt.ylabel(label)
    plt.show()
    
def plotRecords(record1, record2):
    fig = plt.figure(figsize=(20, 10))

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(record1)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(record2)

    plt.savefig('vis/2.jpg')

    # plt.show()

(records, labels) = readData("data")
rec_len = getRecordsMaxLength(records)
print("Record length is %d" % rec_len)
print(records[0].shape)


records = extendRecordsLen(records, rec_len)
print(records[0].shape)
print("done")

records = normalizeRecords(records)
print("done")

labelsBin = np.asarray(pd.get_dummies(labels), dtype = np.int8)

print("Samples: %d" % len(records))

# % get_backend())

plotRecords(records[0], records[1])

# exit()


