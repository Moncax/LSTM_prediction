import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model
from matplotlib import pyplot as plt
from pandas import read_csv
import random
path = ""


'''
下面的split_sequence（）函数实现了滑动窗口这种行为，并将给定的单变量序列分成多个样本，其中每个样本具有指定的时间步长，输出是单个时间步。
'''
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

'''
load_data():
数据导入，filename为文件路径，usecols为选用文件的第几列，header（True/None）表示是否忽略第一行数据
返回值为np.array
'''

def load_data(filename, sep, usecols, header):
    data = read_csv(filename, sep = sep, usecols=[usecols], engine='python', header = header) 
    return data.values.astype('float32')

"""
build_model():
建立模型，可以修改
"""
def build_model():
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))  # 隐藏层，输入，特征维
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

"""
模型的预测，每次调用返回真实值，预测值和将最新一个预测值加入后的源数据值
返回值：true_plot为np.array,pred_plot为np.array,raw_seq为list
"""
def prediction(raw_seq, n_steps, n_features, model):
    # demonstrate prediction
    true = []
    pred = []
    for i in range(len(raw_seq) - n_steps + 1):
        x_input =[]
        for j in range(i, i + n_steps):
            x_input.append(raw_seq[j]) #输入值为一个滑动窗口

        x_input = array(x_input)
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        pred.append(float(yhat))

    for i in range(len(raw_seq)):
        true.append(float(raw_seq[i]))
    true_plot = array(true)
    pred_plot = np.empty_like(true)

    # for i in range(n_steps + 1):
    pred_plot = np.r_[pred_plot, array([np.nan])] #加一行np.array([nan])
    pred_plot[:] = np.nan
    pred_plot[n_steps:] = pred
    raw_seq = np.r_[raw_seq, array([[(pred_plot[-1] * 1)]])] #因为把每次的预测值当成新的源数据去预测，会导致预测的结果误差越来越大，所以需要自动调整每次预测的系数
    # print(true_plot[-1])
    # print(raw_seq[-1])
    # print(pred_plot[-1])
    return true_plot, pred_plot, raw_seq

"""
训练模型并保存
"""
def model_train(X, y, epochs):
    model = build_model()
    # fit model
    model.fit(X, y, epochs=epochs, batch_size=1, verbose=0)  # 迭代次数，批次数，verbose决定是否显示每次迭代
    print("Saving model to disk: \n")
    model.save(path + "model.h5")

if __name__ == '__main__':
############################################ 参数表
    use_test_to_train_jug = 0 # use_test_to_train_jug == 1 选择每次预测数据带入训练模型; use_test_to_train_jug == 0 选择每次预测数据不带入训练，以源数据训练的模型直接预测

    new_model_jug = 1 # new_model_jug == 1 训练新模型来使用; new_model_jug == 0 使用旧模型（已训练好的模型)

    filename = path + 'data_test.csv'

    sep = '\t' #数据导入的分割符
    
    n_steps = 3 # choose a number of time steps

    n_features = 1

    pred_x_num = 40 #预测的个数

    epochs = 50 #迭代轮数
############################################
    raw_seq = load_data(filename = filename, sep = sep, usecols = 1, header = None)

    raw_seq_ori = raw_seq

    # split into samples
    X, y = split_sequence(raw_seq, n_steps)

    # reshape from [samples, timesteps] into [samples, timesteps, features]
    
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    
    if(new_model_jug): #判断是否训练模型
        model_train(X = X, y = y, epochs = epochs)

    model = load_model(path + "model.h5")
   
    for i in range(pred_x_num): #每次都通过模型预测一个值，加入源数据，一共训练pred_x_num次
        if(use_test_to_train_jug):
            X, y = split_sequence(raw_seq, n_steps)
            # print(X, y)
            # reshape from [samples, timesteps] into [samples, timesteps, features]
            n_features = 1
            X = X.reshape((X.shape[0], X.shape[1], n_features))

            model = build_model()
            # fit model
            model.fit(X, y, epochs=20, batch_size=1, verbose=0)  # 迭代次数，批次数，verbose决定是否显示每次迭代

        true_plot, pred_plot, raw_seq = prediction(raw_seq = raw_seq, n_steps = n_steps, n_features = n_features, model = model)

    print("Saving model to disk: \n")
    model.save(path + "model.h5")
    plt.plot(raw_seq_ori, color='black')
    plt.plot(pred_plot, color='green')
    plt.show()
