#!/usr/bin/env python
# coding: utf-8

# In[14]:


# import comet_ml in the top of your file
# from comet_ml import Experiment
    
# # Add the following code anywhere in your machine learning file
# experiment = Experiment(api_key="8kbyc7YDajZyTL0alFDxawj8c",
#                         project_name="sonidos", workspace="hzlclock")
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, SimpleRNN, LSTM, GRU, Lambda, Permute, Input
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D, Concatenate, Conv1D, MaxPooling1D
from keras import backend as K
from keras.utils import plot_model
from keras.models import load_model
from sklearn.metrics import classification_report
import random
import os
import librosa
import numpy as np

import time
import datetime
import pytz
import re
import sys, tqdm

import librosa
import librosa.display
import numpy as np
import scipy
import tqdm
import IPython.display as ipd
import keras.backend as K


# In[2]:


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def load_audio(fdir, window_sz=20, batchsz=500):
    files=os.listdir(fdir)
    random.shuffle(files)
    for f in files:
        musica, sr = librosa.load(fdir+'/'+f, sr=44100)
        print('LOAD', f)
        smusica=smooth(musica, window_len=20)
        #[win-1-win] is our training window
        xb=[]
        yb=[]
        for i in range(0, musica.shape[0]-1-2*window_sz):
            x=smusica[i:i+1+2*window_sz].reshape(-1,1)
            y=musica[i:i+1+2*window_sz]
            x=np.abs(x)
            y=np.abs(y)
            xb.append(x)
            yb.append(y)
#             print(len(xb))
            if len(xb) == batchsz:
                yield np.array(xb), np.array(yb)
                xb=[]
                yb=[]
        if len(xb)>0:
            yield np.array(xb, dtype=np.float16), np.array(yb, dtype=np.float16)
    #     return np.array(xb), np.array(yb)
#             yield musica[i:i+1+2*window_sz].reshape(1,-1), musica[i+window_sz+1].reshape(1,-1)


# In[39]:


window_sz=60
# model = Sequential([
#     Dense(1024, activation='relu', input_shape=(2*window_sz+1,)),
#     Dense(1024, activation='relu'),
#     Dense(1024, activation='relu'),
#     Dense(1, activation='sigmoid'),
# ])
# model.compile(loss=keras.losses.mean_squared_error,
#     optimizer=keras.optimizers.Adam(lr=0.00005, decay=1e-6, beta_1=0.9, beta_2=0.999, amsgrad=False),
#     metrics=['accuracy'])


# regressor = Sequential()

# regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (2*window_sz+1,1)))
# regressor.add(Dropout(0.2))
# regressor.add(LSTM(units = 50, return_sequences = True))
# # regressor.add(Dropout(0.2))
# # regressor.add(LSTM(units = 50, return_sequences = True))
# regressor.add(Dropout(0.2))
# regressor.add(LSTM(units = 50))
# regressor.add(Dropout(0.2))

# regressor.add(Dense(2*window_sz+1, activation='sigmoid'))
# model=regressor

# regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# model=regressor

# model = Sequential()
# model.add(Embedding(1, 32, input_length=2*window_sz+1))
# model.add(Bidirectional(LSTM(256)))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
# model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
# model.build()

model = Sequential()
model.add(Conv1D(64, (4), input_shape=(2*window_sz+1,1), activation='relu', strides=2))
model.add(Conv1D(96, (4), activation='relu'))
model.add(Dropout(0.3))
model.add(Conv1D(128, (4), activation='relu'))
# model.add(Conv1D(128, (4), activation='relu'))
model.add(Flatten())
model.add(Dense(2*window_sz+1, activation='sigmoid'))




# returns train, inference_encoder and inference_decoder models
# def define_models(n_input, n_output, n_units):
# 	# define training encoder
# 	encoder_inputs = Input(shape=(None, n_input))
# 	encoder = LSTM(n_units, return_state=True)
# 	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# 	encoder_states = [state_h, state_c]
# 	# define training decoder
# 	decoder_inputs = Input(shape=(None, n_output))
# 	decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
# 	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
# 	decoder_dense = Dense(n_output, activation='softmax')
# 	decoder_outputs = decoder_dense(decoder_outputs)
# 	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# 	# define inference encoder
# 	encoder_model = Model(encoder_inputs, encoder_states)
# 	# define inference decoder
# 	decoder_state_input_h = Input(shape=(n_units,))
# 	decoder_state_input_c = Input(shape=(n_units,))
# 	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# 	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
# 	decoder_states = [state_h, state_c]
# 	decoder_outputs = decoder_dense(decoder_outputs)
# 	decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
# 	# return all models
# 	return model, encoder_model, decoder_model
# model=define_models(2*window_sz+1, 2*window_sz+1, 128)[0]

def customLoss(yTrue, yPred):
    return K.sum(K.log(yTrue)-K.log(yPred))

model.compile(loss=keras.losses.mean_absolute_error,
    optimizer=keras.optimizers.Adam(lr=0.005, decay=1e-6, beta_1=0.9, beta_2=0.999, amsgrad=False),
    metrics=['accuracy'])
model.summary()

batchsz=5000
train_step=0
test_step=0
dev_step=0
def test():
    predict=[]
    label=[]
    for i in load_audio('test', batchsz=batchsz, window_sz=window_sz):
            loss, acc = model.evaluate(i[0], i[1], verbose=0)
            metrics = {
              'loss_test':loss,
              'accuracy_test':acc
            }
            print(metrics)
    print("====EPOCH====",e,"=====TEST=====")
    print(classification_report(label, predict))

sttime=time.time()
for e in range(0, 10):
#     with experiment.train():
        for loaded in load_audio('train_classic', batchsz=batchsz, window_sz=window_sz):
            train_step+=1
#             print(loaded[0].shape)
#             x=np.abs(loaded[0])
#             x=np.array(x, dtype=np.float16)
#             print(x.reshape(-1, input_shape[0], input_shape[1], input_shape[2]).shape)
#             print(x.dtype)
#             print(loaded[0].shape)
            loss, acc = model.train_on_batch(loaded[0],loaded[1])
            
            metrics = {
                'loss':loss,
                'accuracy':acc
            }
            print("STEP %d\tACC %.9f\tLOSS%.9f|"%(train_step, acc, loss))
#             sys.stdout.write("%d:%.4f-%.4f|"%(train_step, acc, loss))
#             sys.stdout.flush()
#             experiment.set_step(train_step)
#             experiment.log_metrics(metrics)

            if train_step%200==0:
                timestr = datetime.datetime.fromtimestamp(int(time.time()),
                                                          pytz.timezone('Asia/Shanghai')).strftime('UTC8_%Y%m%d_%H_%M_%S')
                model.save("model_%d_"%window_sz + timestr)
        test()