#!/usr/bin/env python
# coding: utf-8

import librosa
# import librosa.display
import numpy as np
import scipy
import tqdm
import math

import random
import numba
import sys
import cv2
from skimage.transform import resize


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


def wave(f, duration):
    fs=sr
    samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)
    return samples

def obstacle(length):
    samples = (np.sin(np.pi*np.arange(length)/(length))).astype(np.float32)
    samples+=0.1*np.random.rand(length)
#     print(length, samples.shape)
    return samples


# In[90]:


musica, sr = librosa.load(sys.argv[1], sr=44100)
musica=smooth(musica, window_len=8)
librosa.output.write_wav(sys.argv[1]+'.smooth.wav', musica, sr)

# In[91]:


# mask=np.random.rand(musica.shape[0]+100)


# In[92]:


# ob_st_freq=200
# ob_end_freq=10000

# minlen=int(sr/ob_end_freq)
# maxlen=int(sr/ob_st_freq)
# for i in tqdm.trange(0, 20000):
#     pos=random.randint(2*maxlen, musica.shape[0]-2*maxlen)
#     le=random.randint(minlen, maxlen)
#     obs=obstacle(le)
# #     print(obs.shape)
#     mask[pos:pos+le]+=obs
    
# mask=smooth(mask, window_len=80)

# mask*=0.02
# musica+=mask[0:musica.shape[0]]


# librosa.output.write_wav('gen.wav', musica, sr)
# librosa.output.write_wav('mask.wav', mask, sr)


state=0.0
epsilon=0.05
slicelen=100
rang=0.2
offset=np.zeros(math.ceil(musica.shape[0]/slicelen))
for i in range(0, offset.shape[0]):
    state+=random.uniform(
        (-epsilon-state*epsilon),
        (epsilon-state*epsilon))
#     state+=random.uniform(
#         rang*-epsilon, rang*epsilon)
    offset[i]=state

offset*=rang
offset=np.power(2, offset)
print(np.max(offset), np.min(offset))



tail=math.ceil(musica.shape[0]/slicelen)*slicelen-musica.shape[0]
tail



bmusica_array=np.zeros((int(musica.shape[0]*2),1))
# print(bmusica_array.shape)

smusica=np.pad(musica, (0, tail), mode='constant')
# badmusica=[]
pos=0
pos_musica=0
# bmusica_array=[]
for idx, i in enumerate(tqdm.trange(math.ceil(musica.shape[0]/slicelen))):
    # badslice=np.array(resize(smusica[pos_musica: pos_musica+slicelen],
    #      (int(slicelen*0.95),1), anti_aliasing=False), copy=True)
    orig=smusica[pos_musica: pos_musica+slicelen].reshape(1,-1)
    # print(orig.shape)
    # print(pos_musica/sr, pos/sr, np.max(orig), np.min(orig))
    badslice=cv2.resize(orig,(int(slicelen*offset[i]),1))
    badslice=badslice.reshape(-1,1)
    # print(badslice.shape)
    # badslice=np.trim_zeros(badslice)
    # if badslice.shape[0] != 1900:
    #     print(badslice.shape[0])
    
    pos_musica+=slicelen
    bmusica_array[pos:pos+badslice.shape[0]]+=badslice
    pos+=badslice.shape[0]
    # bmusica_array.append(badslice)
bmusica_array=np.asarray(bmusica_array)
bmusica_array=bmusica_array.ravel()
print(bmusica_array.shape)
print("SAVE", sys.argv[1]+'.bad.wav')
# np.trim_zeros(bmusica_array)
librosa.output.write_wav(sys.argv[1]+'.bad.wav', np.trim_zeros(bmusica_array), sr)


