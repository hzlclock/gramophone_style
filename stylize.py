#!/usr/bin/env python
# coding: utf-8

import librosa
import soundfile as sf
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
import os


epsilon=0.12 #random pitch mod range
slicelen=1000 #random pitch mod interval
rang=0.10   
pitch_mod_global=0.97 #overall pitch modulation
smooth_window=4

from pysndfx import AudioEffectsChain

fx = (
    AudioEffectsChain()
    .equalizer(20,  db=-12.0)
    .equalizer(40,  db=-9.0)
    .equalizer(80,  db=-6.0)
    .equalizer(120, db=-4.0)
    .equalizer(200, db=-2.0)
    # .equalizer(500, db=3.0)
    # .equalizer(1000,db=3.0)
    # .equalizer(1600,db=3.0)
    # .equalizer(3000,db=3.0)
)


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

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def addobstacle(h,l1,l2,xpos,track):
    l1i=int(l1)
    l2i=int(l2)
    offset_o=np.zeros(l1i+l2i)
    # k1=float(h)/l1
    k1=-float(h)/(l1*l1)
    for i in range(-l1i,0):
        offset_o[l1i+i]=k1*i*i+h

    k2=-float(h)/(l2*l2)
    for i in range(0,l2i):
        offset_o[l1i+i]=k2*i*i+h

    assert(xpos+l1i+l2i<track.shape[0])
    try:
        track[xpos: xpos+l1i+l2i]+=offset_o
    except:
        print(track.shape, xpos, xpos+l1i+l2i)

    return track


musica, sr = librosa.load(sys.argv[1])
musica=smooth(musica, window_len=smooth_window)
musica=fx(musica)

state=0.0

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
offset*=pitch_mod_global
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
    badslice=np.array(cv2.resize(orig,(int(slicelen*offset[i]),1)), copy=True)
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
bmusica_array=np.trim_zeros(bmusica_array)
print(bmusica_array.shape)

for i in range(0,int(len(bmusica_array)/500)):
    bmusica_array=addobstacle(random.uniform(0.001, 0.03),\
                             random.uniform(20,1000),\
                             random.uniform(20,2000),\
                             int(random.random()*(len(bmusica_array)-4000)), bmusica_array)

for i in range(0,int(len(bmusica_array)/10000)):
    bmusica_array=addobstacle(random.uniform(0.1, 0.5),\
                             random.uniform(20,1000),\
                             random.uniform(20,2000),\
                             int(random.random()*(len(bmusica_array)-4000)), bmusica_array)


print(bmusica_array.shape)
print("SAVE", sys.argv[1]+'.bad')
# np.trim_zeros(bmusica_array)
os.makedirs("bad", exist_ok=True)
ensure_dir('bad/'+sys.argv[1]+'.bad.flac')
sf.write('bad/'+sys.argv[1]+'.bad.flac', bmusica_array, sr)


