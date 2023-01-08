from python_speech_features import mfcc
import scipy.io.wavfile
from scipy.fftpack import dct
import numpy as np

from tempfile import TemporaryFile
import os
import pickle
import random 
import operator

import math
import numpy

import glob

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

import pandas as pd
import warnings

from tkinter import filedialog
from tkinter import*
import tkinter as tk

#pre-emphasis
def pre_emphasis(signal):
    pre_emphasis = 0.97
    signal_emphasis = numpy.append(signal[0],signal[1:]-pre_emphasis*signal[:-1])
    return signal_emphasis

#frame Blocking
def framming(signal_emphasis, sample_rate):
    frame_size=0.025
    frame_stride=0.01
    frame_length=frame_size*sample_rate
    frame_step=frame_stride*sample_rate
    signal_length=len(signal_emphasis)
    frames_overlap=frame_length-frame_step
    
    num_frames=numpy.abs(signal_length-frames_overlap)//numpy.abs(frame_length-frames_overlap)
    rest_samples=numpy.abs(signal_length-frames_overlap)%numpy.abs(frame_length-frames_overlap)
    
    pad_signal_length=int(frame_length-rest_samples)
    z=numpy.zeros((pad_signal_length))
    pad_signal=numpy.append(signal_emphasis,z)
    
    frame_length=int(frame_length)
    frame_step=int(frame_step)
    num_frames=int(num_frames)
    
    indices=numpy.tile(numpy.arange(0, frame_length),(num_frames, 1))+numpy.tile(numpy.arange(0, num_frames*frame_step, frame_step),(frame_length,1)).T
    
    frames=pad_signal[indices.astype(numpy.int32,copy=False)]
    return frames, frame_length

#windowing
def windowing(frames,frame_length):
    frames=frames*(numpy.hamming(frame_length))
    return frames

#FFT
def fft(frames):
    NFFT=512
    mag_frames=numpy.absolute(numpy.fft.rfft(frames, NFFT))
    pow_frames=((1.0/NFFT)*((mag_frames)**2))
    return pow_frames, NFFT

#filter bank
def filter_bank(pow_frames, sample_rate, NFFT):
    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB

    return (filter_banks/ numpy.amax(filter_banks))*255

#DCT
def cepstral_liftering(filter_banks):
    num_ceps=12
    cep_lifter=11
    mfcc=dct(filter_banks, type=2,axis=1,norm='ortho')[:,:(num_ceps)]
    (nframes, ncoeff)=mfcc.shape
    n=numpy.arange(ncoeff)
    lift=1+(cep_lifter/2)*numpy.sin(numpy.pi*n/cep_lifter)
    mfcc=(numpy.mean(mfcc,axis=0)+1e-8)
    return mfcc

#generate mfcc to data test and train
def generate_features():
    all_features=[]
    all_labels=[]
    directory = "../SKRIPSI"
    f= open("my.dat" ,'wb')
    i=0
    audios=['ardeidae','cuculidae','laridae','rallidae']
    for audio in audios:
        print(audio)
        sound_files=glob.glob(audio+'/*.wav')
        print('processing %d audio in %s file...'%(len(sound_files),audio))
        for file in sound_files:
            print(file)
            sample_rate, signal = scipy.io.wavfile.read(file)
            signal = signal[0:int(1*sample_rate),0]
            pre=pre_emphasis(signal)
            frames=framming(pre, sample_rate)
            window=windowing(frames[0],frames[1])
            ff=fft(window)
            filter=filter_bank(ff[0],sample_rate, ff[1])
            mfcc=cepstral_liftering(filter)
            mfcc=numpy.ndarray.flatten(mfcc)
            all_features.append(mfcc)
            all_labels.append(audio)
    f.close()
    return all_features, all_labels

X, y = generate_features()

#make dataframes
df=pd.DataFrame(y)
df_new = pd.concat([df, pd.DataFrame(X)], axis=1)
df_new.columns =['famili','0', '1', '2','3','4','5','6','7','8','9','10','11']

#split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df_new[['0', '1', '2','3','4','5','6','7','8','9','10','11']], df_new['famili'], random_state=0, stratify=y)

#normalisasi data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

#new audio input
root = tk.Tk()
root.title("klasifikasi")
root.geometry('300x100')

warnings.filterwarnings('ignore')

def open_file():
    global file
    file = filedialog.askopenfilename(filetypes=(("Audio Files",".wav"), ("All Files", "*.*")))
    print('Reading file \n "%s"' % file) #menginput audio baru
    lbl2.configure(text=(file))

lbl=Label(root, text="Input File Audio :", anchor="e",width=20)
lbl.grid(column=0, row=1)
btn = Button(root, text='open file', command = open_file)
btn.grid(column=1, row=1)

namafile=Label(root, text="Nama File :", anchor="e", width=20)
namafile.grid(column=0, row=2)
lbl2=Label(root, text="...", anchor="w", width=10)
lbl2.grid(column=1, row=2)

lbl3=Label(root, text="hasil (famili) :", anchor="e", width=20)
lbl3.grid(column=0, row=4)
hasil=Label(root, text="...", anchor="w", width=10)
hasil.grid(column=1, row=4)

def klasifikasi():
    global famili_prediction
    sample_rate, signal = scipy.io.wavfile.read(file)
    signal = signal[0:int(1*sample_rate),0] #preprocessing untuk mengubah sinyal analog ke diginal 
    pre=pre_emphasis(signal) #pre-emphasis
    frames=framming(pre, sample_rate) #frame blocking
    window=windowing(frames[0],frames[1]) #windowing
    ff=fft(window) #FFT
    filter=filter_bank(ff[0],sample_rate, ff[1]) #filterbank
    mfcc=cepstral_liftering(filter) #DCT dan cepstral liftering
    mfcc=numpy.ndarray.flatten(mfcc)
    mfcc=mfcc.reshape(1, -1)
    mfcc = scaler.transform(mfcc) #normalisasi data
    print ("Nilai MFCC :\n",mfcc) #hasil ekstrasi

    knn = KNeighborsClassifier(n_neighbors = 13) #knn (k=13)
    knn.fit(X_train, y_train)
    famili_prediction = knn.predict(mfcc)
    print("hasil Klasifikasi :\n",famili_prediction[0])
    hasil.configure(text=(famili_prediction[0]))


btn2= Button(root, text='klasifikasi', command = klasifikasi)
btn2.grid(column=1, row=3)

root.mainloop()
