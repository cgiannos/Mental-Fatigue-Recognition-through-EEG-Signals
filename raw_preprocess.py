import numpy as np
import pandas as pd
import scipy.signal
from scipy.stats import entropy
import matplotlib.pyplot as plt
import mne
import os
# You need the libraries: numpy, pandas, os, matplotlib, scipy, mne
# You need no change the paths in lines 11, 57, 78 

dir = r'C:\Users\giannos\Desktop\Biosignal\fatigued'

channels=['Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3',
          'FC1','C1','C3','C5','T7','TP7','CP5','CP3','CP1','P1','P3','P5','P7',
          'P9','PO7','PO3','O1','Oz','POz','Pz','CPz','Fpz','Fp2','AF8','AF4','AFz',
          'Fz','F2','F4','F6','F8','FT8','FC6','FC4','FC2','FCz','Cz','C2','C4','C6','T8'
          ,'TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10','PO8','PO4','O2']

bands = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 35)
}

chan = 63
sampling_rate = 256
n_samples = 769

def preprocess(eeg):

    info = mne.create_info(ch_names=channels, sfreq=256, ch_types='eeg')

    raw = mne.io.RawArray(eeg.values, info)

    filt = raw.filter(1, 35, fir_design='firwin')

    baseline_mean = filt.get_data().mean(axis=-1)
    baseline = filt.copy()
    baseline._data -= baseline_mean[:, np.newaxis]

    baseline._data=scipy.signal.detrend(baseline._data)

    return np.array(baseline._data)



data = []
labels = []
temp = 'AD'
i=0


for i in range(2):
    if i>0:
        dir = r'C:\Users\giannos\Desktop\Biosignal\rested'
    for file in os.listdir(dir):
        subject, event, _ = os.path.splitext(file)[0].split('_')
        print (subject,event)

        path = os.path.join(dir, file)
        

        eeg_data = pd.read_csv(path, header=None)

        eeg_preprocessed = preprocess(eeg_data)
        #print(eeg_preprocessed.shape)

        data.append(eeg_preprocessed)
        labels.append(i)
        

features = np.asarray(data)
labels = np.asarray(labels)
print(features.shape)
print(labels.shape)
np.savez(r'C:\Users\giannos\Desktop\raw_mental.npz', features=features, labels=labels)