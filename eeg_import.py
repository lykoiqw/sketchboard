import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

import mne

# #### 1) Loading data ####

sample_data_raw_file = 'C:/Users/lykoi/Desktop/BCICIV_2a_gdf/A01T.gdf'
raw = mne.io.read_raw_gdf(sample_data_raw_file)

print(raw.info)

n_time_samps = raw.n_times
time_secs = raw.times
ch_names = raw.ch_names
n_chan = len(ch_names)  # note: there is no raw.n_channels attribute
print('the (cropped) sample data object has {} time samples and {} channels.'
      ''.format(n_time_samps, n_chan))
print('The last time sample is at {} seconds.'.format(time_secs[-1]))

raw.plot()
raw.plot(duration=60, proj=False, n_channels=len(raw.ch_names),
         remove_dc=False)

# plot spectral density
raw.plot_psd(average=True)

# edit channel types & names
raw.set_channel_types({'EOG-left': 'eog', 'EOG-central': 'eog', 'EOG-right': 'eog'})
channel_renaming_dict = {'EEG-0': 'EEG-FC3', 'EEG-1': 'EEG-FC1', 'EEG-2': 'EEG-FCz', 'EEG-3': 'EEG-FC2', 'EEG-4': 'EEG-FC4',
                         'EEG-5': 'EEG-C5', 'EEG-6': 'EEG-C1', 'EEG-7': 'EEG-C2', 'EEG-8': 'EEG-C6',
                         'EEG-9': 'EEG-CP3', 'EEG-10': 'EEG-CP1', 'EEG-11': 'EEG-CPz', 'EEG-12': 'EEG-CP2', 'EEG-13': 'EEG-CP4',
                         'EEG-14': 'EEG-P1', 'EEG-15': 'EEG-P2', 'EEG-16': 'EEG-POz'}
raw.rename_channels(channel_renaming_dict)

eeg_raw = raw.copy().pick_types(eeg=True, eog=False)
eog_raw = raw.copy().pick_types(eeg=False, eog=True)
print(len(raw.ch_names), '→', len(eeg_raw.ch_names), '+', len(eog_raw.ch_names))

# working with events
# read embedded events as annotations
print(raw.annotations)

events_from_annot, event_dict = mne.events_from_annotations(raw)
print(event_dict)
print(events_from_annot[:10])



fig = mne.viz.plot_events(events_from_annot, event_id=event_dict, sfreq=raw.info['sfreq'],
                          first_samp=raw.first_samp)

# subselecting events
events_lh = mne.pick_events(events_from_annot, include=7)
events_rh = mne.pick_events(events_from_annot, include=8)

custom_dict = {'1023': 'rejected trial', '1072': 'eye movement',
                  '276': 'idling/eyes open', '277': 'idling/eyes closed',
                  '32766': 'start of new run', '768': 'start of trial',
                  '769': 'cue onset/left hand', '770': 'cue onset/right hand',
                  '771': 'cue onset/foot', '772': 'cue onset/tongue'}

(events_from_annot, event_dict) = mne.events_from_annotations(raw, event_id=custom_dict)