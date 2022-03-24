import os
import numpy as np
import matplotlib.pyplot as plt
import mne

# #### 3-2) Epoching ####

# load (filtered & downsampled) data
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_filt-0-40_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)

# preprocessing
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)

ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)
ica.exclude = [1, 2]  # ICs with EOG, ECG artifacts
ica.plot_properties(raw, picks=ica.exclude)

# detecting experimental events (using STIM channels)
events = mne.find_events(raw, stim_channel='STI 014')
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'smiley': 5, 'buttonpress': 32}


# epoching
# define reject criteria
reject_criteria = dict(mag=4000e-15,     # 4000 fT
                       grad=4000e-13,    # 4000 fT/cm
                       eeg=150e-6,       # 150 µV
                       eog=250e-6)       # 250 µV

# make Epochs object
epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=0.5,
                    reject=reject_criteria, preload=True)

# pool across left/right stimulus presentations so we can compare auditory versus visual responses
conds_we_care_about = ['auditory/left', 'auditory/right',
                       'visual/left', 'visual/right']

# randomly sample epochs from each condition
epochs.equalize_event_counts(conds_we_care_about)
aud_epochs = epochs['auditory']
vis_epochs = epochs['visual']
del raw, epochs  # free up memory

aud_epochs.plot_image(picks=['MEG 1332', 'EEG 021'])


# epoching with (unfiltered) raw data
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False).crop(tmax=60)

# find events
events = mne.find_events(raw, stim_channel='STI 014')

# epoching
epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7)
print(epochs)
print(epochs.event_id)

# provide event dictionary
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'buttonpress': 32}
epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, event_id=event_dict,
                    preload=True)
print(epochs.event_id)
del raw     # free up memory


# visualization of Epochs object
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False).crop(tmax=120)

events = mne.find_events(raw, stim_channel='STI 014')
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'button': 32}

epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, event_id=event_dict,
                    preload=True)
del raw

# plotting Epochs as time series
catch_trials_and_buttonpresses = mne.pick_events(events, include=[5, 32])
epochs['face'].plot(events=catch_trials_and_buttonpresses, event_id=event_dict,
                    event_color=dict(button='red', face='blue'))

# plotting with sensors
epochs['face'].plot(events=catch_trials_and_buttonpresses, event_id=event_dict,
                    event_color=dict(button='red', face='blue'),
                    group_by='selection', butterfly=True)

# plotting sensor locations
epochs.plot_sensors(kind='3d', ch_type='all')
epochs.plot_sensors(kind='topomap', ch_type='all')

# plotting the power spectrum
epochs['auditory'].plot_psd(picks='eeg')

# plotting spectral estimates as scalp topography
# default parameters plot five freq bands
epochs['visual/right'].plot_psd_topomap()

# specify bands parameter for custom viewing
bands = [(10, '10 Hz'), (15, '15 Hz'), (20, '20 Hz'), (10, 20, '10-20 Hz')]
epochs['visual/right'].plot_psd_topomap(bands=bands, vlim='joint',
                                        ch_type='grad')