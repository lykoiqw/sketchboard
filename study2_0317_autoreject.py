"""
===================================================
Preprocessing workflow with ``autoreject`` and ICA
===================================================

We recommend that you first highpass filter the data,
then run autoreject (local) and supply the bad epochs detected by it
to the ICA algorithm for a robust fit, and finally run
autoreject (local) again.
"""

import os.path as op
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

import mne
import autoreject
import openneuro

dataset = 'ds002778'  # The id code on OpenNeuro for this example dataset
subject_id = 'pd14'

# download data
target_dir = op.join(op.dirname(autoreject.__file__), '..', 'examples')
# target_dir = op.join(op.dirname(autoreject.__file__), 'examples')

# skip this part if already you've already executed this
openneuro.download(dataset=dataset, target_dir=target_dir,
                   include=[f'sub-{subject_id}/ses-off'])

# load raw data from bdf file
raw_fname = op.join(target_dir, f'sub-{subject_id}',
                    'ses-off', 'eeg', 'sub-pd14_ses-off_task-rest_eeg.bdf')
raw = mne.io.read_raw_bdf(raw_fname, preload=True)

# load sensor locations
dig_montage = mne.channels.make_standard_montage('biosemi32')   # built-in montage for Biosemi 32 channel EEG
raw.drop_channels([ch for ch in raw.ch_names
                   if ch not in dig_montage.ch_names])
raw.set_montage(dig_montage)


# high-pass filter for removing slow drift
raw.filter(l_freq=1, h_freq=None)
epochs = mne.make_fixed_length_epochs(raw, duration=3, preload=True)

# autoreject for high-pass filtered data
ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                           n_jobs=1, verbose=True)
ar.fit(epochs[:20])  # fit on a few epochs to save time
epochs_ar, reject_log = ar.transform(epochs, return_log=True)

# compute ICA
ica = mne.preprocessing.ICA(random_state=99)
ica.fit(epochs[~reject_log.bad_epochs])

# plot source components to see blink artifacts
exclude = [0,   # blinks
           2]    # saccades
ica.plot_components(exclude)
ica.exclude = exclude

# plot with and without eyeblink component
ica.plot_overlay(epochs.average(), exclude=ica.exclude)
ica.apply(epochs, exclude=ica.exclude)

# compute channel-level rejections
ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                           n_jobs=1, verbose=True)
ar.fit(epochs[:20])  # fit on the first 20 epochs to save time
epochs_ar, reject_log = ar.transform(epochs, return_log=True)
epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))

# visualize dropped epochs
epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))

# visualize reject log
reject_log.plot('horizontal')

# visualize the cleaned average data & compare it against bad segments
evoked_bad = epochs[reject_log.bad_epochs].average()
plt.figure()
plt.plot(evoked_bad.times, evoked_bad.data.T * 1e6, 'r', zorder=-1)
epochs_ar.average().plot(axes=plt.gca())