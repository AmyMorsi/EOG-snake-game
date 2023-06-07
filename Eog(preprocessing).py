import pandas as pd 
df=pd.read_csv("EOG_dataset.csv")
df
df = df.drop(df[df['label'] == 'örnek'].index)
df
df.duplicated()
df = df[df['id'] != 111]
df
df.tail(60)
df.drop(df.index[:-20])
df = df[:-20]
df
df
DF_V = df[(df['polarity'] == 'v') & (df['label'] != 'örnek')]
DF_V

import matplotlib.pyplot as plt
import numpy as np
temp=df[['id','label','polarity']]
x=df.drop(columns=['id','label','polarity'])
x
 # Replace with the index of the row you want to extract
row_data = x.iloc[:].values.tolist()

# Plot data
plt.figure(figsize=(12,6))
plt.plot(np.arange(0, len(row_data)), row_data)
plt.xlabel('Time(s)')
plt.ylabel('Amp(V)')
plt.show()

# Filtering

from scipy import signal 
from scipy.signal import butter,filtfilt
import statistics
def butter_bandpass_filter(Input_Signal,Low_Cutoff=0.5,High_Cutoff=20.0,Sampling_Rate=176,order=2):
    nyq=0.5*Sampling_Rate
    low=Low_Cutoff/nyq
    high=High_Cutoff/nyq
    numerator,denomirator=butter(order,[low,high],btype='band',output='ba',analog=False,fs=None)
    filtered=filtfilt(numerator,denomirator,Input_Signal)
    return filtered
filtered_Signal=butter_bandpass_filter(row_data,Low_Cutoff=0.5,High_Cutoff=20.0,Sampling_Rate=176,order=2)
plt.figure(figsize=(12,6))
plt.plot(np.arange(0,len(filtered_Signal)),filtered_Signal)
plt.xlabel('Time(s)')
plt.ylabel('Amp(V)')
# Resampling
resampled_Signal=signal.resample(filtered_Signal,50)
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.plot(np.arange(0,len(filtered_Signal)),filtered_Signal)
plt.xlabel('Time(s)')
plt.ylabel('Amp(V)')
plt.subplot(122)
plt.plot(np.arange(0,len(resampled_Signal)),resampled_Signal)
plt.xlabel('Time(s)')
plt.ylabel('Amp(V)')
# import pywt

# row_data
# # Define wavelet function and level of decomposition
# wavelet = 'db4'
# level = 6

# # Perform DWT decomposition
# coeffs = pywt.wavedec(row_data, wavelet, level=level)

# # Plot decomposition coefficients
# plt.figure(figsize=(12,6))
# for i in range(len(coeffs)):
#     plt.subplot(level+1, 1, i+1)
#     plt.plot(coeffs[i])
#     plt.xlim([0, len(row_data)])
#     plt.title('Level {}'.format(i))
# plt.tight_layout()
# plt.show()
# Dc_Signal=[(filtered_Signal[i]+10) for i in range(len(filtered_Signal))]
# Mean=statistics.mean(Dc_Signal)
# RemovedDc_Signal=[(Dc_Signal[i]-Mean) for i in range(len(Dc_Signal))]
# plt.plot(np.arange(0,len(RemovedDc_Signal)),RemovedDc_Signal)
# plt.xlabel('Time(s)')
# plt.ylabel('Amp(V)')
# def DWT(row_data, wavelet='db4', level=6):
#     # Perform DWT decomposition
#     coeffs = pywt.wavedec(row_data, wavelet, level=level)

#     # Return decomposition coefficients
#     return coeffs
filterd=np.apply_along_axis(butter_bandpass_filter, 1, x.values)

filterd.shape
import scipy.signal as signal

df_filterd = pd.DataFrame(filterd, columns=x.columns)
df_filterd.shape

import scipy.signal as signal

data_array = df_filterd.to_numpy()

resampled_data = signal.resample(data_array, 50, axis=1)

resampled_data = signal.resample(resampled_data, 200, axis=0)

# Create a DataFrame with the desired shape
resampled_df = pd.DataFrame(resampled_data)
resampled_df
resampled_df.shape
df=pd.concat([resampled_df, temp], axis=1)
df
dff = pd.read_excel('EOG_dataset.xlsx')
dff
dff.to_csv('EOG(feature extraction).csv', index=False)

#####################################################################################