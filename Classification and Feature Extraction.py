import pandas as pd
import pywt
import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
df=pd.read_csv('EOG(feature extraction).csv')
t=df[['id','label','polarity']]
df=df.drop(['id','label','polarity'],axis=1)
df
# Extract Features by PSD 

from scipy.signal import welch

def compute_psd(signal, fs):
    freqs, psd = welch(signal, fs=fs)
    return psd.tolist()

psd_features = []

eog_wavelet_features = []
statistical_features = []
morphological_features = []
mixed_features = []
psd_features = []
from scipy.signal import welch

for i in range(len(df)):
    signal = df.iloc[i]  # Assuming each row represents a signal

    
    # Compute power spectral density
    fs = 50  # Specify your sampling frequency
    psd = compute_psd(signal, fs)
    psd_features.append(psd)

statistical_df = pd.DataFrame(statistical_features)
morphological_df = pd.DataFrame(morphological_features)
mixed_df = pd.DataFrame(mixed_features)
psd_df = pd.DataFrame(psd_features)

psd_df=pd.concat([psd_df, t], axis=1)
psd_df

axis_h = psd_df[psd_df['polarity'] == 'h']
axis_v = psd_df[psd_df['polarity'] == 'v']
axis_h
merged_feature = pd.merge(axis_v, axis_h, on=['id','label'])
merged_feature

# Classification PSD Features (Bouns)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Example data
X = merged_feature.drop(['label','id','polarity_y','polarity_x'], axis=1, inplace=False)
y = merged_feature['label'] # Target labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.24, random_state=42,shuffle=True)

# Create a logistic regression model
model = LogisticRegression(max_iter=100000)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Create a KNN classifier with the desired number of neighbors
model = KNeighborsClassifier(n_neighbors=5)

# Train the model
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)

train_accuracy = accuracy_score(y_train, y_train_pred)

y_test_pred = model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split    

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
y_test_pred = model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print(y_pred)
y_pred_df = pd.DataFrame(y_pred, columns=['predicted_label'])



from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Create an AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=220)

# Train the model
model.fit(X_train, y_train)

# Predict the labels for the training set
y_train_pred = model.predict(X_train)

# Calculate the training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)

# Predict the labels for the testing set
y_test_pred = model.predict(X_test)

# Calculate the test accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the training and test accuracies
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Create an SVM classifier
svm = SVC(kernel='linear')

# Train the model
svm.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred = svm.predict(X_train)

# Calculate the training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)

# Make predictions on the test set
y_test_pred = model.predict(X_test)

# Calculate the test accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the training and test accuracies
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
# Extract Features by Wevlet 
def extract_wavelet_features(signal, wavelet='db4', levels=4):
    # Apply wavelet decomposition
    coefficients = pywt.wavedec(signal, wavelet, level=levels)
    
    # Extract features from approximation and detail coefficients
    features = []
    for i in range(levels):
        cA = coefficients[i]  # Approximation coefficients
        cD = coefficients[i+1]  # Detail coefficients
        
        # Statistical features from approximation coefficients
        features.append(np.mean(cA))
        features.append(np.std(cA))
        features.append(np.median(cA))
        features.append(np.max(cA))
        features.append(np.min(cA))
        
        # Statistical features from detail coefficients
        features.append(np.mean(cD))
        features.append(np.std(cD))
        features.append(np.median(cD))
        features.append(np.max(cD))
        features.append(np.min(cD))
    
    return features


wavelet_features = []
for i in range(len(df)):
    wavelet_features.append(extract_wavelet_features(df.iloc[i]))
wf =pd.DataFrame(wavelet_features)
wf=pd.concat([wf, t], axis=1)

import pandas as pd


h_data = wf[wf['polarity'] == 'h']
v_data = wf[wf['polarity'] == 'v']

h_data
merged_featurew = pd.merge(v_data, h_data, on=['id','label'])
merged_featurew
# Classification Wevelt Features 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Example data
X = merged_featurew.drop(['label','id','polarity_y','polarity_x'], axis=1, inplace=False)
y = merged_featurew['label'] # Target labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.24, random_state=42,shuffle=True)

# Create a logistic regression model
model2 = LogisticRegression(max_iter=100000)

# Train the model
model2.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model2.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)
print(y_pred)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split    
model2 = RandomForestClassifier()
model2.fit(X_train, y_train)
y_train_pred = model2.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
y_test_pred = model2.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print(y_pred,y_test)

######
# Conclusion
##### As we see accuracy with Random forrest classifier is higer than Logistic and KNN with PSD features or Wevlet features 
# Game
y_pred_df
df_h = y_pred_df[y_pred_df['predicted_label'].isin(['Right', 'Left'])].copy()
df_v = y_pred_df[y_pred_df['predicted_label'].isin(['Up', 'Down'])].copy()

df_h.to_csv('file1h.csv', index=False)
df_v.to_csv('file2v.csv', index=False)
df_h
y_pred_df.to_csv('PredictedLabels.csv', index=False)