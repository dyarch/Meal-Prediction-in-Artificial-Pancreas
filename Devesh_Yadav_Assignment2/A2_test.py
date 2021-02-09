#!/usr/bin/env python
# coding: utf-8

# In[12]:


from A2_train import createFeature
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import scipy.stats
from scipy import fft
import pywt
import tsfresh.feature_extraction.feature_calculators as ts
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import svm, model_selection
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
# setting option values for comfortable viewing experience in jupyter notebook
pd.options.display.max_columns = 100
np.set_printoptions(suppress=True)


# In[17]:


# read test file
test_data = pd.read_csv('test.csv')
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
test_data = imputer.fit_transform(test_data)
dummy_data = pd.DataFrame(np.random.rand(100, 30))
placeholder_feature, feature = createFeature(dummy_data, test_data)
feature = StandardScaler().fit_transform(feature)
classifier1 = pickle.load(open(r'rf_model.pkl', 'rb'))
predicted_labels = classifier1.predict(feature)
print("Length of labels: ", len(predicted_labels))
np.savetxt("Results.csv", predicted_labels, delimiter=",", fmt='%d')
