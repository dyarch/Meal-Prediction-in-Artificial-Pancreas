#!/usr/bin/env python
# coding: utf-8

# In[188]:


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
from sklearn import svm, model_selection
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
# setting option values for comfortable viewing experience in jupyter notebook
pd.options.display.max_columns = 100
np.set_printoptions(suppress=True)


# In[189]:


def load_and_extract(cgmFile, insulinFile):
    # reading data and preparing it
    df_cgm_1 = pd.read_csv(cgmFile, parse_dates=[['Date', 'Time']])
    df_insulin_1 = pd.read_csv(insulinFile, parse_dates=[['Date', 'Time']])
    df_insulin_1.drop(columns=['Index'], inplace = True)
    df_insulin_1['BWZ Carb Input (grams)'] = df_insulin_1['BWZ Carb Input (grams)'].fillna(0.0)
    df_insulin_1.reset_index(inplace = True)
    nan_replacement = 0.0

    # creating a date value list for further processing
    dv_list = df_insulin_1[['Date_Time', 'BWZ Carb Input (grams)', 'index']].values
    dv_list = np.flipud(dv_list)
    # get current time in number of seconds from the start of the data
    reference_pt = dv_list[0, 0]
    for i in range(0, len(dv_list)):
        dv_list[i, 0] = (dv_list[i, 0] - reference_pt).total_seconds()
    dv_list[:, 0] = dv_list[:, 0].astype(int)
    dv_list[:, 1] = dv_list[:, 1].astype(int)

    # print debuggin info
    print('dv_list: ', dv_list)
    x = dv_list[dv_list[:, 1] != nan_replacement]
    print('Possible Meal starting points: ', x)
    print('Lenght: ', len(x))

    # logic for selecting meal start dates according to specification from the Insulin data
    considering = False
    lag = -1
    tm_index = -999
    counter = 0
    meal_start_dates = []
    for i in range(0, len(dv_list)):
        temp = dv_list[i, :]
        if considering:
            lag = temp[0] - tm
        if temp[1] != nan_replacement and considering == False:
            #print(temp)
            tm = temp[0]
            tm_index = temp[2]
            considering = True
            continue
        if temp[1] != nan_replacement and considering and lag < 7200:
            tm = temp[0]
            tm_index = temp[2]
            continue
        if temp[1] != nan_replacement and considering and lag == 7200:
            considering = False
            counter = counter + 1
            listee = df_insulin_1.loc[df_insulin_1['index'] == temp[2]]
            meal_start_dates.append(listee.iloc[0]['Date_Time'])
            continue
        if temp[1] == nan_replacement and considering and lag >= 7200:
            considering = False
            counter= counter + 1
            listee = df_insulin_1.loc[df_insulin_1['index'] == tm_index]
            meal_start_dates.append(listee.iloc[0]['Date_Time'])

    print('Meal starting points according to specification: ', counter)



    # convert meal starting dates in insulin data to meal starting dates in CGM data
    df_cgm_1 = df_cgm_1[['Date_Time', 'Sensor Glucose (mg/dL)']]
    df_cgm_1.reset_index(inplace=True)
    meal_start_dates_cgm = []
    meal_start_dates_cgm_index = []
    for insulin_date in meal_start_dates:
        cgm_record = df_cgm_1.loc[df_cgm_1['Date_Time'] > insulin_date][-1:]
        cgm_record_date = cgm_record.iloc[0]['Date_Time']
        cgm_record_index = cgm_record.iloc[0]['index']
        if len(meal_start_dates_cgm) == 0:
            meal_start_dates_cgm.append(cgm_record_date)
            meal_start_dates_cgm_index.append(cgm_record_index)
        else:
            if meal_start_dates_cgm_index[-1] - cgm_record_index > 31:
                meal_start_dates_cgm.append(cgm_record_date)
                meal_start_dates_cgm_index.append(cgm_record_index)

    # there complications after converting dates from insulin to CGM data
    # tm1 + 2hr and tm2 - 30min might overlap and cause us
    # delete_idx_list = []
    for i in range(1, len(meal_start_dates_cgm)):
        if (meal_start_dates_cgm[i]-meal_start_dates_cgm[i-1]).total_seconds() <= 9000:
            #delete_idx_list.append(i)
            print('found one')
            print(meal_start_dates_cgm[i-1])
            print(meal_start_dates_cgm[i])

    # marking meal data points as 1
    counter = 0
    meal_or_not = np.zeros(len(df_cgm_1))
    for i in range(0, len(meal_start_dates_cgm_index)):
        idx = meal_start_dates_cgm_index[i]
        if idx-6 < 0 or idx+23 >= len(df_cgm_1):
            continue
        else:
            counter = counter + 1
            for j in range(idx-6, idx+24):
                meal_or_not[j] = 1
    print(counter)

    # Debugging code
    # i = 0
    # counter = 0
    # summer = 0
    # while i < len(meal_or_not):
    #     if meal_or_not[i] == 1:
    #         counter = counter + 1
    #         summer = 0
    #         while i < len(meal_or_not) and meal_or_not[i] == 1:
    #             summer = summer + 1
    #             i = i+1
    #         if summer > 30:
    #             print('wtf')
    #     i = i+1
    # print(counter)

    # now we fill the not-meal values in the meal_or_not list - check for a window of size 24 that has only zeros
    helper = 0
    counter = 0
    j = 23
    for i in range(0, j+1):
        helper = helper + meal_or_not[i]
    while True:
        #print(j, ': ', meal_or_not[j], ': ',helper)
        if helper == 0:
            counter = counter+1
            for i in range(j-23,j+1):
                helper = helper + 2
                meal_or_not[i] = 2

        j = j+1
        if j < len(meal_or_not):
            helper = helper + meal_or_not[j]
            helper = helper - meal_or_not[j-24]
        else:
            break


    # some idea on data
    print('We have identified meal and no meal data points from the given data set. Now see these stats.')
    sick = 0
    for i in range(0, len(meal_or_not)):
        if meal_or_not[i] == 2:
            sick = sick + 1
    print('Non-meal data points:', sick)
    print("No-meal data (bugged): ", counter)
    sick = 0
    for i in range(0, len(meal_or_not)):
        if meal_or_not[i] == 1:
            sick = sick + 1
    print('meal data points:', sick)
    print('meal data: ', len(meal_start_dates_cgm_index))

    # Creating data matrix from the meal/no-meal labels assigned to data points in the CGM data
    meal_data = []
    no_meal_data = []
    i = 0
    while i < len(meal_or_not):
        #print('index: ', i, meal_or_not[i])
        flag = False
        while i < len(meal_or_not) and meal_or_not[i] == 0.0 :
            i = i+1
        if i >= len(meal_or_not):
            break
        if meal_or_not[i] == 1.0:
            meal_temp = []
            for j in range(0, 30):
                if np.isnan(df_cgm_1.iloc[i+j]['Sensor Glucose (mg/dL)']) == True:
                    flag = True
                    break
                meal_temp.append(df_cgm_1.iloc[i+j]['Sensor Glucose (mg/dL)'])
            if flag == False:
                meal_data.append(meal_temp)
            i = i + 30
        elif meal_or_not[i] == 2.0:
            #print('yes 2.0')
            no_meal_temp = []
            for j in range(0, 24):
                if np.isnan(df_cgm_1.iloc[i+j]['Sensor Glucose (mg/dL)']) == True:
                    flag = True
                    break
                no_meal_temp.append(df_cgm_1.iloc[i+j]['Sensor Glucose (mg/dL)'])
            if flag == False:
                no_meal_data.append(no_meal_temp)
            i = i + 24

    # converting list to numpy array
    np_meal_data = np.array(meal_data)
    print('Meal data matric size before processing: ', np_meal_data.shape)
    np_no_meal_data = np.array(no_meal_data)
    print('No meal data matrix size before processing: ', np_no_meal_data.shape)
    return np_meal_data, np_no_meal_data


# In[190]:


def createFeature(np_meal_data, np_no_meal_data):
    # Now we start by creating features for meal and no meal data
    df_meal = pd.DataFrame(np_meal_data)
    df_no_meal = pd.DataFrame(np_no_meal_data)
    # initialize the feature dataframes
    feature_meal = pd.DataFrame()
    feature_no_meal = pd.DataFrame()
    # first feature tmin - tmax =diffTime
    # for meal
    ff1 = pd.DataFrame()
    ff1['diffTime'] = df_meal.apply(lambda row: abs(row.idxmax() - row.idxmin()), axis =1)
    # feature_meal = feature_meal.merge(ff1, left_index=True, right_index = True)
    feature_meal = ff1
    # for no meal
    ff1 = pd.DataFrame()
    ff1['diffTime'] = df_no_meal.apply(lambda row: abs(row.idxmax() - row.idxmin()), axis =1)
    # feature_no_meal = feature_no_meal.merge(ff1, left_index=True, right_index = True)
    feature_no_meal = ff1
    # second feature Glucosemin- GlucoseMax
    # for meal
    ff2 = pd.DataFrame()
    ff2['diffGlucose'] = df_meal.apply(lambda row: row.max() - row.min(), axis = 1)
    feature_meal = feature_meal.merge(ff2, left_index=True, right_index=True)
    # for no meal
    ff2 = pd.DataFrame()
    ff2['diffGlucose'] = df_no_meal.apply(lambda row: row.max() - row.min(), axis = 1)
    feature_no_meal = feature_no_meal.merge(ff2, left_index=True, right_index=True)
    # third feature Fourier transform
    def fourier(row):
        val = abs(fft(row))
        val.sort()
        return np.flip(val)[0:3]

    # for meal
    ff31 = pd.DataFrame()
    ff31['FFT'] = df_meal.apply(lambda x: fourier(x), axis=1)
    ff3 = pd.DataFrame(ff31.FFT.tolist(), columns=['FFT1', 'FFT2', 'FFT3'])
    feature_meal = feature_meal.merge(ff3, left_index=True, right_index=True)

    # for no meal
    ff31 = pd.DataFrame()
    ff31['FFT'] = df_no_meal.apply(lambda x: fourier(x), axis=1)
    ff3 = pd.DataFrame(ff31.FFT.tolist(), columns=['FFT1', 'FFT2', 'FFT3'])
    feature_no_meal = feature_no_meal.merge(ff3, left_index=True, right_index=True)

    # fourth feature - CGMVelocity
    # for meal
    feature_meal['CGMVelocity'] = np.nan
    for i in range(len(df_meal)):
        liste_temp = df_meal.loc[i, :].tolist()
        summer = []
        for j in range(1, df_meal.shape[1]):
            summer.append(abs(liste_temp[j]-liste_temp[j-1]))
        feature_meal.loc[i, 'CGMVelocity'] = np.round(np.mean(summer), 2)

    # for no meal
    feature_no_meal['CGMVelocity'] = np.nan
    for i in range(len(df_no_meal)):
        liste_temp = df_no_meal.loc[i, :].tolist()
        summer = []
        for j in range(1, df_no_meal.shape[1]):
            summer.append(abs(liste_temp[j]-liste_temp[j-1]))
        feature_no_meal.loc[i, 'CGMVelocity'] = np.round(np.mean(summer), 2)

    # fourth feature part 2 - tmax
    # for meal
    ff4 = pd.DataFrame()
    ff4['maxTime'] = df_meal.apply(lambda row: row.idxmax(), axis =1)
    feature_meal = feature_meal.merge(ff4, left_index=True, right_index = True)
    # for no meal
    ff4 = pd.DataFrame()
    ff4['maxTime'] = df_no_meal.apply(lambda row: row.idxmax(), axis =1)
    feature_no_meal = feature_no_meal.merge(ff4, left_index=True, right_index = True)

    # fifth feature skewness
    # for meal
    feature_meal['Skewness'] = np.nan
    for i in range(len(df_meal)):
        feature_meal['Skewness'][i] = ts.skewness(df_meal.loc[i, :])
    # for no meal
    feature_no_meal['Skewness'] = np.nan
    for i in range(len(df_no_meal)):
        feature_no_meal['Skewness'][i] = ts.skewness(df_no_meal.loc[i, :])

    # sixth feature entorpy
    # for meal
    # feature_meal['Entropy'] = np.nan
    # for i in range(len(df_meal)):
    #     feature_meal['Entropy'][i] = ts.sample_entropy(np.array(df_meal.iloc[i, :]))
    # # for no meal
    # feature_no_meal['Entropy'] = np.nan
    # for i in range(len(df_no_meal)):
    #     feature_no_meal['Entropy'][i] = ts.sample_entropy(np.array(df_no_meal.iloc[i, :]))

    # seventh feature kurtosis
    # for meal
    feature_meal['Kurt'] = np.nan
    for i in range(len(df_meal)):
        feature_meal['Kurt'][i] = ts.kurtosis(np.array(df_meal.iloc[i, :]))
    # for no meal
    feature_no_meal['Kurt'] = np.nan
    for i in range(len(df_no_meal)):
        feature_no_meal['Kurt'][i] = ts.kurtosis(np.array(df_no_meal.iloc[i, :]))
    return feature_meal, feature_no_meal


# In[ ]:


if __name__ == "__main__":
    np_meal_data, np_no_meal_data = load_and_extract("CGMData.csv", "InsulinData.csv")
    feature_meal, feature_no_meal = createFeature(np_meal_data, np_no_meal_data)
    print('Loading in second set of data - Patient2')
    np_meal_data, np_no_meal_data = load_and_extract("CGMPatient3.csv", "InsulinPatient3.csv")
    feature_meal1, feature_no_meal1 = createFeature(np_meal_data, np_no_meal_data)
    feature_meal = pd.concat([feature_meal, feature_meal1])
    feature_no_meal = pd.concat([feature_no_meal, feature_no_meal1])
    # get things ready for training
    # normalize the features
    feature = pd.concat([feature_meal, feature_no_meal], ignore_index=True)
    feature = StandardScaler().fit_transform(feature)
    # concat data
    # feature = pd.concat([feature_meal, feature_no_meal], ignore_index=True)
    label = pd.concat([pd.DataFrame({'Class': [1]*len(feature_meal.index)}), pd.DataFrame({'Class': [0]*len(feature_no_meal.index)})], ignore_index=True)
    print('Feature dimensions: ', feature.shape)
    lable = label.to_numpy()

    # getting started with KFold and training model
    k_fold = KFold(n_splits=3, shuffle=True, random_state=0)
    scoring ={'precision', 'recall', 'f1', 'accuracy'}
    classifier1 = RandomForestClassifier(n_estimators=50)
    # print ('RF: ', cross_val_score(clf7, X, y, cv=k_fold, n_jobs=1))
    print('Stats about the trained model')
    metrix = model_selection.cross_validate(estimator=classifier1, X=feature, y=label, cv=k_fold, scoring=scoring)
    print('Accuracy: ', np.mean(metrix['test_accuracy'])*100)
    classifier1.fit(feature, label)
    # save the model
    rf_model_file = "rf_model.pkl"
    with open(rf_model_file, 'wb') as f:
        pickle.dump(classifier1, f)


# In[ ]:





# In[ ]:





# In[ ]:




