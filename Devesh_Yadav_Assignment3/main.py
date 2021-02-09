#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import sklearn
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
from scipy.special import entr
import scipy
import warnings
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
# setting option values for comfortable viewing experience in jupyter notebook
pd.options.display.max_columns = 100
np.set_printoptions(suppress=True)
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))


# In[13]:


def load_and_extract(cgmFile, insulinFile):
    # reading data and preparing it
    df_cgm_1 = pd.read_csv(cgmFile, parse_dates=[['Date', 'Time']])
    df_insulin_1 = pd.read_csv(insulinFile, parse_dates=[['Date', 'Time']])
    df_insulin_1.drop(columns=['Index'], inplace = True)
    df_insulin_1['BWZ Carb Input (grams)'] = df_insulin_1['BWZ Carb Input (grams)'].fillna(0.0)
    df_insulin_1.reset_index(inplace = True)
    nan_replacement = 0.0
    
    # generating ground truth values for the insulin data
#     num_clusters = int(np.floor((df_insulin_1['BWZ Carb Input (grams)'].max() - df_insulin_1['BWZ Carb Input (grams)'].min())/20.0)) +1
#     binz = np.arange(df_insulin_1['BWZ Carb Input (grams)'].min(), df_insulin_1['BWZ Carb Input (grams)'].max()+20, 20)
#     labelz = np.arange(1, num_clusters+1, 1)
#     df_insulin_1['GroundTruth'] = pd.cut(df_insulin_1['BWZ Carb Input (grams)'], bins = binz, labels = labelz, include_lowest=True)
    #df_insulin_1.sample(50)

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
    possible_meal_start_dates_insulin_idx = []
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
            possible_meal_start_dates_insulin_idx.append(listee.iloc[0]['index'])
            continue
        if temp[1] == nan_replacement and considering and lag >= 7200:
            considering = False
            counter= counter + 1
            listee = df_insulin_1.loc[df_insulin_1['index'] == tm_index]
            meal_start_dates.append(listee.iloc[0]['Date_Time'])
            possible_meal_start_dates_insulin_idx.append(listee.iloc[0]['index'])

    print('Meal starting points according to specification: ', counter)



    # convert meal starting dates in insulin data to meal starting dates in CGM data
    df_cgm_1 = df_cgm_1[['Date_Time', 'Sensor Glucose (mg/dL)']]
    df_cgm_1.reset_index(inplace=True)
    meal_start_dates_cgm = []
    meal_start_dates_cgm_index = []
    meal_start_dates_insulin_index = []
    yooo = 0
    for insulin_date in meal_start_dates:
        cgm_record = df_cgm_1.loc[df_cgm_1['Date_Time'] > insulin_date][-1:]
        cgm_record_date = cgm_record.iloc[0]['Date_Time']
        cgm_record_index = cgm_record.iloc[0]['index']
        if len(meal_start_dates_cgm) == 0:
            meal_start_dates_cgm.append(cgm_record_date)
            meal_start_dates_cgm_index.append(cgm_record_index)
            meal_start_dates_insulin_index.append(possible_meal_start_dates_insulin_idx[yooo])
        else:
            if meal_start_dates_cgm_index[-1] - cgm_record_index > 31:
                meal_start_dates_cgm.append(cgm_record_date)
                meal_start_dates_cgm_index.append(cgm_record_index)
                meal_start_dates_insulin_index.append(possible_meal_start_dates_insulin_idx[yooo])
        yooo = yooo+1

    # there complications after converting dates from insulin to CGM data
    # tm1 + 2hr and tm2 - 30min might overlap and cause us
    # delete_idx_list = []
    for i in range(1, len(meal_start_dates_cgm)):
        if (meal_start_dates_cgm[i]-meal_start_dates_cgm[i-1]).total_seconds() <= 9000:
            #delete_idx_list.append(i)
            print('found one complication after converting dates from insulin to CGM data')
            print(meal_start_dates_cgm[i-1])
            print(meal_start_dates_cgm[i])

    # marking meal data points as 1
    counter = 0
    ground_truth_val = []
    meal_or_not = np.zeros(len(df_cgm_1))
    for i in range(0, len(meal_start_dates_cgm_index)):
        idx = meal_start_dates_cgm_index[i]
        if idx-6 < 0 or idx+23 >= len(df_cgm_1):
            continue
        else:
            counter = counter + 1
            listee = df_insulin_1.loc[df_insulin_1['index'] == meal_start_dates_insulin_index[i]]
            ground_truth_val.append(listee.iloc[0]['BWZ Carb Input (grams)'])
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
                # remove this comment block to remove nan values in meal data
#                 if np.isnan(df_cgm_1.iloc[i+j]['Sensor Glucose (mg/dL)']) == True:
#                     flag = True
#                     break
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
    ground_truth = np.array(ground_truth_val)
    return np_meal_data, np_no_meal_data, ground_truth


# In[14]:


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


# In[15]:


np_meal_data, np_no_meal_data, ground_truth = load_and_extract("CGMData.csv", "InsulinData.csv")

# combining the meal data and ground truth to eliminate nan values in the meal data, and recreating these matrixes
np_meal_data_temp = np.append(np_meal_data, ground_truth.reshape((ground_truth.shape[0], 1)), 1)
np_meal_data_temp = np_meal_data_temp[~np.isnan(np_meal_data_temp).any(axis=1)]
np_meal_data = np_meal_data_temp[:, :30]
ground_truth1 = np_meal_data_temp[:, -1:]

# creating bins 
ground_truth = ground_truth1
label = pd.DataFrame(ground_truth)
num_clusters = int(np.floor((label[0].max() - label[0].min())/20.0)) +1
binz = np.arange(label[0].min(), label[0].max()+20, 20)
labelz = np.arange(1, num_clusters+1, 1)
label['GT'] = pd.cut(label[0], bins = binz, labels = labelz, include_lowest=True)

# remove clusters with not enough data points - threshold is 3% of the whole data set
gt_cluster_count = np.array(label.groupby(['GT']).agg(number_of_rows=('GT', 'count')))
drop_cluster_id = []
threshold = np.ceil(3*ground_truth.shape[0]/100)
for i in range(0, gt_cluster_count.shape[0]):
    if gt_cluster_count[i] < threshold:
        drop_cluster_id.append(i+1)

# create feature matrix from the extracted data
feature_meal, feature_no_meal = createFeature(np_meal_data, np_no_meal_data)


# In[16]:


# removing the above mentioned data points
temp_feature = feature_meal.merge(label['GT'], left_index=True, right_index = True)
temp_feature = temp_feature[~temp_feature['GT'].isin(drop_cluster_id)]
temp_feature.reset_index(inplace=True)
num_clusters = temp_feature['GT'].nunique()
feature_meal = temp_feature.loc[:, temp_feature.columns != 'GT']
label = pd.DataFrame(temp_feature['GT'])


# In[17]:


# normalizing data
feature = StandardScaler().fit_transform(feature_meal)
pca = PCA(n_components=2, random_state=42)
feature = pca.fit_transform(feature)
# training the kmeans classifier
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(feature)


# In[18]:


# calculating the center of the clusters
cluster_centers = np.zeros((num_clusters, feature.shape[1]))
cluster_size = np.zeros((num_clusters, 1))
for i in range(0, feature.shape[0]):
    cluster_id = kmeans.labels_[i]
    cluster_size[cluster_id] += 1
    for j in range(0, feature.shape[1]):
        cluster_centers[cluster_id][j] += feature[i][j]
        
for i in range(0, num_clusters):
    num_num = cluster_size[i]
    for j in range(0, cluster_centers.shape[1]):
        cluster_centers[i][j] /= num_num
        
# calculating sse
kmeans_sse = 0
for i in range(0, feature.shape[0]):
    cluster_id = kmeans.labels_[i]
    a = cluster_centers[cluster_id]
    b = feature[i]
    temp = np.linalg.norm(a-b)
    kmeans_sse += temp**2
    
# calculating purity
contingency_matrix = sklearn.metrics.cluster.contingency_matrix(label['GT'], kmeans.labels_)
max_from_row = np.zeros((contingency_matrix.shape[0], 1))
total_sum = 0
all_max_sum = 0
for i in range(0, contingency_matrix.shape[0]):
    temp_max = -1
    for j in range(0, contingency_matrix.shape[1]):
        temp_max = max(temp_max, contingency_matrix[i][j])
        total_sum += contingency_matrix[i][j]
    all_max_sum += temp_max
kmeans_purity = np.round(all_max_sum/total_sum, 2)

# calculating entropy
kmeans_entropy = 0
for i in range(0, contingency_matrix.shape[0]):
    kmeans_entropy += scipy.stats.entropy(contingency_matrix[i], base=2)*np.sum(contingency_matrix[i])/total_sum


# In[19]:


# training dbscan classifier
db = DBSCAN(eps=0.47, min_samples=3).fit(feature)
n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
n_noise_ = list(db.labels_).count(-1)
print("DBSCAN - number of clusters : ", n_clusters_)


# In[20]:


# calculating the center of the clusters
cluster_centers = np.zeros((num_clusters, feature.shape[1]))
cluster_size = np.zeros((num_clusters, 1))
for i in range(0, feature.shape[0]):
    cluster_id = db.labels_[i]
    cluster_size[cluster_id] += 1
    for j in range(0, feature.shape[1]):
        cluster_centers[cluster_id][j] += feature[i][j]
        
for i in range(0, num_clusters):
    num_num = cluster_size[i]
    for j in range(0, cluster_centers.shape[1]):
        cluster_centers[i][j] /= num_num

# assigning noise to clusters
db_labels = db.labels_
for i in range(0, feature.shape[0]):
    cluster_id = -1
    min_dist = 200000
    a = feature[i]
    if db.labels_[i] == -1:
        for j in range(0, cluster_centers.shape[0]):
            b = cluster_centers[j]
            calc_dist = np.linalg.norm(a-b)
            if calc_dist < min_dist:
                min_dist = calc_dist
                cluster_id = j
        db_labels[i] = j
        
# calculating sse
db_sse = 0
for i in range(0, feature.shape[0]):
    cluster_id = db_labels[i]
    a = cluster_centers[cluster_id]
    b = feature[i]
    temp = np.linalg.norm(a-b)
    db_sse += temp**2
    
# calculating purity
contingency_matrix = sklearn.metrics.cluster.contingency_matrix(label['GT'], db_labels)
max_from_row = np.zeros((contingency_matrix.shape[0], 1))
total_sum = 0
all_max_sum = 0
for i in range(0, contingency_matrix.shape[0]):
    temp_max = -1
    for j in range(0, contingency_matrix.shape[1]):
        temp_max = max(temp_max, contingency_matrix[i][j])
        total_sum += contingency_matrix[i][j]
    all_max_sum += temp_max
db_purity = np.round(all_max_sum/total_sum, 2)

# calculating entropy
db_entropy = 0
for i in range(0, contingency_matrix.shape[0]):
    db_entropy += scipy.stats.entropy(contingency_matrix[i], base=2)*np.sum(contingency_matrix[i])/total_sum


# In[21]:


res_columns = ['SSE_Kmeans', 'SSE_DBSCAN', 'Entropy_Kmeans', 'Entropy_DBSCAN', 'Purity_Kmeans', 'Purity_DBSCAN']
res_val = []
res_val.append(np.round(kmeans_sse, 2))
res_val.append(np.round(db_sse, 2))
res_val.append(kmeans_entropy)
res_val.append(db_entropy)
res_val.append(kmeans_purity)
res_val.append(db_purity)
results = pd.DataFrame([res_val], columns = res_columns)
print(results)
results.to_csv('results.csv', index = False)


# In[22]:


results


# In[ ]:




