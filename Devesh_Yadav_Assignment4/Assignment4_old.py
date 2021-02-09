#!/usr/bin/env python
# coding: utf-8

# In[120]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import pickle
import csv
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import warnings
warnings.filterwarnings("ignore")
# setting option values for comfortable viewing experience in jupyter notebook
pd.options.display.max_columns = 100
pd.options.display.max_rows = 20
np.set_printoptions(suppress=True)


# In[121]:


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
    dv_list = df_insulin_1[['Date_Time', 'BWZ Carb Input (grams)', 'index', 'BWZ Estimate (U)']].values
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
            ground_truth_val.append(listee.iloc[0]['BWZ Estimate (U)'])
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


# In[122]:


np_meal_data, np_no_meal_data, ground_truth = load_and_extract("CGMData.csv", "InsulinData.csv")
ground_truth = np.round(ground_truth).astype(int)

# combining the meal data and ground truth to eliminate nan values in the meal data, and recreating these matrixes
np_meal_data_temp = np.append(np_meal_data, ground_truth.reshape((ground_truth.shape[0], 1)), 1)
np_meal_data_temp = np_meal_data_temp[~np.isnan(np_meal_data_temp).any(axis=1)]
np_meal_data = np_meal_data_temp[:, :30]
ground_truth = np_meal_data_temp[:, -1:]


# In[123]:


num_bins = int(np.floor((np.amax(np_meal_data) - np.amin(np_meal_data))/20.0)) +1
if (np.amax(np_meal_data) - np.amin(np_meal_data))%20 == 0.0:
    maxi = np.amax(np_meal_data) + 1
else:
    maxi = np.amax(np_meal_data)
binz = np.arange(np.amin(np_meal_data), maxi+20, 20)
labelz = np.arange(1, num_bins+1, 1)

# creating rule set
b_max = pd.DataFrame(np.amax(np_meal_data, 1), columns=['b_max'])
b_meal = pd.DataFrame(np_meal_data[:, 6], columns=['b_meal'])
insulin_bolus = pd.DataFrame(ground_truth, columns=['insulin_bolus'])

# bining the data
b_max['b_max'] = pd.cut(b_max['b_max'], bins = binz, labels = labelz, include_lowest=True)
b_meal['b_meal'] = pd.cut(b_meal['b_meal'], bins = binz, labels = labelz, include_lowest=True)

# combining the data into a single dataframe
df_rule = pd.DataFrame(b_max['b_max'], columns = ['b_max'])
df_rule = df_rule.merge(b_meal, left_index = True, right_index = True)
df_rule = df_rule.merge(insulin_bolus, left_index=True, right_index = True)
df_rule['insulin_bolus'] = df_rule['insulin_bolus'].astype(int)
df_rulez = df_rule.copy()


# In[124]:


rule_list = []
for i in range(df_rule.shape[0]):
    tempe = []
    tempe.append('bmax_' + str(df_rule.values[i, 0]))
    tempe.append('bmeal_' + str(df_rule.values[i, 1]))
    tempe.append('bolus_' + str(df_rule.values[i, 2]))
    rule_list.append(tempe)


# In[125]:


# encoding the data so that it is digestible by our algorithm
encoder = TransactionEncoder()
rule_encoded = encoder.fit(rule_list).transform(rule_list)
df_rule_encoded = pd.DataFrame(rule_encoded, columns = encoder.columns_)

# using apriori algorithm to find itemsets
item_sets = apriori(df_rule_encoded, min_support=0.0001, use_colnames=True)
item_sets['length']=item_sets['itemsets'].apply(lambda x: len(x))

# finding 3-itemsets and generating rules
frequent_sets=item_sets[(item_sets['length']==3)]
df_rule=association_rules(item_sets, metric="confidence", min_threshold=0.0001)
df_rule["antecedent_len"]=df_rule["antecedents"].apply(lambda x: len(x))
df_rule = df_rule[df_rule['antecedent_len'] >= 2]


# In[126]:


# generating list of valid consequents
temp_list = df_rulez['insulin_bolus'].unique().tolist()
valid_consequents = []
for x in temp_list:
    valid_consequents.append('bolus_'+str(x))
valid_consequents_frozen = []
for x in valid_consequents:
    valid_consequents_frozen.append(frozenset([x]))

# removing rules that are invalide according to our problem statement    
df_rule = df_rule[df_rule['consequents'].isin(valid_consequents_frozen)]


# In[127]:


itemlist_list = frequent_sets['itemsets'].tolist()
check_this_out = [list(x) for x in itemlist_list]
final_list_frequent = []
for tempe_list in check_this_out:
    #print('----------------------------------------------------')
    filters = ["bmax_", "bmeal_", "bolus_"]
    #print(tempe_list)
    temp_val = []
    for filter_a in filters:
        b = [xe for xe in tempe_list if filter_a in xe]
        if len(b[0]) == len(filter_a) + 2:
            b = b[0][-2:]
        else:
            b = b[0][-1:]
        temp_val.append(b)
    #print(temp_val)
    temp_str = '('+temp_val[0]+', '+temp_val[1]+', '+temp_val[2]+')'
    #print(temp_str)
    final_list_frequent.append([temp_str])

with open('frequent_itemsets.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerows(final_list_frequent)


# In[128]:


antecedents_list = df_rule[df_rule['confidence'] <=0.15]['antecedents'].tolist()
consequent_list = df_rule[df_rule['confidence'] <=0.15]['consequents'].tolist()
check_this_out = [list(x) for x in antecedents_list]
_2check_this_out = [list(x) for x in consequent_list]
final_list_anomalous = []
for i in range(0, len(check_this_out)):
    #print('-------------------------------------')
    tempe_list = check_this_out[i]
    filters = ["bmax_", "bmeal_"]
    #print(tempe_list)
    temp_val = []
    for filter_a in filters:
        b = [xe for xe in tempe_list if filter_a in xe]
        if len(b[0]) == len(filter_a) + 2:
            b = b[0][-2:]
        else:
            b = b[0][-1:]
        temp_val.append(b)
    # extract bolus value from the list
    filter_a = "bolus_"
    #print(_2check_this_out[i])
    b = [xe for xe in _2check_this_out[i] if filter_a in xe]
    if len(b[0]) == len(filter_a) + 2:
        b = b[0][-2:]
    else:
        b = b[0][-1:]
    temp_val.append(b)
    #print(temp_val)
    temp_str = '{'+temp_val[0]+', '+temp_val[1]+'}->'+temp_val[2]
    #print(temp_str)
    final_list_anomalous.append([temp_str])

with open('anomalous_rules.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerows(final_list_anomalous)


# In[129]:


antecedents_list = df_rule[df_rule['confidence'] == df_rule.confidence.max()]['antecedents'].tolist()
consequent_list = df_rule[df_rule['confidence'] == df_rule.confidence.max()]['consequents'].tolist()
check_this_out = [list(x) for x in antecedents_list]
_2check_this_out = [list(x) for x in consequent_list]
final_list_anomalous = []
for i in range(0, len(check_this_out)):
    #print('-------------------------------------')
    tempe_list = check_this_out[i]
    filters = ["bmax_", "bmeal_"]
    #print(tempe_list)
    temp_val = []
    for filter_a in filters:
        b = [xe for xe in tempe_list if filter_a in xe]
        if len(b[0]) == len(filter_a) + 2:
            b = b[0][-2:]
        else:
            b = b[0][-1:]
        temp_val.append(b)
    # extract bolus value from the list
    filter_a = "bolus_"
    #print(_2check_this_out[i])
    b = [xe for xe in _2check_this_out[i] if filter_a in xe]
    if len(b[0]) == len(filter_a) + 2:
        b = b[0][-2:]
    else:
        b = b[0][-1:]
    temp_val.append(b)
    #print(temp_val)
    temp_str = '{'+temp_val[0]+', '+temp_val[1]+'}->'+temp_val[2]
    #print(temp_str)
    final_list_anomalous.append([temp_str])

with open('largest_confidence_rules.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerows(final_list_anomalous)


# In[ ]:




