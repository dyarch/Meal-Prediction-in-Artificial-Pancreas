#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[14]:

print('-------- Loading Data --------')

df_cgm = pd.read_csv(r'CGMData.csv', parse_dates=[['Date', 'Time']])
df_insulin = pd.read_csv(r'InsulinData.csv', parse_dates=[['Date', 'Time']])


# In[15]:

print('-------- dividing cgm records into auto and manual mode --------')

# dividing cgm records into auto and manual mode
cgm = df_cgm[['Index', 'Date_Time', 'Sensor Glucose (mg/dL)']]
insulin = df_insulin[['Index', 'Date_Time', 'Alarm']]
cgm['Date'] = cgm['Date_Time'].dt.date.astype('datetime64')
auto_mode_start_record = insulin.loc[insulin['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF'][-1:]
auto_start_date = auto_mode_start_record.iloc[0]['Date_Time']
cgm_auto = cgm.loc[cgm['Date_Time'] > auto_start_date]
cgm_manual = cgm.loc[cgm['Date_Time'] < auto_start_date]


# In[16]:


# get average glucose value for each day and replace null values with the respective mean value for the respective day
cgm_auto['SG_Mean'] = cgm_auto.groupby('Date')['Sensor Glucose (mg/dL)'].transform('mean')
cgm_manual['SG_Mean'] = cgm_manual.groupby('Date')['Sensor Glucose (mg/dL)'].transform('mean')


# In[17]:


# replacing null values with the mean value for that day
cgm_auto['Sensor Glucose (mg/dL)'].fillna(cgm_auto.SG_Mean, inplace = True)
cgm_manual['Sensor Glucose (mg/dL)'].fillna(cgm_manual.SG_Mean, inplace = True)


# In[18]:


# removing days that have more than 288 data points or less than 288 - threshold
date_info = cgm_auto.groupby('Date')['Index'].count().reset_index()
threshold = 13
for x in date_info.Date.unique():
    tmp_row = date_info.loc[date_info['Date'] == x, 'Index']
    num_records = tmp_row.iloc[0]
    if num_records > 288 or num_records < 288 - threshold:
        cgm_auto.drop(cgm_auto.loc[cgm_auto['Date'] == x].index ,inplace = True)

# imitating for the manual mode
date_info = cgm_manual.groupby('Date')['Index'].count().reset_index()
threshold = 13
for x in date_info.Date.unique():
    tmp_row = date_info.loc[date_info['Date'] == x, 'Index']
    num_records = tmp_row.iloc[0]
    if num_records > 288 or num_records < 288 - threshold:
        cgm_manual.drop(cgm_manual.loc[cgm_manual['Date'] == x].index ,inplace = True)


# In[19]:


# creating the result matrix
col = ['>180', '>250', '70<=<=180', '70<=<=150', '<70', '<54']
col_prefix = ['night', 'day', 'whole_day']
all_col = []
for pre in col_prefix:
    for c in col:
        val = pre+"("+c+")"
        all_col.append(val)

results = pd.DataFrame(columns=all_col)


# In[20]:

print('-------- Calculating results --------')

# creating a new column - tells us if this records is a day one or night one
cgm_auto['night/day'] = cgm_auto.Date_Time.apply(lambda x:'night' if (int(x.strftime('%H')) < 6) else 'day')
cgm_manual['night/day'] = cgm_manual.Date_Time.apply(lambda x:'night' if (int(x.strftime('%H')) < 6) else 'day')


# In[21]:


date_info = cgm_manual.groupby('Date')['Index'].count().reset_index()
num_days = date_info.shape[0]
final_results = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for x in date_info.Date.unique():
    tmp_results = []
    # night
    tmp_results.append(cgm_manual.loc[(cgm_manual['Date'] == x) & (cgm_manual['Sensor Glucose (mg/dL)'] > 180) & (cgm_manual['night/day'] == 'night')]['Index'].count())
    tmp_results.append(cgm_manual.loc[(cgm_manual['Date'] == x) & (cgm_manual['Sensor Glucose (mg/dL)'] > 250) & (cgm_manual['night/day'] == 'night')]['Index'].count())
    tmp_results.append(cgm_manual.loc[(cgm_manual['Date'] == x) & (cgm_manual['Sensor Glucose (mg/dL)'].between(70, 180)) & (cgm_manual['night/day'] == 'night')]['Index'].count())
    tmp_results.append(cgm_manual.loc[(cgm_manual['Date'] == x) & (cgm_manual['Sensor Glucose (mg/dL)'].between(70, 150)) & (cgm_manual['night/day'] == 'night')]['Index'].count())
    tmp_results.append(cgm_manual.loc[(cgm_manual['Date'] == x) & (cgm_manual['Sensor Glucose (mg/dL)'] < 70) & (cgm_manual['night/day'] == 'night')]['Index'].count())
    tmp_results.append(cgm_manual.loc[(cgm_manual['Date'] == x) & (cgm_manual['Sensor Glucose (mg/dL)'] < 54) & (cgm_manual['night/day'] == 'night')]['Index'].count())

    #day
    tmp_results.append(cgm_manual.loc[(cgm_manual['Date'] == x) & (cgm_manual['Sensor Glucose (mg/dL)'] > 180) & (cgm_manual['night/day'] == 'day')]['Index'].count())
    tmp_results.append(cgm_manual.loc[(cgm_manual['Date'] == x) & (cgm_manual['Sensor Glucose (mg/dL)'] > 250) & (cgm_manual['night/day'] == 'day')]['Index'].count())
    tmp_results.append(cgm_manual.loc[(cgm_manual['Date'] == x) & (cgm_manual['Sensor Glucose (mg/dL)'].between(70, 180)) & (cgm_manual['night/day'] == 'day')]['Index'].count())
    tmp_results.append(cgm_manual.loc[(cgm_manual['Date'] == x) & (cgm_manual['Sensor Glucose (mg/dL)'].between(70, 150)) & (cgm_manual['night/day'] == 'day')]['Index'].count())
    tmp_results.append(cgm_manual.loc[(cgm_manual['Date'] == x) & (cgm_manual['Sensor Glucose (mg/dL)'] < 70) & (cgm_manual['night/day'] == 'day')]['Index'].count())
    tmp_results.append(cgm_manual.loc[(cgm_manual['Date'] == x) & (cgm_manual['Sensor Glucose (mg/dL)'] < 54) & (cgm_manual['night/day'] == 'day')]['Index'].count())

    # whole day
    tmp_results.append(cgm_manual.loc[(cgm_manual['Date'] == x) & (cgm_manual['Sensor Glucose (mg/dL)'] > 180)]['Index'].count())
    tmp_results.append(cgm_manual.loc[(cgm_manual['Date'] == x) & (cgm_manual['Sensor Glucose (mg/dL)'] > 250)]['Index'].count())
    tmp_results.append(cgm_manual.loc[(cgm_manual['Date'] == x) & (cgm_manual['Sensor Glucose (mg/dL)'].between(70, 180))]['Index'].count())
    tmp_results.append(cgm_manual.loc[(cgm_manual['Date'] == x) & (cgm_manual['Sensor Glucose (mg/dL)'].between(70, 150))]['Index'].count())
    tmp_results.append(cgm_manual.loc[(cgm_manual['Date'] == x) & (cgm_manual['Sensor Glucose (mg/dL)'] < 70)]['Index'].count())
    tmp_results.append(cgm_manual.loc[(cgm_manual['Date'] == x) & (cgm_manual['Sensor Glucose (mg/dL)'] < 54)]['Index'].count())

    final_results = np.add(final_results, tmp_results)

# doing operations to get the metrics that we will add to the results data frame
final_results[:] = [float(x) for x in final_results]
divider = float(288*num_days)/100.00
final_results = np.divide(final_results, divider)
results.loc[len(results)] = final_results


# In[22]:


date_info = cgm_auto.groupby('Date')['Index'].count().reset_index()
num_days = date_info.shape[0]
final_results = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for x in date_info.Date.unique():
    tmp_results = []
    # night
    tmp_results.append(cgm_auto.loc[(cgm_auto['Date'] == x) & (cgm_auto['Sensor Glucose (mg/dL)'] > 180) & (cgm_auto['night/day'] == 'night')]['Index'].count())
    tmp_results.append(cgm_auto.loc[(cgm_auto['Date'] == x) & (cgm_auto['Sensor Glucose (mg/dL)'] > 250) & (cgm_auto['night/day'] == 'night')]['Index'].count())
    tmp_results.append(cgm_auto.loc[(cgm_auto['Date'] == x) & (cgm_auto['Sensor Glucose (mg/dL)'].between(70, 180)) & (cgm_auto['night/day'] == 'night')]['Index'].count())
    tmp_results.append(cgm_auto.loc[(cgm_auto['Date'] == x) & (cgm_auto['Sensor Glucose (mg/dL)'].between(70, 150)) & (cgm_auto['night/day'] == 'night')]['Index'].count())
    tmp_results.append(cgm_auto.loc[(cgm_auto['Date'] == x) & (cgm_auto['Sensor Glucose (mg/dL)'] < 70) & (cgm_auto['night/day'] == 'night')]['Index'].count())
    tmp_results.append(cgm_auto.loc[(cgm_auto['Date'] == x) & (cgm_auto['Sensor Glucose (mg/dL)'] < 54) & (cgm_auto['night/day'] == 'night')]['Index'].count())

    #day
    tmp_results.append(cgm_auto.loc[(cgm_auto['Date'] == x) & (cgm_auto['Sensor Glucose (mg/dL)'] > 180) & (cgm_auto['night/day'] == 'day')]['Index'].count())
    tmp_results.append(cgm_auto.loc[(cgm_auto['Date'] == x) & (cgm_auto['Sensor Glucose (mg/dL)'] > 250) & (cgm_auto['night/day'] == 'day')]['Index'].count())
    tmp_results.append(cgm_auto.loc[(cgm_auto['Date'] == x) & (cgm_auto['Sensor Glucose (mg/dL)'].between(70, 180)) & (cgm_auto['night/day'] == 'day')]['Index'].count())
    tmp_results.append(cgm_auto.loc[(cgm_auto['Date'] == x) & (cgm_auto['Sensor Glucose (mg/dL)'].between(70, 150)) & (cgm_auto['night/day'] == 'day')]['Index'].count())
    tmp_results.append(cgm_auto.loc[(cgm_auto['Date'] == x) & (cgm_auto['Sensor Glucose (mg/dL)'] < 70) & (cgm_auto['night/day'] == 'day')]['Index'].count())
    tmp_results.append(cgm_auto.loc[(cgm_auto['Date'] == x) & (cgm_auto['Sensor Glucose (mg/dL)'] < 54) & (cgm_auto['night/day'] == 'day')]['Index'].count())

    # whole day
    tmp_results.append(cgm_auto.loc[(cgm_auto['Date'] == x) & (cgm_auto['Sensor Glucose (mg/dL)'] > 180)]['Index'].count())
    tmp_results.append(cgm_auto.loc[(cgm_auto['Date'] == x) & (cgm_auto['Sensor Glucose (mg/dL)'] > 250)]['Index'].count())
    tmp_results.append(cgm_auto.loc[(cgm_auto['Date'] == x) & (cgm_auto['Sensor Glucose (mg/dL)'].between(70, 180))]['Index'].count())
    tmp_results.append(cgm_auto.loc[(cgm_auto['Date'] == x) & (cgm_auto['Sensor Glucose (mg/dL)'].between(70, 150))]['Index'].count())
    tmp_results.append(cgm_auto.loc[(cgm_auto['Date'] == x) & (cgm_auto['Sensor Glucose (mg/dL)'] < 70)]['Index'].count())
    tmp_results.append(cgm_auto.loc[(cgm_auto['Date'] == x) & (cgm_auto['Sensor Glucose (mg/dL)'] < 54)]['Index'].count())

    final_results = np.add(final_results, tmp_results)

# doing operations to get the metrics that we will add to the results data frame
final_results[:] = [float(x) for x in final_results]
divider = float(288*num_days)/100.00
final_results = np.divide(final_results, divider)
results.loc[len(results)] = final_results


# In[23]:


# refactoring the result data frame
index_dict = {0: 'Manual', 1: 'Auto'}
results = results.rename(mapper = index_dict)
results.index.name = 'Mode'


# In[24]:


results.to_csv('Yadav_results.csv')


# In[ ]:
