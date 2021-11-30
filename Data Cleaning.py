# %%
import pandas as pd
#raw signal export
rawdata=pd.read_csv('/home/danielliu/Insync/2468942@uams.edu/OneDrive Biz/Research/Signal/Signal_final.csv', na_values=['NaN', 'nan'])

# %%
data = rawdata

# %%

cols=data.columns
cols

# %%
#drop type (all provider), SerCID, provider name, service area (all ACH), numerator, denominator
data.drop(cols[[0, 2, 3, 5, -3, -2]], axis=1, inplace=True)

# %%
data.columns

# %%
# setting variable datatypes
data[data.columns[[0, 1, 2, 3, 4, 7]]].astype('category')
data[data.columns[[-3, -4]]]=data[data.columns[[-3,-4]]].apply(pd.to_datetime)
data['Value'].astype(float)
data.info()

# %%
#drop Anesthesia
data=data[data['Specialty'] != 'Anesthesiology']

# %%
#drop Audiology
data=data[data['Specialty'] != 'Audiology']
#drop physical therapy
data=data[data['Specialty'] != 'Physical Therapy']

# %%
# dropping residents due to their variability in schedules
data = data[data['ProviderType'] != 'RESIDENT']

# %%
data.isnull().sum()

# %%
#set NaN values to 0
data.fillna(value='0', inplace=True)
data.isnull().sum()

# %%
Physicians=data[data['ProviderType'] == 'PHYSICIAN']

# %%
Physicians.head()

# %%
emppivot=Physicians.pivot_table(values='Value', index=['EmpCID', 'ReportingPeriodEndDate'], columns='Metric')

# %%
emppivot

# %%
deptpivot=Physicians.pivot_table(values='Value', index=['Specialty', 'EmpCID', 'ReportingPeriodEndDate'], columns='Metric')

# %%
deptpivot.to_csv('DeptPivot-final.csv')

# %%
Physicians.to_csv('Physicians-final.csv')
emppivot.to_csv('PhysiciansPivot-final.csv')
data.to_csv('Data-final.csv')

# %%
import pandas as pd

data=pd.read_csv('Physicians-final.csv', index_col=0)

# %%
data.head()

# %%
data['ReportingPeriodEndDate']=pd.to_datetime(data['ReportingPeriodEndDate'])

# %%
data_before=data[data['ReportingPeriodEndDate'] < pd.to_datetime('1/1/2021')]

# %%
data_after=data[data['ReportingPeriodEndDate'] > pd.to_datetime('1/1/2021')]

# %%
deptpivot_before = data_before.pivot_table(values='Value', index=['Specialty', 'ReportingPeriodEndDate','EmpCID'], columns='Metric')

# %%
deptpivot_after = data_after.pivot_table(values='Value', index=['Specialty', 'ReportingPeriodEndDate','EmpCID'], columns='Metric')

# %%
deptpivot_before.to_csv('deptpivot_before-final.csv')
deptpivot_after.to_csv('deptpicot_after-final.csv')

# %%



