# %%
#Requires pandas, scipy, matplotlib, and seaborn)
import pandas as pd
import matplotlib.pyplot as plt

#cleaned data
physicians=pd.read_csv('Physicians-final.csv', index_col=0)

# %%
#drop those without a specialty or those that are not as outpatient focused
skip_specialty=['Emergency Medicine', 'Burn Surgery', 'Cardiothoracic Surgery', 
                'Pediatric Critical Care Medicine','Radiology', 'Intensive Care', '0']
for i in skip_specialty:
    physicians=physicians[physicians['Specialty']!= i]

# %%
#build a filter for metrics of interest
hyp_less = ['Progress Note Length',
            'Time in Notes per Appointment', 
            'Time in Notes per Day']
hyp_unequal = ['Note Composition Method by Author - Manual',
               'Note Composition Method by Author - Copy/Paste',
               'Note Composition Method by Author - SmartTool',
               'Note Composition Method by Author - NoteWriter', 
               'Note Contribution Source - Text Written by Provider']

# %% [markdown]
# ## Functions

# %% [markdown]
# Cleaned up unused functions, so fewer than before

# %%
def emp_in_spec(df, specialty):
    df=df[df['Specialty']==specialty]
    return df

# %%
def phys_to_metric(a, filter, date='1/1/2021'):
    #set datatype
    a['ReportingPeriodEndDate']=pd.to_datetime(a['ReportingPeriodEndDate'])
    a['EmpCID'].apply(str)
    #apply filter
    fil = a[a['Metric'].isin(filter)]
    #creates before and after dataframes
    before=fil[fil['ReportingPeriodEndDate'] < pd.to_datetime(date)]
    after=fil[fil['ReportingPeriodEndDate'] > pd.to_datetime(date)]
    #pivots to create specialty/metric by report date table
    m_before=before.pivot_table(index=['Specialty', 'Metric', 'EmpCID','ReportingPeriodEndDate'], values='Value')
    m_after=after.pivot_table(index=['Specialty', 'Metric', 'EmpCID','ReportingPeriodEndDate'], values='Value')
    return (m_before, m_after)

# %%
def reformat(df, level='Specialty'):
    df[0].to_csv('temp1.csv')
    df[1].to_csv('temp2.csv')
    #there's probably a better way to do this, but reading a csv gives me the formatting I want
    a = pd.read_csv('temp1.csv')
    b = pd.read_csv('temp2.csv')
    if level == 'Specialty':
        a["S/M"]= a[level] + '/' + a['Metric']
        b["S/M"]= b[level] + '/' + b['Metric']
        a=a[['S/M', 'Value']]
        b=b[['S/M', 'Value']]
    if level == 'Employee':
        lvl= 'EmpCID'
        a["E/M"]= a[lvl].apply(str) + '/' + a['Metric']
        b["E/M"]= b[lvl].apply(str) + '/' + b['Metric']    
        a=a[['E/M', 'Value']]
        b=b[['E/M', 'Value']]
    return a, b

# %%
def rank_sum_testing(df, hyp, level='Specialty'):
    df=reformat(df, level)
    
    #level can be either 'Specialty' or 'EmpCID'
    if level == 'Specialty':
        level = 'S/M'
    if level == 'Employee':
        level = 'E/M'
    #hyp takes the same inputs as ranksums: ‘two-sided’, ‘less’: one-sided, ‘greater’: one-sided
    import numpy as np
    from scipy.stats import ranksums
    ts=[]
    before=df[0]
    after=df[1]
    
    for i in set(before[level]):
        b=list(before[before[level] == i]['Value'])
        a=list(after[after[level]==i]['Value'])
        diff=np.mean(a)-np.mean(b)
        p=ranksums(b, a, alternative = hyp)
        ts.append((i, np.mean(b), np.mean(a), diff, diff/np.mean(b), p[0],p[1]))
    
    ts=pd.DataFrame(ts)
    
    return ts

# %%
def re_index(a, level='Specialty'):
    if level == 'Specialty':
        lvl = 'Specialty/Metric'
        a.columns=[lvl, 'Before Avg', 'After Avg', 'After minus Before', '% change', 'RS-stat', 'p-value']
        a[lvl]=a[lvl].str.split('/').apply(tuple)
        index=pd.MultiIndex.from_tuples(a[a.columns[0]], names=['Specialty', 'Metric'])
        
    if level == 'Employee':
        lvl ='Employee/Metric'
        a.columns=[lvl, 'Before Avg', 'After Avg', 'After minus Before', '% change', 'RS-stat', 'p-value']
        a[lvl]=a[lvl].str.split('/').apply(tuple)
        index=pd.MultiIndex.from_tuples(a[a.columns[0]], names=['Employee', 'Metric'])
        
    a.index=index
    a.drop(lvl, axis=1, inplace=True)
    return a.sort_values(level)

# %%
def ann_heatmap_sb_tot(df1, df2, col, label, level='Specialty'):
    import numpy as np
    import seaborn as sb
    import matplotlib.pyplot as plt
    
    sb.set_theme(palette='flare')
    
    metric1=df1[col].unstack(level)
    metric2=df2[col].unstack(level)
    lab1=df1[label].unstack(level)
    lab2=df2[label].unstack(level)
    
    metric=pd.concat([metric1, metric2])
    label=pd.concat([lab1, lab2]) *100
    
    f, ax = plt.subplots(figsize=(30, 8)) 
    hm=sb.heatmap(metric, annot=label, linewidths=.25, ax=ax, vmin=0, vmax=0.050000001)  
    fig=hm.get_figure()
    return fig


# %%
def ann_heatmap_sb_nc(df1, df2, col, label, level='Specialty'):
    import numpy as np
    import seaborn as sb
    import matplotlib.pyplot as plt
    
    sb.set_theme(palette='flare')
    
    metric1=df1[col].unstack(level)
    metric2=df2[col].unstack(level)
    lab1=df1[label].unstack(level)
    lab2=df2[label].unstack(level)
    
    metric=pd.concat([metric1, metric2])
    label=pd.concat([lab1, lab2]) *100
    
    f, ax = plt.subplots(figsize=(30, 8)) 
    hm=sb.heatmap(metric, annot=label, linewidths=.25, ax=ax, vmin=0, vmax=0.100000001)  
    fig=hm.get_figure()
    return fig


# %%
def ann_heatmap_gr(df1, df2, col, label, level='Specialty'):
    import numpy as np
    import seaborn as sb
    import matplotlib.pyplot as plt
    
    sb.set_theme(palette='flare')
    
    metric1=df1[col].unstack(level)
    lab1=df1[label].unstack(level)
    
    metric=pd.concat([metric1])
    label=pd.concat([lab1]) *100
    
    f, ax = plt.subplots(figsize=(30, 8)) 
    hm=sb.heatmap(metric, annot=label, linewidths=.25, ax=ax, vmin=0, vmax=0.050000001)  
    fig=hm.get_figure()
    return fig

# %%
def one_step_sb_tot(df, Specialty, level):
    df=emp_in_spec(physicians, Specialty)
    df1=phys_to_metric(df, hyp_less)
    ts1=rank_sum_testing(df1, 'greater', level)
    ts1=re_index(ts1, level)
    
    df2=phys_to_metric(df, hyp_unequal)
    ts2=rank_sum_testing(df2, 'two-sided', level)
    ts2=re_index(ts2, level)
    
    fig=ann_heatmap_sb_tot(ts1, ts2, 'p-value', '% change', level)
    return ts1, ts2, fig

# %%
def one_step_gr(df, Specialty, level):
    df=emp_in_spec(physicians, Specialty)
    df1=phys_to_metric(df, hyp_less)
    ts1=rank_sum_testing(df1, 'less', level)
    ts1=re_index(ts1, level)
    
    df2=phys_to_metric(df, hyp_unequal)
    ts2=rank_sum_testing(df2, 'two-sided', level)
    ts2=re_index(ts2, level)
    
    fig=ann_heatmap_sb_tot(ts1, ts2, 'p-value', '% change', level)
    return ts1, ts2, fig

# %%
def one_step_nc(df, Specialty, level):
    df=emp_in_spec(physicians, Specialty)
    df1=phys_to_metric(df, hyp_less)
    ts1=rank_sum_testing(df1, 'two-sided', level)
    ts1=re_index(ts1, level)
    
    df2=phys_to_metric(df, hyp_unequal)
    ts2=rank_sum_testing(df2, 'two-sided', level)
    ts2=re_index(ts2, level)
    
    fig=ann_heatmap_sb_nc(ts1, ts2, 'p-value', '% change', level)
    return ts1, ts2, fig

# %% [markdown]
# ## The below will add employee level results (both csv and images) per specialty to a folder called 'Spec Results'.  If you have any specialties with only one employee, you may need to skip it.  In the code below, pediatric heme onc (specs[20]) only has one employee, so it's skipped using the below lists.

# %%
specs=sorted(list(set(physicians['Specialty'])))

# %% [markdown]
# ## uses 'two-sided' as null hypothesis for hyp_less variables (including progress note length) then performs statistical testing

# %%
nc_results=[]
for i in specs[:20]:
    a=one_step_nc(physicians, i, level='Employee')
    nc_results.append([i,a])
for i in specs[21:]:
    a=one_step_nc(physicians, i, level='Employee')
    nc_results.append([i,a])

# %% [markdown]
# Here, identifies p-values that are significantly shorter, significantly greater, and those with no change (then samples 40 from no_change)

# %%
sig=[]
for (x,y) in nc_results:
    for row in y[0].index:
        if row[1] == 'Progress Note Length' and y[0].loc[row]['% change'] < 0 and y[0].loc[row]['p-value'] < 0.10:
            sig.append(row[0])

# %%
sig_gr=[]
for (x,y) in nc_results:
    for row in y[0].index:
        if row[1] == 'Progress Note Length' and  y[0].loc[row]['% change'] > 0 and y[0].loc[row]['p-value'] < 0.10:
            sig_gr.append(row[0])

# %%
sig_nc=[]
for (x,y) in nc_results:
    for row in y[0].index:
        if row[1] == 'Progress Note Length' and y[0].loc[row]['p-value'] > 0.10: # for two-sided
            sig_nc.append(row[0])

# %% [markdown]
# Random sample from physicians with no significant changes in note length

# %%
import random
random.seed(10)
ran = random.sample(sig_nc, k=50)

# %%
ran

# %% [markdown]
# ## Generating Aggregate Statistics

# %%
len(sig_nc)

# %%
emps=list(set(list(physicians[physicians['Specialty'] != specs[20]]['EmpCID']))) #this speciality only has pre-data and no post-data

# %%
len(emps)

# %%
len(sig)

# %%
len(sig_gr)

# %%
ncs=[i for i in emps if str(i) not in (sig + sig_gr)]

# %%
len(ncs)

# %%
shorter_stats=[]
nc_stats=[]
longer_stats=[]

for (x,y) in nc_results:
    try:
        for row in y[0].index:
            if str(row[0]) in (sig):
                shorter_stats.append([y[0].loc[[row]]])
            elif str(row[0]) in sig_gr:
                longer_stats.append([y[0].loc[[row]]])
            elif str(row[0]) in str(ncs):
                nc_stats.append([y[0].loc[[row]]])
        for row in y[1].index:
            if str(row[0]) in (sig):
                shorter_stats.append([y[1].loc[[row]]])
            elif str(row[0]) in sig_gr:
                longer_stats.append([y[1].loc[[row]]])
            elif str(row[0]) in str(ncs):
                nc_stats.append([y[1].loc[[row]]])
    except KeyError:
        continue


# %%
from scipy.stats import ttest_ind
s_s = []
nc_s = []
l_s = []

sig = [int(i) for i in sig]
sig_gr = [int(i) for i in sig_gr]
ncs = [int(i) for i in ncs]

gr = [sig, sig_gr, ncs]


mets = []

for m in set(list(physicians['Metric'])):
    t = physicians[physicians['Metric'] == m]
    ts = []
    for i in gr:
        d = t[t['EmpCID'].isin(i)]
        d_b = d[pd.to_datetime(d.ReportingPeriodEndDate) < pd.to_datetime('1/1/2021')]
        d_a = d[pd.to_datetime(d.ReportingPeriodEndDate) > pd.to_datetime('1/1/2021')]
        test = ttest_ind(d_b['Value'], d_a['Value'], nan_policy='omit', alternative='two-sided')
        if i == sig:
            v = 'shorter'
        elif i == sig_gr:
            v = 'longer'
        elif i == ncs:
            v = 'no_change'
        ts.append([v, d_b['Value'].mean(), d_a['Value'].mean(), test[0], test[1]])
    mets.append([m, ts])


# %%
# F-test for difference between s, l, n groups both before and after 1/1/2021
s_s = []
nc_s = []
l_s = []

sig = [int(i) for i in sig]
sig_gr = [int(i) for i in sig_gr]
ncs = [int(i) for i in ncs]

gr = [sig, sig_gr, ncs]


mets = []

from scipy.stats import f_oneway

for m in set(list(physicians['Metric'])):
    t = physicians[physicians['Metric'] == m]
    
    s = t[t['EmpCID'].isin(gr[0])]
    l = t[t['EmpCID'].isin(gr[1])]
    n = t[t['EmpCID'].isin(gr[2])]
    
    s_b = s[pd.to_datetime(s.ReportingPeriodEndDate) < pd.to_datetime('1/1/2021')]
    l_b = l[pd.to_datetime(l.ReportingPeriodEndDate) < pd.to_datetime('1/1/2021')]
    n_b = n[pd.to_datetime(n.ReportingPeriodEndDate) < pd.to_datetime('1/1/2021')]
    
    
    s_a = s[pd.to_datetime(s.ReportingPeriodEndDate) > pd.to_datetime('1/1/2021')]
    l_a = l[pd.to_datetime(l.ReportingPeriodEndDate) > pd.to_datetime('1/1/2021')]
    n_a = n[pd.to_datetime(n.ReportingPeriodEndDate) > pd.to_datetime('1/1/2021')]
    
    F_a = f_oneway(s_a['Value'], l_a['Value'], n_a['Value'], axis=0)
    F_b = f_oneway(s_b['Value'], l_b['Value'], n_b['Value'], axis=0)
    
    mets.append([m, 
                 s_b['Value'].mean(), l_b['Value'].mean(), n_a['Value'].mean(), F_b[0], F_b[1],
                 s_a['Value'].mean(), l_a['Value'].mean(), n_a['Value'].mean(), F_a[0], F_a[1]])



# %%
mets=pd.DataFrame(mets)
mets.to_csv('t-tests.csv')
#mets.to_csv('anova.csv')

# %%
sig = [int(i) for i in sig]

# %%
import numpy as np

# %%
s = t[t['EmpCID'].isin(sig)]

# %%
s_b=s[pd.to_datetime(s.ReportingPeriodEndDate) < pd.to_datetime('1/1/2021')]
s_a=s[pd.to_datetime(s.ReportingPeriodEndDate) > pd.to_datetime('1/1/2021')]

# %%
from scipy.stats import ttest_ind

test= ttest_ind(s_b['Value'], s_a['Value'], nan_policy='omit')


# %%
def flatten(lists):
    flat_list = [item for sublist in lists for item in sublist]
    flats = [row.values.tolist() for row in flat_list]
    indexes = [i.index for i in flat_list]
    indexes = [item for sublist in indexes for item in sublist]
    last = [item for sublist in flats for item in sublist]
    return pd.DataFrame(last, index=pd.MultiIndex.from_tuples(indexes, names=('Employee', 'Metric')), columns=flat_list[0].columns)

# %%
lstat=pd.DataFrame(flatten(longer_stats))
#lstat.to_csv('longer_stats.csv')

# %%
sstat=pd.DataFrame(flatten(shorter_stats))
#sstat.to_csv('shorter_stats.csv')

# %%
nstat=pd.DataFrame(flatten(nc_stats))
#nstat.to_csv('nc_stats.csv')

# %% [markdown]
# ## Aggregate Statistical Testing

# %%
set(list(nstat.index.get_level_values('Metric')))

# %%
from scipy.stats import ttest_rel
import numpy as np

def agg_ttest(df):
    comp=[]
    for i in set(list(df.index.get_level_values('Metric'))):
        t = df[df.index.get_level_values('Metric') == i]
        test=ttest_rel(t['Before Avg'], t['After Avg'], nan_policy='omit')
        b=t['Before Avg'].mean()
        a=t['After Avg'].mean()
        comp.append([i, b, a, a-b, (a-b)/b, test[0], test[1]])

    columns=['Measure', 'Before Avg', 'After Avg', 'Diff', '% change', 'T-test', 'p-value']

    return pd.DataFrame(comp, columns=columns).sort_values('Measure')

# %%
n_comp=agg_ttest(nstat)

# %%
n_comp.sort_values('Measure').to_csv('n_comp.csv')

# %%
agg_ttest(sstat).sort_values('Measure').to_csv('s_comp.csv')

# %%
agg_ttest(lstat).sort_values('Measure').to_csv('l_comp.csv')

# %% [markdown]
# ## Reidentifier for survery purposes

# %%
#use the full signal file with employee CID, name, and specialty
full = pd.read_csv('Signal_final.csv')

# %%
ids = full[['EmpCID', 'ProviderName', 'Specialty']]

# %%
ids.drop_duplicates(inplace=True)

# %%
name_lookup = dict(zip(ids['EmpCID'],ids['ProviderName']))

# %%
spec_lookup = dict(zip(ids['EmpCID'],ids['Specialty']))

# %%
def identifier(ids):
    names = [name_lookup[int(x)] for x in ids]
    spec = [spec_lookup[int(x)] for x in ids]
    return pd.DataFrame(zip(ids, names, spec))

# %%
identifier(sig).to_csv('shorter note.csv')

# %%
identifier(sig_gr).to_csv('longer note.csv')

# %% [markdown]
# since this is two-sided, there are physicians that may have significantly shorter notes when looking at a one-sided test, but fall into the non-significant category when looking at a two-sided since alpha remained at 0.05.  So will filter out those physicians from the random sample.
# 

# %%
filtered = [x for x in ran if x not in sig]

# %%
identifier(filtered).to_csv('same_notes.csv')

# %%



