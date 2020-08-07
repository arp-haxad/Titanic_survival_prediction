# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()


# %%
raw_data = pd.read_csv('train.csv')


# %%
raw_data.head()


# %%
pd.options.display.max_columns  = None
pd.options.display.max_rows = None


# %%
raw_data.head()


# %%
data_c = raw_data.copy()


# %%
data_c.head()


# %%
data_c.columns.values


# %%
temp = ['PassengerId','Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked','Survived']
data_c = data_c[temp]
data_c.head()


# %%
pclass = data_c['Pclass']


# %%
pclass.head()


# %%
df =pd.get_dummies(pclass,drop_first= True)
df


# %%
class_2 = df.iloc[:,0]
class_3 = df.iloc[:,1]


# %%
data_c.columns.values


# %%
new_df = pd.concat([data_c,class_2,class_3],axis = 1)
new_df.head()


# %%
new_df.head()


# %%
new_df.columns.values


# %%
save_checkpoint = new_df.copy()


# %%
temp2 = ['Name', 2, 3,'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin',
       'Embarked', 'Survived']
df.shape


# %%
new_df.head()


# %%
new_df = new_df[temp2]


# %%
new_df = new_df.drop('Name',axis = 1)
new_df.head()


# %%
new_df['Sex'] = new_df['Sex'].map({'male':0,'female':1})
new_df.head()


# %%
save_checkpoint.head()


# %%
sac = save_checkpoint.copy()


# %%
sac['Sex'] = sac['Sex'].map({'male':0,'female':1})
sac.head()


# %%
saac_1 = save_checkpoint.copy()


# %%
saac_1['Sex_New'] = saac_1['Sex'].map({'male':0,'female':1})
saac_2 = saac_1.copy()
saac_1.head()


# %%
saac_1 = saac_1.drop(['Sex','Name'],axis  = 1)
saac_1.head()


# %%
saac_1['Parch'].unique()


# %%
saac_1['Ticket'].count()


# %%
saac_1.shape


# %%
saac_1 = saac_1.drop('Ticket',axis = 1)
saac_1.head()


# %%
embarked = saac_1['Embarked']
embarked_dummy = pd.get_dummies(embarked,drop_first=True)
saac_f = pd.concat([saac_1,embarked_dummy],axis = 1)
saac_f.head()


# %%
saac_f = saac_f.drop('Embarked',axis = 1)


# %%
saac_f.head()


# %%
saac_f['Cabin'].count()


# %%
df_with_cabin = saac_f.copy()


# %%
saac_f['Cabin'].unique()


# %%
saac_f  = saac_f.drop('Cabin',axis = 1)
saac_f.head()


# %%

saac_f2 = saac_f.copy()


# %%
saac_f.columns.values


# %%
tmp = [ 2, 3,'Age', 'SibSp', 'Parch', 'Fare','Sex_New', 'Q',
       'S','Survived']
saac_f = saac_f[tmp]


# %%
saac_f.head()


# %%
saac_f.to_csv('non_standardised.csv',index = False)


# %%
df_standardisation1 = saac_f.iloc[:,2]
df_standardisation2 = saac_f.iloc[:,5]
new_df_combined = pd.concat([df_standardisation1,df_standardisation2],axis = 1)
new_df_combined.head()


# %%
from sklearn.preprocessing import StandardScaler


# %%
scaler = StandardScaler()


# %%
scaler.fit(new_df_combined)
# stnd_df = scaler.transform(new_df_combined)


# %%
stnd_df = scaler.transform(new_df_combined)


# %%
stnd_df.shape


# %%
data = stnd_df.shape[0]
# dfage = pd.DataFrame(data,columns='Age')
# dfage.head()
data


# %%
stnd_df


# %%
data_age_fare = pd.DataFrame({'Age_new':stnd_df[:,0],'Fare_new':stnd_df[:,1]})
data_age_fare.head()


# %%
saac_f = pd.concat([saac_f,data_age_fare],axis = 1)
saac_f.head()


# %%
saac_f = saac_f.drop(['Age','Fare'],axis = 1)
saac_f.head()


# %%
saac_f.columns.values


# %%
arr_t =[ 2, 3, 'SibSp', 'Parch', 'Sex_New', 'Q', 'S','Age_new','Fare_new','Survived']
final = saac_f[arr_t]
final.head()


# %%
final['Age_new'].unique()


# %%
final['Age_new'].mode()


# %%
final_checkpoint = final.copy()


# %%
final['Age_new'] = final['Age_new'].map({'nan':-0.392601})
final.head()


# %%
final_checkpoint_2 = final_checkpoint.copy()


# %%
final = final_checkpoint.copy()


# %%
final.head()


# %%
final['Age_new'] = final['Age_new'].fillna(final['Age_new'].mean())
final['Fare_new'] = final['Fare_new'].fillna(final['Fare_new'].mean())


# %%
final_processed = final.copy()


# %%
final_processed.to_csv('processed.csv',index = False)


# %%
inputs = final.iloc[:,0:9]
inputs.head()


# %%
targets = final['Survived']
targets.sum()/891


# %%
final.shape

# %% [markdown]
# ## solving  with balancing and shuffling 

# %%
target_sum = int(np.sum(targets))
target_sum


# %%
list_to_emp = []
counter_0 = 0
for i in range(targets.shape[0]):
    if targets[i]==0:
        counter_0= counter_0+1
        if counter_0>target_sum:
           list_to_emp.append(i)


# %%
# inputs_balanced = inputs.drop(inputs,list_to_emp,axis = 0)
# targets_balanced 
inputs_1 = inputs.copy()
targets_1 = targets.copy()
inputs_balanced = inputs.drop(index = list_to_emp,axis = 0)
target_balanced = targets.drop(index = list_to_emp,axis = 0)


# %%
target_balanced.sum()/target_balanced.shape[0]

# %% [markdown]
# ## shuffling

# %%
# index = np.arange(targets.shape[0])
# # index
# np.random.shuffle(index)

# shuffled_inputs = inputs[index]
# shuffle_targets  = targets[index]

# %% [markdown]
# ## modelling

# %%
from sklearn.linear_model import LogisticRegression


# %%
model = LogisticRegression()


# %%
model.fit(inputs_balanced,target_balanced)


# %%
model.score(inputs_balanced,target_balanced)


# %%
model.coef_


# %%
model.intercept_

# %% [markdown]
# ## summary table

# %%
arr = inputs_balanced.columns.values


# %%
arr


# %%
summary_table = pd.DataFrame(columns = ['Features'],data = arr)
summary_table


# %%
summary_table['coefficient'] = np.transpose(model.coef_)
summary_table


# %%
summary_table.index = summary_table.index+1
summary_table.loc[0] = ['Intercept',model.intercept_[0]]
summary_table  = summary_table.sort_index()
summary_table


# %%
## using the  numpy exp to measure the different coeficient
summary_table['odds_ratio'] = np.exp(summary_table.coefficient)
summary_table

# %% [markdown]
# ## table acc to significance of the features

# %%
summary_table.sort_values('odds_ratio',ascending = False)
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# import numpy as np 
# import pandas as pd 
# from sklearn.preprocessing import StandardScaler


# %%

data = pd.read_csv('test.csv')


# %%
pclass = data['Pclass']


# %%
df =pd.get_dummies(pclass,drop_first= True)


# %%
class_2 = df.iloc[:,0]
class_3 = df.iloc[:,1]


# %%
new_df = pd.concat([data,class_2,class_3],axis = 1)


# %%
new_df = new_df.drop('Name',axis = 1)


# %%
new_df['Sex_New'] = new_df['Sex'].map({'male':0,'female':1})


# %%
new_df.head()


# %%
new_df= new_df.drop(['Sex'],axis  = 1)


# %%
embarked = new_df['Embarked']


# %%
embarked_dummy = pd.get_dummies(embarked,drop_first=True)
new_df = pd.concat([new_df,embarked_dummy],axis = 1)
new_df.head()


# %%
new_df = new_df.drop('Embarked',axis = 1)


# %%
new_df = new_df.drop('Cabin',axis = 1)


# %%
new_df = new_df.drop('PassengerId',axis = 1)
new_df.head()


# %%
new_df = new_df.drop('Pclass',axis = 1)


# %%
new_df.head()


# %%
df_standardisation1 = new_df.iloc[:,0]
df_standardisation2 = new_df.iloc[:,4]
new_df_combined = pd.concat([df_standardisation1,df_standardisation2],axis = 1)


# %%
scaler = StandardScaler()


# %%

scaler.fit(new_df_combined)


# %%
stnd_df = scaler.transform(new_df_combined)


# %%
data_age_fare = pd.DataFrame({'Age_new':stnd_df[:,0],'Fare_new':stnd_df[:,1]})


# %%
new_df= pd.concat([new_df,data_age_fare],axis = 1)
final = new_df.copy()


# %%
final['Age_new'] = final['Age_new'].fillna(final['Age_new'].mean())
final['Fare_new'] = final['Fare_new'].fillna(final['Fare_new'].mean())


# %%
final.head()


# %%
# final = final.drop(['Age','Fare'],axis = 1)
final = final.drop(['Age_new','Fare_new'],axis = 1)


# %%
final.head()


# %%
final = final.drop('Ticket',axis = 1)


# %%
final = pd.concat([final,data_age_fare],axis = 1)
final


# %%
final.columns.values


# %%
final['Age_new'] = final['Age_new'].fillna(final['Age_new'].mean())
final['Fare_new'] = final['Fare_new'].fillna(final['Fare_new'].mean())


# %%



# %%
data_age_fare


# %%
data_age_fare['Age_new'] = data_age_fare['Age_new'].fillna(data_age_fare['Age_new'].mean())
data_age_fare['Fare_new'] = data_age_fare['Fare_new'].fillna(data_age_fare['Fare_new'].mean())


# %%
data_age_fare


# %%
new = new_df.drop(['Age','Ticket','Fare','Age_new','Fare_new'],axis = 1)
new.head()


# %%
fff = pd.concat([new,data_age_fare],axis = 1)
fff.head()


# %%
fff.columns.values


# %%
t =[2, 3,'SibSp', 'Parch','Sex_New', 'Q', 'S', 'Age_new', 'Fare_new']
fff= fff[t]


# %%
fff.head()


# %%
fff.to_csv('test_modifies.csv',index = False)


# %%





# %% [markdown]
# ## testing the model

# %%
test_raw_data = pd.read_csv('test_modifies.csv')
test_raw_data


# %%
predict_outcome = model.predict_proba(test_raw_data)


# %%
predict_outcome


# %%
final_prediction = predict_outcome[:,1]


# %%
final_prediction


# %%
survival_output = np.where(final_prediction>0.5,1,0)
survival_output


# %%
df_output = pd.DataFrame({'Survived':survival_output})
df_output.head()


# %%
df_oo = pd.read_csv('test.csv')
df_oo.head()


# %%
df_id = df_oo.iloc[:,0]


# %%
final_csv  =  pd.concat([df_id,df_output],axis = 1)
final_csv.head()


# %%
prediction_output =  final_csv.to_csv('output_2.csv',index = False)


# %%
ooo = read.csv


