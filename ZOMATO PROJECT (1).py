#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


# In[2]:


df= pd.read_csv(r'C:\Users\Administrator\zomato.csv')


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.dtypes


# In[6]:


df.drop(['url','address','phone','location','dish_liked','reviews_list','menu_item'],axis=1,inplace=True)


# In[7]:


df=df.rename(columns={"name":'Name','rate':'Ratings','votes':'Votes','rest_type':'Rest_Type','cuisines':'Cuisines','approx_cost(for two people)':'Cost','listed_in(type)':'Type','listed_in(city)':'City','online_order':'Takes online orders?','book_table':'Has table booking?'})


# In[8]:


df.sample(10)


# In[9]:


sum(df.duplicated())


# In[10]:


df=df.drop_duplicates()


# In[11]:


def name_clean(text):
    return re.sub(r"[^a-zA-Z0-9 ]", "", text)
df['Name'] = df['Name'].apply(lambda x: name_clean(x))


# In[12]:


df["Ratings"]=df["Ratings"].replace("NEW", np.nan)
df['Ratings']=df['Ratings'].replace('NaN',np.nan)
df['Ratings']=df['Ratings'].replace('-',np.nan)
df['Ratings']=df['Ratings'].replace('nan',np.nan)
def remove_5(value: str):
    if type(value)==str:
        value_new=value.split('/')[0]
        return value_new
    return value
df['Ratings']=df['Ratings'].apply(remove_5)
df['Ratings']=df['Ratings'].astype(float)
print(df['Ratings'].dtypes)


# In[13]:


def cost(value):
    value = str(value)
    if "," in value:
        value = float(value.replace(",",""))
        return value
    else:
        return float(value)
df['Cost'] = df['Cost'].apply(cost)
print(df['Cost'].head())


# In[14]:


print(df.isnull().sum())

print([features for features in df.columns if df[features].isnull().sum()>0])


# In[15]:


df['Ratings'].unique()


# In[16]:


def handelrate(value):
    if(value=="NEW" or value=="-"):
        return np.nan
    else:
        value=str(value).split("/")
        value=value[0]
        return float(value)
df['Ratings']=df['Ratings'].apply(handelrate)
df['Ratings'].head()


# In[17]:


df["Ratings"].fillna(df["Ratings"].mean(),inplace=True)
df["Ratings"].isnull().sum()


# In[18]:


df.info()


# In[19]:


df.dropna(inplace=True)
df.head(15)


# In[20]:


df.rename(columns={"approx_cost(for two people)":"Cost2Plates","listed_in(type)":"Type"},inplace=True)
df.head(15)


# In[21]:


df.describe()


# In[22]:


df.columns.value_counts()


# In[23]:


df["Cost"].unique()


# In[24]:


df["City"].unique()


# In[25]:


def handlecomma(value):
    value=str(value)
    if "," in value:
        value=value.replace(",","")
        return float(value)
    else:
        return float(value)
df["Cost2"]=df["Cost"].apply(handlecomma)
df["Cost2"].unique()


# In[26]:


df.head(15)


# In[27]:


df["Rest_Type"].value_counts()


# In[28]:


rest_type=df["Rest_Type"].value_counts(ascending=False)
rest_type.head(16)


# In[29]:


rest_types_lessthan1000=rest_type[rest_type<1000]
rest_types_lessthan1000


# In[30]:


def handle_rest_type(value):
    if(value in rest_types_lessthan1000):
        return 'others'
    else: 
        return value 
df['Rest_Type']=df["Rest_Type"].apply(handle_rest_type)
df['Rest_Type'].value_counts()


# In[31]:


df.head(16)


# In[32]:


cuisines=df["Cuisines"].value_counts(ascending=False)
cuisines_lessthan100=cuisines[cuisines<100]
def handle_cuisines(value):
    if(value in cuisines_lessthan100):
        return "others"
    else:
        return value
df["Cuisines"]=df["Cuisines"].apply(handle_cuisines)
df["Cuisines"].value_counts()
df.head(16)


# In[33]:


df["Type"].value_counts()


# In[34]:


df["City"].value_counts()


# In[38]:


plt.figure(figsize=(20,10))
graph=sns.countplot(df["City"]);
plt.xticks(rotation=90);


# In[39]:


plt.figure(figsize=(10,10))
sns.boxplot(x="Takes online orders?",y="Ratings",data=df)


# In[40]:


plt.figure(figsize=(5,10))
sns.countplot(df["Has table booking?"])


# In[52]:


plt.figure(figsize=[15,8])
sns.countplot (x = df['Type'])
sns.countplot (x = df['Type']).set_xticklabels(sns.countplot(x = df['Type']).get_xticklabels(),rotation=50,ha='right')
plt.title('Type of Service')


# In[74]:


plt.figure(figsize=[15,10])
chains=df['Name'].value_counts()[:15]
sns.barplot(x=chains,y=chains.index,palette='Set1')
plt.title('Famous Resturant in Bangalore',size = 35 ,pad = 35)
plt.xlabel('outlets',size=35)


# In[66]:


plt.figure(figsize=(15,8))
plt.title('Online Delivery Distribution')
plt.pie(df['Takes online orders?'].value_counts()/9551*100, labels=df['Takes online orders?'].value_counts().index, autopct='%1.1f%%', startangle=180);


# In[80]:


plt.figure(figsize=(12,6))
sns.scatterplot(x="Ratings", y="Rest_Type", hue='Cost2', data=df)

plt.xlabel("Ratings")
plt.ylabel("Rest_Type")
plt.title('Ratings vs Rest Type');


# In[90]:


plt.figure(figsize=(9,7))

sns.distplot(df['Ratings'],bins=20)


# In[92]:


plt.figure(figsize=(15,7))
rest=df['Rest_Type'].value_counts()[:20]
sns.barplot(rest,rest.index)
plt.title("Restaurant types")
plt.xlabel("count")


# In[ ]:




