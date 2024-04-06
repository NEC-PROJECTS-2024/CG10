#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd # Pandas (version : 1.1.5) 
import numpy as np # Numpy (version : 1.19.2)
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans # Scikit Learn (version : 0.23.2)
import seaborn as sns
plt.style.use('seaborn')
import matplotlib.pyplot as plt
import warnings
#get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[4]:


#loading data set
import pandas as pd
df=data =pd.read_csv('Mall_Customers.csv')
print(df)


# In[5]:


data


# In[7]:


data.head()


# In[6]:


data.tail()


# In[8]:


len(data)


# In[9]:


data.shape


# In[10]:


data.columns


# In[11]:


data.dtypes


# In[12]:


data.info()


# In[13]:


data.describe()


# In[14]:


data.isnull()


# In[15]:


data.isnull().sum()


# In[16]:


data.columns


# In[20]:


X = data[['Annual Income (k$)','Spending Score (1-100)' ]]


# In[21]:


X


# In[17]:


data = data.drop('CustomerID', axis=1)
data.head()


# In[4]:


import pandas as pd # Pandas (version : 1.1.5) 
import numpy as np # Numpy (version : 1.19.2)
import matplotlib.pyplot as plt
labels=data['Gender'].unique()
values=data['Gender'].value_counts(ascending=True)

fig, (ax0,ax1) = plt.subplots(ncols=2,figsize=(15,8))
bar = ax0.bar(x=labels, height=values, width=0.4, align='center', color=['#42a7f5','#d400ad'])
ax0.set(title='Count difference in Gender Distribution',xlabel='Gender', ylabel='No. of Customers')
ax0.set_ylim(0,130)
ax0.axhline(y=data['Gender'].value_counts()[0], color='#d400ad', linestyle='--', label=f'Female ({data.Gender.value_counts()[0]})')
ax0.axhline(y=data['Gender'].value_counts()[1], color='#42a7f5', linestyle='--', label=f'Male ({data.Gender.value_counts()[1]})')
ax0.legend()


ax1.pie(values,labels=labels,colors=['#42a7f5','#d400ad'],autopct='%1.1f%%')
ax1.set(title='Ratio of Gender Distribution')
fig.suptitle('Gender Distribution', fontsize=30);
plt.show()


# In[6]:


import seaborn as sns
fig, ax = plt.subplots(figsize=(5,8))
sns.set(font_scale=1.5)
ax = sns.boxplot(y=data["Age"], color="#f73434")
ax.axhline(y=data['Age'].max(), linestyle='--',color='#c90404', label=f'Max Age ({data.Age.max()})')
ax.axhline(y=data['Age'].describe()[6], linestyle='--',color='#f74343', label=f'75% Age ({data.Age.describe()[6]:.2f})')
ax.axhline(y=data['Age'].median(), linestyle='--',color='#eb50db', label=f'Median Age ({data.Age.median():.2f})')
ax.axhline(y=data['Age'].describe()[4], linestyle='--',color='#eb50db', label=f'25% Age ({data.Age.describe()[4]:.2f})')
ax.axhline(y=data['Age'].min(), linestyle='--',color='#046ebf', label=f'Min Age ({data.Age.min()})')
ax.legend(fontsize='xx-small', loc='upper right')
ax.set_ylabel('No. of Customers')

plt.title('Age Distribution', fontsize = 20)
plt.show()


# In[7]:


import seaborn as sns 
fig, ax = plt.subplots(figsize=(20,8))
sns.set(font_scale=1.5)
ax = sns.countplot(x=data['Age'], palette='spring')
ax.axhline(y=data['Age'].value_counts().max(), linestyle='--',color='#c90404', label=f'Max Age Count ({data.Age.value_counts().max()})')
ax.axhline(y=data['Age'].value_counts().mean(), linestyle='--',color='#eb50db', label=f'Average Age Count ({data.Age.value_counts().mean():.1f})')
ax.axhline(y=data['Age'].value_counts().min(), linestyle='--',color='#046ebf', label=f'Min Age Count ({data.Age.value_counts().min()})')
ax.legend(loc ='right')
ax.set_ylabel('No. of Customers')

plt.title('Age Distribution', fontsize = 20)
plt.show()


# In[13]:


maxi = data[data['Gender']=='Male'].Age.value_counts().max()
mean = data[data['Gender']=='Male'].Age.value_counts().mean()
mini = data[data['Gender']=='Male'].Age.value_counts().min()
fig, ax = plt.subplots(figsize=(20,8))
sns.set(font_scale=1.5)
ax = sns.countplot(x=data[data['Gender']=='Male'].Age, palette='spring')
ax.axhline(y=maxi, linestyle='--',color='#c90404', label=f'Max Age Count ({maxi})')
ax.axhline(y=mean, linestyle='--',color='#eb50db', label=f'Average Age Count ({mean:.1f})')
ax.axhline(y=mini, linestyle='--',color='#046ebf', label=f'Min Age Count ({mini})')
ax.set_ylabel('No. of Customers')

ax.legend(loc ='right')
plt.title('Age Count Distribution in Male Customers', fontsize = 20)
plt.show()


# In[8]:


maxi_female = data[data['Gender']=='Female'].Age.value_counts().max()
mean_female = data[data['Gender']=='Female'].Age.value_counts().mean()
mini_female = data[data['Gender']=='Female'].Age.value_counts().min()

fig, ax = plt.subplots(figsize=(20,8))
sns.set(font_scale=1.5)

ax = sns.countplot(x=data[data['Gender']=='Female'].Age, palette='spring')

ax.axhline(y=maxi_female, linestyle='--', color='#c90404', label=f'Max Age Count ({maxi_female})')
ax.axhline(y=mean_female, linestyle='--', color='#eb50db', label=f'Average Age Count ({mean_female:.1f})')
ax.axhline(y=mini_female, linestyle='--', color='#046ebf', label=f'Min Age Count ({mini_female})')

ax.set_ylabel('No. of Customers')
ax.legend(loc='right')

plt.title('Age Count Distribution in Female Customers', fontsize=20)
plt.show()


# In[141]:


data = data.rename(columns={'Annual Income (k$)':'Annual_Income','Spending Score (1-100)':'Spending_Score'})
data.head()


# In[142]:


# In[143]:


fig, ax = plt.subplots(figsize=(5,8))
sns.set(font_scale=1.5)
ax = sns.boxplot(y=data["Annual_Income"], color="#f73434")
ax.axhline(y=data["Annual_Income"].max(), linestyle='--',color='#c90404', label=f'Max Income ({data.Annual_Income.max()})')
ax.axhline(y=data["Annual_Income"].describe()[6], linestyle='--',color='#f74343', label=f'75% Income ({data.Annual_Income.describe()[6]:.2f})')
ax.axhline(y=data["Annual_Income"].median(), linestyle='--',color='#eb50db', label=f'Median Income ({data.Annual_Income.median():.2f})')
ax.axhline(y=data["Annual_Income"].describe()[4], linestyle='--',color='#eb50db', label=f'25% Income ({data.Annual_Income.describe()[4]:.2f})')
ax.axhline(y=data["Annual_Income"].min(), linestyle='--',color='#046ebf', label=f'Min Income ({data.Annual_Income.min()})')
ax.legend(fontsize='xx-small', loc='upper right')
ax.set_ylabel('No. of Customers')

plt.title('Annual Income (in Thousand USD)', fontsize = 20)
plt.show()


# In[12]:


fig, ax = plt.subplots(figsize=(15,7))
sns.set(font_scale=1.5)
ax = sns.scatterplot(y=data['Annual_Income'], x=data['Age'], color='#f73434', s=70,edgecolor='black', linewidth=0.3)
ax.set_ylabel('Annual Income (in Thousand USD)')
plt.title('Annual Income per Age', fontsize = 20)
plt.show()


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame

# Set the font scale and create a pairplot
sns.set(font_scale=1.5)
pairplot = sns.pairplot(data, height=4, aspect=1.5, palette='viridis')

# Set the title
pairplot.fig.suptitle('Pairplot of Annual Income and Age', y=1.02, fontsize=20)

# Display the plot
plt.show() 


# In[14]:


#Visualizing annual Income per Age by Gender on a scatterplot.

fig, ax = plt.subplots(figsize=(15,7))
sns.set(font_scale=1.5)
ax = sns.scatterplot(y=data['Annual_Income'], x=data['Age'], hue=data['Gender'], palette='seismic', s=70,edgecolor='black', linewidth=0.3)
ax.set_ylabel('Annual Income (in Thousand USD)')
ax.legend(loc ='upper right')

plt.title('Annual Income per Age by Gender', fontsize = 20)
plt.show()


# In[144]:


fig, ax = plt.subplots(figsize=(5,8))
sns.set(font_scale=1.5)
ax = sns.boxplot(y=data['Spending_Score'], color="#f73434")
ax.axhline(y=data['Spending_Score'].max(), linestyle='--',color='#c90404', label=f'Max Spending ({data["Spending_Score"].max()})')
ax.axhline(y=data['Spending_Score'].describe()[6], linestyle='--',color='#f74343', label=f'75% Spending ({data["Spending_Score"].describe()[6]:.2f})')
ax.axhline(y=data['Spending_Score'].median(), linestyle='--',color='#eb50db', label=f'Median Spending ({data["Spending_Score"].median():.2f})')
ax.axhline(y=data['Spending_Score'].describe()[4], linestyle='--',color='#eb50db', label=f'25% Spending ({data["Spending_Score"].describe()[4]:.2f})')
ax.axhline(y=data['Spending_Score'].min(), linestyle='--',color='#046ebf', label=f'Min Spending ({data["Spending_Score"].min()})')
ax.legend(fontsize='xx-small', loc='upper right')
ax.set_ylabel('Spending Score (1-100)')

plt.title('Spending Score (1-100)', fontsize = 20)
plt.show()


# In[20]:


fig, ax = plt.subplots(figsize=(15,7))
sns.set(font_scale=1.5)
ax = sns.scatterplot(y=data['Spending_Score'], x=data['Age'], s=70, color='#f73434', edgecolor='black', linewidth=0.3)
ax.set_ylabel('Spending Scores')

plt.title('Spending Scores per Age', fontsize = 20)
plt.show()


# In[21]:


fig, ax = plt.subplots(figsize=(10,8))
sns.set(font_scale=1.5)
ax = sns.boxplot(x=data['Gender'], y=data["Spending_Score"], hue=data['Gender'], palette='seismic')
ax.set_ylabel('Spending Score')

plt.title('Spending Score Distribution by Gender', fontsize = 20)
plt.show()


# In[22]:


fig, ax = plt.subplots(figsize=(15,7))
sns.set(font_scale=1.5)
ax = sns.scatterplot(y=data['Spending_Score'], x=data['Age'], hue=data['Gender'], palette='seismic', s=70,edgecolor='black', linewidth=0.3)
ax.set_ylabel('Spending Scores')
ax.legend(loc ='upper right')

plt.title('Spending Score per Age by Gender', fontsize = 20)
plt.show()


# In[23]:


fig, ax = plt.subplots(figsize=(15,7))
sns.set(font_scale=1.5)
ax = sns.scatterplot(y=data['Spending_Score'],x=data['Annual_Income'], s=70, color='#f73434', edgecolor='black', linewidth=0.3)
ax.set_ylabel('Spending Scores')
ax.set_xlabel('Annual Income (in Thousand USD)')
plt.title('Spending Score per Annual Income', fontsize = 20)
plt.show()


# In[26]:


clustering_data = data.iloc[:,[2,3]]
clustering_data.head()


# In[34]:


import numpy as np
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 30):
    kms = KMeans(i)
    kms.fit(clustering_data)
    wcss.append(kms.inertia_)

np.array(wcss)


# In[36]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import numpy as np

numeric_data = data.select_dtypes(include=[np.number])


categorical_data = data.select_dtypes(exclude=[np.number])

label_encoder = LabelEncoder()
categorical_data_encoded = categorical_data.apply(label_encoder.fit_transform)


preprocessed_data = pd.concat([numeric_data, categorical_data_encoded], axis=1)

wcss = []
for i in range(1, 30):
    kms = KMeans(n_clusters=i)
    kms.fit(preprocessed_data)
    wcss.append(kms.inertia_)

np.array(wcss)


# In[38]:


fig, ax = plt.subplots(figsize=(15,7))
ax = plt.plot(range(1,30),wcss, linewidth=2, color="red", marker ="8")
plt.axvline(x=5, ls='--')
plt.ylabel('WCSS')
plt.xlabel('No. of Clusters (k)')
plt.title('The Elbow Method', fontsize = 20)
plt.show()


# In[37]:


clusters = clustering_data.copy()
clusters['Cluster_Prediction'] = kms.fit_predict(clustering_data)
clusters.head()


# In[36]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Assuming 'data' is your DataFrame with mixed data types
numeric_data = data.select_dtypes(include=[np.number])
categorical_data = data.select_dtypes(exclude=[np.number])

# One-hot encode categorical variables
encoder = OneHotEncoder()
categorical_data_encoded = encoder.fit_transform(categorical_data)

# Concatenate numeric and encoded categorical data
clustering_data_processed = np.hstack((numeric_data, categorical_data_encoded.toarray()))

# Fit KMeans model
kms = KMeans(n_clusters=5, init='k-means++')
kms.fit(clustering_data_processed)


# In[54]:


kms.cluster_centers_


# In[42]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the Mall_Customers.csv dataset into a DataFrame
data = pd.read_csv('Mall_Customers.csv')

# Extract relevant features for clustering ('Annual Income (k$)' and 'Spending Score (1-100)')
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Determine the maximum number of clusters based on the number of samples
k = min(len(X), 5)  # Number of clusters (maximum of 5 clusters or the number of samples in X)
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Get cluster predictions
cluster_predictions = kmeans.labels_

# Plotting the clusters and centroids
fig, ax = plt.subplots(figsize=(15,7))

# Scatter plot for each cluster
for cluster_num in range(k):
    cluster_data = X[cluster_predictions == cluster_num]  # Filter data for each cluster
    
    plt.scatter(
        x=cluster_data['Annual Income (k$)'],
        y=cluster_data['Spending Score (1-100)'],
        s=70, edgecolor='black', linewidth=0.3,
        label=f'Cluster {cluster_num + 1}'
    )

# Scatter plot for centroids
plt.scatter(
    x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1],
    s=120, c='yellow', label='Centroids',
    edgecolor='black', linewidth=0.3
)

plt.legend(loc='right')
plt.xlim(0, 140)
plt.ylim(0, 100)
plt.xlabel('Annual Income (in Thousand USD)')
plt.ylabel('Spending Score')
plt.title('Clusters', fontsize=20)
plt.show()


# In[43]:


# Create a DataFrame to store cluster assignments along with original data
clusters = data.copy()  # Copy the original DataFrame
clusters['Cluster_Prediction'] = cluster_predictions  # Add cluster predictions column

# Check the first few rows of the clusters DataFrame
print(clusters.head())



# In[44]:


print(clusters.columns)


# In[155]:


fig, ax = plt.subplots(figsize=(15,7)) 

# Scatter plot for each cluster
for cluster_num in range(5):
    cluster_data = clusters[clusters['Cluster_Prediction'] == cluster_num]  # Filter data for each cluster
    
    plt.scatter(
        x=cluster_data['Annual Income (k$)'],
        y=cluster_data['Spending Score (1-100)'],
        s=70, edgecolor='black', linewidth=0.3,
        label=f'Cluster {cluster_num + 1}'
    )

# Scatter plot for centroids
plt.scatter(
    x=kms.cluster_centers_[:, 0], y=kms.cluster_centers_[:, 1],
    s=120, c='yellow', label='Centroids',
    edgecolor='black', linewidth=0.3
)

plt.legend(loc='right')
plt.xlim(0, 140)
plt.ylim(0, 100)
plt.xlabel('Annual Income (in Thousand USD)')
plt.ylabel('Spending Score')
plt.title('Clusters', fontsize=20)
plt.show()


# In[157]:


X = data[['Annual Income (k$)','Spending Score (1-100)' ]]


# In[158]:


k_means = KMeans(n_clusters=5,random_state=42)
y_means = k_means.fit_predict(X)


# In[159]:


y_means


# In[160]:


plt.scatter(X.iloc[y_means==0,0],X.iloc[y_means==0,1],s=100,c='red',label="Cluster 1")
plt.scatter(X.iloc[y_means==1,0],X.iloc[y_means==1,1],s=100,c='yellow',label="Cluster 2")
plt.scatter(X.iloc[y_means==2,0],X.iloc[y_means==2,1],s=100,c='green',label="Cluster 3")
plt.scatter(X.iloc[y_means==3,0],X.iloc[y_means==3,1],s=100,c='blue',label="Cluster 4")
plt.scatter(X.iloc[y_means==4,0],X.iloc[y_means==4,1],s=100,c='black',label="Cluster 5")
plt.scatter(k_means.cluster_centers_[:,0],k_means.cluster_centers_[:,1],s=100, c="magenta")
plt.title("Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()





# In[161]:


k_means.predict([[15,39]])


# In[162]:


import joblib


# In[163]:


joblib.dump(k_means,"customer_segmentation")


# In[164]:


model=joblib.load("customer_segmentation")


# In[165]:


model.predict([[15,39]])


# In[ ]:


import pandas as pd
from sklearn.cluster import KMeans

# Load your dataset
df = pd.read_csv('Mall_Customers.csv')

# Select relevant features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Perform KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['Zone'] = kmeans.fit_predict(X)

# Map cluster labels to zone names
zone_mapping = {
    0: 'Normal Customers',
    1: 'Spenders',
    2: 'Target Customers',
    3: 'Pinch Penny Customers',
    4: 'Balanced Customers'
}
df['Zone'] = df['Zone'].map(zone_mapping)

# Save the updated dataset
df.to_csv('Mall_Customers_with_Zone.csv', index=False)


# In[45]:


import pandas as pd

# Load your dataset
df = pd.read_csv('Mall_Customers_with_Zone.csv')

# Function to get valid integer input
def get_valid_integer_input(prompt):
    while True:
        try:
            value = int(input(prompt))
            return value
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

# Function to get valid float input
def get_valid_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Input age, gender, and annual income from the user
age = get_valid_integer_input("Enter age: ")
gender = input("Enter gender (Male/Female): ")
annual_income = get_valid_float_input("Enter annual income (in thousands): ")

# Filter the dataset based on the input age, gender, and annual income
filtered_data = df[(df['Age'] == age) & (df['Gender'] == gender) & (df['Annual Income (k$)'] == annual_income)]

# Check if there are any matching records
if not filtered_data.empty:
    # Get the actual spending score from the filtered data
    actual_spending_score = filtered_data['Spending Score (1-100)'].iloc[0]
    print('Actual Spending Score:', actual_spending_score)
else:
    print('No matching records found for the given input.')

# In[2]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv('Mall_Customers_with_Zone.csv')

# Preprocess gender column (convert to numerical)
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Convert gender to numerical

# Select features (age, gender, annual income) and target variable (spending score)
X = df[['Age', 'Gender', 'Annual Income (k$)']]
y = df['Spending Score (1-100)']

# Initialize and train the Random Forest regression model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X, y)

# Input age, gender, and annual income from the user
age = int(input("Enter age: "))
gender = input("Enter gender (Male/Female): ")
annual_income = float(input("Enter annual income (in thousands): "))

# Encode gender input
gender_encoded = label_encoder.transform([gender])[0]
# Predict spending score for the input data using the Random Forest model
predicted_score = rf_regressor.predict([[age, gender_encoded, annual_income]])
print('Predicted Spending Score:', predicted_score[0])
# Find actual spending score from the dataset based on input values
actual_spending_score = df[(df['Age'] == age) & (df['Gender'] == gender_encoded) & (df['Annual Income (k$)'] == annual_income)]['Spending Score (1-100)'].iloc[0]
print('Actual Spending Score:', actual_spending_score)


# In[3]:


import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv('Mall_Customers_with_Zone.csv')

# Preprocess gender column (convert to numerical)
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Convert gender to numerical

# Select features (age, gender, annual income) and target variable (spending score)
X = df[['Age', 'Gender', 'Annual Income (k$)']]
y = df['Spending Score (1-100)']

# Initialize and train the KNN regression model
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X, y)

# Input age, gender, and annual income from the user
age = int(input("Enter age: "))
gender = input("Enter gender (Male/Female): ")
annual_income = float(input("Enter annual income (in thousands): "))

# Encode gender input
gender_encoded = label_encoder.transform([gender])[0]

# Predict spending score for the input data using the KNN model
predicted_score = knn_regressor.predict([[age, gender_encoded, annual_income]])
print('Predicted Spending Score:', predicted_score[0])

# Find actual spending score from the dataset based on input values
actual_spending_score = df[(df['Age'] == age) & (df['Gender'] == gender_encoded) & (df['Annual Income (k$)'] == annual_income)]['Spending Score (1-100)'].iloc[0]
print('Actual Spending Score:', actual_spending_score)


# In[4]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv('Mall_Customers_with_Zone.csv')


label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Convert gender to numerical
X = df[['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)']]
y = df['Zone']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression classifier
log_reg_classifier = LogisticRegression(max_iter=1000, random_state=42)
log_reg_classifier.fit(X_train, y_train)

# Predict zones for the test data
y_pred = log_reg_classifier.predict(X_test)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Input age, gender, annual income, and spending score from the user
age = int(input("Enter age: "))
gender = input("Enter gender (Male/Female): ")
annual_income = float(input("Enter annual income (in thousands): "))
spending_score = float(input("Enter spending score: "))

# Encode gender input
gender_encoded = label_encoder.transform([gender])[0]

# Predict zone for the input data using the Logistic Regression model
predicted_zone = log_reg_classifier.predict([[age, gender_encoded, annual_income, spending_score]])
print('Predicted Zone:', predicted_zone[0])


# In[5]:


import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset
df = pd.read_csv('Mall_Customers_with_Zone.csv')

# Preprocess gender column (convert to numerical)
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Convert gender to numerical

# Select features (age, gender, annual income, spending score) and target variable (zone)
X = df[['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)']]
y = df['Zone']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Predict zones for the test data
y_pred = nb_classifier.predict(X_test)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Input age, gender, annual income, and spending score from the user
age = int(input("Enter age: "))
gender = input("Enter gender (Male/Female): ")
annual_income = float(input("Enter annual income (in thousands): "))
spending_score = float(input("Enter spending score: "))

# Encode gender input
gender_encoded = label_encoder.transform([gender])[0]

# Predict zone for the input data using the Naive Bayes model
predicted_zone = nb_classifier.predict([[age, gender_encoded, annual_income, spending_score]])
print('Predicted Zone:', predicted_zone[0])




