#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import plotly.express as px
import plotly.graph_objects as go


# In[6]:


house_data = pd.read_csv("C:\\Users\\asrit\\Downloads\\train.csv")
test_data = pd.read_csv("C:\\Users\\asrit\\Downloads\\test.csv")


# In[7]:


print(len(house_data))
print(len(house_data.columns))
house_data.head()


# In[8]:


house_data.info()


# In[9]:


house_data.describe()


# In[10]:


house_data.isnull().sum()


# In[11]:


columns_null = house_data.columns[house_data.isnull().any()]

for col in columns_null:
    print(col)


# In[12]:


missing_values = house_data.isnull().sum()
columns_with_missing_values = missing_values[missing_values > 0]
print("\nColumns with Missing Values:")
print(columns_with_missing_values)


# In[13]:


duplicates = house_data.duplicated().sum()

house_data.drop_duplicates(inplace=True)

duplicates_after = house_data.duplicated().sum()
print(duplicates_after)


# In[14]:


column_info = house_data.dtypes

for col_name, data_type in column_info.items():  # Use items() instead of iteritems()
    print(f"{col_name}: {data_type}\t", end='')


# In[15]:


house_data.columns


# In[16]:


fig = px.scatter_3d(house_data,x = 'LotArea', y='LotFrontage',z = 'SalePrice')
fig.show()


# In[17]:


#  list of categorical columns to exclude
categorical_columns = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

# Select numerical columns
numerical_columns = [col for col in house_data.columns if col not in categorical_columns]

# Create a DataFrame with only numerical features and the target variable
numerical_data = house_data[numerical_columns + ['SalePrice']]

# Calculate the correlation matrix
correlation_matrix = numerical_data.corr()

# Step 2: Generate a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix[['SalePrice']], annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation with SalePrice")
plt.show()


# In[18]:


# Split the data into training and testing sets
X = house_data[['TotalBsmtSF', 'BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath',"FullBath", "HalfBath"]]
y = house_data['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


# Select the columns of interest (features and target)
features = house_data[['TotalBsmtSF', 'BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']]
target = house_data[['SalePrice']]

# Create a new DataFrame with only the selected columns
data_subset = pd.concat([features, target], axis=1)  # Use square brackets and specify axis=1

# Calculate the correlation matrix
correlation_matrix = data_subset.corr()

# Create a heatmap for the correlation matrix
plt.figure(figsize=(8, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Greens', fmt=".2f")
plt.title("Correlation Heatmap: Features vs. Target")
plt.show()


# In[20]:


# Check for missing values in the training dataset
missing_train = house_data.isnull().sum()
print("missing train data",missing_train)

# Check for missing values in the testing dataset
missing_test = test_data.isnull().sum()
print("missing test data",missing_test)


# In[21]:


# Missing values in selected features
features_missing_values = X.isnull().sum()
features_missing_values


# In[22]:


#create a linear regression model
model = LinearRegression()
# Fit the model to the training data
model.fit(X_train, y_train)
print(model)


# In[23]:


y_pred = model.predict(X_test)


# In[24]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[25]:


print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


# In[26]:


X.sample(5)


# In[27]:


# Predict the price of a new house
new_house = np.array([[3000, 4, 2,1,1,2]])
predicted_price = model.predict(new_house)
print(f"Predicted Price for the New House: ${predicted_price[0]:.2f}")


# In[28]:


# Cross-validation to assess model performance
cv_scores = cross_val_score(model, X, y, cv=5)
print('Cross-Validation Scores:', cv_scores)
print('Mean CV Score:', cv_scores.mean())


# In[29]:


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()


# In[ ]:




