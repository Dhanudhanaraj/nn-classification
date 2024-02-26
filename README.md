# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
![Screenshot 2024-02-25 131238](https://github.com/Dhanudhanaraj/nn-classification/assets/119218812/e8729c86-2fe4-4ef4-9c38-37cb0dfd8f26)

## DESIGN STEPS

### STEP 1:
Import the necessary packages & modules
### STEP 2:
Load and read the dataset
### STEP 3:
Perform pre processing and clean the dataset
### STEP 4:
Encode categorical value into numerical values using ordinal/label/one hot encoding
### STEP 5:
Visualize the data using different plots in seaborn
### STEP 6:
Normalize the values and split the values for x and y
### STEP 7:
Build the deep learning model with appropriate layers and depth
### STEP 8:
Analyze the model using different metrics
### STEP 9:  
Plot a graph for Training Loss, Validation Loss Vs Iteration & for Accuracy, Validation Accuracy vs Iteration
### STEP 10:
Save the model using pickle
### STEP 11:
Using the DL model predict for some random inputs

## PROGRAM
```
Name:DHANUMALYA D 
Register Number:212222230030
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt

df = pd.read_csv('/content/customers (1).csv')
df.head()
df.columns
df.dtypes
df.shape
df.isnull().sum()
df_cleaned=df.dropna(axis=0)
df_cleaned.isnull().sum()
df_cleaned.shape
df_cleaned.dtypes
df_cleaned['Gender'].unique()
df_cleaned['Ever_Married'].unique()
df_cleaned['Graduated'].unique()
df_cleaned['Family_Size'].unique()
df_cleaned['Var_1'].unique()
df_cleaned['Spending_Score'].unique()
df_cleaned['Profession'].unique()
df_cleaned['Segmentation'].unique()

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

categories_list=[['Male','Female'],
                 ['No','Yes'],
                 ['No','Yes'],
                 ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
                 'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
                 ['Low','Average','High']
                 ]
enc = OrdinalEncoder(categories = categories_list)

cust_1=df_cleaned.copy()
cust_1[['Gender','Ever_Married','Graduated','Profession','Spending_Score']]=enc.fit_transform(cust_1[['Gender','Ever_Married','Graduated','Profession','Spending_Score']])

cust_1.dtypes


le = LabelEncoder()
     

cust_1['Segmentation'] = le.fit_transform(cust_1['Segmentation'])
     

cust_1.dtypes
     

cust_1 = cust_1.drop('ID',axis=1)
cust_1 = cust_1.drop('Var_1',axis=1)
     

cust_1.dtypes
     

# Calculate the correlation matrix
corr = cust_1.corr()

# Plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)

sns.pairplot(cust_1)

cust_1.describe()
cust_1['Segmentation'].unique()

X=cust_1[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size']].values

y1 = cust_1[['Segmentation']].values
one_hot_enc = OneHotEncoder()
one_hot_enc.fit(y1)
y1.shape
y = one_hot_enc.transform(y1).toarray() 
y.shape    
y1[0]
y[0]
X.shape
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.33,
                                               random_state=50)
X_train[0]
X_train.shape
scaler_age = MinMaxScaler()
scaler_age.fit(X_train[:,2].reshape(-1,1))
X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)

# To scale the Age column
X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)


# Creating the model
ai_brain = Sequential([
  Dense(6,input_shape=(8,)),
  Dense(10,activation='relu'),
  Dense(10,activation='relu'),
  Dense(4,activation='softmax'),
])

ai_brain.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=2)


ai_brain.fit(x=X_train_scaled,y=y_train,
             epochs=2000,batch_size=256,
             validation_data=(X_test_scaled,y_test),
             )
metrics = pd.DataFrame(ai_brain.history.history)

metrics.head()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(ai_brain.predict(X_test_scaled), axis=1)
x_test_predictions.shape
y_test_truevalue = np.argmax(y_test,axis=1)
y_test_truevalue.shape
print(confusion_matrix(y_test_truevalue,x_test_predictions))
print(classification_report(y_test_truevalue,x_test_predictions))
     
# Saving the Model
ai_brain.save('customer_classification_model.h5')

# Saving the data
with open('customer_data.pickle', 'wb') as fh:
   pickle.dump([X_train_scaled,y_train,X_test_scaled,y_test,cust_1,df_cleaned,scaler_age,enc,one_hot_enc,le], fh)

ai_brain = load_model('customer_classification_model.h5')

# Loading the data
with open('customer_data.pickle', 'rb') as fh:
   [X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le]=pickle.load(fh)
     
#Prediction for a single input
x_single_prediction = np.argmax(ai_brain.predict(X_test_scaled[1:2,:]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))

```

## Dataset Information
![Screenshot 2024-02-25 173943](https://github.com/Dhanudhanaraj/nn-classification/assets/119218812/31305c89-d032-4d89-b439-233c57a59a6d)


## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![Screenshot 2024-02-26 142021](https://github.com/Dhanudhanaraj/nn-classification/assets/119218812/c17fba38-c42a-4ccb-9af8-5d2f9034510b)


### Classification Report
![Screenshot 2024-02-26 142041](https://github.com/Dhanudhanaraj/nn-classification/assets/119218812/901d5a3e-c74f-436c-8dff-bda05a194430)


### Confusion Matrix
![Screenshot 2024-02-26 142030](https://github.com/Dhanudhanaraj/nn-classification/assets/119218812/ca62c422-31d8-4508-a66d-969fbe8bd6c5)


### New Sample Data Prediction
![Screenshot 2024-02-25 174133](https://github.com/Dhanudhanaraj/nn-classification/assets/119218812/28406ef7-88d8-443f-89fc-8bcd9bf8d1f7)

## RESULT
A neural network classification model is developed for the given dataset.

