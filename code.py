
#Black Friday Practice Problem by Analytics Vidhya 

#Importing Libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import statistics  

#Importing dataset
dataset = pd.read_csv('train.csv')
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 11]


#Missing Values

x.apply(lambda x: sum(x.isnull()))
#We found two variables with missing values – Product_Category_2 and Product_Category_3
#They both have very high missing values so I am deleting them
#And also deleting product catogery 1
del x['Product_Category_1']
del x['Product_Category_2']
del x['Product_Category_3']

#mean1 = x["Item_Weight"].mean()
#x["Item_Weight"] = x["Item_Weight"].replace(np.nan, mean1)

# Lets impute Outlet_Size with the mode of the Outlet_Size for the particular type of outlet
#mode1 = statistics.mode(x["Outlet_Size"])
#x["Outlet_Size"] = x["Outlet_Size"].replace(np.nan, mode1)


#Feature Engineering 

#Deleting User_ID and Product_ID 
del x['User_ID']
del x['Product_ID']


# Categorical Variable

#from sklearn.preprocessing import LabelEncoder
#labelencoder_x = LabelEncoder()
#x['Item_Fat_Content'] = labelencoder_x.fit_transform(x['Item_Fat_Content'])

# But this line of code will give different countries different numbers and by that our ML algo will 
#understand that one country is greater than another. This method is good for categories like Small, medium, large etc



#To solve this issue we will use Dummy Variables 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x['Gender'] = labelencoder_x.fit_transform(x['Gender'])
x['Age'] = labelencoder_x.fit_transform(x['Age'])
x['Occupation'] = labelencoder_x.fit_transform(x['Occupation'])
x['City_Category'] = labelencoder_x.fit_transform(x['City_Category'])
x['Stay_In_Current_City_Years'] = labelencoder_x.fit_transform(x['Stay_In_Current_City_Years'])


#One Hot Coding:
x = pd.get_dummies(x, columns=['Gender','Age','Occupation','City_Category',
                              'Stay_In_Current_City_Years','Marital_Status'])

#If y is also categorical then 
#But for dependent variable we don't have to use OneHotEncoder as ML knows this
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)

#---------------------------------------------------------------------------------------------------------------

#All the above steps will be done for test set also 

 
#Importing dataset
testset = pd.read_csv('Test.csv')

#Missing Values

testset.apply(lambda testset: sum(testset.isnull()))
#We found two variables with missing values
del testset['Product_Category_1']
del testset['Product_Category_2']
del testset['Product_Category_3']





#for submission purpose 
submittion_item_identifier = testset[['User_ID', 'Product_ID']]


#Deleting Item identifier as it has no use 
del testset['User_ID']
del testset['Product_ID']



# Categorical Variable

#from sklearn.preprocessing import LabelEncoder
#labelencoder_x = LabelEncoder()
#x['Item_Fat_Content'] = labelencoder_x.fit_transform(x['Item_Fat_Content'])

# But this line of code will give different countries different numbers and by that our ML algo will 
#understand that one country is greater than another. This method is good for categories like Small, medium, large etc



#To solve this issue we will use Dummy Variables 
from sklearn.preprocessing import LabelEncoder
labelencoder_testset = LabelEncoder()
testset['Gender'] = labelencoder_testset.fit_transform(testset['Gender'])
testset['Age'] = labelencoder_testset.fit_transform(testset['Age'])
testset['Occupation'] = labelencoder_testset.fit_transform(testset['Occupation'])
testset['City_Category'] = labelencoder_testset.fit_transform(testset['City_Category'])
testset['Stay_In_Current_City_Years'] = labelencoder_testset.fit_transform(testset['Stay_In_Current_City_Years'])


#One Hot Coding:
testset = pd.get_dummies(testset, columns=['Gender','Age','Occupation','City_Category',
                              'Stay_In_Current_City_Years','Marital_Status'])

#Feature Scaling ## No feature scaling for y in categorical 
#from sklearn.preprocessing import StandardScaler
#sc_x = StandardScaler()
#x = sc_x.fit_transform(x)
#testset = sc_x.transform(testset)

#----------------------------------------------------------------------------------------------------------------


#Model Building
#1-baseline model
mean_sales = y.mean()

#Define a dataframe with IDs for submission:

submittion_item_identifier['Purchase'] = mean_sales

#Export submission file
submittion_item_identifier.to_csv("alg0.csv",index=False)

##splitting the dataset
#from sklearn.model_selection import train_test_split 
#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = .2, random_state = 0)


#Linear Regression Model 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)

#predicting the testset result 
y_pred = regressor.predict(testset)

#Define a dataframe with IDs for submission:

submittion_item_identifier['Purchase'] = y_pred

#Export submission file
submittion_item_identifier.to_csv("linearregression.csv",index=False)


#SVR model 
from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
regressor.fit(x,y)
y_pred = regressor.predict(testset)

base3 = testset[['Item_Identifier','Outlet_Identifier']]
base3['Item_Outlet_Sales'] = y_pred

#Export submission file
base3.to_csv("svr_poly.csv",index=False)

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(x,y)

y_pred = regressor.predict(testset)

submittion_item_identifier['Purchase'] = y_pred

#Export submission file
submittion_item_identifier.to_csv("tree.csv",index=False)

#Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 8)

regressor.fit(x,y)

y_pred = regressor.predict(testset)

submittion_item_identifier['Purchase'] = y_pred
#Export submission file
submittion_item_identifier.to_csv("Forest.csv",index=False)




















