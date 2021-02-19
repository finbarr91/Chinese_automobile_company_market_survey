import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from sklearn.preprocessing import scale
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE
import statsmodels.api as sm


# reading of the dataset
cars = pd.read_csv("CarPrice_Assignment.csv")
print(cars.columns)

# summary of the dataset: 205 rows, 26 columns, no null values
print(cars.info())

"""
Understanding the Data Dictionary
The data dictionary contains the meaning of various attributes; some non-obvious ones are:
"""

print(cars['symboling'].astype('category').value_counts())

# aspiration: An (internal combustion) engine property showing
# whether the oxygen intake is through standard (atmospheric pressure)
# or through turbocharging (pressurised oxygen intake).
print(cars['aspiration'].astype('category').value_counts())

# drivewheel: frontwheel, rarewheel or four-wheel drive
print(cars['drivewheel'].astype('category').value_counts())

# wheelbase: distance between centre of front and rarewheels
sns.displot(cars['wheelbase'])
plt.show()

# curbweight: weight of car without occupants or baggage
sns.displot(cars['curbweight'])
plt.show()

# stroke: volume of the engine (the distance traveled by the
# piston in each cycle)
sns.displot(cars['stroke'])
plt.show()

# compression ration: ration of volume of compression chamber
# at largest capacity to least capacity
sns.displot(cars['compressionratio'])
plt.show()

# target variable: price of car
sns.displot(cars['price'])
plt.show()

print(cars['car_ID'].dtype)
print(cars['symboling'].dtype)

"""
Data Exploration
To perform linear regression, the (numeric) target variable should be linearly related to at least one another numeric variable. Let's see whether that's true in this case.

We'll first subset the list of all (independent) numeric variables, and then make a pairwise plot.
"""

# all numeric (float and int) variables in the dataset
cars_numeric = cars.select_dtypes(include=['float64', 'int64'])
print('\nCars Numeric:\n', cars_numeric.head())
print('\n Car Numeric Columns\n', cars_numeric.columns)

# Here, although the variable symboling is numeric (int), we'd rather treat it as categorical since it has only 6 discrete values.
# Also, we do not want 'car_ID'.

# dropping symboling and car_ID
drop_colums = ['symboling', 'car_ID']
cars_numeric = cars_numeric.drop(drop_colums, axis=1)
print(cars_numeric.head())
# Let's now make a pairwise scatter plot and observe linear relationships.

# pairwise scatter plot
plt.figure(figsize=(20, 10))
sns.pairplot(cars_numeric)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()

# This is quite hard to read, and we can rather plot correlations between variables.
# Also, a heatmap is pretty useful to visualise multiple correlations in one plot.

# correlation matrix
cor = cars_numeric.corr()
print(cor)

# plotting correlations on heatmap

# figure size
plt.figure(figsize=(20,15))
sns.heatmap(cor,cmap='YlGnBu', annot=True)
plt.show()
"""Let's now conduct some data cleaning steps.

We've seen that there are no missing values in the dataset. We've also seen that variables are in the correct format, except symboling, which should rather be a categorical variable (so that dummy variable are created for the categories).

Note that it can be used in the model as a numeric variable also.
"""

# variable formats
print(cars.info())

# converting symboling to categorical
cars['symboling'] = cars['symboling'].astype('object')
print(cars.info())

# To extraxt the company name from the column carname
print(cars['CarName'][0:])

"""Notice that the carname is what occurs before a space, e.g. alfa-romero, audi, chevrolet, dodge, bmx etc.

Thus, we need to simply extract the string before a space. There are multiple ways to do that."""
# Extracting carname
# Method 1: str.split() by space
carnames = cars['CarName'].apply(lambda x: x.split(" ")[0])
print(carnames[:30])

# regex: any alphanumeric sequence before a space, may contain a hyphen
p = re.compile(r'\w+-?\w+')
carnames = cars['CarName'].apply(lambda x: re.findall(p, x)[0])
print(carnames)

# Let's create a new column to store the compnay name and check whether it looks okay.

# New column car_company
cars['car_company'] = cars['CarName'].apply(lambda x: re.findall(p, x)[0])
# look at all values
print(cars['car_company'].astype('category').value_counts())

"""Notice that some car-company names are misspelled - vw and vokswagen should be volkswagen, 
porcshce should be porsche, toyouta should be toyota, 
Nissan should be nissan, maxda should be mazda etc.
This is a data quality issue, let's solve it."""

# replacing misspelled car_company names

# volkswagen
cars.loc[(cars['car_company'] == "vw") |
         (cars['car_company'] == "vokswagen")
         , 'car_company'] = 'volkswagen'

# porsche
cars.loc[cars['car_company'] == "porcshce", 'car_company'] = 'porsche'

# toyota
cars.loc[cars['car_company'] == "toyouta", 'car_company'] = 'toyota'

# nissan
cars.loc[cars['car_company'] == "Nissan", 'car_company'] = 'nissan'

# mazda
cars.loc[cars['car_company'] == "maxda", 'car_company'] = 'mazda'

print(cars['car_company'].astype('category').value_counts())

# The car_company variable looks okay now. Let's now drop the car name variable.

# drop carname variable
cars = cars.drop('CarName', axis=1)
print(cars.info())

# outliers
print(cars.describe())

print(cars.info())

# Data Preparation
# Let's now prepare the data and build the model.
# split into X and y
X = cars.loc[:, ['symboling', 'fueltype', 'aspiration', 'doornumber',
       'carbody', 'drivewheel', 'enginelocation', 'wheelbase', 'carlength',
       'carwidth', 'carheight', 'curbweight', 'enginetype', 'cylindernumber',
       'enginesize', 'fuelsystem', 'boreratio', 'stroke', 'compressionratio',
       'horsepower', 'peakrpm', 'citympg', 'highwaympg',
       'car_company']]

y = cars['price']

# creating dummy variables for categorical variables

# subset all categorical variables
cars_categorical = X.select_dtypes(include=['object'])
print(cars_categorical.head())

# convert into dummies
cars_dummies = pd.get_dummies(cars_categorical, drop_first=True)
print(cars_dummies.head())

# drop categorical variables
X = X.drop(list(cars_categorical.columns), axis=1)

# concat dummy variables with X
X = pd.concat([X, cars_dummies], axis=1)


# storing column names in cols, since column names are (annoyingly) lost after
# scaling (the df is converted to a numpy array)
cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
print(X.columns)

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.7,test_size = 0.3, random_state=100)

# Building the first model with all the features
# instantiate
lm = LinearRegression()
# fit
lm.fit(X_train, y_train)
# print coefficients and intercept
print(lm.coef_)
print(lm.intercept_)
# predict
y_pred = lm.predict(X_test)

# metrics
print(r2_score(y_true=y_test, y_pred=y_pred))

# RFE with 15 features
lm = LinearRegression()
rfe_15 = RFE(lm, 15)

# fit with 15 features
rfe_15.fit(X_train, y_train)

# Printing the boolean results
print(rfe_15.support_)
print(rfe_15.ranking_)

# making predictions using rfe model
y_pred = rfe_15.predict(X_test)

# r-squared
print(r2_score(y_test, y_pred))

# RFE with 6 features
lm = LinearRegression()
rfe_6 = RFE(lm, 6)

# fit with 6 features
rfe_6.fit(X_train, y_train)

# predict
y_pred = rfe_6.predict(X_test)

# r-squared
print(r2_score(y_test, y_pred))

# subset the features selected by rfe_15
col_15 = X_train.columns[rfe_15.support_]

# subsetting training data for 15 selected columns
X_train_rfe_15 = X_train[col_15]

# add a constant to the model
X_train_rfe_15 = sm.add_constant(X_train_rfe_15)
print(X_train_rfe_15.head())

# fitting the model with 15 variables
lm_15 = sm.OLS(y_train, X_train_rfe_15).fit()
print(lm_15.summary())

# making predictions using rfe_15 sm model
X_test_rfe_15 = X_test[col_15]


# # Adding a constant variable
X_test_rfe_15 = sm.add_constant(X_test_rfe_15, has_constant='add')
print(X_test_rfe_15.info())
# Making predictions
y_pred = lm_15.predict(X_test_rfe_15)

# r-squared
r2_score(y_test, y_pred)

# subset the features selected by rfe_6
col_6 = X_train.columns[rfe_6.support_]

# subsetting training data for 6 selected columns
X_train_rfe_6 = X_train[col_6]

# add a constant to the model
X_train_rfe_6 = sm.add_constant(X_train_rfe_6)

# fitting the model with 6 variables
lm_6 = sm.OLS(y_train, X_train_rfe_6).fit()
print(lm_6.summary())

# making predictions using rfe_6 sm model
X_test_rfe_6 = X_test[col_6]

# Adding a constant
X_test_rfe_6 = sm.add_constant(X_test_rfe_6, has_constant='add')
X_test_rfe_6.info()

# Making predictions
y_pred = lm_6.predict(X_test_rfe_6)

# r2_score for 6 variables
r2_score(y_test, y_pred)

"""Choosing the optimal number of features
Now, we have seen that the adjusted r-squared varies from about 93.3 to 88 as we go from 15 to 6 features, 
one way to choose the optimal number of features is to make a plot between n_features and adjusted r-squared, and then choose the value of n_features."""

n_features_list = list(range(4, 20))
adjusted_r2 = []
r2 = []
test_r2 = []

for n_features in range(4, 20):
       # RFE with n features
       lm = LinearRegression()

       # specify number of features
       rfe_n = RFE(lm, n_features)

       # fit with n features
       rfe_n.fit(X_train, y_train)

       # subset the features selected by rfe_6
       col_n = X_train.columns[rfe_n.support_]

       # subsetting training data for 6 selected columns
       X_train_rfe_n = X_train[col_n]

       # add a constant to the model
       X_train_rfe_n = sm.add_constant(X_train_rfe_n)

       # fitting the model with 6 variables
       lm_n = sm.OLS(y_train, X_train_rfe_n).fit()
       adjusted_r2.append(lm_n.rsquared_adj)
       r2.append(lm_n.rsquared)

       # making predictions using rfe_15 sm model
       X_test_rfe_n = X_test[col_n]

       # # Adding a constant variable
       X_test_rfe_n = sm.add_constant(X_test_rfe_n, has_constant='add')

       # # Making predictions
       y_pred = lm_n.predict(X_test_rfe_n)

       test_r2.append(r2_score(y_test, y_pred))

# plotting adjusted_r2 against n_features
plt.figure(figsize=(10, 8))
plt.plot(n_features_list, adjusted_r2, label="adjusted_r2")
plt.plot(n_features_list, r2, label="train_r2")
plt.plot(n_features_list, test_r2, label="test_r2")
plt.legend(loc='upper left')
plt.show()

# Final Model
# Let's now build the final model with 6 features.
# RFE with n features
lm = LinearRegression()

n_features = 6

# specify number of features
rfe_n = RFE(lm, n_features)

# fit with n features
rfe_n.fit(X_train, y_train)

# subset the features selected by rfe_6
col_n = X_train.columns[rfe_n.support_]

# subsetting training data for 6 selected columns
X_train_rfe_n = X_train[col_n]

# add a constant to the model
X_train_rfe_n = sm.add_constant(X_train_rfe_n)

# fitting the model with 6 variables
lm_n = sm.OLS(y_train, X_train_rfe_n).fit()
adjusted_r2.append(lm_n.rsquared_adj)
r2.append(lm_n.rsquared)

# making predictions using rfe_15 sm model
X_test_rfe_n = X_test[col_n]

# # Adding a constant variable
X_test_rfe_n = sm.add_constant(X_test_rfe_n, has_constant='add')

# # Making predictions
y_pred = lm_n.predict(X_test_rfe_n)

test_r2.append(r2_score(y_test, y_pred))

# summary
print(lm_n.summary())
# results
print(r2_score(y_test, y_pred))

""""
Final Model Evaluation
Let's now evaluate the model in terms of its assumptions. We should test that:
The error terms are normally distributed with mean approximately 0
There is little correlation between the predictors
Homoscedasticity, i.e. the 'spread' or 'variance' of the error term (y_true-y_pred) is constant
"""

# Error terms
c = [i for i in range(len(y_pred))]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('ytest-ypred', fontsize=16)                # Y-label
plt.show()

# Plotting the error terms to understand the distribution.
fig = plt.figure()
sns.distplot((y_test-y_pred),bins=50)
fig.suptitle('Error Terms', fontsize=20)                  # Plot heading
plt.xlabel('y_test-y_pred', fontsize=18)                  # X-label
plt.ylabel('Index', fontsize=16)                          # Y-label
plt.show()

# mean
print(np.mean(y_test-y_pred))

sns.distplot(cars['price'],bins=50)
plt.show()

# multicollinearity
predictors = ['carwidth', 'curbweight', 'enginesize',
             'enginelocation_rear', 'car_company_bmw', 'car_company_porsche']

cors = X.loc[:, list(predictors)].corr()
sns.heatmap(cors, annot=True)
plt.show()