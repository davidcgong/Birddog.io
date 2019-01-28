import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, model_selection, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNet, ElasticNetCV,
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score


file_to_read = "state_top_data.csv"
df = pd.read_csv(file_to_read)
#we want to put it in html for web
df.to_html('state_data.html')

iterated_once = False
#useful for X for our regressor
count = 0
# This stores all the dates (1996-4, ... 2018-12)
date_list = []
# ID counter for region
id_counter = 0
#This stores the IDs for regions
state_id_dict = dict()
# This stores the state and price value {Georgia: [199k, 200k, 205k ...], ...}
state_val_dict = defaultdict(list)
for index, row in df.iterrows():
    month_count = 3
    #we add the id for the state here
    state_id_dict[row['RegionName']] = id_counter
    while month_count != len(df.columns):
        state_val_dict[row['RegionName']].append(df.iloc[index, month_count])
        if iterated_once == False:
#            date_list.append(df.columns[month_count])
            date_list.append(count)
            count += 1
        month_count += 1
    iterated_once = True
    id_counter += 1
    
#print(state_val_dict)
#print(date_list)
#print(state_id_dict)

total_entries = len(df.columns)
valid_state = False

while valid_state == False:
    try:
        state_input = input("Please enter the state for which you would like to see housing data (full name i.e. California, New York, Georgia, etc.): ")
        state_id = state_id_dict[state_input]
        valid_state = True
    except KeyError:
        print("Invalid state. Please make sure punctuations and spellings are correct.")

X = date_list
X = np.array(X)
X = X.reshape(-1, 1)
y = df.iloc[state_id, 3:len(df.columns)]

#Split for training and testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.85)

#regressor = RandomForestRegressor(n_estimators = 100)
#regressor = KNeighborsRegressor()
#regressor = LinearRegression()


#------------Elastic Net Regression----------------
regressor = ElasticNetCV()
regressor.fit(X_train, y_train)


predictions = regressor.predict(X)
y_pred = regressor.predict(len(X)+30)
r2_score = regressor.score(X_test, y_test)
mae_score = model_selection.cross_val_score(regressor, X, y, scoring='neg_mean_absolute_error')
print("y_pred: ", y_pred)
#print("predictions: ", predictions)
print("R^2 Score: ", r2_score)
print("Mean Absolute Error: ",mae_score.mean())
# Visualising the Random Forest Regression results in higher resolution and smoother curve
X_Grid = np.arange(min(X), max(X), 0.01)
X_Grid = X_Grid.reshape((len(X_Grid), 1))

plt.plot(state_val_dict[state_input], label=state_input)
plt.plot(X_Grid, regressor.predict(X_Grid), color = 'orange', label='Forecast')
plt.scatter(X_train, y_train, color='red', label='Selected training points')
plt.legend()
plt.axis('auto')
plt.title("ElasticNetCV (" + state_input + ")")
plt.xlabel('Months since 1996-04')
plt.ylabel('Price ($)')
plt.show()

#------------Lasso Regression----------------
regressor = LassoCV()
regressor.fit(X_train, y_train)


predictions = regressor.predict(X)
y_pred = regressor.predict(len(X)+30)
r2_score = regressor.score(X_test, y_test)
mae_score = model_selection.cross_val_score(regressor, X, y, scoring='neg_mean_absolute_error')
print("y_pred: ", y_pred)
#print("predictions: ", predictions)
print("R^2 Score: ", r2_score)
print("Mean Absolute Error: ",mae_score.mean())
# Visualising the Random Forest Regression results in higher resolution and smoother curve
X_Grid = np.arange(min(X), max(X), 0.01)
X_Grid = X_Grid.reshape((len(X_Grid), 1))

plt.plot(state_val_dict[state_input], label=state_input)
plt.plot(X_Grid, regressor.predict(X_Grid), color = 'orange', label='Forecast')
plt.scatter(X_train, y_train, color='red', label='Selected training points')
plt.legend()
plt.axis('auto')
plt.title("LassoCV (" + state_input + ")")
plt.xlabel('Months since 1996-04')
plt.ylabel('Price ($)')
plt.show()

#--------Linear Regression-------------

regressor = LinearRegression()
regressor.fit(X_train, y_train)


predictions = regressor.predict(X)
y_pred = regressor.predict(len(X)+30)
r2_score = regressor.score(X_test, y_test)
mae_score = model_selection.cross_val_score(regressor, X, y, scoring='neg_mean_absolute_error')
print("y_pred: ", y_pred)
#print("predictions: ", predictions)
print("R^2 Score: ", r2_score)
print("Mean Absolute Error: ",mae_score.mean())
# Visualising the Random Forest Regression results in higher resolution and smoother curve
X_Grid = np.arange(min(X), max(X), 0.01)
X_Grid = X_Grid.reshape((len(X_Grid), 1))


plt.plot(state_val_dict[state_input], label=state_input)
plt.plot(X_Grid, regressor.predict(X_Grid), color = 'orange', label='Forecast')
plt.scatter(X_train, y_train, color='red', label='Selected training points')
plt.legend()
plt.axis('auto')
plt.title("Linear (" + state_input + ")")
plt.xlabel('Months since 1996-04')
plt.ylabel('Price ($)')
plt.show()

#--------Random Forest-------------

regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)


predictions = regressor.predict(X)
y_pred = regressor.predict(len(X)+30)
r2_score = regressor.score(X_test, y_test)
mae_score = model_selection.cross_val_score(regressor, X, y, scoring='neg_mean_absolute_error')
print("y_pred: ", y_pred)
#print("predictions: ", predictions)
print("R^2 Score: ", r2_score)
print("Mean Absolute Error: ",mae_score.mean())
# Visualising the Random Forest Regression results in higher resolution and smoother curve
X_Grid = np.arange(min(X), max(X), 0.01)
X_Grid = X_Grid.reshape((len(X_Grid), 1))


plt.plot(state_val_dict[state_input], label=state_input)
plt.plot(X_Grid, regressor.predict(X_Grid), color = 'orange', label='Forecast')
plt.scatter(X_train, y_train, color='red', label='Selected training points')
plt.legend(prop={'size':6})
plt.axis('auto')
plt.title("Random Forest (" + state_input + ")")
plt.xlabel('Months since 1996-04')
plt.ylabel('Price ($)')

plt.show()


