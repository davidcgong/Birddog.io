import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, model_selection, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNet, ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score


file_to_read = "state_top_data.csv"
df = pd.read_csv(file_to_read)
print(df.head())

#useful for X for our regressor for indexing
count = 0
date_list = []
for i in range(0, len(df.columns)-3):
    date_list.append(i)

#Get x-ticker for matplotlib
X_vals = df.columns.values[3:len(df.columns)].tolist()


valid_state = False
total_entries = len(df.columns)

region_list = df['RegionName']
region_list = region_list[0: len(region_list)].to_dict()
region_list = {v: k for k, v in region_list.items()}

while valid_state == False:
    try:
        state_input = input("Please enter the state for which you would like to see housing data (full name i.e. California, New York, Georgia, etc.): ")
        state_id = region_list[state_input]
        valid_state = True
    except KeyError:
        print("Invalid state. Please make sure punctuations and spellings are correct.")

#includes all the pricing data for the state from 1996-04 to Present        
state_val_list = df.iloc[state_id, 3:len(df.columns)].values.tolist()

X = date_list
X = np.array(X)
X = X.reshape(-1, 1)
y = df.iloc[state_id, 3:len(df.columns)]
#Split for training and testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.85)


# This function prints out the metrics and does predictions and draws them out on matplotlib
def graph_metrics(regressor, predictions, X, state_val_list, state_id, regressionType):

    y_pred = regressor.predict(len(X)+12)
    
    #get metrics (predicted value a year from now, r^2 score, MAE, and Forecast percentage change)

    #percentage_change is getting the percentage difference between the predicted value and the most recent value, and then rounded to 3 decimal places.
    #this lets us calculate the forecast percentage change over a year
    percentage_change = round(((y_pred[0] / predictions[len(state_val_list) - 1]) - 1.00) * 100, 3)
    r2_score = regressor.score(X_test, y_test)
    mae_score = model_selection.cross_val_score(regressor, X, y, scoring='neg_mean_absolute_error')
    print("y_pred: ", y_pred)
    print("R^2 Score: ", r2_score)
    print("Mean Absolute Error: ",mae_score.mean())
    print("Forecast percentage change for next year: ", percentage_change)
    X_Grid = np.arange(min(X), max(X), 0.01)
    X_Grid = X_Grid.reshape((len(X_Grid), 1))
    
    #plotting out the graph on matplotlib
    plt.figure(figsize=(16, 9))
    plt.plot(X_vals, state_val_list, label="True")
    plt.plot(X_Grid, regressor.predict(X_Grid), color = 'orange', label='Predictions')
    plt.scatter(X_train, y_train, color='red', label='Selected training points')
    plt.legend(prop={'size':8})
    plt.xticks(np.arange(0, len(df.columns), step = 65))
    plt.axis('auto')
    plt.title(regressionType + " (" + state_input +")")
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.show()

#------------Elastic Net Regression----------------
regressor = ElasticNetCV()
regressor.fit(X_train, y_train)
predictions = regressor.predict(X)

graph_metrics(regressor, predictions, X, state_val_list, state_id, "ElasticNetCV")


#------------Lasso Regression----------------
regressor = LassoCV()
regressor.fit(X_train, y_train)
predictions = regressor.predict(X)

graph_metrics(regressor, predictions, X, state_val_list, state_id, "LassoCV")

#--------Linear Regression-------------

regressor = LinearRegression()
regressor.fit(X_train, y_train)
predictions = regressor.predict(X)

graph_metrics(regressor, predictions, X, state_val_list, state_id, "LinearRegression")

#--------Random Forest-------------

regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)
predictions = regressor.predict(X)

graph_metrics(regressor, predictions, X, state_val_list, state_id, "Random Forest")

# Forecasting with the last 12 months
a = []
for i in range (0, 12):
    a.append(i)
#print(last_12_months)
anX = a
anX = np.array(anX)
anX = anX.reshape(-1, 1)
aY = df.iloc[state_id, len(df.columns)-12:len(df.columns)]
regressor = LinearRegression()
regressor.fit(anX, aY)
predictions = regressor.predict(anX)
y_pred = regressor.predict(len(anX) + 12)
percentage_change = round(((y_pred[0] / predictions[len(anX) - 1]) - 1.00) * 100, 3)
print(predictions)
print("Forecast percentage change for next year: ", percentage_change)



