import keras
import pandas as pd
import numpy as np
import csv

dataset = pd.read_csv('homes_clean_nocounty.csv')
dataset = np.array(dataset)

row_split = int(len(dataset) * .7)


output = []


for i in range(1298, 1431):
	#split input and output
	X_train = dataset[i:i+1, 2:len(dataset[0]) - 1]
	y_train = dataset[i:i+1, len(dataset[0]) - 1]
	X_test = dataset[i:i+1, 3:len(dataset[0])]
	X_train = np.reshape(X_train, (len(X_train), len(X_train[0]), -1))
	X_test = np.reshape(X_test, (len(X_test), len(X_test[0]), -1))

	from keras.models import Sequential
	from keras.layers import LSTM
	from keras.layers import Dense
	from keras.layers import Activation
	from keras.layers import Dropout
	from keras import optimizers

	# model
	model = Sequential()

	model.add(LSTM(273, input_shape=(273, 1)))

	model.add(Dense(20, activation='relu'))

	model.add(Dense(1))

	sgd = optimizers.SGD(lr=0.3, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_squared_error', optimizer=sgd)

	model.fit(X_train, y_train, epochs=41, verbose=0)

	ans = model.predict(X_test)[0][0]
	ans = ans * .98
	print(ans)
	output.append(ans)

	
with open('predictions_tx.csv', 'w', newline='') as csvfile:

    filewriter = csv.writer(csvfile, delimiter=',')
    for i in range(len(output)):
    	x = ""

    	for j in range(len(dataset[0])):
    		x += str(dataset[i+1298][j]).strip() + "," 

    	x += str(output[i])
    	filewriter.writerow([x])
