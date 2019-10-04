import json
import numpy as np
import joblib
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout


def load_data(file_list):
    arr = []
    arr = np.array(arr)
    for i in range(0, 300):
        for file_name in file_list:
            with open("/Users/AntonioShen/PycharmProjects/Awesome/" + file_name + ".json", 'r') as open_file:
                data = open_file.read()
            val = json.loads(data)
            data = val[file_name[len(file_name)-1]]
            data = np.array(data)
            data = data[i]
            arr = np.concatenate((arr, data), axis=0)
    arr = np.array(arr)
    arr = np.reshape(arr, (2400, 28))
    return arr


train_X = load_data(["result0", "result1", "result2", "result3", "result4", "result5", "result6", "result7"])
train_Y = np.array([0, 1, 2, 3, 4, 5, 6, 7])
for i in range(0, 299):
    train_Y = np.concatenate((train_Y, np.array([0, 1, 2, 3, 4, 5, 6, 7])), axis=0)
train_Y = to_categorical(train_Y)
print(train_X)
print(train_Y)

model = Sequential()
model.add(Dense(128, activation='tanh', input_shape=(28,)))
model.add(Dense(512, activation='tanh'))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(8, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_X, train_Y, epochs=100, verbose=2)

joblib.dump(model, '/Users/AntonioShen/awesome.pkl')
exit()
