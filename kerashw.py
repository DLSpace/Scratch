import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation


model = Sequential()
#input layer
model.add(Dense(units=1, input_dim=1))
model.add(Activation('softmax'))
#hidden layer 1
model.add(Dense(units=10))
model.add(Activation('sigmoid'))
#hidden layer 2
model.add(Dense(units=10))
model.add(Activation('sigmoid'))
#output layer
model.add(Dense(units=1))
model.add(Activation('softmax'))

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])


# Generate dummy data
import numpy as np
# data = np.random.random((1000, 1))
data = np.array([[0],
                [0.523598776],
                [0.785398163],
                [1.047197551],
                [1.570796327]])
print(data)
# labels = np.random.randint(2, size=(5, 1))
labels = np.array([[0],
                [0.5],
                [0.707106781],
                [0.866025404],
                [1]])
print(labels)
# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=1000, batch_size=10)

pred = model.predict([
    [0.034906585],
    [0.06981317],
    [0.104719755],
    [0.13962634],
    [0.174532925],
    [0.20943951],
    [0.244346095],
    [0.27925268]
],batch_size=10)
print(pred)