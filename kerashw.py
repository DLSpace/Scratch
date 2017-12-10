import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation


model = Sequential()
model.add(Dense(units=64, input_dim=1))
model.add(Activation('relu'))
model.add(Dense(units=1))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
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
model.fit(data, labels, epochs=10, batch_size=10)

