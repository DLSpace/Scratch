import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation


model = Sequential()
model.add(Dense(units=64, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

x_train = np.array([0,0.523598776,0.785398163,1.047197551,1.570796327])
y_train = np.array([0,0.5,0.707106781,0.866025404,1])

model.fit(x_train, y_train, epochs=5, batch_size=32)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)


#classes = model.predict(x_test, batch_size=128)

