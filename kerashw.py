import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
import math
import matplotlib.pyplot as plt
import datetime


model = Sequential()
#input layer
model.add(Dense(units=1, input_dim=1))
model.add(Activation('linear'))
#hidden layer 1
model.add(Dense(units=10))
model.add(Activation('sigmoid'))
#hidden layer 2
model.add(Dense(units=10))
model.add(Activation('sigmoid'))
#output layer
model.add(Dense(units=1))
model.add(Activation('linear'))

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])


# Generate dummy data
import numpy as np
# data = np.random.random((1000, 1))
data = []
labels = [];
for i in range(0,90,1):
    data.extend([[math.radians(i)]])
    labels.extend([[math.sin(math.radians(i))]])
# data = np.array([[0],
#                 [0.523598776],
#                 [0.785398163],
#                 [1.047197551],
#                 [1.570796327]])
# print(data)
# labels = np.random.randint(2, size=(5, 1))

# labels = np.array([[0],
#                 [0.5],
#                 [0.707106781],
#                 [0.866025404],
#                 [1]])
# print(labels)
# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=2000, batch_size=5)

model_json = model.to_json()
with open("kerastestModel.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("kerastestModel.h5")

testData = [];
for i in range(0,180,2):
    testData.extend([[math.radians(i)]])

pred = model.predict(testData,batch_size=1)

fig, ax = plt.subplots()
plt.ylabel('Predict');
plt.xlabel(datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"));
plt.ioff()
plt.autoscale(enable=True,axis='both');

for i in range(0,len(pred),1):
    # ax.plot(epoch,self.avgCost,'.');
    ax.plot(testData[i][0],pred[i][0],'b,');
    ax.plot(testData[i][0],math.sin(testData[i][0]),'r,');
    print('sin('+str(testData[i][0])+') = '+str(pred[i][0])+' :act: ' + str(math.sin(testData[i][0])))

plt.savefig("kerastest.png")
plt.close()