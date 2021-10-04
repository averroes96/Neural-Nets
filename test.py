from model import Model
from layer import Layer
from data import DataSet

model = Model()
h_layer = Layer(2,6)
o_layer = Layer(6,1)
model.addLayer(None)
model.addLayer(h_layer)
model.addLayer(o_layer)

input1 = [0, 0]
input2 = [0, 1]
input3 = [1, 0]
input4 = [1, 1]

outputs = [0, 1, 1, 0]

model.data.append(DataSet(input1, outputs[0]))
model.data.append(DataSet(input2, outputs[1]))
model.data.append(DataSet(input3, outputs[2]))
model.data.append(DataSet(input4, outputs[3]))

model.train(1000, 0.05)