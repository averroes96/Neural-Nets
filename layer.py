from neuron import Neuron
import utils

class Layer:

    def __init__(self, input, n_neurons=None) -> None: # Constructor for the hidden / output neurons
        super().__init__()

        self.neurons = []

        if n_neurons == None:
            for i in range(0, len(input)):
                self.neurons.append(Neuron(input[i]))
        else:
            for i in range(0, n_neurons):
                weights = []
                for j in range(0, input):
                    weights.append(utils.randomFloat(Neuron.minWeightValue, Neuron.maxWeightValue))
                self.neurons.append(Neuron(weights, utils.randomFloat(0, 1)))

    # def __init__(self, input) -> None: # Constructor for the input layer
    #     super().__init__()
        
    #     self.neurons = []
    #     for i in range(0, len(input)):
    #         self.neurons.append(Neuron(input[i]))

# inputs = np.random.randint(0,100, size=(1,10))
# print(inputs)
# input_layer = Layer(inputs[0])

# print(input_layer.neurons[0].value)