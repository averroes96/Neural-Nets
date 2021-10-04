from layer import Layer
import utils

class Model:
    
    def __init__(self) -> None: # Constructor for the hidden / output neurons
        super().__init__()

        self.data = []
        self.layers = []

    def forward(self, inputs):

        self.layers.append(Layer(inputs))

        for i in range(0, len(self.layers)):
            for j in range(0, len(self.layers[i].neurons)):
                sum = 0.0
                for k in range(0, len(self.layers[i-1].neurons)):
                    sum += self.layers[i-1].neurons[k].value * self.layers[i].neurons[j].weights[k]
                self.layers[i].neurons[j].value = utils.sigmoid(sum)
                # print(self.layers[i].neurons[j].value)
    
    def backward(self, learning_rate, data):
        
        n_layers = len(self.layers)
        output_indx = n_layers - 1

        for i in range(0, len(self.layers[output_indx])):
            output = self.layers[output_indx].neurons[i].value
            target = self.data.target[i]
            derivative = output - target
            delta = derivative * (output * (1 - output))
            self.layers[output_indx].neurons[i].gradient = delta
            for j in range(0, len(self.layers[output_indx].neurons[i].weights)):
                previous_output = self.layers[output_indx - 1]
                error = delta * previous_output
                self.layers[output_indx].neurons[i].cached_weights[j] = self.layers[output_indx].neurons[i].weights[j] - learning_rate * error

        for i in range(output_indx - 1, 0, -1):
            for j in range(0, len(self.layers[i].neurons)):
                output = self.layers[i].neurons[j].value
                gradient_sum = self.gradientSum(j, i-1)
                delta = (gradient_sum) * (output*(1-output))
                self.layers[i].neurons[j].gradient = delta
                for k in range(0, len(self.layers[i].neurons[j].weights)):
                    previous_output = self.layers[i-1].neurons[k].value
                    error = delta * previous_output
                    self.layers[i].neurons[j].cache_weights[k] = self.layers[i].neurons[j].weights[k] - learning_rate * error
            
        for i in range(0, len(self.layers)):
            for j in range(0, len(self.layers[i].neurons)):
                self.layers[i].neurons[j].update_weight()

    def gradientSum(self, n_index, l_index):

        gradient_sum = 0
        current_layer = self.layers[l_index]

        for i in range(0, len(current_layer.neurons)):
            current_neuron = current_layer.neurons[i]
            gradient_sum += current_neuron.weights[n_index] * current_neuron.gradient

        return gradient_sum

    def train(self, epochs, learning_rate):

        print("============")
        print("Output before training")
        print("============")
        self.init()

        for i in range(0, epochs):
            for j in range(0, len(self.data)):
                self.forward(self.data[j].data)
                self.backward(learning_rate, self.data[j])

        print("============")
        print("Output after training")
        print("============")
        self.init()
    
    def init(self):
        for i in range(0, len(self.data)):
            self.forward(self.data[i].input)
            print(self.layers[2].neurons[0].value)

    def addLayer(self, layer):
        self.layers.append(layer)