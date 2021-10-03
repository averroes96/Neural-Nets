class Neuron:

    minWeightValue = -1.0
    maxWeightValue = 1.0

    def __init__(self, weights, bias) -> None: # Constructor for the hidden / output neurons
        super().__init__()

        self.weights = weights
        self.bias = bias
        self.cached_weights = weights
        self.gradient = 0

    def __init__(self, value) -> None: # Constructor for the input neurons
        super().__init__()
        
        self.weights = None
        self.bias = -1
        self.cached_weights = self.weights
        self.gradient = -1
        self.value = value

    def updateWeights(self):
        self.weights = self.cached_weights

    @classmethod
    def setWeightsRange(cls, min, max): # Static function to set min and max weight for all variables
        cls.minWeightValue = min
        cls.maxWeightValue = max

neurone = Neuron(7)

neurone.setWeightsRange(5,-5)
print(neurone.maxWeightValue)
print(neurone.minWeightValue)