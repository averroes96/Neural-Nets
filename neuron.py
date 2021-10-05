class Neuron:

    minWeightValue = -1.0
    maxWeightValue = 1.0

    def __init__(self, weights, bias=-1) -> None: # Constructor for the hidden / output neurons
        super().__init__()
        
        if bias != -1:
            self.weights = weights
            self.bias = bias
            self.cached_weights = weights
            self.gradient = 0
            self.value = 0
        else:
            self.weights = None
            self.bias = -1
            self.cached_weights = self.weights
            self.gradient = -1
            self.value = weights

    def updateWeights(self):
        self.weights = self.cached_weights

    def __str__(self): # toString
        return f"Weights: {self.weights} | Bias: {self.weights} | cached_weights: {self.cached_weights} | value: {self.value}"

    @classmethod
    def setWeightsRange(cls, min, max): # Static function to set min and max weight for all variables
        cls.minWeightValue = min
        cls.maxWeightValue = max

neurone = Neuron(7)

# neurone.setWeightsRange(5,-5)
# print(neurone.maxWeightValue)
# print(neurone.minWeightValue)