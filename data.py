class DataSet:

    def __init__(self, input, target) -> None: # Constructor for the hidden / output neurons
        super().__init__()

        self.input = input
        self.target = target