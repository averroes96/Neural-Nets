import random
import math

def randomFloat(minimum, maximum):
    a = random.uniform(0, 1)
    num = minimum + random.uniform(minimum, maximum) * (maximum - minimum)
    if a < 0.5:
        return num
    else:
        return -num

def sigmoid(x): # Sigmoid function
    return (1 / (1 + math.exp(-x)))

def sigmoidDerivative(x): # Derivative of the sigmoid function
    return sigmoid(x) * (1 - sigmoid(x))

def squaredError(output, target): # Used for the BackPropagation
    return (0.5 * pow(2, (target - output)))

def sumSquaredError(outputs, targets): # Used to calculate the overall error rate
    sum = 0
    for i in range(0, len(outputs)):
        sum += squaredError(outputs[i],targets[i])
    
    return sum

# print(randomFloat(0,100))
# print(sigmoid(-30))
# print(sigmoidDerivative(1))
# print(squaredError(0.99, 0.13))