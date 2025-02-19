import numpy as np
 
def softmax(vec):
  exponential = np.exp(vec)
  probabilities = exponential / np.sum(exponential)
  return probabilities
 
vector = np.array([0.45, -0.21, 0.67, 0.33, -0.29, 0.92, 0.44, -0.21, 0.67, 0.33, 0.45, -0.21, 0.67, 0.33, -0.29, 0.92, 0.44])
probabilities = softmax(vector)
print("Probability Distribution is:")
print(probabilities)