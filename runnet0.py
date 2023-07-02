import pickle
import numpy as np

X_train = np.array([])

class NeuralNetwork:
    def __init__(self, weights):
        self.weights = weights

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self):
        y_hat = []
        for sample in X_train:
            activations = sample
            for i in range(len(self.weights[0])):
                w = self.weights[0][i]
                b = self.weights[1][i]
                # Calculate the dot product of inputs and weights
                weighted_sum = np.dot(activations, w) + b
                # Apply activation function (e.g., sigmoid or ReLU)
                activations = self.sigmoid(weighted_sum)
            if activations <= 0.5 :
                y_hat.append(0)
            else :
                y_hat.append(1)
        return y_hat

def extract_network():
    with open("wnet0.pkl", 'rb') as file:
        return pickle.load(file)

def extract_data():
    global X_train
    # Step 1: Read the file
    filename = 'testnet0.txt'
    data = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            binary_string = line
            data.append(binary_string)

    # Step 2: Convert binary strings to numerical arrays
    X = np.array([list(map(int, sample)) for sample in data])
    X_train = X

    print("X_train shape:", X_train.shape)

def main():
    nn = extract_network()
    extract_data()
    y_hat = nn.forward_propagation()
    np.savetxt('output0.txt', y_hat, fmt='%d', newline='\n')

if __name__ == '__main__':
    main()