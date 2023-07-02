import numpy as np
import random
import pickle

POPULATION_SIZE = 250
MAX_ITERATIONS = 160
MUTATION_RATE = 0.2
SATISFACTORY_FITNESS = 400
tournament_size = 25
probability = 0.4
INPUT_SIZE = 16
HIDDEN_LAYERS_SIZE = [32]
OUTPUT_SIZE = 1
X_train = np.array([])
y_train = np.array([])
X_test = np.array([])
y_test = np.array([])
def extract_data():
    global X_train, y_train, X_test,y_test
    # Step 1: Read the file
    filename = 'training1.txt'
    data = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip().split()
            binary_string = line[0]
            classification = int(line[1])
            data.append((binary_string, classification))

    # Step 2: Convert binary strings to numerical arrays
    X = np.array([list(map(int, sample[0])) for sample in data])
    y = np.array([sample[1] for sample in data])
    X_train = X # Step 3: Normalize the input data to [-1, 1]
    y_train = y

    filename1 = 'test1.txt'
    data1 = []
    with open(filename1, 'r') as file1:
        for line in file1:
            line = line.strip().split()
            binary_string = line[0]
            classification = int(line[1])
            data1.append((binary_string, classification))

    # Step 2: Convert binary strings to numerical arrays
    X1 = np.array([list(map(int, sample[0])) for sample in data1])
    y1 = np.array([sample[1] for sample in data1])
    X_test = X1  # Step 3: Normalize the input data to [-1, 1]
    y_test = y1


    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    return X_train,y_train

def initialize_population(hidden_size):
    population = []

    for _ in range(POPULATION_SIZE):
        # Initialize the weights for each individual in the population
        weights = []
        biases = []

        # Input layer to first hidden layer
        weights.append(np.random.randn(INPUT_SIZE, hidden_size[0]))
        biases.append(np.random.randn(hidden_size[0]))

        # Hidden layers
        for i in range(len(hidden_size)-1):
            weights.append(np.random.randn(hidden_size[i], hidden_size[i + 1]))
            biases.append(np.random.randn(hidden_size[i+1]))

        # Last hidden layer to output layer
        weights.append(np.random.randn(hidden_size[-1], OUTPUT_SIZE))
        biases.append(np.random.randn(OUTPUT_SIZE))

        # Add the weights to the population
        population.append((weights,biases))

    return population

# class NumpyArrayEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             if obj.ndim == 1:
#                 return obj.tolist()
#             else:
#                 return [arr.tolist() for arr in obj]
#         return super().default(obj)

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

    def __str__(self):
        return str(self.fitness)

    # def to_json(self):
    #     json_data = {
    #         'weights': self.weights,
    #         'fitness': self.fitness
    #     }
    #     return json.dumps(json_data, indent=4, cls=NumpyArrayEncoder)
    #
    # @classmethod
    # def from_json(cls, json_str):
    #     json_data = json.loads(json_str)
    #     weights = [np.array(arr) for arr in json_data['weights']]
    #     fitness = json_data['fitness']
    #     return cls(weights, fitness)


def fitness (pop):
    nn = NeuralNetwork(pop)
    y_hat = nn.forward_propagation()
    fit_score = accuracy(y_hat)
    return fit_score

def accuracy(y):
    y_hat = np.array(y)
    # Calculate the number of correct predictions
    num_correct = np.sum(y_hat == y_train)
    # Calculate the total number of predictions
    total_predictions = len(y_train)
    # Calculate accuracy as the ratio of correct predictions to total predictions
    accuracy_score = num_correct / total_predictions
    return accuracy_score

def loss(y_hat):
    epsilon = [1e-15]  # Small constant to avoid log(0)
    # Calculate the binary cross entropy loss
    loss = -np.mean(y_train * np.log(y_hat + epsilon) + (1 - y_train) * np.log(1 - y_hat + epsilon))
    return loss

def selection(population):
    tournament = random.sample(range(len(population)), tournament_size)
    parent1_idx = max(tournament, key=lambda i: population[i][1])
    tournament.remove(parent1_idx)
    parent2_idx = max(tournament, key=lambda i: population[i][1])

    parent1 = population[parent1_idx]
    parent2 = population[parent2_idx]

    return parent1, parent2

def crossover(parent1, parent2):
    if random.random() < probability:

        child1_listw = []
        child2_listw = []
        child1_listb = []
        child2_listb = []
        for p1, p2 in zip(parent1[0][0], parent2[0][0]):
            point = random.randint(1, p1.shape[0] - 2)
            child1 = np.concatenate((p1[:point, :], p2[point:, :]), axis=0)
            child2 = np.concatenate((p2[:point, :], p1[point:, :]), axis=0)
            child1_listw.append(child1)
            child2_listw.append(child2)

        for pp1, pp2 in zip(parent1[0][1], parent2[0][1]):
            if pp1.shape[0] != 1 :
                point = random.randint(1, pp1.shape[0] - 2)
                child1 = np.concatenate((pp1[:point], pp2[point:]))
                child2 = np.concatenate((pp2[:point], pp1[point:]))
                child1_listb.append(child1)
                child2_listb.append(child2)
            else:
                if parent1[1] > parent2[1]:
                    child1_listb.append(np.array([parent1[1]]))
                    child2_listb.append(np.array([parent1[1]]))
                else:
                    child1_listb.append(np.array([parent2[1]]))
                    child2_listb.append(np.array([parent2[1]]))

        child1 = (child1_listw,child1_listb)
        child2 = (child2_listw,child2_listb)

        return child1, child2
    else:
        return parent1[0], parent2[0]

def mutation(child):
    # weights = child[0]
    # biases = child[1]
    # num_of_weights = len(weights)
    # num_of_biases = len(biases)
    # chance_to_pick_weight = num_of_weights / (num_of_weights + num_of_biases)
    #
    # while random.random() < MUTATION_RATE:
    #     if random.random() <= chance_to_pick_weight:
    #         # pick a random weight layer
    #         layer = random.randint(0, num_of_weights - 1)
    #         # pick a random weight
    #         weight = random.randint(0, len(weights[layer]) - 1)
    #
    #         # add random value to the weight
    #         weights[layer][weight]  = change_weight(weights[layer][weight])
    #     else:
    #         # pick a random bias layer
    #         layer = random.randint(0, num_of_biases - 1)
    #         # pick a random bias
    #         bias = random.randint(0, len(biases[layer]) - 1)
    #
    #         # add random value to the bias
    #         biases[layer][bias] = change_weight(biases[layer][bias])
    #
    # return child
    # Perform mutation operation
    if random.random() < MUTATION_RATE:
        mutated_weights = []
        mutated_biases = []
        con_child = [(sublist1, sublist2) for sublist1, sublist2 in zip(*child)]

        for layer_weights, layer_bias in con_child:
            mutated_layer_weights = list(layer_weights)
            mutated_layer_bias = list(layer_bias)
            index1, index2 = random.sample(range(len(mutated_layer_weights)), 2)
            mutated_layer_weights[index1], mutated_layer_weights[index2] = mutated_layer_weights[index2], \
                                                                           mutated_layer_weights[index1]
            if len(mutated_layer_bias) > 2 :
                index1b, index2b = random.sample(range(len(mutated_layer_bias)), 2)
                mutated_layer_bias[index1b], mutated_layer_bias[index2b] = mutated_layer_bias[index2b], mutated_layer_bias[index1b]
            mutated_weights.append(np.array(mutated_layer_weights))
            mutated_biases.append(np.array(mutated_layer_bias))
        mutated_child = (mutated_weights, mutated_biases)
    else:
        mutated_child = child
    return mutated_child

def change_weight(weight):
    random_number = random.random()
    if random_number > 0.75:
        return weight + random.gauss(0, 1)
    elif random_number > 0.5:
        return weight * random.gauss(1, 0.25)
    elif random_number > 0.25:
        return 0
    else:
        return random.gauss(0, 1)

def genetic_algo():
    global MUTATION_RATE, probability
    x_train, y_train = extract_data()
    threshold = 5
    consecutive_generations = 0
    population=initialize_population(HIDDEN_LAYERS_SIZE)
    # min_fitness_gen = 999999999
    max_fitness_gen = 0
    # Lists to store fitness statistics for each generation
    lowest_fitnesses = []
    average_fitnesses = []
    highest_fitnesses = []

    for gen in range(MAX_ITERATIONS):
        fitness_scores = []
        networks = []
        for pop in population:
            fit=fitness(pop)
            fitness_scores.append(fit)

        # Calculate fitness statistics for this generation
        lowest_fitness = min(fitness_scores)
        average_fitness = sum([fs for fs in fitness_scores]) / len(fitness_scores)
        highest_fitness = max(fitness_scores)

        # Append fitness statistics to their respective lists
        lowest_fitnesses.append(lowest_fitness)
        average_fitnesses.append(average_fitness)
        highest_fitnesses.append(highest_fitness)

        best_fitness = max(fitness_scores)

        if best_fitness > max_fitness_gen:
            max_fitness_gen = best_fitness
            consecutive_generations = 0
        else:
            consecutive_generations += 1
        if consecutive_generations == int(0.05 * POPULATION_SIZE):
            MUTATION_RATE = 1
            probability = 1
        best_weightes = population[fitness_scores.index(best_fitness)]
        print(f"Generation {gen + 1}: Best Fitness = {best_fitness}")
        zipped_population = [(pop, fit) for pop, fit in zip(population, fitness_scores)]
        zipped_population = sorted(zipped_population, key=lambda x: x[1], reverse=True)

        offspring = []
        for i in range(POPULATION_SIZE):
            if i < 0.1 * POPULATION_SIZE:
                offspring.append(best_weightes)
            else:
                parent1, parent2 = selection(zipped_population)
                child1, child2 = crossover(parent1, parent2)
                child3 = mutation(child1)
                child4 = mutation(child2)
                offspring.append(child3)
                offspring.append(child4)
                i+=1
        population = offspring

        if consecutive_generations % int(0.1 * POPULATION_SIZE) == 2:
            probability = 0.4
            MUTATION_RATE = 0.2

        # if gen > threshold and consecutive_generations >= threshold:
        #     print("Reached a local maximum. Stopping the genetic algorithm.")
        #     break

    return NeuralNetwork(best_weightes), lowest_fitnesses, average_fitnesses, highest_fitnesses

def save_neural_network(nn):
    # Export the object to a JSON file
    with open('wnet1.pkl', 'wb') as file:
        pickle.dump(nn, file)

def main():
    best_weights, lowest_fitnesses, average_fitnesses, highest_fitnesses = genetic_algo()
    save_neural_network(best_weights)

if __name__ == '__main__':
    main()