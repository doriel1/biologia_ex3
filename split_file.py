# import json
#
# # Reading the object from a file
# with open('best_weights.json', 'r') as file:
#     best_weights = json.load(file)

def split_file(input_file, training_file, test_file, training_lines=15000):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    train_lines = lines[:training_lines]
    test_lines = lines[training_lines:]

    with open(training_file, 'w') as file:
        file.writelines(train_lines)

    with open(test_file, 'w') as file:
        file.writelines(test_lines)



def main():
    # Example usage
    split_file('nn1.txt', 'training1.txt', 'test1.txt', training_lines=15000)


if __name__ == '__main__':
    main()