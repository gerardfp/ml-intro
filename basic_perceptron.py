import random

def perceptron(inputs, weights, threshold):
    weighted_sum = sum(x * w for x, w in zip(inputs, weights))
    return 1 if weighted_sum >= threshold else 0

def train_perceptron(data, learning_rate=0.1, max_iter=1000):

    weights = [random.random() for _ in range(len(data[0][0]))]
    threshold = random.random()
   
    for _ in range(max_iter):

        num_errors = 0
        
        for inputs, desired_output in data:
            output = perceptron(inputs, weights, threshold)
            error = desired_output - output
            
            if error != 0:
                num_errors += 1
                for i in range(len(weights)):
                    weights[i] += learning_rate * error * inputs[i]
                threshold -= learning_rate * error
        
        if num_errors == 0:
            break
    
    return weights, threshold

print("=== AND ===")
and_data = [
 [[0, 0],  0],
 [[0, 1],  0],
 [[1, 0],  0], 
 [[1, 1],  1]
]

and_weights, and_threshold = train_perceptron(and_data)
print("Weights:", and_weights)
print("Threshold:", and_threshold) 

print(perceptron((0,0), and_weights, and_threshold))
print(perceptron((0,1), and_weights, and_threshold))
print(perceptron((1,0), and_weights, and_threshold))
print(perceptron((1,1), and_weights, and_threshold))

print("=== NOT ===")
not_data = [
 [[0],  1],
 [[1],  0],
]
not_weights, not_threshold = train_perceptron(not_data)

print("Weights:", not_weights)
print("Threshold:", not_threshold) 

print(perceptron([0], not_weights, not_threshold))
print(perceptron([1], not_weights, not_threshold))
