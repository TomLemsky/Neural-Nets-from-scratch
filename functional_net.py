import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

def compute_gradients(weight_vector, x, y, output_only=False, cost_only=False):
    W_1, b_1, W_2, b_2 = vector_to_weights(weight_vector)
    # activation function (sigmoid/logistic)
    f  = lambda xx: 1/(1+np.exp(-xx)) # For ReLU: np.maximum(0,xx)
    # yy is f(xx) (faster to compute that way)
    df = lambda xx,yy: yy * (1-yy) #For ReLU: 1*(yy > 0) # 1 if xx>0 else 0

    # input vector
    a_0 = x
    # first layer preactivation
    p_1 = a_0 @ W_1 + b_1
    # first layer activation
    a_1 = f(p_1)

    # second layer preactivation
    p_2 = a_1 @ W_2 + b_2
    # second layer activation
    a_2 = f(p_2)

    if output_only:
        return a_2

    # Cost
    a_y = (a_2 - y).ravel()
    C = 1/2 * (np.dot(a_y,a_y))


    if cost_only:
        return C


    # Derivative of the cost w.r.t. the second layer activation
    dC_da2 = (a_2 - y)
    # Derivative of the cost w.r.t. the second layer preactivation (elementwise multiplication)
    dC_dp2 = np.multiply(dC_da2, df(p_2,a_2))
    # Derivative of the cost w.r.t. the second layer biases
    dC_db2 = dC_dp2
    # Derivative of the cost w.r.t. the second layer weights
    dC_dW2 = a_1.T @ dC_dp2

    # Derivative of the cost w.r.t. the first layer activation
    dC_da1 = dC_dp2 @ W_2.T
    # Derivative of the cost w.r.t. the first layer preactivation (elementwise multiplication)
    dC_dp1 = np.multiply(dC_da1, df(p_1,a_1))

    # Derivative of the cost w.r.t. the first layer biases
    dC_db1 = dC_dp1
    # Derivative of the cost w.r.t. the first layer weights
    dC_dW1 = a_0.T @ dC_dp1

    return weights_to_vector(dC_dW1, dC_db1, dC_dW2, dC_db2), a_2, C

def train_model(epochs, batch_size , learning_schedule, momentum_schedule, X_train, y_train, X_val, y_val):
    if type(learning_schedule)==float or type(learning_schedule)==int: # extend a single float to a list
        learning_schedule = [learning_schedule]*epochs
    elif len(learning_schedule) < epochs: # extend schedule if too short
        learning_schedule = list(learning_schedule) + [learning_schedule[-1]]*epochs

    if type(momentum_schedule)==float or type(momentum_schedule)==int: # extend a single float to a list
        momentum_schedule = [momentum_schedule]*epochs
    elif len(learning_schedule) < epochs: # extend schedule if too short
        momentum_schedule = list(momentum_schedule) + [momentum_schedule[-1]]*epochs

    # first layer weights and biases
    W_1 = np.random.normal(0, 1, size=(input_size, hidden_size)).astype(float_type)
    b_1 = np.zeros((1, hidden_size)).astype(float_type)
    #b_1 = np.random.normal(0, 1, size=(1, hidden_size)).astype(float_type)

    # second layer weights and biases
    W_2 = np.random.normal(0, 1, size=(hidden_size, output_size)).astype(float_type)
    b_2 = np.zeros((1, output_size)).astype(float_type)
    #b_2 = np.random.normal(0, 1, size=(1, output_size)).astype(float_type)

    weight_vector = weights_to_vector(W_1, b_1, W_2, b_2).astype(float_type)

    accuracy = test_model(weight_vector, X_val, y_val)
    print(accuracy)

    #learning_schedule = [0.5]*4 + [0.1]*8 + [0.02]*10 + [0.005]*10 + [0.001]*8 + [0.0005]*50
    #mu_schedule = [0]*100

    v = np.zeros(len(weight_vector)) # velocity
    #mu = 0.5 # momentum/ friction hyperparameter
    for e in range(0,epochs):
        X_train, y_train = shuffle(X_train, y_train)
        learning_rate = learning_schedule[e]
        mu = momentum_schedule[e]
        result = []
        for b in range(len(y_train)//batch_size):
            accumulated_gradients = np.zeros(weight_vector.shape)
            for i in range(b*batch_size,(b+1)*batch_size):
                gradient_vector, output, cost = compute_gradients(weight_vector, X_train[i].reshape((1, input_size)), y_train[i])
                accumulated_gradients += gradient_vector
                result.append(np.argmax(output)==np.argmax(y_train[i]))
                if (i+1)%1000==0:
                    training_accuracy = sum(result[-1000:])/1000
                    print(f"Training accuracy: {training_accuracy} (Trained on {i+1} samples)")
                if (i+1)%10000==0:
                    test_accuracy = test_model(weight_vector, X_val[:1000], y_val[:1000])
                    print(f"Test accuracy: {test_accuracy} (Epoch {e+1})")
                    print(f"Learning rate: {learning_rate}; Momentum: {mu}")
            gradient = accumulated_gradients
            # momentum update
            v = np.multiply(mu, v) - np.multiply(learning_rate, gradient)
            weight_vector += v

    return weight_vector

def test_model(weight_vector,X,y):
    correct_guesses = 0
    for i in range(len(y)):
        output = compute_gradients(weight_vector, X[i], None, output_only=True)
        if np.argmax(output)==np.argmax(y[i]):
            correct_guesses += 1
    return correct_guesses/len(y)

def weights_to_vector(W_1, b_1, W_2, b_2):
    return np.hstack((W_1.ravel(),b_1.ravel(),W_2.ravel(),b_2.ravel()))

def vector_to_weights(weight_vector):
    shapes = [(input_size, hidden_size),
              (1, hidden_size),
              (hidden_size, output_size),
              (1, output_size)]
    result = []
    index = 0
    for s in shapes:
        vec_part = weight_vector[index:index+s[0]*s[1]]
        result.append(vec_part.reshape(s))
        index += s[0]*s[1]

    return result

# for gradient checking
def numerical_gradient(weight_vector, x, y):
    epsilon = 0.000000001
    gradient = np.zeros(weight_vector.size)
    for i in range(weight_vector.size):
        if(i%25==0):
            print(i)
        direction = np.zeros(weight_vector.size)
        direction[i] = 1
        C_minus = compute_gradients(weight_vector - direction * epsilon, x, y, cost_only=True)
        C_plus  = compute_gradients(weight_vector + direction * epsilon, x, y, cost_only=True)
        gradient[i] = (C_plus - C_minus)/(2*epsilon)
    return gradient

def gradient_checking(weight_vector, x, y):
    gradient_vector, output, cost = compute_gradients(weight_vector, x, y, output_only=False)
    numerical_gradient = numerical_gradient(weight_vector, x, y)
    difference = gradient_vector - numerical_gradient
    for i in range(gradient_vector.size): # input_size*hidden_size + hidden_size
        print(gradient_vector[i], numerical_gradient[i])
    maximum_error = np.max(difference)
    print(maximum_error)
    return maximum_error <= 0.01

def one_hot_encoding(vec, n):
    result = np.zeros((len(vec),n))
    result[np.arange(len(vec)), vec] = 1
    return result

def load_data(training_size, test_size, dataset_name):
    mnist = fetch_openml(name=dataset_name, data_home="data") # https://www.openml.org/d/554
    # normalize pixel values to be between 0 and 1
    X = mnist['data']/255
    # convert each row to a vector with exactly one 1 in the correct place
    targets = mnist['target'].astype(np.dtype(np.int16))
    y = one_hot_encoding(targets,output_size)
    X_train = X[:training_size].astype(float_type)
    y_train = y[:training_size].astype(float_type)

    X_test = X[-test_size:].astype(float_type)
    y_test = y[-test_size:].astype(float_type)
    return X_train, y_train, X_test, y_test

# single-precsion float for faster computation
float_type = np.float32

epochs = 50
batch_size = 20
learning_rate = [0.05]*20 + [0.01]*20 + [0.005]*20
momentum = 0


input_size = 28*28
hidden_size = 100
output_size = 10

training_size = 60000
test_size     = 10000
dataset_name = "mnist_784"

print(f"Loading dataset {dataset_name}")
X_train, y_train, X_test, y_test = load_data(training_size, test_size, dataset_name)
print(f"Finished loading {dataset_name}")

weight_vector = train_model(epochs, batch_size, learning_rate, momentum, X_train, y_train, X_test, y_test)
print("Finished training!")

print("Type: ", weight_vector.dtype)

test_accuracy = test_model(weight_vector, X_test, y_test)
print(f"Test accuracy on full test set: {test_accuracy} (After {epochs} Epochs)")
