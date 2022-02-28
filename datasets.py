import h5py
import numpy as np

def shuffle(a,b):
    """Shuffle two numpy lists in the same way"""
    length = max(len(a),len(b))
    indexes = np.arange(length)
    np.random.shuffle(indexes)
    a = a[indexes]
    b = b[indexes]
    return a, b

def one_hot_encoding(vec, n):
    """One-hot encoding of the input vector vec with n classes"""
    result = np.zeros((len(vec),n))
    result[np.arange(len(vec)), vec] = 1
    return result

def generate_XOR_dataset(training_set_size):
    """Generates a simple dataset for the XOR problem"""
    input_size  = 2
    output_size = 1
    X_train = np.zeros((training_set_size, input_size))
    X_train[0::4,:] = [0,0] # every fourth entry gets the same input config
    X_train[1::4,:] = [0,1]
    X_train[2::4,:] = [1,0]
    X_train[3::4,:] = [1,1]
    y_train = np.zeros((training_set_size, output_size))
    #y_train[:,9] = 2
    y_train[1::4,0] = 1 # 0 XOR 1 = 1
    y_train[2::4,0] = 1 # 1 XOR 0 = 1
    return X_train, y_train

def load_mnist(path="./data/mnist_dataset.h5"):
    """Load the MNIST dataset from a .h5 file"""
    print("Loading MNIST dataset...")
    n_classes = 10
    float_type = np.float32
    int_type   = np.int16
    pixels = 28*28
    with h5py.File(path) as f:
        X_train = f["train_data"][:].astype(float_type)
        X_test = f["test_data"][:].astype(float_type)

        y_train = f["train_labels"][:].astype(int_type)
        y_test = f["test_labels"][:].astype(int_type)

    X_train = X_train / 255.0
    X_test  = X_test  / 255.0

    y_train = one_hot_encoding(y_train,n_classes).astype(float_type)
    y_test  = one_hot_encoding(y_test, n_classes).astype(float_type)

    X_train = X_train.reshape((X_train.shape[0],pixels))
    X_test  = X_test .reshape((X_test .shape[0],pixels))

    return X_train, y_train, X_test, y_test



def load_openml(training_size, test_size, dataset_name):
    """Load a dataset from the openml website"""
    print(f"Loading dataset: {dataset_name}")
    from sklearn.datasets import fetch_openml

    float_type = np.float32
    mnist = fetch_openml(name=dataset_name, data_home="data") # https://www.openml.org/d/554
    # normalize pixel values to be between 0 and 1
    X = mnist['data']/255.0
    # convert each row to a vector with exactly one 1 in the correct place
    targets = mnist['target'].astype(np.dtype(np.int16))
    y = one_hot_encoding(targets,10)
    X_train = X[:training_size].astype(float_type)
    y_train = y[:training_size].astype(float_type)

    X_test = X[-test_size:].astype(float_type)
    y_test = y[-test_size:].astype(float_type)
    print(f"Finished loading {dataset_name}")
    return X_train, y_train, X_test, y_test
