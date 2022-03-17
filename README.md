# Neural Networks from scratch

These are implementations of neural networks I implemented completly from scratch. Only the numpy library is used for matrix multiplications. The correctness of the gradient computation is verified with numerical gradient checking.

- functional_net.py is a hardcoded two-layer fully-connected network. It is implemented with a completely functional approach, with the model's weights as an input argument to every function. This makes this version pretty slow to run.

- net.py is a modular feedforward network that different layers can be added to. It uses an object-oriented approach to store weights and compute gradients in the layer objects. To the outside, the models have a Keras-like API to train and test them.

## Requirements

- Python3
- Numpy (for matrix math)
- h5py (to load data)

## Command-line Usage

Command-line usage to train functional_net.py on the MNIST dataset:

```
usage: functional_net.py [-h] [--lr LR] [--momentum MOMENTUM]
                         [--hidden_neurons HIDDEN_NEURONS] [--epochs EPOCHS]
                         [--batch_size BATCH_SIZE]

Run two-layer fully-connected network.

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Learning rate
  --momentum MOMENTUM   Momentum for gradient descent with momentum
  --hidden_neurons HIDDEN_NEURONS
                        Number of hidden neurons
  --epochs EPOCHS       How many epochs to train for
  --batch_size BATCH_SIZE
                        Batch size during training

```

Command-line usage to train net.py on the MNIST dataset:

```
usage: net.py [-h] [--lr LR] [--momentum MOMENTUM] [--epochs EPOCHS] [--batch_size BATCH_SIZE]

Run two-layer fully-connected network.

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Learning rate
  --momentum MOMENTUM   Momentum for gradient descent with momentum
  --epochs EPOCHS       How many epochs to train for
  --batch_size BATCH_SIZE
                        Batch size during training
```

## Usage as library

```
from net import FeedForwardNet, InputLayer, FullyConnectedLayer, ReLuLayer

# load_data
X_train, y_train, X_test, y_test = load_data()
input_size = 28*28
n_classes  = 10

# Make a FeedForwardNet object and add layers to it
model = FeedForwardNet()
model.add(InputLayer(input_size))
model.add(FullyConnectedLayer(100))
model.add(ReLuLayer())
model.add(FullyConnectedLayer(50))
model.add(ReLuLayer())
model.add(FullyConnectedLayer(n_classes))

# Train the model on the training data
learning_rate = 0.02
epochs = 2
batch_size = 10
model.train(X_train, y_train, learning_rate, epochs, batch_size=batch_size)

# Calculate the accuracy on the test data
test_accuracy = model.test(X_test, y_test)
print('Accuracy on test set    : ', test_accuracy)

# Get an example output
output = model.forward(X_train[:1].T)
prediction   = np.argmax(output)
ground_truth = np.argmax(y_train[:1].T)
print(f"Example prediction: {prediction} (Ground truth: {ground_truth}))")
```
