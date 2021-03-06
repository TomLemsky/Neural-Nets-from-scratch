import numpy as np
import argparse

from datasets import load_mnist

class FullyConnectedLayer:
    """ A fully connected layer"""
    previous_layer = None
    def __init__(self, neurons, **kwargs):
        """ neurons: number of neurons in this layer"""
        self.size = neurons

    def initialize_weights(self):
        """ Initialize weights and biases"""
        self.input_size = self.previous_layer.size
        self.W = np.random.randn(self.size, self.input_size) * np.sqrt(1 / (self.size + self.input_size))
        #np.random.normal(0, 1, size=(self.size, self.input_size)) #.astype(np.float32)
        # biases slightly above 0 to prevent dead ReLUs
        self.b = 0.1 * np.ones((self.size, 1))

    def forward(self, x):
        """ Compute the output of the layer given its input"""
        self.x = x
        self.p = np.dot(self.W, x) + self.b
        return self.p

    def backward(self, dC_dp):
        """Given the gradient of the cost function w.r.t. the layer's input,
        Compute the gradient w.r.t its inputs.
        At the same time calculate and store the gradient w.r.t its weights"""
        # gradient w.r.t. outputs
        self.dC_dp = dC_dp

        # gradient w.r.t. weights and biases
        self.dC_dW = np.dot(dC_dp, self.x.T) #
        self.dC_db = dC_dp # np.sum(dC_dp, axis=1)

        # gradient w.r.t. inputs
        self.DC_dx = np.dot(self.W.T, dC_dp)

        return self.DC_dx

    def weight_update(self, learning_rate, batch_size):
        """ update weights and biases """
        #print("x", self.p)
        #print("W", self.W)
        self.W -= (learning_rate / batch_size) * self.dC_dW
        self.b -= (learning_rate / batch_size) * ( self.dC_db @ np.ones((self.dC_db.shape[1],1)))

class InputLayer:
    """ A fully connected layer"""
    previous_layer = None
    def __init__(self, neurons, **kwargs):
        """ neurons: number of neurons in this layer"""
        self.size = neurons

    def initialize_weights(self):
        """ Initialize weights and biases"""
        pass

    def forward(self, x):
        """ Compute the output of the layer given its input"""
        self.x = x
        return self.x

    def backward(self, dC_dp):
        """Given the gradient of the cost function w.r.t. the layer's input,
        Compute the gradient w.r.t it's inputs.
        At the same time calculate and store the gradient w.r.t it's weights"""

        return dC_dp

    def weight_update(self, learning_rate, batch_size):
        """ update weights and biases """
        pass

class ReLuLayer:
    """ ReLu activation layer"""
    previous_layer = None
    def __init__(self, **kwargs):
        pass

    def initialize_weights(self):
        """ Initialize weights and biases"""
        self.input_size = self.previous_layer.size
        self.size = self.previous_layer.size

    def forward(self, x):
        """ Compute the output of the layer given its input"""
        self.x = x
        self.p = np.maximum(0.0,x)
        return self.p

    def backward(self, dC_dp):
        """Given the gradient of the cost function w.r.t. the layer's input,
        Compute the gradient w.r.t it's inputs."""
        # gradient w.r.t. inputs
        self.Dp_dx = 1.0*(self.p > 0.0)
        self.DC_dx = np.multiply(dC_dp, self.Dp_dx)
        return self.DC_dx

    def weight_update(self, learning_rate, batch_size):
        pass

class LeakyReLuLayer:
    """ ReLu activation layer"""
    previous_layer = None
    def __init__(self, slope=0.01, **kwargs):
        self.slope = slope
        pass

    def initialize_weights(self):
        """ Initialize weights and biases"""
        self.input_size = self.previous_layer.size
        self.size = self.previous_layer.size

    def forward(self, x):
        """ Compute the output of the layer given its input"""
        self.x = x
        #print(x)
        self.p = np.maximum(0.0,x) + (x <= 0.0) * (self.slope * x)
        return self.p

    def backward(self, dC_dp):
        """Given the gradient of the cost function w.r.t. the layer's input,
        Compute the gradient w.r.t it's inputs."""
        # gradient w.r.t. inputs
        self.Dp_dx = 1.0*(self.p > 0.0) + self.slope * (self.p <= 0.0)
        self.DC_dx = np.multiply(dC_dp, self.Dp_dx)
        return self.DC_dx

    def weight_update(self, learning_rate, batch_size):
        pass

class AbsLayer:
    """ ReLu activation layer"""
    previous_layer = None
    def __init__(self, **kwargs):
        pass

    def initialize_weights(self):
        """ Initialize weights and biases"""
        self.input_size = self.previous_layer.size
        self.size = self.previous_layer.size

    def forward(self, x):
        """ Compute the output of the layer given its input"""
        self.x = x
        self.p = np.abs(x)
        return self.p

    def backward(self, dC_dp):
        """Given the gradient of the cost function w.r.t. the layer's input,
        Compute the gradient w.r.t it's inputs."""
        # gradient w.r.t. inputs
        self.Dp_dx = 1.0*(np.sign(self.p))
        self.DC_dx = np.multiply(dC_dp, self.Dp_dx)
        return self.DC_dx

    def weight_update(self, learning_rate, batch_size):
        pass


class SigmoidLayer:
    """ Sigmoid activation layer"""
    previous_layer = None
    def __init__(self, **kwargs):
        pass

    def initialize_weights(self):
        """ Initialize weights and biases"""
        self.input_size = self.previous_layer.size
        self.size = self.previous_layer.size

    def forward(self, x):
        """ Compute the output of the layer given its input"""
        self.x = x
        self.p = 1/(1+np.exp(-x))
        return self.p

    def backward(self, dC_dp):
        """Given the gradient of the cost function w.r.t. the layer's input,
        Compute the gradient w.r.t it's inputs."""
        # gradient w.r.t. inputs
        self.Dp_dx = self.p * (1 - self.p)
        self.DC_dx = np.multiply(dC_dp, self.Dp_dx)
        return self.DC_dx

    def weight_update(self, learning_rate, batch_size):
        pass

class TanhLayer:
    """ Sigmoid activation layer"""
    previous_layer = None
    def __init__(self, **kwargs):
        pass

    def initialize_weights(self):
        """ Initialize weights and biases"""
        self.input_size = self.previous_layer.size
        self.size = self.previous_layer.size

    def forward(self, x):
        """ Compute the output of the layer given its input"""
        self.x = x
        self.p = (np.exp(2*x)-1)/(np.exp(2*x)+1)
        return self.p

    def backward(self, dC_dp):
        """Given the gradient of the cost function w.r.t. the layer's input,
        Compute the gradient w.r.t it's inputs."""
        # gradient w.r.t. inputs
        self.Dp_dx = 1 - self.p**2
        self.DC_dx = np.multiply(dC_dp, self.Dp_dx)
        return self.DC_dx

    def weight_update(self, learning_rate, batch_size):
        pass


class MeanSquaredError:
    """ Mean squared error cost function"""
    def __init__(self):
        pass

    def forward(self, a, y):
        """ Return the value of the cost function"""
        d = (a - y).ravel()
        return 1/2 * (np.dot(d,d))

    def backward(self, a, y):
        """ Return the gradient of the cost function"""
        return (a - y)

class FeedForwardNet:
    def __init__(self):
        self.layers = []

    def add(self, new_layer):
        """ Add a new layer to the network """
        if len(self.layers)>0:
            new_layer.previous_layer = self.layers[-1]
        self.layers.append(new_layer)
        new_layer.initialize_weights()

    def forward(self, x):
        a = x
        for l in self.layers:
            a = l.forward(a)
        return a

    def train(self, X_train, y_train, learning_rate, epochs, batch_size=10, cost_function=None):
        acc_no = 100 # number to average the training accuracy over
        if not cost_function:
            cost_function = MeanSquaredError()
        for e in range(epochs):
            result = []
            print(f"Epoch {e}")
            for b in range(len(y_train)//batch_size):
                #for i in range(b*batch_size,(b+1)*batch_size):
                x = X_train[b*batch_size:(b+1)*batch_size].T #.reshape((len(X_train[i]),1))
                y = y_train[b*batch_size:(b+1)*batch_size].T #.reshape((len(y_train[i]),1))
                # forward pass through whole network
                a = self.forward(x)
                #print(a, y)
                C = cost_function.forward(a,y)
                #print(f"Cost: {C}")
                dC_da = cost_function.backward(a,y)
                for l in self.layers[::-1]:
                    dC_da = l.backward(dC_da)
                    l.weight_update(learning_rate, batch_size)
                result.append(np.sum(np.equal(np.argmax(a,axis=0),np.argmax(y,axis=0)))) # Count correct guesses
                if (b-1)%acc_no==0:
                    training_accuracy = sum(result[-acc_no:])/(batch_size * min(acc_no,len(result)))
                    print(f"Training accuracy over the last {acc_no} batches: {training_accuracy}")

    def test(self, X_test, y_test):
        correct_guesses = 0
        for i in range(len(y_test)):
            x = X_test[i].reshape((len(X_test[i]),1))
            y = y_test[i].reshape((len(y_test[i]),1))
            # forward pass through whole network
            a = self.forward(x)
            if np.argmax(a)==np.argmax(y):
                correct_guesses += 1
        return correct_guesses/len(y_test)

    def gradient_check(self, x, y, cost_function=None):
        """ Check gradient from backpropagation empirically """
        x = x.reshape((len(x),1))
        y = y.reshape((len(y),1))
        epsilon = 0.0000001
        if not cost_function:
            cost_function = MeanSquaredError()
        # forward propagate through network
        a = self.forward(x)
        C_old = cost_function.forward(a,y)
        # backward backpropagation
        dC_da = cost_function.backward(a,y)
        for l in self.layers[::-1]:
            dC_da = l.backward(dC_da)

        numerical_gradient = []
        backprop_gradient  = []
        weights = []

        for l in self.layers:
            if not hasattr(l, 'W'):
                continue
            print(type(l).__name__)
            backprop_gradient += list((l.dC_dW).ravel())
            for i in range(l.W.shape[0]):
                for j in range(l.W.shape[1]):
                    # change variable "infinitessimally"
                    w_orig = l.W[i,j]
                    l.W[i,j] = w_orig + epsilon
                    # propagate through net
                    a = self.forward(x)
                    C_plus = cost_function.forward(a,y)
                    # change variable to minus epsilon
                    l.W[i,j] = w_orig - epsilon
                    # propagate through net
                    a = self.forward(x)
                    C_minus = cost_function.forward(a,y)
                    # change variable back
                    l.W[i,j] = w_orig
                    # calculate Derivative
                    derivative = (C_plus - C_minus) / (2*epsilon)
                    numerical_gradient.append(float(derivative))

                    weights.append(l.W[i,j])

        biggest_difference = 0
        for i in range(len(numerical_gradient)):
            n = numerical_gradient[i]
            b = backprop_gradient[i]
            diff = abs(n-b)
            if(diff>biggest_difference):
                biggest_difference = diff
            print(i, weights[i], n, b)
        print(biggest_difference)

def one_hot_encoding(vec, n):
    result = np.zeros((len(vec),n))
    result[np.arange(len(vec)), vec] = 1
    return result

def main(args):
    learning_rate = args.lr #0.02
    epochs = args.epochs #2
    batch_size = args.batch_size # 10

    X_train, y_train, X_test, y_test = load_mnist()
    input_size = 28*28
    n_classes  = 10

    model = FeedForwardNet()
    model.add(InputLayer(input_size))
    model.add(FullyConnectedLayer(100))
    model.add(ReLuLayer())
    model.add(FullyConnectedLayer(50))
    model.add(ReLuLayer())
    model.add(FullyConnectedLayer(n_classes))

    # print("Example forward pass with randomly initialized weights: ")
    # a=model.forward(X_train[:1].T)
    # print(a)
    # print(a.shape)

    model.train(X_train, y_train, learning_rate, epochs, batch_size=batch_size)
    print(f"\nFinished training! \n")

    output = model.forward(X_train[:1].T)
    prediction   = np.argmax(output)
    ground_truth = np.argmax(y_train[:1].T)
    print(f"Example prediction: {prediction} (Ground truth: {ground_truth}))")

    print('Accuracy on training set: ', model.test(X_train, y_train))
    print('Accuracy on test set    : ', model.test(X_test, y_test))

    #model.gradient_check(X_train[4],y_train[4])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run two-layer fully-connected network.')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.1, type=float, help='Momentum for gradient descent with momentum')
    parser.add_argument('--epochs', default=5, type=int, help='How many epochs to train for')
    parser.add_argument('--batch_size', default=10, type=int, help='Batch size during training')

    args = parser.parse_args()
    main(args)
