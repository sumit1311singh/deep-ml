import numpy as np

class NeuralNetwork:
    '''
    Build a neural network from scratch using only NumPy.
    Required architecture: 784 → 128 (ReLU) → 10 (Softmax)
    '''
    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01):
        '''
        Initialize network parameters.
        Use small random initialization (e.g., Xavier/He initialization).
        '''
        self.lr = lr
        # TODO: Initialize weights and biases
        # Hint: Use np.random.randn() with proper scaling
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (hidden_size + output_size))
        self.b2 = np.zeros(output_size)

        
    
    def forward(self, X):
        '''
        Forward pass through the network.
        
        Args:
            X: Input batch, shape (N, 784)
        
        Returns:
            probs: Class probabilities, shape (N, 10)
        
        Must cache intermediate values for backward pass!
        '''
        # TODO: Implement forward pass
        # Layer 1: X @ W1 + b1, then ReLU
        # Layer 2: hidden @ W2 + b2, then Softmax
        layer_1 = X @ self.W1 + self.b1
        act_layer_1 = np.maximum(layer_1, 0)

        layer_2 = act_layer_1 @ self.W2 + self.b2 # logits
        
        shifted_logits = layer_2 - np.max(layer_2, axis=1, keepdims=True)
        exp_layer_2 = np.exp(shifted_logits)
        act_layer_2 = exp_layer_2 / np.sum(exp_layer_2, axis=1, keepdims=True)

        self.cache = {
            "X": X,
            "layer_1": layer_1,
            "act_layer_1": act_layer_1,
            "layer_2": layer_2,
            "act_layer_2": act_layer_2
        }

        return act_layer_2
    
    def backward(self, X, y, probs):
        '''
        Backward pass - compute gradients for all parameters.
        
        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,)
            probs: Predicted probabilities from forward pass, shape (N, 10)
        
        Returns:
            loss: Scalar cross-entropy loss
        
        Must update self.W1, self.b1, self.W2, self.b2 using computed gradients!
        '''
        # TODO: Implement backward pass
        # 1. Compute loss
        # 2. Compute gradient of loss w.r.t. softmax output
        # 3. Backprop through linear layer 2
        # 4. Backprop through ReLU
        # 5. Backprop through linear layer 1
        # 6. Update all parameters using gradients
        ce_loss = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-8))

        dZ2 = probs.copy()
        dZ2[np.arange(len(y)), y] -= 1
        dZ2 /= len(y)

        dW2 = self.cache['act_layer_1'].T @ dZ2
        db2 = np.sum(dZ2, axis=0)
        dA1 = dZ2 @ self.W2.T

        dZ1 = dA1 * (self.cache['layer_1'] > 0)

        dW1 = self.cache['X'].T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        return ce_loss

    
    def train_step(self, X, y):
        '''
        Complete training step: forward + backward + update.
        
        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,)
        
        Returns:
            loss: Scalar loss value
        '''      

        probs = self.forward(X)
        loss = self.backward(X, y, probs)
        return loss

    def predict(self, X):
        '''
        Predict class labels.
        
        Args:
            X: Input batch, shape (N, 784)
        
        Returns:
            predictions: Predicted class labels, shape (N,)
        '''
        probs = self.forward(X)
        return np.argmax(probs, axis=1)