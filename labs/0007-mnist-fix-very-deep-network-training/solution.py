import torch
import torch.nn as nn

class DeepNetwork(nn.Module):
    '''
    A very deep network that trains poorly!
    
    This 30-layer network achieves only ~20% accuracy due to:
    - Vanishing gradients
    - No skip connections
    - No normalization
    - Poor gradient flow
    
    Your task: Fix this network to achieve 90%+ accuracy while keeping ≥30 layers!
    
    You can:
    - Add skip/residual connections (ResNet style)
    - Add dense connections (DenseNet style)
    - Add batch normalization or layer normalization
    - Change activation functions
    - Reorganize the architecture
    - Add any architectural improvements you know
    
    You CANNOT:
    - Reduce the number of layers below 30
    '''
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()

        # Input projection
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # 30 hidden layers
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(30)
        ])
        
        # One normalization per hidden layer (pre-norm residual style)
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) for _ in range(30)
        ])
        self.final_norm = nn.BatchNorm1d(hidden_size)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_classes)
        
        # Activation
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Input projection
        x = self.activation(self.input_layer(x))
        
        # 30 layers, each with a skip connection: x = x + Linear(ReLU(LayerNorm(x)))
        for layer, norm in zip(self.layers, self.norms):
            x = x + 0.5 * layer(self.activation(norm(x)))
        
        # Output
        x = self.output_layer(self.activation(self.final_norm(x)))
        
        return x
