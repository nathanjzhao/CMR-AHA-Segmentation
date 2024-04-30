import torch
import torch.nn as nn

class BlankDropout(nn.Module):
    """A blank dropout layer for validation or model testing without dropout"""
    def __init__(self, p=0.0):
        super(BlankDropout, self).__init__()

    def forward(self, x):
        return x

class RegressionCNN(nn.Module):
    """Regression CNN model with options for testing various models"""
    def __init__(self, num_features, relu, num_classes):
        super(RegressionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, num_features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_features, 2 * num_features, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(5000 * num_features , 2 * num_features * 128)
        
        # Embedding layer 
        self.embedding = nn.Embedding(3, 1) # (embedding categories, values embedded)
        
        self.fc2 = nn.Linear(2 * num_features * 128 + 1, num_classes) # fc1 + values embedded
        
        if relu:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Sigmoid()

    def forward(self, x, categorical_data, num_features, dropout=None):
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = BlankDropout()

        # Model architecture
        x = self.activation(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = self.activation(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.activation(self.fc1(x))
        x = x.view(-1, 2 * num_features * 128)

        # Apply the embedding layer to the categorical data
        categorical_embedding = self.embedding(categorical_data)
        
        # Concatenate the categorical embedding with the output of the first fully connected layer
        combined_data = torch.cat((x, categorical_embedding), dim=1)
        x = self.fc2(combined_data) # NO DROPOUT BEFORE LAST LAYER

        return x