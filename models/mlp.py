import torch
import torch.nn as nn
import numpy as np

class BlackBoxModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=10):
        super(BlackBoxModel, self).__init__()

        self.name = "dnn"

        # First fully connected layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Second fully connected layer (hidden layer)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layer
        self.fc3 = nn.Linear(hidden_dim, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

    def predict(self, x):
        x = torch.FloatTensor(x)
        return (self(x).reshape(-1) > 0.5).float().detach().numpy()

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x)
            probs = self(x_t).reshape(-1, 1).cpu().numpy()
            return np.concatenate([1 - probs, probs], axis=1)



class MNISTModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
