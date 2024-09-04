import torch 
from torch import nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

class Neural_Network(nn.Module):
    def __init__(self):
        super(Neural_Network, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        return self.network(inputs)

class Scaler:
    def __init__(self):
        self.scaler_y = MinMaxScaler()
        self.scaler_x = MinMaxScaler()

    def scale_x(self, inputs):
        return self.scaler_x.fit_transform(inputs)
    
    def scale_y(self, inputs):
        return self.scaler_y.fit_transform(inputs.reshape(-1, 1)).ravel()

class System:
    def __init__(self, network, epochs, learning_rate):
        self.epochs = epochs
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def calculate_accuracy(self, X, y):
        with torch.no_grad():
            output = self.network(X)
            _, predicted = torch.max(output, 1)
            accuracy = (predicted == y).float().mean()
        return accuracy 
    
    def train(self, train_loader, X_test, y_test):
        run = True
        while run == True:
            for epoch in range(self.epochs):
                for X_train_batch, y_train_batch in train_loader:
                    output = self.network(X_train_batch)
                    loss = self.criterion(output, y_train_batch)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                train_accuracy = self.calculate_accuracy(X_train_batch, y_train_batch)
                test_accuracy = self.calculate_accuracy(X_test, y_test)

                print(f"Epoch: {epoch + 1}/{self.epochs} - Train Accuracy: {train_accuracy:.4f} - Test Accuracy: {test_accuracy:.4f}")

                if train_accuracy > 0.900 and test_accuracy > 0.900:
                    run = False
                    
def convert_tensor(inputs):
    return torch.tensor(inputs, dtype=torch.float32)

data = pd.read_csv("diabetes_detection/diabetes.csv")
X_train = data.drop(columns=["Outcome"]).values
y_train = data["Outcome"].values

scaler = Scaler()
X_train = scaler.scale_x(X_train)
y_train = scaler.scale_y(y_train)

X_train = convert_tensor(X_train)
y_train = convert_tensor(y_train).long()

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

network = Neural_Network()
system = System(network, 1, 0.001)
system.train(train_loader, X_train, y_train)
