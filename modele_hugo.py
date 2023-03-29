
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import dataset_test 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""""
# Define the LSTM model
class EmotionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(EmotionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
"""
# Set hyperparameters and instantiate the model, loss function, and optimizer
input_size = 165
hidden_size = 128
num_layers = 2
num_classes = 2
learning_rate = 0.01
num_epochs = 100
batch_size = 32

#model = EmotionModel(input_size, hidden_size, num_layers, num_classes)



# Create DataLoaders for training and validation sets
training_dataloader = dataset_test.training_dataloader
test_dataloader = dataset_test.test_dataloader
 
# Train the model 
class LSTMModel(nn.Module):
    def __init__(self, input_size=165, hidden_size1=512, hidden_size2=256, output_size=2):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size2, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

model = LSTMModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    print(f'Starting epoch {epoch+1}/{num_epochs}')
    for i, (data, labels) in enumerate(training_dataloader):
        # Forward pass
        print('Forward pass')
        outputs = model(data)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted)
        print(labels)
        # Backward pass and optimization
        print('Backward pass')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress after each batch iteration
        print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(training_dataloader)}], Loss: {loss.item():.4f}')

    # Evaluate the model on the validation set
    with torch.no_grad():
        correct = 0
        total = 0
        for data, labels in test_dataloader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            print(labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Validation accuracy: {100 * correct / total:.2f}%')