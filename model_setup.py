
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Custom dataset class
class EmotionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Define the LSTM model
class EmotionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(EmotionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Set hyperparameters and instantiate the model, loss function, and optimizer
input_size = 165
hidden_size = 128
num_layers = 2
num_classes = 7
learning_rate = 0.001
num_epochs = 100
batch_size = 32

model = EmotionModel(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create DataLoaders for training and validation sets
train_dataset = EmotionDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = EmotionDataset(val_data, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Train the model
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate the model on the validation set
    with torch.no_grad():
        correct = 0
        total = 0
        for data, labels in val_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Validation accuracy: {100 * correct / total:.2f}%')