import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import dataset_setup 
import base
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the LSTM model

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=512, hidden_size2=256, output_size=len(dataset_setup.label_map)):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size2, 512)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(512, output_size)


    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size1=512, hidden_size2=256, output_size=len(dataset_setup.label_map)):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size1, batch_first=True)
        self.gru2 = nn.GRU(hidden_size1, hidden_size2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size2, 512)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

# Set hyperparameters and instantiate the model, loss function, and optimizer
input_size = len(base.points_interet) * 3
hidden_size1= 512
hidden_size2= 256
num_layers = 2
num_classes = len(dataset_setup.emotions_of_interest)
learning_rate = 0.0005
num_epochs = 12
batch_size = 32


model = GRUModel(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create DataLoaders for training and validation sets
training_dataloader = dataset_setup.training_dataloader
test_dataloader = dataset_setup.test_dataloader

a = next(iter(training_dataloader))

print(a[0][0].shape)
 
#Train the model 

Training_loss_evolution = []
Testing_accuracy_evolution = []
Training_accuracy_evolution = []

nb_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('The model has {} parameters'.format(nb_param))
print(f'input size is {input_size}')
print(f'output size is {num_classes}')

for epoch in range(num_epochs) : 
    
    print(f'Starting epoch {epoch+1}/{num_epochs}')
    correct = 0
    total = 0
    for i, (data, labels) in enumerate(training_dataloader):
        # Forward pass
        print('Forward pass')
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        print(outputs)
        print(predicted)
        print(labels)
        loss = criterion(outputs, labels)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Backward pass and optimization
        print('Backward pass')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress after each batch iteration
        Training_loss_evolution.append(loss.item())
        print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(training_dataloader)}], Loss: {loss.item():.4f}', f'TrainAcc: {100 * correct/total}')
        
    Training_accuracy_evolution.append(100*correct / total)
    print(f'TrainAccEnd : {Training_accuracy_evolution}')
    # Evaluate the model on the validation set
    with torch.no_grad():
        correct = 0
        total = 0
        for data, labels in test_dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        Testing_accuracy_evolution.append((100 * correct) /total)
        print(f'TestAcc : {Testing_accuracy_evolution}')
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation accuracy: {100 * correct / total:.2f}%')

def plotting_figure(Y_list, x_label, y_label, labels, title):
    # Create a new figure
    fig = plt.figure()
    # Add a subplot to the figure
    ax = fig.add_subplot(1, 1, 1)
    # Plot each Y list on the same X axis
    for i, Y in enumerate(Y_list):
        ax.plot(Y, label=labels[i])
    # Add labels to the X and Y axes
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # Add a title to the plot
    ax.set_title(title)
    # Add a legend to the plot
    ax.legend()
    # Show the plot
    plt.show()


plotting_figure([Testing_accuracy_evolution, Training_accuracy_evolution], x_label='Epochs', y_label='Accuracy', labels=['Testing Acc', 'Training Acc'], title='Accuracy')

plotting_figure([Training_loss_evolution], x_label='Epochs', y_label='Training Loss', labels=['Training Loss'], title='Training Loss')
