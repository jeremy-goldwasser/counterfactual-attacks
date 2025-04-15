import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Initialize weights using He initialization suited for ReLUs
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 1)
        
        # Initialize weights for fully connected layers
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.fc2.weight)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the input for the fully connected layer
        x = x.view(-1, 128 * 32 * 32)
        
        # Fully connected layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
def evaluate(model, test_loader, criterion=None, device=None):
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss() # binary classification
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    running_loss = 0.0
    total_correct = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Reshape labels to be compatible with model outputs
            labels = labels.float().unsqueeze(1)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Calculate loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) >= 0.5
            total_correct += torch.sum(preds == labels)
    
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = total_correct.double() / len(test_loader.dataset)
    
    print(f'Test Loss: {epoch_loss:.4f}, Test Acc: {epoch_acc:.4f}')
    return epoch_loss, epoch_acc