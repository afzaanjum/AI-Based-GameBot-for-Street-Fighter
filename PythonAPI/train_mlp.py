import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Defining the MLP model class with Dropout for regularization
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout layer with 30% dropout rate
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout layer with 30% dropout rate
            nn.Linear(64, 12),
            nn.Sigmoid()  # Sigmoid activation to output between 0 and 1
        )

    def forward(self, x):
        return self.layers(x)

# Load data from game_log.csv
data = pd.read_csv('game_log.csv', quoting=csv.QUOTE_NONNUMERIC)

# Input feature columns
input_columns = ['P1_Health', 'P1_X', 'P1_Y', 'P2_Health', 'P2_X', 'P2_Y']

# Define button order
button_order = ['Up', 'Down', 'Right', 'Left', 'Select', 'Start', 'Y', 'B', 'X', 'A', 'L', 'R']

# Parse button JSON strings into binary vectors
def parse_buttons(button_str):
    try:
        button_dict = json.loads(button_str)
        return [1 if button_dict.get(button, False) else 0 for button in button_order]
    except json.JSONDecodeError as e:
        print(f"Error parsing button string: {button_str}")
        print(f"Error message: {e}")
        return [0] * len(button_order)

# Apply parsing to P1_Buttons
y = np.array([parse_buttons(button_str) for button_str in data['P1_Buttons']]).astype(np.float32)

# Extract and standardize input features
X = data[input_columns].values.astype(np.float32)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to PyTorch tensors
X = torch.tensor(X)
y = torch.tensor(y)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model, loss, and optimizer
model = MLP()
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lists to store losses for plotting
train_losses = []
val_losses = []

# Training loop with validation loss tracking
for epoch in range(100):
    model.train()  # Set model to training mode
    optimizer.zero_grad()
    
    # Training step
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    # Validation step
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
    
    # Store losses for later plotting
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    # Print losses every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'mlp_model.pth')
print("\nModel trained and saved as 'mlp_model.pth'.")

# Print Training vs Validation Accuracy
with torch.no_grad():
    # Training Accuracy
    model.train()
    train_outputs = model(X_train)
    train_predicted = (train_outputs > 0.5).float()
    train_accuracy = (train_predicted == y_train).float().mean()
    
    # Validation Accuracy
    model.eval()
    val_outputs = model(X_val)
    val_predicted = (val_outputs > 0.5).float()
    val_accuracy = (val_predicted == y_val).float().mean()

    # Overall Accuracy
    all_outputs = model(torch.cat([X_train, X_val]))  # Combine both train and validation sets
    all_predicted = (all_outputs > 0.5).float()
    overall_accuracy = (all_predicted == torch.cat([y_train, y_val])).float().mean()

print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
print(f'Overall Accuracy: {overall_accuracy * 100:.2f}%')
