from command import Command
from buttons import Buttons
import torch
import torch.nn as nn
import numpy as np

# Define the correct MLP model (with dropout)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 12),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# Load the trained model
model = MLP()
model.load_state_dict(torch.load('mlp_model.pth', map_location=torch.device('cpu')))
model.eval()

class Bot:
    def __init__(self):
        self.my_command = Command()
        self.buttn = Buttons()

    def fight(self, current_game_state, player):
        input_data = np.array([
            current_game_state.player1.health, current_game_state.player1.x_coord, current_game_state.player1.y_coord,
            current_game_state.player2.health, current_game_state.player2.x_coord, current_game_state.player2.y_coord
        ], dtype=np.float32)

        input_tensor = torch.tensor(input_data).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
        predicted_buttons = (output > 0.5).int().tolist()[0]

        # Use all lowercase to match Buttons class attributes
        button_names = ['up', 'down', 'right', 'left', 'select', 'start', 'y', 'b', 'x', 'a', 'l', 'r']
        for i, name in enumerate(button_names):
            setattr(self.buttn, name, bool(predicted_buttons[i]))

        if player == '1':
            self.my_command.player_buttons = self.buttn
        else:
            self.my_command.player2_buttons = self.buttn

        return self.my_command
