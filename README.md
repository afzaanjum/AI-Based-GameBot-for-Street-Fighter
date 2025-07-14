Game Bot – AI Project (Spring 2025)

In this project we had to design a game bot for Street Fighter Game and we had to use MLP model for training.

Project Folder Structure

PythonAPI/
│
├── pycache/ # Compiled Python files

├── bot.py # Main bot logic (uses trained model)

├── buttons.py # SNES button definitions

├── command.py # Sends button commands to emulator

├── controller.py # Controls game loop and emulator connection

├── game_log.xlsx # Saved gameplay logs

├── game_log.txt # Raw game log for debugging/training

├── game_state.py # Manages the GameState class

├── mlp_model.pth # Trained MLP model (PyTorch)

├── player.py # Player information (health, position, etc.)

├── train_mlp.py # MLP training script (PyTorch)

├── training.ipynb # Jupyter notebook for data prep/training

└── README.md # This file


 System Requirements
 
Operating System: Windows 7 or above (64-bit)

Python Version: 3.6.3 or later

Emulator: BizHawk Emulator (EmuHawk)

ROM File: Street Fighter II Turbo (U).smc


 Python Dependencies

Install the following Python packages:

pip install numpy pandas scikit-learn joblib

You may add tensorflow or torch if you're using deep learning.


How to Run the Bot

Step-by-Step Instructions

Download and Extract API

Extract contents into a folder.


Launch the Emulator

Run EmuHawk.exe from the single-player or two-player folder.

File → Open ROM → Load Street Fighter II Turbo (U).smc.


Open Toolbox

Tools → Tool Box (or press Shift + T)

Run the Python Bot

Open CMD in the folder where your project files are.


Run:

python controller.py 1

Use 1 for Player 1 and 2 for Player 2.


Game Controls

After execution, go to the game and select your character in Normal Mode.

Use emulator settings to configure/select controllers if needed.


Establish Connection

In EmuHawk, click the Gyroscope Bot icon (second icon on the top row).

You should see CONNECTED SUCCESSFULLY in the terminal.


Gameplay

The game bot will control the character using your ML/DL logic.

The program stops after one round – rerun as needed.


AI/ML Implementation

The bot uses a Machine Learning / Deep Learning model to make decisions.

bot.py contains:

A modified fight() function using game state input

Logic to load and use trained model

Model trained on recorded dataset from actual gameplay (game_log.csv).


 Notes

Make sure the emulator is not minimized or in the background when running the bot.

Always run the emulator before the bot.

The bot must be able to play with any random character, so generalization is crucial.
