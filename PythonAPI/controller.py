import socket
import json
import sys
import csv
from game_state import GameState
from bot import Bot

def connect(port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("127.0.0.1", port))
    server_socket.listen(5)
    (client_socket, _) = server_socket.accept()
    print("Connected to game!")
    return client_socket

def send(client_socket, command):
    command_dict = command.object_to_dict()
    payload = json.dumps(command_dict).encode()
    client_socket.sendall(payload)

def receive(client_socket):
    payload = client_socket.recv(4096)
    input_dict = json.loads(payload.decode())
    return GameState(input_dict)

def log_game_state(csv_writer, game_state):
    # Prepare the row data
    row = [
        game_state.timer,
        game_state.player1.health,
        game_state.player1.x_coord,
        game_state.player1.y_coord,
        game_state.player1.move_id,
        game_state.player2.health,
        game_state.player2.x_coord,
        game_state.player2.y_coord,
        game_state.player2.move_id,
        json.dumps(game_state.player1.player_buttons.object_to_dict()),
        json.dumps(game_state.player2.player_buttons.object_to_dict())
    ]
    # Write the row to CSV
    csv_writer.writerow(row)

def main():
    port = 9999 if sys.argv[1] == '1' else 10000
    client_socket = connect(port)
    bot = Bot()

    current_game_state = None
    last_logged_state = None

    # Open CSV file for writing
    csv_filename = "game_log.csv"
    with open(csv_filename, "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
        # Write header
        csv_writer.writerow([
            "Timer",
            "P1_Health", "P1_X", "P1_Y", "P1_MoveID",
            "P2_Health", "P2_X", "P2_Y", "P2_MoveID",
            "P1_Buttons",
            "P2_Buttons"
        ])

        while True:
            try:
                # Receive current game state from the game
                current_game_state = receive(client_socket)

                # Break the loop if round is over
                if current_game_state.is_round_over:
                    print("Round over. Ending loop.")
                    break

                # Skip if round has not started or if the game state hasn't changed
                if not current_game_state.has_round_started or current_game_state == last_logged_state:
                    continue

                # Get bot command for the player based on the game state
                bot_command = bot.fight(current_game_state, sys.argv[1])
                send(client_socket, bot_command)

                # Log the current game state to the CSV file
                log_game_state(csv_writer, current_game_state)
                last_logged_state = current_game_state

            except Exception as e:
                print(f"Error: {e}")
                break

    print(f"Game log saved as {csv_filename}.")

if __name__ == "__main__":
    main()
