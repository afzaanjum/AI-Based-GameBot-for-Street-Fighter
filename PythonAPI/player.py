from buttons import Buttons

class Player:
    def __init__(self, player_dict):
        self.dict_to_object(player_dict)

    def dict_to_object(self, player_dict):
        self.player_id = player_dict['character']
        self.health = player_dict['health']
        self.x_coord = player_dict['x']
        self.y_coord = player_dict['y']
        self.is_jumping = player_dict['jumping']
        self.is_crouching = player_dict['crouching']
        self.player_buttons = Buttons(player_dict['buttons'])
        self.is_player_in_move = player_dict['in_move']
        self.move_id = player_dict['move']

    def __eq__(self, other):
        if not isinstance(other, Player):
            return False
        return (
            self.health == other.health and
            self.x_coord == other.x_coord and
            self.y_coord == other.y_coord and
            self.is_jumping == other.is_jumping and
            self.is_crouching == other.is_crouching and
            self.is_player_in_move == other.is_player_in_move and
            self.move_id == other.move_id and
            self.player_buttons == other.player_buttons
        )

# from buttons import Buttons

# class Player:

#     def __init__(self, player_dict):
        
#         self.dict_to_object(player_dict)
    
#     def dict_to_object(self, player_dict):
        
#         self.player_id = player_dict['character']
#         self.health = player_dict['health']
#         self.x_coord = player_dict['x']
#         self.y_coord = player_dict['y']
#         self.is_jumping = player_dict['jumping']
#         self.is_crouching = player_dict['crouching']
#         self.player_buttons = Buttons(player_dict['buttons'])
#         self.is_player_in_move = player_dict['in_move']
#         self.move_id = player_dict['move']
