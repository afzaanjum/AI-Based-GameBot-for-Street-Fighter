from player import Player

class GameState:
    def __init__(self, input_dict):
        self.dict_to_object(input_dict)

    def dict_to_object(self, input_dict):
        self.player1 = Player(input_dict['p1'])
        self.player2 = Player(input_dict['p2'])
        self.timer = input_dict['timer']
        self.fight_result = input_dict['result']
        self.has_round_started = input_dict['round_started']
        self.is_round_over = input_dict['round_over']

    def __eq__(self, other):
        if not isinstance(other, GameState):
            return False
        return (
            self.timer == other.timer and
            self.player1 == other.player1 and
            self.player2 == other.player2 and
            self.fight_result == other.fight_result and
            self.has_round_started == other.has_round_started and
            self.is_round_over == other.is_round_over
        )














# from player import Player

# class GameState:

#     def __init__(self, input_dict):

#         self.dict_to_object(input_dict)

#     def dict_to_object(self, input_dict):

#         self.player1 = Player(input_dict['p1'])
#         self.player2 = Player(input_dict['p2'])
#         self.timer = input_dict['timer']
#         self.fight_result = input_dict['result']
#         self.has_round_started = input_dict['round_started']
#         self.is_round_over = input_dict['round_over']