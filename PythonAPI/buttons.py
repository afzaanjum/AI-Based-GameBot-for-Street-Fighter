class Buttons:

    def __init__(self, buttons_dict=None):
        if buttons_dict is not None:
            self.dict_to_object(buttons_dict)
        else:
            self.init_buttons()

    def init_buttons(self):
        self.up = False
        self.down = False
        self.right = False
        self.left = False
        self.select = False
        self.start = False
        self.y = False
        self.b = False
        self.x = False
        self.a = False
        self.l = False
        self.r = False

    def dict_to_object(self, buttons_dict):
        self.up = buttons_dict.get('Up', False)
        self.down = buttons_dict.get('Down', False)
        self.right = buttons_dict.get('Right', False)
        self.left = buttons_dict.get('Left', False)
        self.select = buttons_dict.get('Select', False)
        self.start = buttons_dict.get('Start', False)
        self.y = buttons_dict.get('Y', False)
        self.b = buttons_dict.get('B', False)
        self.x = buttons_dict.get('X', False)
        self.a = buttons_dict.get('A', False)
        self.l = buttons_dict.get('L', False)
        self.r = buttons_dict.get('R', False)

    def object_to_dict(self):
        return {
            'Up': self.up,
            'Down': self.down,
            'Right': self.right,
            'Left': self.left,
            'Select': self.select,
            'Start': self.start,
            'Y': self.y,
            'B': self.b,
            'X': self.x,
            'A': self.a,
            'L': self.l,
            'R': self.r
        }
