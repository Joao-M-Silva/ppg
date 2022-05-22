from enum import Enum, auto

class Colors(Enum):
    BLUE = 'B'
    GREEN = 'G'
    RED = 'R'
    BLUE_F = 'B_F'
    GREEN_F = 'G_F'
    RED_F = 'R_F'

class TimeUnit(Enum):
    SECONDS = auto()
    MILISECONDS = auto()