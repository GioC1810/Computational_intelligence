import random

from exam.utils import Utilities
from game import Player, Move


class RandomPlayer(Player):
    def __init__(self, player_id) -> None:
        super().__init__()
        self.player_number = player_id

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        valid_moves = Utilities.generate_valid_moves(game, self.player_number)
        return random.choice(valid_moves)