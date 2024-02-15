from collections import defaultdict
from copy import deepcopy
from typing import Tuple

import numpy as np

from exam.game import Player, Move
from exam.utils import Utilities


class MinMaxPlayer(Player):
    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self.state_memorized = 0
        self.player_number = 1

    def make_move(self, game) -> tuple[tuple[int, int], 'Move']:

        _, best_move = self.minimax(game, self.depth, self.player_number, True, float('-inf'), float('inf'))
        return best_move


    def minimax(self, game, depth, player_id, maximizing_player, alpha, beta):
        if depth == 0 or game.check_winner() != -1:
            return self.heuristic(game, player_id), None

        best_move = None
        if maximizing_player:
            assert player_id == self.player_number
            max_eval = float('-inf')
            for move in Utilities.generate_valid_moves(game, player_id):
                new_game_state = deepcopy(game)
                new_game_state._Game__move(move[0], move[1], player_id)
                next_player_number = 2 if player_id == 1 else 1
                eval, _ = self.minimax(new_game_state, depth-1 , next_player_number, False, alpha, beta)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            assert player_id == 3-self.player_number
            for move in Utilities.generate_valid_moves(game, player_id):
                new_game_state = deepcopy(game)
                new_game_state._Game__move(move[0], move[1], player_id)
                next_player_number = 2 if player_id == 1 else 1
                eval, _ = self.minimax(new_game_state, depth-1 , next_player_number, True, alpha, beta)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def heuristic(self, game, player_id):

        if game.check_winner() == 1:
            return 10
        elif game.check_winner() == 2:
            return -10
        else:
            score = 0
            board = game.get_board()
            # Check rows
            for row in board:
                player_cells = np.sum(row == player_id)
                if player_cells == 4:
                    score += 1
                if player_cells == 3:
                    score += 0.5

            # Check columns
            for col in board.T:
                player_cells = np.sum(col == player_id)
                if player_cells == 4:
                    score += 1
                if player_cells == 3:
                    score += 0.5

            # Check diagonals
            diag1 = board.diagonal()
            diag2 = np.fliplr(board).diagonal()
            for diag in [diag1, diag2]:
                player_cells = np.sum(diag == player_id)
                if player_cells == 4:
                    score += 1
                if player_cells == 3:
                    score += 0.5

            return score
