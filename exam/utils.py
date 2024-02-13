from copy import deepcopy
from typing import Tuple, List

import numpy as np

from exam.game import Move, Game

MC_TABLE_FIRST = "mc_first.pkl"
MC_TABLE_SECOND = "mc_second.pkl"

TRANSFORMATIONS = [
        (0, None),  # No transformation
        (0, 0),  # Reflection along the vertical axis
        (0, 1),  # Reflection along the horizontal axis
        (1, None),  # 90 degrees rotation
        (2, None),  # 180 degrees rotation
        (3, None),  # 270 degrees rotation
    ]


class Utilities():


    @staticmethod
    def generate_all_moves() -> list[Tuple[Tuple[int, int], Move]]:
        all_moves = []
        for i in range(5):
            for j in range(5):
                for move in Move:
                    all_moves.append(((i, j), move))
        return all_moves

    @staticmethod
    def generate_valid_moves(game: Game, player_number) -> list[Tuple[Tuple[int, int], Move]]:

        valid_moves = []

        for move in Utilities.generate_all_moves():
            copy = deepcopy(game)
            if copy._Game__move(move[0], move[1], player_number):
                valid_moves.append(move)

        return valid_moves

    @staticmethod
    def convert_state(state):
        flattened_arr = state.flatten()
        return ''.join(map(lambda x: str(0) if x == -1 else str(x), flattened_arr))

    @staticmethod
    def convert_string_to_array(state_str) -> np.ndarray:
        # Convert string to numpy array
        arr = np.array(list(map(int, state_str)))

        arr = arr.reshape((5, 5))

        arr = np.where(arr == 0, -1, arr)

        return arr

    @staticmethod
    def apply_rotation(state_np, rotations):
        # Rotate the state by the specified number of rotations (clockwise)
        return np.rot90(state_np.reshape((5, 5)), k=rotations).flatten()

    @staticmethod
    def apply_reflection(state_np, axis):
        # Reflect the state along the specified axis (0 for vertical, 1 for horizontal)
        return np.flip(state_np.reshape((5, 5)), axis=axis).flatten()

    @staticmethod
    def apply_symmetry(state_str, rotations=0, reflection_axis=None):
        state_np = np.array([int(char) for char in state_str])

        # Apply rotations
        state_np = Utilities.apply_rotation(state_np, rotations)

        # Apply reflection if specified
        if reflection_axis is not None:
            state_np = Utilities.apply_reflection(state_np, axis=reflection_axis)

        return ''.join(map(str, state_np))

    @staticmethod
    def convert_action_to_triplet(action):
        return ((action // 20) % 5, (action // 4) % 5), Move(action % 4)

    @staticmethod
    def convert_action_to_scalar(action):
        return action[0][0] * 20 + action[0][1] * 4 + action[1].value

    @staticmethod
    def get_canonical_state(state_str):
        state_str = Utilities.convert_state(state_str)
        state_np = np.array([int(char) for char in state_str])
        state_str = ''.join(map(str, state_np))
        canonical_state = state_str
        transformations = (0, None)

        for rotations, reflection_axis in TRANSFORMATIONS:
            sym_state = Utilities.apply_symmetry(state_np, rotations, reflection_axis)
            if sym_state < canonical_state:
                canonical_state = sym_state
                transformations = (rotations, reflection_axis)

        return canonical_state, transformations

    @staticmethod
    def get_reverse_action(action, transformations):
        canonical_action = action
        rotations, reflection_axis = transformations

        # Reverse the reflection if it was applied
        if reflection_axis is not None:
            canonical_action = Utilities.reverse_reflection(canonical_action, reflection_axis)

        # Reverse the rotations
        for _ in range(rotations % 4):
            canonical_action = Utilities.reverse_rotation(canonical_action)

        return canonical_action

    @staticmethod
    def reverse_reflection(action, axis):
        if axis == 1:  # Reflection along the vertical axis
            from_pos = (action[0][0], 4 - action[0][1])
            if action[1] in [Move.TOP, Move.BOTTOM]:
                return (from_pos, action[1])
            else:
                if action[1] == Move.LEFT:
                    move = Move.RIGHT
                else:
                    move = Move.LEFT
                return (from_pos, move)
        elif axis == 0:  # Reflection along the horizontal axis
            from_pos = (4 - action[0][0], action[0][1])
            if action[1] in [Move.LEFT, Move.RIGHT]:
                return (from_pos, action[1])
            else:
                if action[1] == Move.TOP:
                    move = Move.BOTTOM
                else:
                    move = Move.TOP
                return (from_pos, move)

    @staticmethod
    def reverse_rotation(action):
        return ((action[0][1], 4 - action[0][0]), action[1].clockwise())

    @staticmethod
    def generate_canonical_transitions(game, player_number) -> List[Tuple[str, Tuple[Tuple[int, int], Move]]]:

        state_transitions = []
        for move in Utilities.generate_valid_moves(game, player_number):
            copy = deepcopy(game)
            copy._Game__move(move[0], move[1], player_number)
            next_canonical_state = Utilities.get_canonical_state(copy.get_board())[0]
            state_transitions.append((next_canonical_state, move))

        return state_transitions

    @staticmethod
    def play_game_with_random(player1, player2, first_player=True):
        game = Game()
        player1.player_number = 1 if first_player else 2
        player2.player_number = 2 if first_player else 1
        players = [player1, player2]
        turn = 0 if first_player else 1
        counter = 0
        while game.check_winner() == -1 and counter < 200:
            move = players[turn].make_move(game)
            game._Game__move(move[0], move[1], players[turn].player_number)
            turn = 1 - turn
            counter += 1
        return game.check_winner()

    @staticmethod
    def play_multiple_games(n_matches, player1, player_2):
        wins = 0
        for i in range(n_matches):
            winner = Utilities.play_game_with_random(player1, player_2)
            if winner == player1.player_number:
                wins += 1
        print(f"number of victories: {wins}, out of {n_matches}")
        return wins / n_matches





