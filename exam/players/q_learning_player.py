import pickle
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np

from exam.game import Player, Move, Game
from exam.utils import Utilities

POSSIBLE_MOVES = 100

class QPlayer(Player):
    def __init__(self, learning_rate, discount_rate, exploration_rate, min_exploration_rate, exploration_decay):
        super().__init__()
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay
        self.player_id = 1
        self.q_table = defaultdict(lambda: np.zeros(POSSIBLE_MOVES))

    def get_game_reward(self, game_result):
        if game_result == self.player_id:
            return 10
        elif game_result == -1:
            return 0
        else:
            return -10

    def make_move_train(self, game) -> tuple[tuple[int, int], Move]:


        if np.random.rand() < self.exploration_rate:
            action = random.choice(Utilities.generate_valid_moves(game, self.player_number))
        else:
            canonical_state, transformations = Utilities.get_canonical_state(game.get_board())
            game_copy = deepcopy(game)
            game_copy._board = Utilities.convert_string_to_array(canonical_state)

            q_values = self.q_table[canonical_state]
            valid_moves = [Utilities.convert_action_to_scalar(action) for action in
                           Utilities.generate_valid_moves(game_copy, self.player_number)]
            valid_q_values = [q_values[action] for action in valid_moves]
            max_q_value = max(valid_q_values)
            max_index_es = [action for action, q_value in zip(valid_moves, valid_q_values) if q_value == max_q_value]
            action = np.random.choice(max_index_es)
            action = Utilities.convert_action_to_triplet(action)
            action = Utilities.get_reverse_action(action, transformations)
            if not deepcopy(game)._Game__move(action[0], action[1], self.player_number):
                action = random.choice(Utilities.generate_valid_moves(game, self.player_number))
                self.update_q_table(canonical_state, Utilities.convert_action_to_scalar(action), -50, None)
        return action

    def make_move(self, game) -> tuple[tuple[int, int], Move]:

        canonical_state, transformations = Utilities.get_canonical_state(game.get_board())
        game_copy = deepcopy(game)
        game_copy._board = Utilities.convert_string_to_array(canonical_state)

        q_values = self.q_table[canonical_state]
        valid_moves = [Utilities.convert_action_to_scalar(action) for action in
                       Utilities.generate_valid_moves(game_copy, self.player_number)]
        valid_q_values = [q_values[action] for action in valid_moves]
        max_q_value = max(valid_q_values)
        max_index_es = [action for action, q_value in zip(valid_moves, valid_q_values) if q_value == max_q_value]
        action = np.random.choice(max_index_es)
        action = Utilities.convert_action_to_triplet(action)
        action = Utilities.get_reverse_action(action, transformations)
        if not deepcopy(game)._Game__move(action[0], action[1], self.player_number):
            action = random.choice(Utilities.generate_valid_moves(game, self.player_number))
        return action

    def update_q_table(self, state, action, reward, next_state=None):
        if next_state is not None:
            self.q_table[state][action] = ((1 - self.learning_rate) * self.q_table[state][action] +
                                           self.learning_rate * (reward + self.discount_rate * (
                        -np.max(self.q_table[next_state]))))
        else:
            self.q_table[state][action] = reward

    def execute_move(self, game: Game, states_action_traversed):

        canonical_state, transformations = Utilities.get_canonical_state(game.get_board())
        action = self.make_move_train(game)
        game._Game__move(action[0], action[1], self.player_number)
        next_state = Utilities.get_canonical_state(game.get_board())[0]
        states_action_traversed.append((canonical_state, Utilities.convert_action_to_scalar(action), next_state))


    def train(self, n_episodes, opponent, first_Player=True):
        game = Game()
        self.player_number = 1 if first_Player else 2
        opponent.player_number = 2 if first_Player else 1
        players = [self, opponent]
        turn = 0 if first_Player else 1
        episode_count = 0
        for episode in range(n_episodes):
            states_action_traversed = []
            counter = 0
            while game.check_winner() == -1 and counter < 200:
                if turn == self.player_number-1:
                    self.execute_move(game, states_action_traversed)
                else:
                    ok = False
                    while not ok:
                        move = players[turn].make_move(game)
                        ok = game._Game__move(move[0], move[1], players[turn].player_number)
                turn = 1 - turn
                counter += 1
            turn = 0 if first_Player else 1
            game_reward = self.get_game_reward(game.check_winner())
            reward_decrementation = game_reward / len(states_action_traversed)
            for state, action, next_state in states_action_traversed[::-1]:
                self.update_q_table(state, action, game_reward, next_state)
                game_reward -= reward_decrementation if game_reward > 0 else -reward_decrementation
            game = Game()
            self.exploration_rate = np.clip(
                np.exp(-self.exploration_decay * episode), self.min_exploration_rate, 1)
            episode_count += 1
            if episode_count % 100 == 0:
                print(f"Episode: {episode_count} with exploration rate: {self.exploration_rate}")

    def test(self, n_matches, opponent, first_player=True):
        game = Game()
        self.player_number = 1 if first_player else 2
        opponent.player_number = 2 if first_player else 1
        players = [self, opponent]
        rl_victories = 0
        for _ in range(n_matches):
            turn = 0 if first_player else 1
            counter = 0
            while game.check_winner() == -1 and counter < 200:
                move = players[turn].make_move(game)
                ok = game._Game__move(move[0], move[1], players[turn].player_number)
                if not ok:
                    print("Invalid move")
                turn = 1 - turn
                counter += 1
            if game.check_winner() == self.player_number:
                rl_victories += 1
            game = Game()
        print(f"number of victories: {rl_victories} out of {n_matches}")
        return rl_victories / n_matches

    def save_q_table(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, file_name):
        with open(file_name, "rb") as f:
            self.q_table = pickle.load(f)