import pickle
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from exam.game import Player, Move, Game
from exam.utils import Utilities


class MonteCarloPlayer(Player):
    def __init__(self, exploration_rate, learning_rate, discount_rate, min_exploration_rate):
        super().__init__()
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.min_exploration_rate = min_exploration_rate
        self.states_traversed = []
        self.state_value = defaultdict(float)
        self.wins = 0
        self.win_rates = []


    def get_game_reward(self, game_result):
        if game_result == self.player_number:
            return 10
        elif game_result == -1:
            return 0
        else:
            return -10
    def make_move(self, game) -> tuple[tuple[tuple[int, int], Move], str]:

        if np.random.rand() < self.exploration_rate:
            action = random.choice(Utilities.generate_valid_moves(game, self.player_number))
            game_copy = deepcopy(game)
            game_copy._Game__move(action[0], action[1], self.player_number)
            next_state = Utilities.get_canonical_state(game_copy.get_board())[0]
        else:
            canonical_transition = Utilities.generate_canonical_transitions(game, self.player_number)
            state_chosen, action = max(canonical_transition, key=lambda t: self.state_value[t[0]])
            next_state = state_chosen

        return action, next_state

    def save_q_table(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.state_value, f)

    def load_q_table(self, file_path):
        with open(file_path, 'rb') as f:
            self.state_value = pickle.load(f)

    def update_q_table(self, game_reward):
        reward = game_reward
        for state in reversed(self.states_traversed):
            self.state_value[state] = self.state_value[state] + self.learning_rate * (
                    self.discount_rate * reward - self.state_value[state])
            reward = self.state_value[state]
        self.states_traversed = []

    def train(self, n_episodes, opponent, first_Player=True):
        game = Game()
        self.player_number = 1 if first_Player else 2
        opponent.player_number = 2 if first_Player else 1
        players = [self, opponent]
        turn = 0 if first_Player else 1
        episode_count = 0
        self.exploration_decay = -np.log(self.min_exploration_rate) / n_episodes

        for episode in range(n_episodes):
            counter = 0

            while game.check_winner() == -1 and counter < 200:
                if players[turn].__class__.__name__ == "MonteCarloPlayer":
                    move, next_state = players[turn].make_move(game)
                    game._Game__move(move[0], move[1], players[turn].player_number)
                    self.states_traversed.append((next_state))
                else:
                    move = players[turn].make_move(game)
                    game._Game__move(move[0], move[1], players[turn].player_number)
                turn = 1 - turn
                counter += 1

            turn = 0 if first_Player else 1
            game_reward = self.get_game_reward(game.check_winner())

            self.wins += 1 if game_reward > 0 else 0
            win_rate = self.wins / (episode + 1)
            self.win_rates.append(win_rate)

            self.update_q_table(game_reward)

            game = Game()
            self.exploration_rate = np.clip(
                np.exp(-self.exploration_decay * episode), self.min_exploration_rate, 1)

            episode_count += 1

            if episode_count % 100000 == 0:
                print(f"Episode: {episode_count} with exploration rate: {self.exploration_rate}")


    def test(self, n_matches, opponent, first_player=True):
        self.exploration_rate = 0
        game = Game()
        self.player_number = 1 if first_player else 2
        opponent.player_number = 2 if first_player else 1
        players = [self, opponent]
        agent_victories = 0
        draws = 0
        turn = 0 if first_player else 1

        for episode in range(n_matches):
            counter = 0
            while game.check_winner() == -1 and counter < 200:
                if players[turn].__class__.__name__ == "MonteCarloPlayer":
                    move, next_state = players[turn].make_move(game)
                    game._Game__move(move[0], move[1], players[turn].player_number)
                else:
                    move = players[turn].make_move(game)
                    game._Game__move(move[0], move[1], players[turn].player_number)
                turn = 1 - turn
                counter += 1

            turn = 0 if first_player else 1
            result = self.get_game_reward(game.check_winner())
            if result > 0:
                agent_victories += 1
            elif result == 0:
                draws += 1
            game = Game()
        print(f"number of victories: {agent_victories}, draws: {draws} out of {n_matches}")

    def plot_metrics(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 2)
        plt.plot(self.win_rates)
        plt.title('Win Rate per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')

        plt.tight_layout()
        plt.show()




