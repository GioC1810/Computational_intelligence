import random
from itertools import product

import numpy as np
import pickle


from game import Game, Move, Player


class RandomPlayer(Player):
    def __init__(self, player_id) -> None:
        super().__init__()
        self.player_id = player_id

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        print(f"Random player: position: {from_pos}, move: {move}")
        return from_pos, move


class MyPlayer(Player):
    def __init__(self, learning_rate, discount_rate, exploration_rate, min_exploration_rate, exploration_decay):
        super().__init__()
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay
        self.player_id = 1
        self.q_table = {}

    def get_game_reward(self, game_result):
        if game_result == self.player_id:
            return 10
        elif game_result == 2:
            return -10
        else:
            return 0

    def convert_state(self, state):
        return "".join(str(_) for _ in state.flatten())

    def convert_action(self, action):
        return action[0][0] * 16 + action[0][1] * 4 + action[1].value

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        current_state = game.get_board()
        converted_state = self.convert_state(current_state)

        if converted_state not in self.q_table:
            self.q_table[converted_state] = np.zeros((83,))

        if np.random.rand() < self.exploration_rate:
            # Explore: Choose a random action
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
            action = from_pos, move
        else:
            # Exploit: Choose the action with the highest Q-value
            possible_moves = generate_valid_moves(game, self.player_id)
            converted_moves = [self.convert_action(action) for action in possible_moves]
            q_values = [self.q_table[converted_state][action] for action in converted_moves]
            max_index = np.where(q_values == np.max(q_values))[0]
            action = np.random.choice(max_index)
            action = possible_moves[action]
        return action

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros((83,))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros((83,))
            self.q_table[state][action] = ((1 - self.learning_rate) * self.q_table[state][action] +
                                           self.learning_rate * (reward + self.discount_rate * (
                        -np.max(self.q_table[next_state]))))

    def save_q_table(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, file_name):
        with open(file_name, "rb") as f:
            self.q_table = pickle.load(f)

    def train_backprop_incremental(self, n_episodes, opponent, first_Player=True):
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
                actual_state = self.convert_state(game.get_board())
                print(f"actual state: {actual_state}")
                if turn == self.player_number - 1:
                    print(f"turn agent")
                    action = players[turn].make_move(game)
                    states_action_traversed.append((actual_state, self.convert_action(action)))
                    game._Game__move(action[0], action[1], turn + 1)
                    next_state = self.convert_state(game.get_board())
                else:
                    print(f"turn random")
                    move = players[turn].make_move(game)
                    game._Game__move(move[0], move[1], turn + 1)
                turn = 1 - turn
                counter += 1
            print(f"game ended with state: {game.check_winner()}")
            turn = 0 if first_Player else 1
            game_reward = self.get_game_reward(game.check_winner())
            #self.update_q_table(actual_state, self.convert_action(action), game_reward, next_state)
            reward_step_decrement = game_reward / len(states_action_traversed)
            print(f"reward decrement: {reward_step_decrement}")
            for state, action in states_action_traversed[::-1]:
                self.update_q_table(state, action, game_reward, next_state)
                game_reward -= reward_step_decrement if game_reward > 0 else -reward_step_decrement
            game = Game()
            self.exploration_rate = np.clip(
                np.exp(-self.exploration_decay * episode), self.min_exploration_rate, 1)
            episode_count += 1
            if episode_count % 100 == 0:
                print(f"Episode: {episode_count}")

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
                game._Game__move(move[0], move[1], turn + 1)
                turn = 1 - turn
                counter += 1
            if game.check_winner() == self.player_number:
                rl_victories += 1
            game = Game()
        print(f"number of victories: {rl_victories} out of {n_matches}")
        return rl_victories / n_matches

# utility fnuctions

def apply_rotation(state_np, rotations):
    # Rotate the state by the specified number of rotations (clockwise)
    return np.rot90(state_np.reshape((5, 5)), k=rotations).flatten()

def apply_reflection(state_np, axis):
    # Reflect the state along the specified axis (0 for vertical, 1 for horizontal)
    return np.flip(state_np.reshape((5, 5)), axis=axis).flatten()

def apply_symmetry(state_str, rotations=0, reflection_axis=None):
    state_np = np.array([int(char) for char in state_str])

    # Apply rotations
    state_np = apply_rotation(state_np, rotations)

    # Apply reflection if specified
    if reflection_axis is not None:
        state_np = apply_reflection(state_np, axis=reflection_axis)

    return ''.join(map(str, state_np))

def print_state_board(state_str):
    state_np = np.array([int(char) for char in state_str])
    return state_np.reshape((5, 5))

def get_canonical_state(state_str):
    state_np = np.array([int(char) for char in state_str])
    state_str = ''.join(map(str, state_np))
    canonical_state = state_str
    transformations = (0, None)
    for rotations in range(4):
        for reflection_axis in [None, 0, 1]:
            sym_state = apply_symmetry(state_np, rotations, reflection_axis)
            if sym_state < canonical_state:
                canonical_state = sym_state
                transformations = (rotations, reflection_axis)

    return canonical_state, transformations

def convert_action_to_triplet(action):
        return ((action // 20) % 5, (action // 4) % 5), Move(action % 4)

def get_reverse_action(action, transformations):
    # Convert the action to its triplet form
    canonical_action = convert_action_to_triplet(action)

    # Extract the transformations
    rotations, reflection_axis = transformations

    # Reverse the reflection if it was applied
    if reflection_axis is not None:
        if reflection_axis == 1:  # Reflection along the vertical axis
            canonical_action = ((canonical_action[0][0], 4 - canonical_action[0][1]), canonical_action[1])
            if canonical_action[1] == Move.LEFT:
                canonical_action = (canonical_action[0], Move.RIGHT)
            elif canonical_action[1] == Move.RIGHT:
                canonical_action = (canonical_action[0], Move.LEFT)
        elif reflection_axis == 0:  # Reflection along the horizontal axis
            canonical_action = ((4 - canonical_action[0][0], canonical_action[0][1]), canonical_action[1])
            if canonical_action[1] == Move.TOP:
                canonical_action = (canonical_action[0], Move.BOTTOM)
            elif canonical_action[1] == Move.BOTTOM:
                canonical_action = (canonical_action[0], Move.TOP)

    # Reverse the rotations
    for _ in range(rotations % 4):
        canonical_action = ((canonical_action[0][1], 4 - canonical_action[0][0]), canonical_action[1])
        if canonical_action[1] == Move.TOP:
            canonical_action = (canonical_action[0], Move.RIGHT)
        elif canonical_action[1] == Move.RIGHT:
            canonical_action = (canonical_action[0], Move.BOTTOM)
        elif canonical_action[1] == Move.BOTTOM:
            canonical_action = (canonical_action[0], Move.LEFT)
        elif canonical_action[1] == Move.LEFT:
            canonical_action = (canonical_action[0], Move.TOP)

    return canonical_action

# if __name__ == '__main__':
# g = Game()
# g.print()
# player1 = MyPlayer()
# player2 = RandomPlayer()
# winner = g.play(player1, player2)
# g.print()
# print(f"Winner: Player {winner}")
