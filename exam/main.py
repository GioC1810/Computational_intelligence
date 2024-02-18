from players.monte_carlo_player import MonteCarloPlayer
from players.minmax_player import MinMaxPlayer
from utils import Utilities, MC_TABLE_FIRST
from players.random_player import RandomPlayer


if __name__ == '__main__':

    monteCarloPlayer = MonteCarloPlayer(0,0,0,0)
    monteCarloPlayer.load_q_table(MC_TABLE_FIRST)
    mimMaxPlayer = MinMaxPlayer(depth=2)
    randomPlayer = RandomPlayer(2)

    print("Monte Carlo vs Random")
    monteCarloPlayer.test(100, randomPlayer)
    print("MinMax vs Random")
    winRate = Utilities.play_multiple_games(100, mimMaxPlayer, randomPlayer)
