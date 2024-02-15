
# Quixo exam
Quixo is a game similar to tic-tac-toe, but with a 5x5 board and a different set of rules. 
The goal of Quixo is to be the first player to create a line of five of their own cubes, either horizontally, vertically, or diagonally.
Gameplay:
* Players take turns either:
  - Moving one of their cubes to an adjacent empty space, changing its orientation, or
  - Pushing an entire row or column of cubes, with the moved cube becoming the last in the row or column.
* The chosen action must result in a cube with its marked face up.
* The game alternates until a player achieves a line of five of their cubes in a row.

## Players 
### Monte carlo agent
It is a reinforcement learning agent based on a Monte Carlo technique that use a method called Temporal Difference method
To reduce the size of the table I created a dictionary based on 'defaultdict' but that overrides the __missing method that does not add a 
new key if it is not found to drastically reduce the final size of the q table
In addition to that i also employ some transformations on the board state to reduce the number of states to be stored in the table
More info about formula update, training and parameters can be found in the mc_agent_training jupyter notebook

### MinMax agent
The agent uses the minmax algorithm with alpha-beta pruning to make the best move. 
To reduce the computational aspect and the efficiency of the algorithm i decide to limit the maximum depth to 2 since the result are very good (almost 100% of win rate against a random player)
The agent uses a heuristic to evaluate the board and make the best move. 
The heuristic is based on the number of possible lines composed by 3 or 4 cubes, gives a score to the number of line with these characteristics.

### Q learning agent
I tried initially to develop an agent that uses a vanilla q learning approach.
Since the state space is very large I tried to reduce the number of them using some symmetries to reduce every state to its canonical form
Despite that the performance of the agent was very poor (just a little more than 50%) and the table was too large and not portable (about 19GB) 
so I don't include the training and the results

## Resource used
* [Reinforcement Learning: An Inreoduction by Richard S. Sutton and Andrew G. Barto](http://incompleteideas.net/book/RLbook2018.pdf)
* [Quixo is solved](https://www.researchgate.net/publication/343390362_Quixo_Is_Solved)
* [Essentials of Metaheuristics](https://cs.gmu.edu/~sean/book/metaheuristics/Essentials.pdf)
* Slides of the course
