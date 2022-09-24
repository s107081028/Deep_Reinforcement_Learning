def MCTS(observation, configuration):
    import time
    import math
    import random
    import numpy as np
    from copy import deepcopy
    init_time = time.time()
    EMPTY = 0
    T_max = configuration.timeout - 0.34
    global current_s
    def opponent(current):
        return 2 if current == 1 else 1

    class State:
        def __init__(self, board, player, winner = 0):
            self.board = board.copy()
            self.player = player
            self.winner = winner
        
        def get_available_actions(self):
            return [act for act in range(configuration.columns) if self.board[act] == 0]

        def check_terminate(self, action, has_played = True):
            columns = configuration.columns
            rows = configuration.rows
            inarow = configuration.inarow - 1
            sum = []
            for r in range(rows):
              sum.append(self.board[action + r * columns])
            if not opponent(self.player) in sum:
              has_played = False
            row = (
                min([r for r in range(rows) if self.board[action + (r * columns)] == opponent(self.player)])
                if has_played
                else max([r for r in range(rows) if self.board[action + (r * columns)] == EMPTY])
            )

            def count(offset_row, offset_column):
                for i in range(1, inarow + 1):
                    r = row + offset_row * i
                    c = action + offset_column * i
                    if (
                        r < 0
                        or r >= rows
                        or c < 0
                        or c >= columns
                        or self.board[c + (r * columns)] != opponent(self.player)
                    ):
                        return i - 1
                return inarow

            if (
                count(1, 0) >= inarow  # vertical.
                or (count(0, 1) + count(0, -1)) >= inarow  # horizontal.
                or (count(-1, -1) + count(1, 1)) >= inarow  # top left diagonal.
                or (count(-1, 1) + count(1, -1)) >= inarow  # top right diagonal.
            ):
                self.winner = opponent(self.player)
            elif not EMPTY in self.board:
              self.winner = 3

        def next_state(self, action):
            for i in range(configuration.rows - 1, -1, -1):
                if(self.board[i * configuration.columns + action] == 0):
                    self.board[i * configuration.columns + action] = self.player
                    self.player = opponent(self.player)
                    break

    class Node:
        def __init__(self, state: State, parent = None):
            self.state = deepcopy(state)
            self.available_actions = state.get_available_actions()
            self.parent = parent  # parent is input
            self.children = {}    # value is child node, key is action
            self.Q = 0
            self.visited = 0
        
        def UCB_func(self, c_param = 1.4):
            if self.visited == 0:
                return math.inf
            return -self.Q / self.visited + c_param * np.sqrt(2 * np.log(self.parent.visited) / self.visited)
        
        def random_action(self, available_actions):
            return available_actions[np.random.choice(range(len(available_actions)))]
        
        def select_best(self, c_param = 1.4):
            """use list of UCBs to select best"""
            Max = -1
            action  = None
            for child in self.children.keys():
                temp = self.children[child].UCB_func(c_param)
                if temp > Max:
                    Max = temp
                    action = child
            return action, self.children[action]

        def expand(self):
            action = self.available_actions.pop()
            current = self.state.player
            next_board = self.state.board.copy()
            for i in range(configuration.rows - 1, -1, -1):
                if(next_board[i * configuration.columns + action] == 0):
                    next_board[i * configuration.columns + action] = current
                    break
            next_player = opponent(current)
            state = State(next_board, next_player)
            state.check_terminate(action)
            child = Node(state, self)
            self.children[action] = child
            return child

        def not_root(self):
            return False if self.parent == None else True

        def update(self, champian):
            self.visited += 1
            oppo = opponent(self.state.player)
            if champian == self.state.player:
                self.Q += 1
            elif champian == oppo:
                self.Q -= 1
            
            if self.not_root():
                self.parent.update(champian)
        
        def roll_out(self):
            current_state = deepcopy(self.state)
            while True:
                if current_state.winner:
                    return current_state.winner
                available_actions = current_state.get_available_actions()
                action = self.random_action(available_actions)
                current_state.next_state(action)
                current_state.check_terminate(action)

        def fully_expanded(self):
            return len(self.available_actions) == 0

    def simulation():
        leaf = simulation_policy()
        winner = leaf.roll_out()
        leaf.update(winner)

    def simulation_policy():
        current = current_s
        while True:
            if current.state.winner:
                break
            if current.fully_expanded():
                _, current = current.select_best()
            else:
                return current.expand()
        leaf = current
        return leaf

    CurrentState = State(observation.board, observation.mark, 0)
    current_s = Node(CurrentState)
    while time.time() - init_time <= T_max:
        simulation()
    action, next_ = current_s.select_best(0.0)
    return action
